"""D16：幂等 Upsert + 文档版本管理 + 按文档删除/重建。

验收（方案 §17.3）：
- 同一文档导入两次后 collection 点数不变（重复点 == 0）；
- 删除后按 `document_id` 过滤计数 == 0。

幂等的关键不是「先删后插」，而是**同一个 chunk 永远落在同一个 point id**
（`point_id_for` 的 UUID5），所以重复导入是覆盖而不是新增。
版本变化时新旧 chunk id 不同，靠「本文件已存在的点集 − 本次点集」算出陈旧点再删。
"""

import unittest
import uuid

import numpy as np

from src.rag.embedding.service import EmbeddingService
from src.rag.store.document_store import DocumentStore
from src.rag.store.indexer import Indexer
from src.rag.store.qdrant_store import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    collection_config,
    make_client,
    point_id_for,
    qdrant_url,
)


def _server_available() -> bool:
    try:
        make_client(timeout=3).get_collections()
        return True
    except Exception:
        return False


SERVER = _server_available()
DOC_A = "doc_" + "a" * 32
DOC_B = "doc_" + "b" * 32


class FakeBackend:
    """记录被编码的文本，返回确定性向量（D16 不该真的调模型）。"""

    def __init__(self):
        self.seen: list[str] = []

    def embed(self, texts):
        self.seen.extend(texts)
        out = []
        for text in texts:
            vec = np.zeros(VECTOR_SIZE, dtype=np.float32)
            vec[0] = 1.0
            vec[1] = (sum(map(ord, text)) % 97) / 97.0
            out.append(vec.tolist())
        return out


def make_document(document_id=DOC_A, version="sha256:v1", source="md/a.md"):
    return {
        "document_id": document_id,
        "source_uri": source,
        "file_name": source.split("/")[-1],
        "file_type": "md",
        "title": "Doc A",
        "language": "en",
        "document_version": version,
        "parser_version": "md_v1",
        "created_at": "2026-11-02T00:00:00Z",
        "status": "active",
    }


def make_parents(document_id=DOC_A, version="sha256:v1", n=3, tag="v1"):
    return [
        {
            "parent_id": "par_" + uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}/{tag}/{i}").hex,
            "document_id": document_id,
            "document_version": version,
            "language": "en",
            "title": f"S{i}",
            "section_path": ["Retrieval", f"S{i}"],
            "content": f"父块 {tag}-{i} 原文",
            "token_count": 100 + i,
            "source_start": i * 100,
            "source_end": i * 100 + 99,
            "chunker_version": "recursive_parent_child_v1",
            "parent_index": i,
        }
        for i in range(n)
    ]


def make_children(parents, version="sha256:v1", per_parent=2, tag="v1"):
    children = []
    for index, parent in enumerate(parents):
        for k in range(per_parent):
            key = f"{parent['parent_id']}/{tag}/{k}"
            children.append({
                "chunk_id": "chk_" + uuid.uuid5(uuid.NAMESPACE_URL, key).hex,
                "parent_id": parent["parent_id"],
                "document_id": parent["document_id"],
                "document_version": version,
                "language": "en",
                "title": parent["title"],
                "section_path": parent["section_path"],
                "content": f"子块 {tag}-{index}-{k} 正文",
                "embedding_text": f"Document: Doc A\nSection: Retrieval / S{index}\nContent: 子块 {tag}-{index}-{k} 正文",
                "token_count": 50 + k,
                "source_start": parent["source_start"] + k * 10,
                "source_end": parent["source_start"] + k * 10 + 9,
                "chunker_version": "recursive_child_v1",
                "embedding_model": "bge-m3",
                "embedding_version": "bge-m3-v1",
                "content_hash": f"sha256:{key}",
                "status": "active",
            })
    return children


@unittest.skipUnless(SERVER, f"Qdrant 服务端不可达（{qdrant_url()}），跳过 D16 用例")
class IndexerTestBase(unittest.TestCase):
    def setUp(self):
        self.client = make_client(timeout=15)
        self.collection = f"rag_chunks_d16_{uuid.uuid4().hex[:8]}"
        self.tmp_dir = __import__("tempfile").TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.store = DocumentStore(f"{self.tmp_dir.name}/d.sqlite3")
        self.backend = FakeBackend()
        self.embedder = EmbeddingService(
            backend=self.backend, cache_path=f"{self.tmp_dir.name}/c.sqlite3"
        )
        self.indexer = Indexer(
            client=self.client, collection=self.collection, store=self.store, embedder=self.embedder
        )
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        try:
            names = [c.name for c in self.client.get_collections().collections]
            if self.collection in names:
                self.client.delete_collection(self.collection)
        except Exception:
            pass
        finally:
            self.client.close()
            self.store.close()

    def index(self, document=None, parents=None, children=None, **kwargs):
        document = document or make_document()
        parents = parents if parents is not None else make_parents()
        children = children if children is not None else make_children(parents)
        return self.indexer.index_document(document, parents, children, **kwargs)


class BasicIndexTest(IndexerTestBase):
    def test_one_point_per_child(self):
        parents = make_parents(n=3)
        children = make_children(parents, per_parent=2)
        result = self.index(make_document(), parents, children)
        self.assertEqual(result["upserted"], 6)
        self.assertEqual(self.indexer.count(), 6)
        self.assertEqual(result["deleted_stale"], 0)

    def test_point_id_is_deterministic_uuid5(self):
        parents = make_parents(n=1)
        children = make_children(parents, per_parent=2)
        self.index(make_document(), parents, children)
        hits = self.client.retrieve(self.collection, ids=[point_id_for(children[0]["chunk_id"])], with_payload=True)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].payload["chunk_id"], children[0]["chunk_id"])
        self.assertEqual(hits[0].payload["text"], children[0]["content"])

    def test_payload_has_no_null_fields(self):
        parents = make_parents(n=1)
        children = make_children(parents, per_parent=1)
        children[0]["language"] = None
        self.index(make_document(), parents, children)
        hit = self.client.retrieve(self.collection, ids=[point_id_for(children[0]["chunk_id"])],
                                   with_payload=True)[0]
        self.assertNotIn("language", hit.payload)

    def test_embeds_embedding_text_not_content(self):
        parents = make_parents(n=1)
        children = make_children(parents, per_parent=1)
        self.index(make_document(), parents, children)
        self.assertEqual(self.backend.seen, [children[0]["embedding_text"]])

    def test_provided_vectors_bypass_embedder(self):
        parents = make_parents(n=2)
        children = make_children(parents, per_parent=1)
        vectors = {c["chunk_id"]: np.eye(1, VECTOR_SIZE, 0).ravel() for c in children}
        self.index(make_document(), parents, children, vectors=vectors)
        self.assertEqual(self.backend.seen, [], "给了向量就不该调模型")
        self.assertEqual(self.indexer.count(), 2)

    def test_child_from_other_document_rejected(self):
        parents = make_parents(n=1)
        children = make_children(parents, per_parent=1)
        children[0]["document_id"] = DOC_B
        with self.assertRaises(ValueError):
            self.index(make_document(), parents, children)


class IdempotencyTest(IndexerTestBase):
    def test_reindex_same_content_does_not_add_points(self):
        """验收：同一文档导入两次后 collection 点数不变（重复点 == 0）。"""
        parents = make_parents(n=3)
        children = make_children(parents, per_parent=2)
        self.index(make_document(), parents, children)
        before = self.indexer.count()
        result = self.index(make_document(), make_parents(n=3), children)
        after = self.indexer.count()
        self.assertEqual(before, after, "重复导入不允许新增点")
        self.assertEqual(after, 6)
        self.assertEqual(result["deleted_stale"], 0, "内容没变就不该删任何点")

    def test_reindex_is_cached_in_embedding_layer(self):
        parents = make_parents(n=2)
        children = make_children(parents, per_parent=2)
        self.index(make_document(), parents, children)
        calls = len(self.backend.seen)
        self.index(make_document(), parents, children)
        self.assertEqual(len(self.backend.seen), calls, "重复导入应命中 embedding 缓存")

    def test_version_change_removes_stale_points(self):
        parents_v1 = make_parents(n=2, tag="v1")
        children_v1 = make_children(parents_v1, per_parent=2, tag="v1")
        self.index(make_document(), parents_v1, children_v1)
        old_ids = {point_id_for(c["chunk_id"]) for c in children_v1}

        parents_v2 = make_parents(n=3, version="sha256:v2", tag="v2")
        children_v2 = make_children(parents_v2, version="sha256:v2", per_parent=2, tag="v2")
        result = self.index(make_document(version="sha256:v2"), parents_v2, children_v2)

        self.assertEqual(self.indexer.count(), 6)
        self.assertEqual(result["deleted_stale"], 4, "旧的 4 个点必须被删掉，否则成为孤儿向量")
        left = self.client.retrieve(self.collection, ids=list(old_ids), with_payload=True)
        self.assertEqual(left, [])

    def test_store_version_follows_index(self):
        parents_v2 = make_parents(n=2, version="sha256:v2", tag="v2")
        children_v2 = make_children(parents_v2, version="sha256:v2", per_parent=1, tag="v2")
        self.index(make_document(version="sha256:v2"), parents_v2, children_v2)
        doc = self.store.get_document(DOC_A)
        self.assertEqual(doc["document_version"], "sha256:v2")
        # 任意命中都能回溯到父块原文
        row = self.store.get_parent_for_chunk(children_v2[0]["chunk_id"])
        self.assertEqual(row["content"], parents_v2[0]["content"])


class DeleteTest(IndexerTestBase):
    def test_delete_document_leaves_zero_points(self):
        """验收：删除后按 document_id 过滤计数 == 0。"""
        parents = make_parents(n=2)
        children = make_children(parents, per_parent=2)
        self.index(make_document(), parents, children)
        self.assertEqual(self.indexer.count(DOC_A), 4)
        self.indexer.delete_document(DOC_A)
        self.assertEqual(self.indexer.count(DOC_A), 0)
        self.assertEqual(self.indexer.count(), 0)
        self.assertIsNone(self.store.get_document(DOC_A))

    def test_delete_only_affects_target_document(self):
        parents_a = make_parents(n=1)
        children_a = make_children(parents_a, per_parent=2)
        self.index(make_document(DOC_A, source="md/a.md"), parents_a, children_a)

        parents_b = make_parents(n=1)
        parents_b[0]["parent_id"] = "par_" + "c" * 32
        parents_b[0]["document_id"] = DOC_B
        children_b = make_children(parents_b, per_parent=2, tag="b")
        self.index(make_document(DOC_B, source="md/b.md"), parents_b, children_b)

        self.assertEqual(self.indexer.count(), 4)
        self.indexer.delete_document(DOC_A)
        self.assertEqual(self.indexer.count(), 2)
        self.assertEqual(self.indexer.count(DOC_B), 2)

    def test_reindex_after_delete_restores_points(self):
        parents = make_parents(n=2)
        children = make_children(parents, per_parent=1)
        self.index(make_document(), parents, children)
        self.indexer.delete_document(DOC_A)
        self.index(make_document(), parents, children)
        self.assertEqual(self.indexer.count(), 2)


class BatchTest(IndexerTestBase):
    def test_small_batches_still_complete(self):
        indexer = Indexer(
            client=self.client, collection=self.collection, store=self.store,
            embedder=self.embedder, batch_size=2,
        )
        parents = make_parents(n=3)
        children = make_children(parents, per_parent=2)
        indexer.index_document(make_document(), parents, children)
        self.assertEqual(indexer.count(), 6)
        self.assertEqual(indexer.count(DOC_A), 6)


if __name__ == "__main__":
    unittest.main()
