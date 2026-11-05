"""D15：Qdrant 部署形态、Collection 配置与 Payload Index。

三条实测事实决定了本文件的写法（不是猜的）：

1. **本地模式（`:memory:` / path）的 payload index 是空转**，client 会直接警告
   「Payload indexes have no effect in the local Qdrant」。所以「6 个索引都存在」
   只能在**服务端**上验，本地模式只验 collection 配置。
2. **Qdrant 只接受 UUID 或整数作为 point id**，而我们的 `chunk_id` 形如
   `chk_<32 hex>`，直接塞进去会被 400 拒绝。必须用 `point_id_for()` 做确定性映射
   （UUID5），否则 D16 的「同一 chunk 重复导入不增点」无从保证。
3. **`ef_construct` 是 collection 级参数**，写进 `VectorParams(hnsw_config=...)`
   会被静默忽略（实测仍是默认 100），验收要求 `ef_construct=200` 就必须放在
   `create_collection(hnsw_config=...)`。
"""

import os
import unittest
import uuid

from src.rag.store.qdrant_store import (
    COLLECTION_NAME,
    EF_CONSTRUCT,
    HNSW_M,
    PAYLOAD_INDEXES,
    VECTOR_SIZE,
    CollectionMismatchError,
    collection_config,
    ensure_collection,
    make_client,
    payload_from_child,
    point_id_for,
    qdrant_url,
)
from tests._slow import run_slow, slow


def _server_available() -> bool:
    try:
        client = make_client(timeout=3)
        client.get_collections()
        return True
    except Exception:
        return False


SERVER = run_slow() and _server_available()   # 默认不探：省 3 s 且不联网


class PointIdTest(unittest.TestCase):
    """chunk_id → Qdrant point id 的确定性映射（不依赖服务端）。"""

    def test_is_valid_uuid(self):
        pid = point_id_for("chk_" + "a" * 32)
        uuid.UUID(pid)  # 解析失败会抛 ValueError

    def test_deterministic(self):
        cid = "chk_" + "b" * 32
        self.assertEqual(point_id_for(cid), point_id_for(cid))

    def test_different_chunks_get_different_points(self):
        a = point_id_for("chk_" + "a" * 32)
        b = point_id_for("chk_" + "b" * 32)
        self.assertNotEqual(a, b)

    def test_namespace_separated_from_bare_uuid5(self):
        """不能直接用裸 chunk_id 做 UUID5：不同项目/前缀会撞同一个 UUID。"""
        cid = "chk_" + "c" * 32
        self.assertNotEqual(point_id_for(cid), str(uuid.uuid5(uuid.NAMESPACE_URL, cid)))


class PayloadTest(unittest.TestCase):
    """Child 记录 → Qdrant payload。"""

    def _child(self):
        return {
            "chunk_id": "chk_" + "d" * 32,
            "parent_id": "par_" + "e" * 32,
            "document_id": "doc_" + "f" * 32,
            "document_version": "sha256:abc",
            "language": "en",
            "title": "Agent Guide",
            "section_path": ["Retrieval", "HNSW"],
            "content": "HNSW uses a hierarchical graph.",
            "embedding_text": "Document: Agent Guide\nSection: Retrieval > HNSW\nContent: HNSW uses...",
            "token_count": 9,
            "source_start": 100,
            "source_end": 133,
            "chunker_version": "recursive_child_v1",
            "embedding_model": "bge-m3",
            "embedding_version": "bge-m3-v1",
            "content_hash": "sha256:def",
            "status": "active",
        }

    def test_text_comes_from_content(self):
        payload = payload_from_child(self._child())
        self.assertEqual(payload["text"], "HNSW uses a hierarchical graph.")

    def test_embedding_text_is_not_duplicated_into_payload(self):
        """embedding_text 只用于编码；存进 payload 会让 Qdrant 里多一份重复文本。"""
        self.assertNotIn("embedding_text", payload_from_child(self._child()))

    def test_filter_fields_present(self):
        payload = payload_from_child(self._child())
        for field in ("document_id", "language", "status", "chunk_id", "parent_id"):
            self.assertIn(field, payload, f"过滤字段 {field} 缺失")

    def test_section_path_kept_as_list(self):
        payload = payload_from_child(self._child())
        self.assertEqual(payload["section_path"], ["Retrieval", "HNSW"])

    def test_none_fields_are_omitted(self):
        """None 值写进去能存但不能用 MatchValue 过滤（实测），所以直接剔除。"""
        child = self._child()
        child["language"] = None
        payload = payload_from_child(child)
        self.assertNotIn("language", payload)
        self.assertIn("document_id", payload)


class SpecConstantsTest(unittest.TestCase):
    """常量必须与方案 D15 验收口径一致，改动要落在文档里。"""

    def test_collection_name(self):
        self.assertEqual(COLLECTION_NAME, "rag_chunks_v1")

    def test_vector_params(self):
        self.assertEqual(VECTOR_SIZE, 1024)
        self.assertEqual((HNSW_M, EF_CONSTRUCT), (16, 200))

    def test_six_payload_indexes(self):
        self.assertEqual(len(PAYLOAD_INDEXES), 6)
        self.assertEqual(
            sorted(PAYLOAD_INDEXES),
            ["document_id", "document_type", "language", "status", "subject", "tenant_id"],
        )


@unittest.skipUnless(SERVER, f"Qdrant 服务端不可达（{qdrant_url()}），跳过服务端用例")
@slow
class ServerCollectionTest(unittest.TestCase):
    def setUp(self):
        self.client = make_client(timeout=10)
        # 每个用例独占一个 collection 名：同名 collection 删除后立刻重建会 400
        # （"Collection data already exists at ./storage/collections/<name>"，目录残留），
        # 测试里不做这种竞态，产品路径的 recreate 由 _wait_collection_gone 负责。
        self.collection = f"rag_chunks_test_{uuid.uuid4().hex[:8]}"
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        # 清理失败不能掩盖测试结论：删不掉就留着（名字带随机后缀，不会互相干扰）
        try:
            if self.collection in [c.name for c in self.client.get_collections().collections]:
                self.client.delete_collection(self.collection)
        except Exception:
            pass
        finally:
            self.client.close()

    def test_creates_spec_compliant_collection(self):
        ensure_collection(self.client, collection=self.collection)
        cfg = collection_config(self.client, self.collection)
        self.assertEqual(cfg["vector_size"], 1024)
        self.assertEqual(cfg["distance"], "Cosine")
        self.assertEqual(cfg["hnsw_m"], 16)
        self.assertEqual(cfg["ef_construct"], 200, "ef_construct 必须建在 collection 级，否则被静默忽略成 100")

    def test_creates_six_payload_indexes(self):
        ensure_collection(self.client, collection=self.collection)
        cfg = collection_config(self.client, self.collection)
        self.assertEqual(sorted(cfg["payload_indexes"]), sorted(PAYLOAD_INDEXES))

    def test_idempotent(self):
        ensure_collection(self.client, collection=self.collection)
        ensure_collection(self.client, collection=self.collection)  # 不应抛错
        names = [c.name for c in self.client.get_collections().collections]
        self.assertEqual(names.count(self.collection), 1)

    def test_recreate_drops_existing_points(self):
        from qdrant_client import models

        ensure_collection(self.client, collection=self.collection)
        self.client.upsert(
            self.collection,
            points=[models.PointStruct(id=point_id_for("chk_" + "1" * 32), vector=[0.1] * 1024,
                                       payload={"document_id": "doc_x"})],
        )
        self.assertEqual(collection_config(self.client, self.collection)["points_count"], 1)
        ensure_collection(self.client, collection=self.collection, recreate=True)
        self.assertEqual(collection_config(self.client, self.collection)["points_count"], 0)

    def test_recreate_survives_transient_delete_failure(self):
        """删除偶发瞬时失败必须重试（实测整套测试连跑时会出现）。"""
        ensure_collection(self.client, collection=self.collection)
        original = self.client.delete_collection
        state = {"failed": 0}

        def flaky(collection_name, **kwargs):
            if state["failed"] == 0:
                state["failed"] += 1
                raise RuntimeError("transient: collection is being flushed")
            return original(collection_name, **kwargs)

        self.client.delete_collection = flaky  # type: ignore[method-assign]
        ensure_collection(self.client, collection=self.collection, recreate=True)
        self.assertEqual(state["failed"], 1, "应该恰好失败一次后被重试吞掉")
        self.assertEqual(collection_config(self.client, self.collection)["points_count"], 0)

    def test_vector_size_mismatch_is_rejected(self):
        """D21 的地基：换 embedding 模型（维度变了）不能往旧 collection 里写。"""
        ensure_collection(self.client, collection=self.collection)
        with self.assertRaises(CollectionMismatchError) as ctx:
            ensure_collection(self.client, collection=self.collection, vector_size=768)
        self.assertIn("1024", str(ctx.exception))

    def test_filtered_search_returns_only_matching_document(self):
        from qdrant_client import models

        ensure_collection(self.client, collection=self.collection)
        vec = [0.1] * 1024
        points = [
            models.PointStruct(id=point_id_for("chk_" + str(i) * 32), vector=vec,
                               payload={"document_id": doc, "language": "en", "text": f"t{i}"})
            for i, doc in enumerate(["doc_a", "doc_a", "doc_b", "doc_c"])
        ]
        self.client.upsert(self.collection, points=points)
        hits = self.client.query_points(
            self.collection, query=vec, limit=10,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="document_id", match=models.MatchValue(value="doc_b"))]),
        ).points
        self.assertEqual(len(hits), 1, "payload 过滤没生效（本地模式下索引是空转，必须连服务端）")
        self.assertEqual(hits[0].payload["document_id"], "doc_b")


class MakeClientProxyTest(unittest.TestCase):
    """httpx 默认信任系统代理（Windows 注册表），本机 Qdrant 会被绕到代理上 → 随机超时。"""

    def test_local_urls_do_not_use_system_proxy(self):
        from src.rag.store.qdrant_store import is_local_url, make_client

        self.assertTrue(is_local_url("http://localhost:6333"))
        self.assertTrue(is_local_url("http://127.0.0.1:6333"))
        self.assertTrue(is_local_url("http://127.1.2.3:6333"))
        self.assertFalse(is_local_url("https://qdrant.example.com:6333"))

        client = make_client("http://localhost:6333", timeout=1)
        try:
            # qdrant-client 1.16 的层级：QdrantClient._client(QdrantRemote).openapi_client(SyncApis).client(httpx)
            httpx_client = client._client.openapi_client.client._client
            self.assertFalse(httpx_client.trust_env, "本机地址必须直连")
        finally:
            client.close()

    def test_remote_url_keeps_environment_proxy(self):
        from src.rag.store.qdrant_store import make_client

        client = make_client("https://qdrant.example.com:6333", timeout=1)
        try:
            self.assertTrue(client._client.openapi_client.client._client.trust_env, "远端可能真的需要代理")
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
