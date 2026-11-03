"""D19：Dense 检索（Qdrant）+ 与精确检索基准对照。

验收（§17.3）：同一 query 的 Qdrant Top50 ∩ numpy 精确 Top50 计算 ANN Recall；
1 万点 ≥0.99、10 万点 ≥0.95。

为什么必须留着 numpy 暴力实现：**没有标准答案就谈不上 ANN 的召回**。
Qdrant 的 HNSW 是近似算法，`ef` 调小会变快也会丢结果，而"丢了"这件事
在返回结果里完全看不出来——只有和精确 Top-K 求交集才知道。

两个容易踩的实现细节：
1. 精确检索必须用**同一批向量**（从 Qdrant 里 scroll 出来），不能用别的来源，
   否则比的是两个不同的数据集；
2. 余弦相似度要先归一化再点积，Qdrant 的 Cosine 距离返回的是 **similarity**
   （不是距离），比较时别把分数当距离排序。
"""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.rag.store.search import (
    SearchResult,
    ann_recall,
    exact_topk,
    normalize_rows,
    qdrant_search,
)
from src.rag.store.qdrant_store import VECTOR_SIZE, make_client, point_id_for, qdrant_url

SCROLL = 1000


def _server_available() -> bool:
    try:
        make_client(timeout=3).get_collections()
        return True
    except Exception:
        return False


SERVER = _server_available()


class PrecisionTest(unittest.TestCase):
    def test_exact_topk_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        matrix = normalize_rows(rng.normal(size=(50, 8)).astype(np.float32))
        query = normalize_rows(rng.normal(size=(1, 8)).astype(np.float32))[0]
        got = exact_topk(matrix, query, k=5)
        scores = matrix @ query
        expected = np.argsort(-scores)[:5].tolist()
        self.assertEqual(got.ids, expected)
        np.testing.assert_allclose(got.scores, scores[expected], rtol=1e-6)

    def test_exact_topk_handles_k_larger_than_n(self):
        rng = np.random.default_rng(1)
        matrix = normalize_rows(rng.normal(size=(3, 8)).astype(np.float32))
        query = normalize_rows(rng.normal(size=(1, 8)).astype(np.float32))[0]
        self.assertEqual(len(exact_topk(matrix, query, k=10).ids), 3)

    def test_normalize_rows_is_safe_for_zero_vectors(self):
        matrix = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        out = normalize_rows(matrix)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out[1], [0.6, 0.8], rtol=1e-6)


class RecallMetricTest(unittest.TestCase):
    def test_full_overlap_is_one(self):
        self.assertEqual(ann_recall([1, 2, 3], [1, 2, 3]), 1.0)

    def test_partial_overlap(self):
        self.assertEqual(ann_recall([1, 2, 3], [2, 3, 4]), 2 / 3)

    def test_empty_exact_set_is_zero_not_crash(self):
        self.assertEqual(ann_recall([1, 2], []), 0.0)

    def test_ann_recall_reports_against_exact_size(self):
        """分母用精确结果的条数：ANN 少返回时不能被算成满分。"""
        self.assertEqual(ann_recall([1], [1, 2, 3, 4]), 0.25)


@unittest.skipUnless(SERVER, f"Qdrant 服务端不可达（{qdrant_url()}），跳过 D19 服务端用例")
class ServerSearchTest(unittest.TestCase):
    def setUp(self):
        from qdrant_client import models

        self.models = models
        # 10k 点 ×1024 维走 REST JSON 上传要十几秒，超时给小了会整批失败（实测）
        self.client = make_client(timeout=180)
        self.collection = f"rag_chunks_d19_{uuid.uuid4().hex[:8]}"
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
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

    def _seed(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        matrix = normalize_rows(rng.normal(size=(n, VECTOR_SIZE)).astype(np.float32))
        ids = [point_id_for(f"chk_{uuid.uuid5(uuid.NAMESPACE_URL, f'{self.collection}/{i}').hex}") for i in range(n)]
        payloads = [{"chunk_id": f"chunk-{i}", "document_id": f"doc-{i % 5}", "text": f"文本 {i}"} for i in range(n)]
        for start in range(0, n, 500):
            self.client.upsert(
                self.collection,
                points=[self.models.PointStruct(id=ids[i], vector=matrix[i].tolist(), payload=payloads[i])
                        for i in range(start, min(start + 500, n))],
                wait=True,
            )
        return matrix, ids, payloads

    def _scroll_all(self):
        points, offset = [], None
        while True:
            batch, offset = self.client.scroll(
                self.collection, limit=SCROLL, offset=offset, with_payload=True, with_vectors=False)
            points.extend(batch)
            if offset is None:
                return points

    def test_search_returns_payload_and_score(self):
        matrix, _ids, payloads = self._seed(50)
        hits = qdrant_search(self.client, self.collection, matrix[0], k=5)
        self.assertEqual(len(hits), 5)
        self.assertIsInstance(hits[0], SearchResult)
        self.assertIn(hits[0].chunk_id, {p["chunk_id"] for p in payloads})
        self.assertGreaterEqual(hits[0].score, hits[-1].score, "分数必须降序")

    def test_top1_is_the_planted_vector(self):
        """种一个「完美匹配」进去：query == 该向量，Top1 必须是它。"""
        matrix, _ids, _payloads = self._seed(30)
        query = matrix[7].copy()
        hits = qdrant_search(self.client, self.collection, query, k=3)
        exact = exact_topk(matrix, query, k=3)
        self.assertEqual(hits[0].chunk_id, f"chunk-{exact.ids[0]}")

    def test_filter_by_document(self):
        from qdrant_client import models

        matrix, _ids, _ = self._seed(40)
        hits = qdrant_search(
            self.client, self.collection, matrix[0], k=10,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="document_id", match=models.MatchValue(value="doc-3"))]),
        )
        self.assertTrue(hits)
        self.assertTrue(all(hit.payload["document_id"] == "doc-3" for hit in hits))

    def test_ann_recall_is_high_on_10k(self):
        """D19 的核心数字：1 万点时 Qdrant Top50 与精确 Top50 的召回 ≥0.99。"""
        matrix, ids, _ = self._seed(10_000, seed=7)
        recalls = []
        for q in range(10):
            query = matrix[q * 137]
            hits = qdrant_search(self.client, self.collection, query, k=50)
            got = [point_id_index(self.client, self.collection, hit.point_id) for hit in hits]
            exact = exact_topk(matrix, query, k=50)
            recalls.append(ann_recall(got, exact.ids))
        self.assertGreaterEqual(float(np.mean(recalls)), 0.99, f"实测召回 {np.mean(recalls):.3f}")


def point_id_index(client, collection, point_id: str) -> int:
    """把 point id 映射回种数据时的下标（测试辅助）。"""
    from qdrant_client import models

    hit = client.retrieve(collection, ids=[point_id], with_payload=["chunk_id"])
    return int(hit[0].payload["chunk_id"].split("-")[1])


if __name__ == "__main__":
    unittest.main()
