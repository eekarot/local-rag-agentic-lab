"""`eval_bench` 的计分口径：**必须用父块 id**。

BUG 实证（N62）：N60 把 `retrieve_with_meta()` 的 `hit.chunk_id` 从「父块 id 顶替」
改成了「真子块 id」，这本身是对的；但 `scripts/eval_bench.py` 一直用
`[hit.chunk_id for hit in outcome.hits]` 去和 `gold_chunks`（父块 id，`par_…`）比，
于是 180 题**全题不命中**：hit@10 0.8571 → 0.0000，而默认套件里没有一条用例
能把这两件事对上（评估器的计分口径没被任何用例钉住）。

这个文件就钉它：同一道题同时给出 `gold_chunks`（父块）与 `gold_points`（子块），
只要计分用错了 id 空间，`hit` 立刻变 False。
"""

from __future__ import annotations

import unittest


class _FakeCounter:
    def count(self, text):
        return len(str(text).split())

    def count_batch(self, texts):
        return [self.count(t) for t in texts]


def _retriever(hits_by_route):
    class _Retriever:
        def retrieve(self, query, k, query_en=None):
            from src.rag.retrieval.orchestrator import RetrievalResult
            from src.rag.retrieval.routes import RouteHit, RouteResult

            routes = {
                name: RouteResult(route=name, enabled=True, candidates=len(hits),
                                  hits=[RouteHit(parent_id=p, chunk_id=c, score=s, route=name)
                                        for p, c, s in hits])
                for name, hits in hits_by_route.items()
            }
            return RetrievalResult(query_zh=query, query_en=query_en, k=k, routes=routes)

    return _Retriever()


QUESTION = {
    "qid": "k8s-fake-001",
    "question": "测试题",
    "type": "single_hop",
    "expected_evidence_lang": "zh-cn",
    # 两套 id 空间互不相交：父块 par_A / 子块 chk_A1
    "gold_chunks": ["par_A"],
    "gold_points": ["chk_A1"],
}


class EvalBenchScoresByParentIdTest(unittest.TestCase):
    def _run(self, hits_by_route):
        import scripts.eval_bench as bench

        original = bench._build_retriever
        bench._build_retriever = lambda *a, **kw: _retriever(hits_by_route)
        try:
            return bench.evaluate(
                [dict(QUESTION)], k=10, pool=10, rerank=False, warmup=False,
                parents={"par_A": {"text": "父块正文"}}, counter=_FakeCounter(), reranker=None,
            )
        finally:
            bench._build_retriever = original

    def test_retrieved_ids_are_parent_ids(self):
        result = self._run({"dense_zh": [("par_A", "chk_A1", 0.9)]})
        row = result["rows"][0]
        self.assertEqual(row["retrieved"], ["par_A"],
                         "计分必须落在父块 id 上：gold_chunks 是父块，子块 id 是另一套空间")
        self.assertTrue(row["hit"], "命中的就是 gold 父块，hit 必须为 True")
        self.assertEqual(result["overall"]["hit@10"], 1.0)

    def test_child_id_that_matches_gold_points_only_is_not_a_hit(self):
        """子块 id 命中 `gold_points` 但父块不是金标父块 ⇒ 这一题就该算没中。"""
        result = self._run({"dense_zh": [("par_other", "chk_A1", 0.9)]})
        row = result["rows"][0]
        self.assertEqual(row["retrieved"], ["par_other"])
        self.assertFalse(row["hit"], "别把 gold_points 当成计分空间：那是子块 id")

    def test_negative_question_recall_is_none(self):
        question = {"qid": "k8s-fake-neg", "question": "测试反例", "type": "negative",
                    "expected_evidence_lang": None, "gold_chunks": [], "gold_points": []}
        import scripts.eval_bench as bench

        original = bench._build_retriever
        bench._build_retriever = lambda *a, **kw: _retriever(
            {"dense_zh": [("par_A", "chk_A1", 0.9)]})
        try:
            result = bench.evaluate([question], k=10, pool=10, rerank=False, warmup=False,
                                    parents={"par_A": {"text": "父块正文"}},
                                    counter=_FakeCounter(), reranker=None)
        finally:
            bench._build_retriever = original
        self.assertIsNone(result["rows"][0]["recall"])


if __name__ == "__main__":
    unittest.main()
