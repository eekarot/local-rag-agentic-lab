"""D24：三路召回编排（zh dense / en dense / en BM25）。

设计要点（决定了测试长什么样）：
- 每个 Route 只知道「拿一段文本检索」，**不关心文本是中文还是英文**；哪一路用哪段文本
  由编排层决定。这样"关掉一路"不影响其它路的输入，天然满足
  「任一路可单独关闭且其余路结果与单独运行逐条一致」。
- Route 的协作者用可调用对象注入（`embedder` / `searcher` / `index`），
  所以不用起 Qdrant 就能测编排逻辑。
"""

import unittest

import numpy as np


def _fake_rows(*rows):
    """searcher 返回值：(chunk_id, parent_id, score)。"""

    def searcher(_vector, k):
        return list(rows)[:k]

    return searcher


class DenseRouteTest(unittest.TestCase):
    def _route(self, rows, name="dense_zh", pool_factor=4, input_lang="zh"):
        from src.rag.retrieval.routes import DenseRoute

        seen = {}

        def embed(texts):
            seen["texts"] = list(texts)
            return np.zeros((len(texts), 4), dtype=np.float32)

        def searcher(_vector, k):
            seen["k"] = k
            return list(rows)[:k]

        route = DenseRoute(name=name, embedder=embed, searcher=searcher,
                           pool_factor=pool_factor, input_lang=input_lang)
        return route, seen

    def test_collapses_chunks_to_parents_keeping_best_score(self):
        rows = [
            ("c1", "pA", 0.90),
            ("c2", "pA", 0.70),     # 同一父块的第二个 child：不能重复出现
            ("c3", "pB", 0.80),
        ]
        route, _seen = self._route(rows)
        hits = route.search("HNSW 是什么", 10)
        self.assertEqual([h.parent_id for h in hits], ["pA", "pB"])
        self.assertAlmostEqual(hits[0].score, 0.90)
        self.assertEqual(hits[0].chunk_id, "c1")
        self.assertEqual(hits[0].route, "dense_zh")

    def test_ranks_are_one_based_and_contiguous(self):
        rows = [("c1", "pA", 0.9), ("c2", "pB", 0.8), ("c3", "pC", 0.7)]
        route, _seen = self._route(rows)
        hits = route.search("q", 10)
        self.assertEqual([h.rank for h in hits], [1, 2, 3])

    def test_oversamples_by_pool_factor_before_collapsing(self):
        rows = [("c%d" % i, "p%d" % (i // 2), 1.0 - i * 0.01) for i in range(10)]
        route, seen = self._route(rows, pool_factor=4)
        route.search("q", 5)
        self.assertEqual(seen["k"], 20, "必须向 Qdrant 多要 k*pool_factor，否则父块收敛后不够 k 条")

    def test_truncates_to_k_after_collapse(self):
        rows = [("c%d" % i, "p%d" % i, 1.0 - i * 0.01) for i in range(10)]
        route, _seen = self._route(rows)
        self.assertEqual(len(route.search("q", 3)), 3)

    def test_empty_result_is_empty_list(self):
        route, _seen = self._route([])
        self.assertEqual(route.search("q", 5), [])


class LexicalRouteTest(unittest.TestCase):
    def test_uses_english_tokenizer_index_and_collapses(self):
        from src.rag.retrieval.routes import LexicalRoute
        from src.rag.store.lexical import BM25Index

        index = BM25Index()
        index.add("c1", "pA", "HNSW graph construction parameters")
        index.add("c2", "pA", "HNSW graph neighbors")
        index.add("c3", "pB", "totally unrelated content about cooking")
        index.build()

        route = LexicalRoute(index=index)
        hits = route.search("HNSW graph", 10)
        self.assertEqual(hits[0].parent_id, "pA")
        self.assertEqual(hits[0].route, "lexical_en")
        self.assertEqual(len({h.parent_id for h in hits}), len(hits), "父块不能重复")

    def test_no_match_returns_empty(self):
        from src.rag.retrieval.routes import LexicalRoute
        from src.rag.store.lexical import BM25Index

        index = BM25Index()
        index.add("c1", "pA", "alpha beta")
        index.build()
        self.assertEqual(LexicalRoute(index=index).search("zzz", 5), [])


class MultiRouteRetrieverTest(unittest.TestCase):
    """三路编排：输入映射、开关一致性、debug 字段。"""

    def _retriever(self):
        from src.rag.retrieval.orchestrator import MultiRouteRetriever
        from src.rag.retrieval.routes import DenseRoute, LexicalRoute
        from src.rag.store.lexical import BM25Index

        index = BM25Index()
        index.add("c3", "pLex", "HNSW graph construction")
        index.build()

        zh_embed_seen, en_embed_seen = [], []
        zh_rows = [("c1", "pZh", 0.95), ("c2", "pShared", 0.90)]
        en_rows = [("c3", "pEn", 0.93), ("c2", "pShared", 0.85)]

        def zh_embed(texts):
            zh_embed_seen.append(list(texts))
            return np.zeros((len(texts), 4), dtype=np.float32)

        def en_embed(texts):
            en_embed_seen.append(list(texts))
            return np.zeros((len(texts), 4), dtype=np.float32)

        routes = [
            DenseRoute(name="dense_zh", embedder=zh_embed,
                       searcher=lambda _v, k: zh_rows[:k], input_lang="zh"),
            DenseRoute(name="dense_en", embedder=en_embed,
                       searcher=lambda _v, k: en_rows[:k], input_lang="en"),
            LexicalRoute(index=index, input_lang="en"),
        ]
        return MultiRouteRetriever(routes), zh_embed_seen, en_embed_seen

    def test_routes_get_the_right_query_text(self):
        retriever, zh_seen, en_seen = self._retriever()
        retriever.retrieve("HNSW 是什么", query_en="What is HNSW?", k=5)
        self.assertEqual(zh_seen[0], ["HNSW 是什么"])
        self.assertEqual(en_seen[0], ["What is HNSW?"])

    def test_missing_query_en_falls_back_to_chinese_with_note(self):
        retriever, _zh, en_seen = self._retriever()
        result = retriever.retrieve("HNSW 是什么", query_en=None, k=5)
        self.assertEqual(en_seen[0], ["HNSW 是什么"])
        self.assertIn("回退", result.routes["dense_en"].note)
        self.assertIn("回退", result.routes["lexical_en"].note)

    def test_all_three_routes_present_even_when_disabled(self):
        retriever, _zh, _en = self._retriever()
        result = retriever.retrieve("q", query_en="qq", k=5, disable=("dense_en",))
        self.assertEqual(set(result.routes), {"dense_zh", "dense_en", "lexical_en"})
        self.assertFalse(result.routes["dense_en"].enabled)
        self.assertEqual(result.routes["dense_en"].hits, [])

    def test_disabling_a_route_does_not_change_other_routes(self):
        """验收原文：任一路可单独关闭且其余路结果与单独运行逐条一致。"""
        retriever, _zh, _en = self._retriever()
        full = retriever.retrieve("q", query_en="qq", k=5)
        for target in ("dense_zh", "dense_en", "lexical_en"):
            with self.subTest(disabled=target):
                partial = retriever.retrieve("q", query_en="qq", k=5, disable=(target,))
                for route in full.routes:
                    if route == target:
                        continue
                    self.assertEqual(
                        partial.routes[route].hits, full.routes[route].hits,
                        f"关掉 {target} 后 {route} 的结果变了",
                    )

    def test_single_route_run_matches_multi_route_run(self):
        """反向确认：多路里的一路 == 只开这一路。"""
        retriever, _zh, _en = self._retriever()
        full = retriever.retrieve("q", query_en="qq", k=5)
        from src.rag.retrieval.orchestrator import MultiRouteRetriever

        for target in ("dense_zh", "dense_en", "lexical_en"):
            with self.subTest(only=target):
                others = tuple(n for n in full.routes if n != target)
                alone = MultiRouteRetriever(retriever.routes_list).retrieve(
                    "q", query_en="qq", k=5, disable=others
                )
                self.assertEqual(alone.routes[target].hits, full.routes[target].hits)

    def test_debug_has_per_route_candidate_counts_and_query_used(self):
        retriever, _zh, _en = self._retriever()
        result = retriever.retrieve("HNSW 是什么", query_en="What is HNSW?", k=50)
        debug = result.debug
        self.assertEqual(debug["k"], 50)
        self.assertEqual(debug["rrf_k"], 60)
        self.assertEqual(set(debug["candidates"]), {"dense_zh", "dense_en", "lexical_en"})
        self.assertEqual(debug["query_used"]["dense_zh"], "HNSW 是什么")
        self.assertEqual(debug["query_used"]["dense_en"], "What is HNSW?")
        for route in debug["candidates"]:
            self.assertIn(route, debug["latency_ms"])

    def test_fused_list_has_no_duplicates_and_covers_routes(self):
        retriever, _zh, _en = self._retriever()
        result = retriever.retrieve("q", query_en="qq", k=10)
        ids = [hit.parent_id for hit in result.fused]
        self.assertEqual(len(ids), len(set(ids)))
        # pShared 在 dense_zh 与 dense_en 都出现，必须只出现一次且分数累加
        shared = [hit for hit in result.fused if hit.parent_id == "pShared"]
        self.assertEqual(len(shared), 1)
        self.assertEqual(set(shared[0].routes), {"dense_zh", "dense_en"})

    def test_disable_all_routes_returns_empty_fused(self):
        retriever, _zh, _en = self._retriever()
        result = retriever.retrieve("q", query_en="qq", k=5,
                                    disable=("dense_zh", "dense_en", "lexical_en"))
        self.assertEqual(result.fused, [])

    def test_unknown_route_in_disable_raises(self):
        retriever, _zh, _en = self._retriever()
        with self.assertRaises(ValueError):
            retriever.retrieve("q", query_en="qq", k=5, disable=("dense_typo",))


class CollapseParityTest(unittest.TestCase):
    """`collapse_to_parents` 与评测侧 `parent_ranked` 不能各写一套跑偏。"""

    def test_same_result_as_eval_store_parent_ranked(self):
        from src.eval_store import parent_ranked
        from src.rag.retrieval.routes import collapse_to_parents

        rows = [
            {"parent_id": "pA", "chunk_id": "c1", "score": 0.9},
            {"parent_id": "pA", "chunk_id": "c2", "score": 0.7},
            {"parent_id": "pB", "chunk_id": "c3", "score": 0.8},
        ]
        legacy = [row["parent_id"] for row in parent_ranked([dict(r) for r in rows], 10)]
        new = [hit.parent_id for hit in collapse_to_parents(rows, 10)]
        self.assertEqual(new, legacy)


if __name__ == "__main__":
    unittest.main()
