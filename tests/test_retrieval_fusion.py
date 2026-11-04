"""D26：应用层 RRF 三路融合 + 跨列表按 id 去重 + 同 id 多路分数累加。

RRF 定义：`score(d) = Σ_route 1 / (rrf_k + rank_route(d))`，rank 从 1 开始。
`rrf_k=60` 是标准取值（Cormack et al. 2009），本项目 A/B 也用它。
"""

import unittest


class RrfFuseRoutesTest(unittest.TestCase):
    def test_single_list_uses_standard_rrf(self):
        from src.rag.retrieval.fusion import rrf_fuse_routes

        fused = rrf_fuse_routes({"dense_zh": ["a", "b"]}, rrf_k=60)
        self.assertAlmostEqual(fused[0].score, 1 / 61)
        self.assertAlmostEqual(fused[1].score, 1 / 62)
        self.assertEqual([hit.parent_id for hit in fused], ["a", "b"])

    def test_same_id_in_two_routes_accumulates_score(self):
        from src.rag.retrieval.fusion import rrf_fuse_routes

        fused = rrf_fuse_routes(
            {"dense_zh": ["a", "b"], "dense_en": ["b", "c"]}, rrf_k=60
        )
        by_id = {hit.parent_id: hit for hit in fused}
        # b 排在第 2 / 第 1 → 1/62 + 1/61，明显高于只出现一次的 a(1/61) 与 c(1/62)
        self.assertAlmostEqual(by_id["b"].score, 1 / 62 + 1 / 61)
        self.assertAlmostEqual(by_id["a"].score, 1 / 61)
        self.assertEqual([hit.parent_id for hit in fused], ["b", "a", "c"])
        self.assertEqual(set(by_id["b"].routes), {"dense_zh", "dense_en"})

    def test_three_routes_all_hit_accumulate_three_terms(self):
        from src.rag.retrieval.fusion import rrf_fuse_routes

        fused = rrf_fuse_routes(
            {"dense_zh": ["x"], "dense_en": ["x"], "lexical_en": ["x"]}, rrf_k=60
        )
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].score, 3 / 61)
        self.assertEqual(fused[0].routes, ("dense_zh", "dense_en", "lexical_en"))

    def test_output_has_no_duplicate_ids(self):
        from src.rag.retrieval.fusion import rrf_fuse_routes

        lists = {
            "dense_zh": ["a", "b", "c"],
            "dense_en": ["c", "b", "a"],
            "lexical_en": ["b", "d"],
        }
        ids = [hit.parent_id for hit in rrf_fuse_routes(lists, rrf_k=60)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {"a", "b", "c", "d"})

    def test_ties_are_deterministic(self):
        """两路各命中不同 id 的第 1 名 → 同分，必须按固定顺序输出。"""
        from src.rag.retrieval.fusion import rrf_fuse_routes

        first = rrf_fuse_routes({"dense_zh": ["z"], "dense_en": ["a"]}, rrf_k=60)
        second = rrf_fuse_routes({"dense_en": ["a"], "dense_zh": ["z"]}, rrf_k=60)
        self.assertEqual([hit.parent_id for hit in first], [hit.parent_id for hit in second])
        self.assertEqual([hit.parent_id for hit in first], ["a", "z"], "并列按 id 升序兜底")

    def test_best_rank_breaks_ties_before_id(self):
        from src.rag.retrieval.fusion import rrf_fuse_routes

        # b: dense_zh 第 2 名 + dense_en 第 1 名 = 1/62 + 1/61 ≈ 0.03252
        # a: 只有 dense_zh 第 1 名 = 1/61 ≈ 0.01639 → 多路命中必须胜过单路第一名
        fused = rrf_fuse_routes(
            {"dense_zh": ["a", "b", "c"], "dense_en": ["b", "c"]}, rrf_k=60
        )
        self.assertEqual(fused[0].parent_id, "b")
        self.assertAlmostEqual(fused[0].score, 1 / 62 + 1 / 61)

    def test_empty_lists(self):
        from src.rag.retrieval.fusion import rrf_fuse_routes

        self.assertEqual(rrf_fuse_routes({}, rrf_k=60), [])
        self.assertEqual(rrf_fuse_routes({"dense_zh": []}, rrf_k=60), [])
        self.assertEqual(
            rrf_fuse_routes({"dense_zh": [], "dense_en": [], "lexical_en": []}, rrf_k=60), []
        )

    def test_top_k_truncates_after_fusion_not_before(self):
        from src.rag.retrieval.fusion import rrf_fuse_routes

        lists = {"dense_zh": ["a", "b", "c"], "dense_en": ["c", "d"]}
        fused = rrf_fuse_routes(lists, rrf_k=60, top_k=2)
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].parent_id, "c", "c 两路命中，必须先于单路命中的 a/b")

    def test_duplicate_inside_one_list_is_counted_once(self):
        from src.rag.retrieval.fusion import rrf_fuse_routes

        fused = rrf_fuse_routes({"dense_zh": ["a", "a", "b"]}, rrf_k=60)
        by_id = {hit.parent_id: hit for hit in fused}
        self.assertAlmostEqual(by_id["a"].score, 1 / 61, msg="同列表重复只能算一次，取最好名次")

    def test_rrf_k_zero_guard(self):
        from src.rag.retrieval.fusion import rrf_fuse_routes

        with self.assertRaises(ValueError):
            rrf_fuse_routes({"dense_zh": ["a"]}, rrf_k=0)


class UnionPoolTest(unittest.TestCase):
    """消融矩阵 C/D 用的是「各路 Top-k 的并集上限」，不是融合。"""

    def test_union_dedupes_and_keeps_route_order(self):
        from src.rag.retrieval.fusion import union_pool

        pool = union_pool({"dense_zh": ["a", "b"], "dense_en": ["b", "c"]})
        self.assertEqual(pool, ["a", "b", "c"])

    def test_union_top_k_per_route(self):
        from src.rag.retrieval.fusion import union_pool

        pool = union_pool({"dense_zh": ["a", "b", "c"], "dense_en": ["d"]}, per_route_k=1)
        self.assertEqual(pool, ["a", "d"])

    def test_union_empty(self):
        from src.rag.retrieval.fusion import union_pool

        self.assertEqual(union_pool({}), [])


class FusedHitShapeTest(unittest.TestCase):
    def test_fused_hit_carries_routes_and_best_rank(self):
        from src.rag.retrieval.fusion import rrf_fuse_routes

        hit = rrf_fuse_routes({"dense_zh": ["a"], "lexical_en": ["a"]}, rrf_k=60)[0]
        self.assertEqual(hit.parent_id, "a")
        self.assertEqual(hit.best_rank, 1)
        self.assertEqual(hit.routes, ("dense_zh", "lexical_en"))
        self.assertIsInstance(hit.score, float)


if __name__ == "__main__":
    unittest.main()
