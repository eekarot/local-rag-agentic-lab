"""四路口径的开关与门禁（B05 消融的前置：把 `lexical_zh` 变成可评估的一路）。

纪律背景：
- 新增/切换路由会改变融合输入，属于**口径变化**，必须进可比性指纹，否则门禁会把
  「三路 → 四路」的差异当成正常抖动放过去。
- 老基线（`docs/baseline-k8s-v1.json`）本来就没写 `routes` 键，但它是用默认三路跑的，
  所以「缺键」必须按默认三路理解，否则所有历史证据一次性变成不可比。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RoutesFlagTest(unittest.TestCase):
    """`--routes` 把路由集合变成显式参数，默认值不许漂。"""

    def test_default_is_the_three_routes(self):
        from scripts.eval_bench import build_parser
        from src.rag.retrieval.routes import DEFAULT_ROUTES

        self.assertEqual(tuple(build_parser().parse_args([]).routes), tuple(DEFAULT_ROUTES))

    def test_four_routes_can_be_requested(self):
        from scripts.eval_bench import build_parser

        args = build_parser().parse_args(
            ["--routes", "dense_zh,dense_en,lexical_en,lexical_zh"])
        self.assertEqual(args.routes, ("dense_zh", "dense_en", "lexical_en", "lexical_zh"))

    def test_unknown_route_name_is_rejected_at_parse_time(self):
        from scripts.eval_bench import build_parser

        with self.assertRaises(SystemExit) as ctx:
            build_parser().parse_args(["--routes", "dense_zh,lexical_xx"])
        self.assertNotEqual(ctx.exception.code, 0, "打错路名必须当场报错，不能静默少一路")

    def test_empty_routes_is_rejected(self):
        from scripts.eval_bench import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--routes", " , "])


class EvaluatePlumbsRoutesTest(unittest.TestCase):
    """`evaluate()` 必须把路由集合传给装配层，并写进结果 JSON。"""

    def test_routes_reach_the_retriever(self):
        import scripts.eval_bench as bench

        seen: dict[str, object] = {}
        original = bench._build_retriever

        def fake(collection, store_path, warmup_texts, log, **kwargs):
            seen.update(kwargs)
            seen["collection"] = collection

            class R:
                route_names = ["dense_zh", "dense_en", "lexical_en", "lexical_zh"]

            return R()

        bench._build_retriever = fake
        try:
            result = bench.evaluate([], k=1, pool=1, rerank=False, warmup=False,
                                    parents={}, counter=object(), reranker=None,
                                    routes=("dense_zh", "dense_en", "lexical_en", "lexical_zh"))
        finally:
            bench._build_retriever = original
        self.assertEqual(tuple(seen["routes"]),
                         ("dense_zh", "dense_en", "lexical_en", "lexical_zh"))
        self.assertEqual(result["routes"],
                         ["dense_zh", "dense_en", "lexical_en", "lexical_zh"])

    def test_default_run_records_three_routes(self):
        import scripts.eval_bench as bench

        original = bench._build_retriever

        class R:
            route_names = ["dense_zh", "dense_en", "lexical_en"]

        bench._build_retriever = lambda *a, **kw: R()
        try:
            result = bench.evaluate([], k=1, pool=1, rerank=False, warmup=False,
                                    parents={}, counter=object(), reranker=None)
        finally:
            bench._build_retriever = original
        self.assertEqual(result["routes"], ["dense_zh", "dense_en", "lexical_en"])


class ComparabilityRoutesTest(unittest.TestCase):
    def test_routes_is_a_comparability_key(self):
        from scripts.eval_bench import COMPARABILITY_KEYS

        self.assertIn("routes", COMPARABILITY_KEYS)

    def test_changing_routes_makes_runs_incomparable(self):
        from scripts.eval_bench import _comparability_reasons

        base = {"questions_sha256": "abc", "routes": ["dense_zh", "dense_en", "lexical_en"]}
        current = {"questions_sha256": "abc",
                   "routes": ["dense_zh", "dense_en", "lexical_en", "lexical_zh"]}
        reasons = _comparability_reasons(current, base)
        self.assertTrue(reasons, "多一路显著改变融合输入，必须报不可比")
        self.assertTrue(any("routes" in reason for reason in reasons))

    def test_old_baseline_without_routes_means_the_default_three(self):
        """历史基线没写 `routes`，但它确实是用默认三路跑的 —— 不许因此变不可比。"""
        from scripts.eval_bench import _comparability_reasons

        base = {"questions_sha256": "abc"}
        current = {"questions_sha256": "abc",
                   "routes": ["dense_zh", "dense_en", "lexical_en"]}
        self.assertEqual(_comparability_reasons(current, base), [])

    def test_missing_routes_on_the_current_side_also_defaults(self):
        from scripts.eval_bench import _comparability_reasons

        base = {"questions_sha256": "abc", "routes": ["dense_zh", "dense_en", "lexical_en"]}
        current = {"questions_sha256": "abc"}
        self.assertEqual(_comparability_reasons(current, base), [])


class BaselineDeclaresRoutesTest(unittest.TestCase):
    """新增口径键必须同步基线，否则默认对比路径一律 not_comparable。"""

    def test_baseline_json_declares_the_three_routes(self):
        baseline = json.loads((ROOT / "docs" / "baseline-k8s-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(baseline.get("routes"), ["dense_zh", "dense_en", "lexical_en"])


class ResolveRouteNamesTest(unittest.TestCase):
    """装配层的路由名解析：纯函数，不起 Qdrant 就能测。"""

    def test_none_means_default(self):
        from src.rag.retrieval.store_wiring import resolve_route_names

        self.assertEqual(resolve_route_names(None),
                         ("dense_zh", "dense_en", "lexical_en"))

    def test_include_zh_lexical_appends_the_fourth(self):
        from src.rag.retrieval.store_wiring import resolve_route_names

        self.assertEqual(resolve_route_names(None, include_zh_lexical=True),
                         ("dense_zh", "dense_en", "lexical_en", "lexical_zh"))

    def test_explicit_order_is_kept(self):
        from src.rag.retrieval.store_wiring import resolve_route_names

        self.assertEqual(resolve_route_names(("lexical_zh", "dense_zh")),
                         ("lexical_zh", "dense_zh"))

    def test_unknown_name_lists_the_valid_ones(self):
        from src.rag.retrieval.store_wiring import resolve_route_names

        with self.assertRaises(ValueError) as ctx:
            resolve_route_names(("dense_zh", "lexical_xx"))
        message = str(ctx.exception)
        self.assertIn("lexical_xx", message)
        self.assertIn("lexical_zh", message, "报错要给出可选路名，人才不用翻代码")

    def test_duplicate_name_is_rejected(self):
        from src.rag.retrieval.store_wiring import resolve_route_names

        with self.assertRaises(ValueError):
            resolve_route_names(("dense_zh", "dense_zh"))

    def test_no_routes_at_all_is_rejected(self):
        from src.rag.retrieval.store_wiring import resolve_route_names

        with self.assertRaises(ValueError):
            resolve_route_names(())


if __name__ == "__main__":
    unittest.main()
