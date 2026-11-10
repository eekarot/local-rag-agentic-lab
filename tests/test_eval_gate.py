"""回归门禁：`eval_bench --compare 基线.json`（ROADMAP C8 / F36）。

门禁最容易写错的三处，这里各有用例守着：

1. **容差不是拍脑袋的 1%**：用**基线一侧**的 95% bootstrap CI 半宽。基线自己就抖 ±0.04 的指标，
   掉 0.01 就报红只会让人把门禁关掉。
2. **不可比必须说清楚、而且不返非零**：题集/k/pool/重排开关任一不同，比的就不是同一件事。
   这时退 0 并打印 `not_comparable` + 原因 —— 退出码非零会在 CI 里被当成「代码坏了」，
   而真实情况是「你换了题」。
3. **只报均值 delta 不够**：必须同时给「逐题更好 / 更差 / 持平」计数。
   均值持平但 10 题换手，与真的没变，是两件不同的事。
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _row(qid, rank, *, hit=None, full=None, recall=None, qtype="single_hop"):
    hit = bool(rank) if hit is None else hit
    return {
        "qid": qid, "question": f"问题 {qid}", "type": qtype,
        "expected_evidence_lang": "zh-cn", "gold": ["par_x"],
        "first_hit_rank": rank, "hit": hit,
        "full": hit if full is None else full,
        "recall": (1.0 if hit else 0.0) if recall is None else recall,
        "latency_ms": 100.0,
    }


def _rows(ranks, prefix="q"):
    return [_row(f"{prefix}{i + 1}", rank) for i, rank in enumerate(ranks)]


def _result(rows, *, questions_sha="abc", k=10, pool=40, rerank=True):
    from scripts.eval_bench import summarize

    return {
        "collection": "bench_k8s_v1",
        "questions_sha256": questions_sha,
        "k": k, "pool": pool, "rerank": rerank,
        "overall": summarize(rows, k),
        "rows": rows,
    }


class BootstrapTest(unittest.TestCase):
    def test_constant_series_has_zero_width(self):
        from scripts.eval_bench import bootstrap_half_width

        self.assertEqual(bootstrap_half_width([1.0] * 40), 0.0)

    def test_width_is_deterministic(self):
        from scripts.eval_bench import bootstrap_half_width

        values = [1.0] * 30 + [0.0] * 10
        self.assertEqual(bootstrap_half_width(values), bootstrap_half_width(values))

    def test_width_grows_with_variance(self):
        from scripts.eval_bench import bootstrap_half_width

        tight = bootstrap_half_width([1.0] * 38 + [0.0] * 2)
        loose = bootstrap_half_width([1.0] * 20 + [0.0] * 20)
        self.assertLess(tight, loose)
        self.assertGreater(loose, 0.0)

    def test_empty_series_is_zero(self):
        from scripts.eval_bench import bootstrap_half_width

        self.assertEqual(bootstrap_half_width([]), 0.0)


class CompareTest(unittest.TestCase):
    def setUp(self):
        from scripts.eval_bench import compare_results

        self.compare = compare_results
        self.baseline_rows = _rows([1] * 30 + [5] * 10)
        self.baseline = _result(self.baseline_rows)

    def test_identical_runs_pass_and_count_all_same(self):
        cmp = self.compare(_result(_rows([1] * 30 + [5] * 10)), self.baseline,
                           self.baseline_rows)
        self.assertTrue(cmp["comparable"])
        self.assertEqual(cmp["verdict"], "ok")
        counts = cmp["per_question"]
        self.assertEqual((counts["better"], counts["worse"], counts["same"]), (0, 0, 40))
        for metric in cmp["metrics"].values():
            self.assertAlmostEqual(metric["delta"], 0.0, places=6)

    def test_big_drop_fails(self):
        current = _result(_rows([5] * 30 + [9] * 10))
        cmp = self.compare(current, self.baseline, self.baseline_rows)
        self.assertTrue(cmp["comparable"])
        self.assertEqual(cmp["verdict"], "regression")
        self.assertEqual(cmp["metrics"]["mrr"]["verdict"], "regression")

    def test_drop_inside_baseline_ci_passes(self):
        """掉一点点要放过 —— 容差是基线自己的 CI 半宽，不是 0。"""
        current = _result(_rows([1] * 29 + [2] + [5] * 10))
        cmp = self.compare(current, self.baseline, self.baseline_rows)
        self.assertEqual(cmp["metrics"]["mrr"]["verdict"], "ok", cmp["metrics"]["mrr"])

    def test_per_question_counts_mixed_changes(self):
        # 基线 [1]*30+[5]*10 vs 现在 [1]*28+[3,3]+[2]*10
        # → 前 28 题不变；第 29、30 题 1→3 变差；后 10 题 5→2 变好
        current = _result(_rows([1] * 28 + [3, 3] + [2] * 10))
        cmp = self.compare(current, self.baseline, self.baseline_rows)
        counts = cmp["per_question"]
        self.assertEqual((counts["better"], counts["worse"], counts["same"]), (10, 2, 28))

    def test_improvement_is_reported_as_such(self):
        current = _result(_rows([1] * 40))
        cmp = self.compare(current, self.baseline, self.baseline_rows)
        self.assertEqual(cmp["metrics"]["mrr"]["verdict"], "improved")

    def test_different_questions_are_not_comparable(self):
        cmp = self.compare(_result(_rows([1] * 40), questions_sha="other"),
                           self.baseline, self.baseline_rows)
        self.assertFalse(cmp["comparable"])
        self.assertTrue(any("题集" in reason for reason in cmp["reasons"]))
        self.assertEqual(cmp["exit_code"], 0, "不可比不许退非零")

    def test_different_pool_or_rerank_is_not_comparable(self):
        for override in ({"pool": 20}, {"rerank": False}, {"k": 20}):
            cmp = self.compare(_result(_rows([1] * 40), **override), self.baseline,
                               self.baseline_rows)
            self.assertFalse(cmp["comparable"], override)
            self.assertEqual(cmp["exit_code"], 0)

    def test_baseline_without_fingerprint_is_not_comparable(self):
        """老基线没有 questions_sha256：宁可说不可比，也不许当可比硬比。"""
        baseline = _result(self.baseline_rows)
        baseline.pop("questions_sha256")
        cmp = self.compare(_result(_rows([1] * 40)), baseline, self.baseline_rows)
        self.assertFalse(cmp["comparable"])
        self.assertTrue(any("questions_sha256" in reason for reason in cmp["reasons"]))

    def test_tolerance_override_is_honoured(self):
        current = _result(_rows([1] * 29 + [3] + [5] * 10))
        loose = self.compare(current, self.baseline, self.baseline_rows, tolerance=0.5)
        tight = self.compare(current, self.baseline, self.baseline_rows, tolerance=0.0)
        self.assertEqual(loose["verdict"], "ok")
        self.assertEqual(tight["verdict"], "regression")

    def test_missing_row_is_not_comparable(self):
        cmp = self.compare(_result(_rows([1] * 39)), self.baseline, self.baseline_rows)
        self.assertFalse(cmp["comparable"])
        self.assertTrue(any("逐题" in reason for reason in cmp["reasons"]))


class ReportFormatTest(unittest.TestCase):
    def test_markdown_contains_table_and_counts(self):
        from scripts.eval_bench import compare_results, format_comparison

        baseline_rows = _rows([1] * 30 + [5] * 10)
        cmp = compare_results(_result(_rows([1] * 28 + [3, 3] + [2] * 10)),
                              _result(baseline_rows), baseline_rows)
        text = "\n".join(format_comparison(cmp))
        self.assertIn("| 指标 |", text)
        self.assertIn("更好", text)
        self.assertIn("更差", text)
        self.assertIn("持平", text)
        self.assertIn("容差", text)

    def test_not_comparable_report_lists_reasons(self):
        from scripts.eval_bench import compare_results, format_comparison

        baseline_rows = _rows([1] * 40)
        cmp = compare_results(_result(_rows([1] * 40), pool=20),
                              _result(baseline_rows), baseline_rows)
        text = "\n".join(format_comparison(cmp))
        self.assertIn("not_comparable", text)


class CliContractTest(unittest.TestCase):
    def test_compare_flag_exists(self):
        from scripts.eval_bench import build_parser

        args = build_parser().parse_args(["--compare", "docs/baseline-k8s-v1.json"])
        self.assertEqual(Path(args.compare), Path("docs/baseline-k8s-v1.json"))
        self.assertIsNone(args.out, "对比时默认不写输出，免得把基线覆盖掉")


class BaselineCoversFingerprintTest(unittest.TestCase):
    """新增口径键时必须同步更新基线，否则默认路径也会被判 not_comparable。"""

    def test_baseline_json_has_every_comparability_key(self):
        import json
        from pathlib import Path

        from scripts.eval_bench import COMPARABILITY_KEYS

        root = Path(__file__).resolve().parents[1]
        baseline = json.loads((root / "docs" / "baseline-k8s-v1.json").read_text(encoding="utf-8"))
        missing = [key for key in COMPARABILITY_KEYS if key not in baseline]
        self.assertEqual(missing, [],
                         f"基线缺口径键 {missing}：补进 docs/baseline-k8s-v1.json 后再跑门禁")


if __name__ == "__main__":
    unittest.main()


class ResolvePathsTest(unittest.TestCase):
    """可选路径参数不 resolve 成 `Path(None)`（带 --compare 时 --out 默认是 None）。"""

    def test_none_paths_stay_none(self):
        from scripts.eval_bench import _resolve_paths, build_parser

        args = _resolve_paths(build_parser().parse_args(["--compare", "docs/baseline-k8s-v1.json"]))
        self.assertIsNone(args.out)
        self.assertIsNone(args.report)
        self.assertTrue(args.compare.is_absolute())
