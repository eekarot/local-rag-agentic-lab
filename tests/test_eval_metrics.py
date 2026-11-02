"""检索层指标（L1）的单元测试：数字全部手算，不依赖任何模型或索引。

    python -m unittest discover -s tests -v
"""

import math
import unittest

from src.eval_metrics import (
    bootstrap_ci,
    recall_fraction,
    first_relevant_rank,
    mcnemar_exact,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
)


class TestRankMetrics(unittest.TestCase):
    def test_first_relevant_rank_counts_from_one(self):
        self.assertEqual(first_relevant_rank(["a", "b", "c"], {"b"}), 2)
        self.assertEqual(first_relevant_rank(["a", "b", "c"], {"a"}), 1)

    def test_first_relevant_rank_none_when_absent(self):
        self.assertIsNone(first_relevant_rank(["a", "b"], {"z"}))
        self.assertIsNone(first_relevant_rank([], {"a"}))

    def test_recall_distinguishes_any_hit_from_full_coverage(self):
        # top-2 里只有一个 gold：算命中，但不算覆盖全部
        self.assertEqual(recall_at_k(["x", "a", "b"], {"a", "b"}, 2), (True, False))
        self.assertEqual(recall_at_k(["x", "a", "b"], {"a", "b"}, 3), (True, True))
        self.assertEqual(recall_at_k(["x", "y", "z"], {"a"}, 3), (False, False))

    def test_recall_fraction_is_continuous_for_multi_gold(self):
        # gold 有 2 条，top-2 只捞到 1 条 → 0.5；top-3 全捞到 → 1.0
        self.assertAlmostEqual(recall_fraction(["x", "a", "b"], {"a", "b"}, 2), 0.5)
        self.assertEqual(recall_fraction(["x", "a", "b"], {"a", "b"}, 3), 1.0)
        self.assertEqual(recall_fraction(["a"], {"a", "b"}, 4), 0.5)
        self.assertEqual(recall_fraction(["z"], {"a"}, 4), 0.0)
        self.assertEqual(recall_fraction(["a"], set(), 4), 0.0)

    def test_recall_with_empty_gold_is_false(self):
        self.assertEqual(recall_at_k(["a"], set(), 4), (False, False))

    def test_mrr_respects_cutoff(self):
        self.assertAlmostEqual(mrr_at_k(["x", "a"], {"a"}, 10), 0.5)
        self.assertEqual(mrr_at_k(["x", "a"], {"a"}, 1), 0.0)

    def test_ndcg_binary_gain_with_single_gold(self):
        # 唯一 gold 排在第二位 → DCG = 1/log2(3)，IDCG = 1
        self.assertAlmostEqual(ndcg_at_k(["x", "a"], {"a"}, 10), 1 / math.log2(3))
        self.assertEqual(ndcg_at_k(["a", "x"], {"a"}, 10), 1.0)

    def test_ndcg_with_two_gold_uses_ideal_ordering(self):
        # ranked=[a,x,b] gold={a,b} → 1 + 1/log2(4) = 1.5；IDCG = 1 + 1/log2(3)
        expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(["a", "x", "b"], {"a", "b"}, 10), expected)

    def test_ndcg_zero_when_nothing_relevant(self):
        self.assertEqual(ndcg_at_k(["x", "y"], {"a"}, 10), 0.0)


class TestUncertainty(unittest.TestCase):
    def test_bootstrap_ci_is_degenerate_for_constant_data(self):
        ci = bootstrap_ci([1.0] * 10, n_boot=200, seed=0)
        self.assertEqual((ci["mean"], ci["lo"], ci["hi"]), (1.0, 1.0, 1.0))

    def test_bootstrap_ci_is_reproducible_and_brackets_mean(self):
        values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
        first = bootstrap_ci(values, n_boot=500, seed=0)
        second = bootstrap_ci(values, n_boot=500, seed=0)
        self.assertEqual(first, second)
        self.assertLessEqual(first["lo"], first["mean"])
        self.assertGreaterEqual(first["hi"], first["mean"])

    def test_bootstrap_ci_widens_for_noisier_data(self):
        tight = bootstrap_ci([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], n_boot=500, seed=0)
        wide = bootstrap_ci([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], n_boot=500, seed=0)
        self.assertGreater(wide["hi"] - wide["lo"], tight["hi"] - tight["lo"])

    def test_paired_bootstrap_detects_consistent_improvement(self):
        better = [1] * 20
        worse = [0] * 20
        result = paired_bootstrap(better, worse, n_boot=500, seed=0)
        self.assertEqual(result["diff"], 1.0)
        self.assertLess(result["p_value"], 0.01)

    def test_paired_bootstrap_reports_no_effect_for_identical_series(self):
        series = [1, 0, 1, 1, 0, 1, 0, 1]
        result = paired_bootstrap(series, series, n_boot=200, seed=0)
        self.assertEqual(result["diff"], 0.0)
        self.assertGreater(result["p_value"], 0.5)

    def test_paired_bootstrap_requires_equal_length(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 0], [1], n_boot=10)

    def test_mcnemar_exact_matches_binomial_tail(self):
        # 6 个不一致且全部偏向 B → p = 2 * 0.5^6 = 0.03125
        result = mcnemar_exact(a_only=0, b_only=6)
        self.assertAlmostEqual(result["p_value"], 0.03125)
        # 势均力敌 → p=1
        self.assertEqual(mcnemar_exact(a_only=4, b_only=4)["p_value"], 1.0)
        # 完全一致 → 无差异证据
        self.assertEqual(mcnemar_exact(a_only=0, b_only=0)["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
