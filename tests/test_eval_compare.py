"""基线对比（回归门禁）与模式间配对比较的测试。"""

import unittest

from src.eval_compare import compare_modes, diff_baseline


def _metric(mean, half_width, n=60):
    return {"n": n, "mean": mean, "lo": mean - half_width, "hi": mean + half_width, "half_width": half_width}


def _baseline(hit4=0.8, half_width=0.05, content_hash="aaa", k=4):
    return {
        "content_hash": content_hash,
        "dataset_sha256": "d1",
        "ks": [k],
        "metrics": {"by_mode": {"hybrid": {"n": 60, "hit": {k: _metric(hit4, half_width)}}}},
    }


def _current(hit4=0.8, half_width=0.05, content_hash="aaa", k=4):
    return {
        "content_hash": content_hash,
        "dataset_sha256": "d1",
        "ks": [k],
        "metrics": {"by_mode": {"hybrid": {"n": 60, "hit": {k: _metric(hit4, half_width)}}}},
    }


class TestDiffBaseline(unittest.TestCase):
    def test_drop_beyond_half_width_is_regression(self):
        result = diff_baseline(_baseline(hit4=0.80), _current(hit4=0.70))
        self.assertEqual(result["verdict"], "regression")
        self.assertAlmostEqual(result["deltas"][0]["delta"], -0.10)
        self.assertTrue(result["deltas"][0]["regressed"])

    def test_drop_within_half_width_is_noise(self):
        result = diff_baseline(_baseline(hit4=0.80), _current(hit4=0.77))
        self.assertEqual(result["verdict"], "ok")
        self.assertFalse(result["deltas"][0]["regressed"])

    def test_gain_is_reported_as_improvement(self):
        result = diff_baseline(_baseline(hit4=0.70), _current(hit4=0.82))
        self.assertEqual(result["verdict"], "improved")

    def test_different_content_hash_disables_gating(self):
        result = diff_baseline(_baseline(hit4=0.80, content_hash="aaa"), _current(hit4=0.50, content_hash="bbb"))
        self.assertEqual(result["verdict"], "not_comparable")
        self.assertFalse(result["deltas"][0]["regressed"])

    def test_missing_slice_is_reported(self):
        base = _baseline()
        current = _baseline()
        current["metrics"]["by_mode"]["hybrid"]["hit"] = {}
        result = diff_baseline(base, current)
        self.assertEqual(result["verdict"], "not_comparable")
        self.assertIn("缺失", " ".join(result["notes"]))


class TestCompareModes(unittest.TestCase):
    def _scored(self, hybrid_hits, bm25_hits):
        rows = []
        qids = [f"q{i}" for i in range(1, len(hybrid_hits) + 1)]
        for qid, a, b in zip(qids, hybrid_hits, bm25_hits):
            rows.append({"qid": qid, "mode": "hybrid", "type": "fact", "lang": "zh", "hit": {4: a}, "full": {4: a}})
            rows.append({"qid": qid, "mode": "bm25", "type": "fact", "lang": "zh", "hit": {4: b}, "full": {4: b}})
        return {"positives": rows, "negatives": [], "unmatched": []}

    def test_counts_discordant_pairs(self):
        scored = self._scored([True, True, False, False], [True, False, True, False])
        result = compare_modes(scored, k=4)
        hybrid_vs_bm25 = result[("hybrid", "bm25")]
        self.assertEqual(hybrid_vs_bm25["a_only"], 1)
        self.assertEqual(hybrid_vs_bm25["b_only"], 1)
        self.assertEqual(hybrid_vs_bm25["both"], 1)
        self.assertEqual(hybrid_vs_bm25["neither"], 1)
        self.assertAlmostEqual(hybrid_vs_bm25["p_value"], 1.0)

    def test_one_sided_dominance_is_significant(self):
        scored = self._scored([True] * 8, [False] * 8)
        result = compare_modes(scored, k=4)
        self.assertLess(result[("hybrid", "bm25")]["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()


class RunsComparableTest(unittest.TestCase):
    """逐题配对的前提：同一个检索器 + 同一套题。"""

    def test_same_everything(self):
        from src.eval_compare import runs_comparable

        meta = {"content_hash": "a", "dataset_sha256": "d"}
        self.assertEqual(runs_comparable(meta, dict(meta))[0], True)

    def test_index_changed_is_not_comparable(self):
        from src.eval_compare import runs_comparable

        ok, reason = runs_comparable(
            {"content_hash": "a", "dataset_sha256": "d"},
            {"content_hash": "b", "dataset_sha256": "d"},
        )
        self.assertFalse(ok)
        self.assertIn("content_hash", reason)

    def test_dataset_changed_is_not_comparable(self):
        from src.eval_compare import runs_comparable

        ok, reason = runs_comparable(
            {"content_hash": "a", "dataset_sha256": "d1"},
            {"content_hash": "a", "dataset_sha256": "d2"},
        )
        self.assertFalse(ok)
        self.assertIn("题目集", reason)
