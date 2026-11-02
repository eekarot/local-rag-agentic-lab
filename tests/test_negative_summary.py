"""负样本汇总口径：用「相对全库的 z 分数」代替「绝对余弦 0.75」。

背景（2026-11-02 实测）：把 embedding 从 nomic-embed-text 换成 bge-m3 后，
负样本的 top1 绝对余弦从 0.682 降到 0.619，报表面上的「≥0.75 比例 16.7% → 0%」
看起来像变好了；但换成 z 分数后是 +5.74 → +5.08，反而略差。
根因是余弦尺度本身随模型变（正样本 gold 的绝对余弦也从 0.682 降到 0.590），
所以跨模型比较绝对阈值没有意义。
"""
import unittest

from src.evaluate import _negative_summary


def _row(qid, cos, cos_z=None):
    return {"qid": qid, "top1_cos": cos, "top1_cos_z": cos_z}


class NegativeSummaryTest(unittest.TestCase):
    def test_reports_z_distribution_and_alert_share(self):
        rows = [_row("n1", 0.70, 5.0), _row("n2", 0.60, 2.0), _row("n3", 0.65, 4.0)]
        out = _negative_summary(rows)
        self.assertEqual(out["n"], 3)
        self.assertEqual(out["n_with_z"], 3)
        self.assertAlmostEqual(out["top1_z_p50"], 4.0)
        self.assertAlmostEqual(out["top1_z_max"], 5.0)
        self.assertAlmostEqual(out["share_z_ge_alert"], 2 / 3)
        self.assertAlmostEqual(out["top1_cos_max"], 0.70)

    def test_separation_compares_against_positive_top1(self):
        """负样本的「自信程度」要和正样本 top1 比，否则看不出向量分能不能做弃答。"""
        rows = [_row("n1", 0.70, 5.1), _row("n2", 0.69, 5.7)]
        positives = [_row("p1", 0.60, 4.3), _row("p2", 0.58, 4.4)]
        out = _negative_summary(rows, positives)
        # 中位数沿用「排序后取 len//2」的既有约定，不取平均
        self.assertAlmostEqual(out["positives_top1_z_p50"], 4.4)
        self.assertAlmostEqual(out["separation"], 4.4 - 5.7)
        self.assertFalse(out["separable"])

    def test_marks_separable_when_negatives_are_clearly_weaker(self):
        rows = [_row("n1", 0.30, 1.0), _row("n2", 0.32, 1.5)]
        positives = [_row("p1", 0.60, 4.3)]
        out = _negative_summary(rows, positives)
        self.assertTrue(out["separable"])

    def test_degrades_gracefully_for_old_logs(self):
        rows = [_row("n1", 0.70), _row("n2", 0.60)]
        out = _negative_summary(rows)
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["n_with_z"], 0)
        self.assertIsNone(out["top1_z_p50"])
        self.assertIsNone(out["share_z_ge_alert"])
        self.assertIsNone(out["separation"])
        self.assertAlmostEqual(out["top1_cos_p50"], 0.70)  # 排序后取 len//2

    def test_empty_rows(self):
        self.assertEqual(_negative_summary([]), {})


if __name__ == "__main__":
    unittest.main()
