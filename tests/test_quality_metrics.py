"""切片质量指标的纯函数测试（G3 / 方案 P1.3 + P1.4）。

两个指标都**输入区间的整数坐标**，不碰 tokenizer、不碰向量库 —— 这样它们
既能在建库前跑，也能在用户自己的语料上跑（那时没有任何金标可测）。

口径来自上游 `ekimetrics/adaptive-chunking`（LREC 2026）的 SC / BI：
* **SC**（Size Compliance）：落在 `[min_tokens, max_tokens]` 的块占比。
  只有分位数不够 —— p90=400 与「有 30% 的块越界」可以同时成立。
* **BI**（Block Integrity）：多少比例的**结构块**（段落/表格/代码）没被切断。
  它的「金标」不是人写的答案，而是 parser 自己给出的块边界 —— 所以也不需金标。

`tolerance_chars` 的存在理由：块边界附近的切分（差几个字符）在阅读上等同于没切断，
上游用 5；这里保留同口径并在文档里写明，免得「差 1 个字符」被判成大问题。
"""

import unittest

from src.rag.ingestion.quality_metrics import (
    block_integrity,
    find_coverage_gaps,
    size_compliance,
)


class SizeComplianceTest(unittest.TestCase):
    def test_all_in_range(self):
        self.assertEqual(size_compliance([100, 200, 300], min_tokens=100, max_tokens=300), 1.0)

    def test_all_out_of_range(self):
        self.assertEqual(size_compliance([99, 301], min_tokens=100, max_tokens=300), 0.0)

    def test_partial(self):
        self.assertAlmostEqual(
            size_compliance([50, 100, 300, 999], min_tokens=100, max_tokens=300), 0.5
        )

    def test_bounds_are_inclusive(self):
        self.assertEqual(size_compliance([100, 300], min_tokens=100, max_tokens=300), 1.0)

    def test_empty_returns_none(self):
        """空输入不许报 0%（与 N36 同一条纪律：分母为 0 时不给数）。"""
        self.assertIsNone(size_compliance([], min_tokens=100, max_tokens=300))

    def test_bad_bounds_raise(self):
        with self.assertRaises(ValueError):
            size_compliance([100], min_tokens=300, max_tokens=100)


class BlockIntegrityTest(unittest.TestCase):
    def test_no_split_keeps_every_block(self):
        self.assertEqual(
            block_integrity([(0, 10), (10, 20)], [(0, 10), (10, 20)], tolerance_chars=5), 1.0
        )

    def test_split_in_the_middle_breaks_the_block(self):
        # 一块 [0,20) 被在 10 处切开
        self.assertEqual(block_integrity([(0, 10), (10, 20)], [(0, 20)], tolerance_chars=5), 0.0)

    def test_split_within_tolerance_is_not_a_break(self):
        # 在离左边 2 个字符处切：容忍
        self.assertEqual(block_integrity([(0, 2), (2, 20)], [(0, 20)], tolerance_chars=5), 1.0)

    def test_partial_integrity(self):
        # 两个块：第二个 [10,20) 被 15 切开（tolerance=0 才算切）
        score = block_integrity([(0, 10), (10, 15), (15, 20)], [(0, 10), (10, 20)], tolerance_chars=0)
        self.assertAlmostEqual(score, 0.5)

    def test_block_broken_by_many_splits_counts_once(self):
        """一块被切两次也只算「一块被破坏」，不许算成 -1 / 扣两次分。"""
        score = block_integrity(
            [(0, 3), (3, 6), (6, 10), (10, 20)], [(0, 10), (10, 20)], tolerance_chars=0
        )
        self.assertAlmostEqual(score, 0.5)

    def test_single_chunk_is_always_intact(self):
        self.assertEqual(block_integrity([(0, 100)], [(0, 40), (40, 100)]), 1.0)

    def test_empty_chunks_or_blocks_return_none(self):
        self.assertIsNone(block_integrity([], [(0, 10)]))
        self.assertIsNone(block_integrity([(0, 10)], []))

    def test_none_spans_are_ignored(self):
        self.assertEqual(block_integrity([(None, None), (0, 20)], [(0, 20)]), 1.0)


class CoverageGapTest(unittest.TestCase):
    def test_contiguous_chunks_have_no_gap(self):
        self.assertEqual(find_coverage_gaps([(0, 10), (10, 20)], end=20), [])

    def test_overlapping_chunks_have_no_gap(self):
        self.assertEqual(find_coverage_gaps([(0, 10), (8, 20)], end=20), [])

    def test_middle_gap_is_reported(self):
        self.assertEqual(find_coverage_gaps([(0, 10), (12, 20)], end=20), [(10, 12)])

    def test_leading_gap_is_reported(self):
        self.assertEqual(find_coverage_gaps([(2, 20)], end=20), [(0, 2)])

    def test_trailing_gap_is_reported(self):
        """最后一块没到末尾 —— 最容易被漏掉的一类丢字。"""
        self.assertEqual(find_coverage_gaps([(0, 18)], end=20), [(18, 20)])

    def test_empty_input(self):
        self.assertEqual(find_coverage_gaps([], end=0), [])
        self.assertEqual(find_coverage_gaps([], end=5), [(0, 5)])


if __name__ == "__main__":
    unittest.main()
