"""Docling vs 现有 PyMuPDF 解析器对照探针的纯函数用例（N42 纪律：只读、不写库）。

这些用例不联网、不需要装 docling：只验收「怎么从两种产物里数出同一组指标」。
真正的对照跑在 `experiments/probe_parser_docling.py`，产物进 docs/evidence/。
"""

from __future__ import annotations

import unittest

from experiments.probe_parser_docling import (
    coverage_ratio,
    docling_chars,
    summarize_blocks,
    summarize_docling,
)


def _docling_doc() -> dict:
    """最小 DoclingDocument：2 页、3 个 text、1 个 table、1 个 picture。"""
    return {
        "version": "1.11.0",
        "pages": {"1": {"page_no": 1}, "2": {"page_no": 2}},
        "texts": [
            {"label": "section_header", "text": "标题一",
             "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}]},
            {"label": "text", "text": "正文两句话。第二句。",
             "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}]},
            # 故意给一条没有 prov 的：bbox 覆盖率不能算成 100%
            {"label": "text", "text": "无坐标文本", "prov": []},
        ],
        "tables": [
            {"label": "table", "prov": [{"page_no": 2, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}],
             "data": {"num_rows": 2, "num_cols": 3,
                      "table_cells": [{"text": "a"}, {"text": "b"}]}},
        ],
        "pictures": [{"label": "picture",
                      "prov": [{"page_no": 2, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}]}],
    }


class SummarizeDoclingTest(unittest.TestCase):
    def test_counts_pages_texts_tables_pictures(self):
        stats = summarize_docling(_docling_doc())
        self.assertEqual(stats["pages"], 2)
        self.assertEqual(stats["texts"], 3)
        self.assertEqual(stats["tables"], 1)
        self.assertEqual(stats["pictures"], 1)

    def test_counts_chars_from_text_and_table_cells(self):
        # 标题一(3) + 正文两句话。第二句。(10) + 无坐标文本(5) + 表格 a/b(2)
        self.assertEqual(docling_chars(_docling_doc()), 20)
        self.assertEqual(summarize_docling(_docling_doc())["chars"], 20)

    def test_bbox_coverage_is_fraction_with_prov_and_bbox(self):
        # 4 个可定位条目里有 3 个带 bbox（第 3 条 text 是空 prov，但那是因为它有自己的
        # 位置，所以分母是 texts+tables+pictures 的有效条目 = 5）
        stats = summarize_docling(_docling_doc())
        self.assertAlmostEqual(stats["bbox_coverage"], 4 / 5, places=4)

    def test_page_coverage_counts_pages_with_any_bbox(self):
        # 第 1 页有 2 条、第 2 页有 2 条 → 2 页都覆盖
        self.assertEqual(summarize_docling(_docling_doc())["pages_with_content"], 2)

    def test_labels_are_counted(self):
        self.assertEqual(summarize_docling(_docling_doc())["labels"]["section_header"], 1)

    def test_tolerates_missing_sections(self):
        stats = summarize_docling({"pages": {}, "texts": [{"label": "text", "text": "x"}]})
        self.assertEqual(stats["tables"], 0)
        self.assertEqual(stats["pictures"], 0)
        self.assertEqual(stats["pages"], 0)
        self.assertEqual(stats["bbox_coverage"], 0.0)


class SummarizeBlocksTest(unittest.TestCase):
    """现有解析器的产物是 DocumentBlock 列表；两边的指标必须能对齐比较。"""

    def test_counts_and_chars(self):
        class _B:
            def __init__(self, text, block_type="paragraph", page_number=1):
                self.text, self.block_type, self.page_number = text, block_type, page_number

        blocks = [_B("一二三"), _B("四五六", "heading", 2), _B("七八", "table", 2)]
        stats = summarize_blocks(blocks)
        self.assertEqual(stats["blocks"], 3)
        self.assertEqual(stats["chars"], 8)
        self.assertEqual(stats["pages_with_content"], 2)
        self.assertEqual(stats["tables"], 1)
        self.assertEqual(stats["labels"]["heading"], 1)

    def test_block_without_page_counts_as_no_coverage(self):
        class _B:
            text, block_type, page_number = "x", "paragraph", None

        self.assertEqual(summarize_blocks([_B()])["pages_with_content"], 0)


class CoverageRatioTest(unittest.TestCase):
    def test_ratio(self):
        self.assertAlmostEqual(coverage_ratio(80, 100), 0.8)

    def test_zero_baseline_does_not_explode(self):
        # 基线 0 字符（空文档 / 纯扫描件）时不能除零，也不能报 0 骗人
        self.assertIsNone(coverage_ratio(10, 0))
        self.assertIsNone(coverage_ratio(0, 0))


if __name__ == "__main__":
    unittest.main()
