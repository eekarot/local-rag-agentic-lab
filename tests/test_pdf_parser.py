"""D04 PDF Parser 的单元测试。

D04 验收行要求：
  - 抽样 30 页人工表：`page_number` 有值率 100%；
  - 抽取字符数与 pypdf 对比 ±5% 内（或写明差异原因）。

人工表由 `python -m src.rag.ingestion.parser.pdf_parser report --pages 1-30` 生成；
这里用**可复现的合成 PDF**（测试现场用 pymupdf 生成，不入库）锁住行为，
再用仓库里那本真 PDF 做与 pypdf 的字符数对照。
"""

import statistics
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:                       # pymupdf 现在是可选依赖（AGPL，见 NOTICE 第三节）
    import pymupdf
    HAVE_PYMUPDF = True
except ImportError:        # pragma: no cover - 干净环境下 PDF 用例整体跳过
    pymupdf = None
    HAVE_PYMUPDF = False

from src.config import DATA_DIR
from src.rag.ingestion.errors import IngestionError
from src.rag.ingestion.parser import get_parser, register_builtin_parsers, reset_registry
from src.rag.ingestion.parser.pdf_parser import PdfParser, cli_main

REAL_PDF = DATA_DIR / "AI-Agents-in-Depth-zh-CN.pdf"
BODY_SIZE = 10.0
HEADING_SIZES = {1: 24.0, 2: 18.0, 3: 14.0}


def build_pdf(path, pages):
    """pages: [[(text, size), ...], ...]，每页一组 (文本, 字号)。"""
    doc = pymupdf.open()
    for spec in pages:
        page = doc.new_page()
        y = 90.0
        for text, size in spec:
            page.insert_text((72, y), text, fontsize=size, fontname="china-s")
            y += size * 1.8
    doc.save(str(path))
    doc.close()
    return str(path)


@unittest.skipUnless(HAVE_PYMUPDF, "需要 pymupdf：pip install -r requirements-pdf.txt")
class SyntheticPdfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.simple = build_pdf(
            base / "simple.pdf",
            [
                [
                    ("第一章 检索基础", HEADING_SIZES[1]),
                    ("这是第一章的正文段落，用来测试正文提取。", BODY_SIZE),
                    ("第二段正文，仍属于第一章的同一个主题。", BODY_SIZE),
                    ("1.1 向量检索", HEADING_SIZES[2]),
                    ("向量检索把文本映射到向量空间。", BODY_SIZE),
                    ("第 1 页 / 共 2 页", 8.0),  # 页脚，应在多页重复后被抓出
                ],
                [
                    ("第二章 切分策略", HEADING_SIZES[1]),
                    ("切分粒度决定召回上限。", BODY_SIZE),
                    ("第 2 页 / 共 2 页", 8.0),
                ],
            ],
        )
        cls.blank_page = build_pdf(base / "blank.pdf", [[("只有一页正文。", BODY_SIZE)], []])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def parse(self, path=None):
        parser = PdfParser()
        blocks = parser.parse(path or self.simple)
        return parser, blocks

    def test_every_block_has_page_number(self):
        _, blocks = self.parse()
        self.assertTrue(blocks)
        for block in blocks:
            self.assertIsNotNone(block.page_number, f"缺少页码: {block.text[:30]!r}")
            self.assertIn(block.page_number, (1, 2))

    def test_page_numbers_are_one_based_and_in_order(self):
        _, blocks = self.parse()
        pages = [b.page_number for b in blocks]
        self.assertEqual(pages, sorted(pages), "页码必须单调不减")
        self.assertEqual(pages[0], 1, "页码从 1 开始（不是 0）")
        self.assertIn(2, pages, "第二页的内容丢了")

    def test_body_text_is_captured(self):
        _, blocks = self.parse()
        joined = "".join(b.text for b in blocks)
        for expected in ("这是第一章的正文段落", "向量检索把文本映射到向量空间", "切分粒度决定召回上限"):
            self.assertIn(expected, joined)

    def test_headings_are_detected_by_font_size(self):
        _, blocks = self.parse()
        headings = [(b.heading_level, b.text) for b in blocks if b.block_type == "heading"]
        texts = [text for _, text in headings]
        self.assertIn("第一章 检索基础", texts)
        self.assertIn("1.1 向量检索", texts)
        self.assertIn("第二章 切分策略", texts)
        # 字号越大层级越高
        levels = dict((text, level) for level, text in headings)
        self.assertLess(levels["第一章 检索基础"], levels["1.1 向量检索"], "h1 的层级数值应小于 h2")

    def test_body_text_is_not_marked_as_heading(self):
        _, blocks = self.parse()
        for block in blocks:
            if block.block_type == "heading":
                continue
            self.assertGreaterEqual(len(block.text), 5)

    def test_repeated_footer_is_dropped(self):
        parser, blocks = self.parse()
        joined = "".join(b.text for b in blocks)
        self.assertNotIn("共 2 页", joined, "重复页脚应被识别为 margin 并剔除")
        self.assertTrue(parser.stats["margins_removed"], "剔除行为要留痕，不能静默")

    def test_section_path_is_never_empty(self):
        _, blocks = self.parse()
        for block in blocks:
            self.assertTrue(block.section_path, f"section_path 为空: {block.text[:30]!r}")

    def test_section_path_follows_detected_headings(self):
        _, blocks = self.parse()
        target = next(b for b in blocks if "向量检索把文本" in b.text)
        self.assertEqual(target.section_path, ["第一章 检索基础", "1.1 向量检索"])
        second = next(b for b in blocks if "切分粒度" in b.text)
        self.assertEqual(second.section_path, ["第二章 切分策略"])

    def test_blank_page_produces_no_blocks(self):
        _, blocks = self.parse(self.blank_page)
        self.assertTrue(all(b.page_number == 1 for b in blocks), f"空页产生了 block: {[b.text for b in blocks]}")

    def test_metadata_records_page_and_font(self):
        _, blocks = self.parse()
        body = next(b for b in blocks if "这是第一章的正文段落" in b.text)
        self.assertEqual(body.metadata.get("page"), 1)
        self.assertAlmostEqual(body.metadata.get("font_size"), BODY_SIZE, places=1)
        self.assertIn("bbox", body.metadata)

    def test_block_ids_are_unique(self):
        _, blocks = self.parse()
        self.assertEqual(len(blocks), len({b.block_id for b in blocks}))

    def test_text_argument_is_rejected(self):
        with self.assertRaises(ValueError):
            PdfParser().parse("x.pdf", text="PDF 没有线性文本，不能只给字符串")

    def test_missing_file_raises_ingestion_error(self):
        with self.assertRaises(IngestionError):
            PdfParser().parse("no/such/file.pdf")

    def test_non_pdf_file_raises_ingestion_error(self):
        with TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake.pdf"
            fake.write_text("我不是 PDF", encoding="utf-8")
            with self.assertRaises(IngestionError):
                PdfParser().parse(str(fake))


@unittest.skipUnless(REAL_PDF.exists(), f"缺少真实 PDF {REAL_PDF}")
@unittest.skipUnless(HAVE_PYMUPDF, "需要 pymupdf：pip install -r requirements-pdf.txt")
class RealPdfTest(unittest.TestCase):
    PAGES = 30

    @classmethod
    def setUpClass(cls):
        cls.parser = PdfParser()
        cls.blocks = cls.parser.parse(str(REAL_PDF), page_limit=cls.PAGES)

    def test_page_number_coverage_is_full(self):
        """D04 验收：`page_number` 有值率 100%。"""
        missing = [b for b in self.blocks if b.page_number is None]
        self.assertEqual(missing, [], f"{len(missing)} 个 block 缺页码")
        coverage = sum(1 for b in self.blocks if b.page_number is not None) / len(self.blocks)
        print(f"\n  [D04] {self.PAGES} 页 → {len(self.blocks)} 个 block，page_number 覆盖率 {coverage:.1%}")

    def test_char_count_within_5_percent_of_pypdf(self):
        """D04 验收：抽取字符数与 pypdf 对比 ±5%。"""
        from pypdf import PdfReader

        reader = PdfReader(str(REAL_PDF))
        pypdf_chars = sum(len(reader.pages[i].extract_text() or "") for i in range(self.PAGES))
        our_chars = sum(len(b.text) for b in self.blocks)
        ratio = our_chars / pypdf_chars if pypdf_chars else 0
        print(
            f"  [D04] pymupdf 抽取 {our_chars:,} 字符 / pypdf {pypdf_chars:,} 字符，比值 {ratio:.3f}"
        )
        self.assertGreater(abs(ratio - 1), 0.0, "两者完全相等反而可疑，需确认不是同一实现")
        self.assertLess(abs(ratio - 1), 0.05, f"字符数偏差 {abs(ratio - 1):.1%} 超出 ±5%")

    def test_some_headings_are_detected(self):
        headings = [b for b in self.blocks if b.block_type == "heading"]
        ratio = len(headings) / len(self.blocks)
        print(f"  [D04] 识别到标题 {len(headings)} 个（占 {ratio:.1%}）")
        self.assertGreater(len(headings), 0, "一个标题都没识别到，字号启发式失效")

    def test_section_path_is_never_empty(self):
        for block in self.blocks:
            self.assertTrue(block.section_path)

    def test_blocks_are_page_ordered(self):
        pages = [b.page_number for b in self.blocks]
        self.assertEqual(pages, sorted(pages))


@unittest.skipUnless(HAVE_PYMUPDF, "需要 pymupdf：pip install -r requirements-pdf.txt")
class CliReportTest(unittest.TestCase):
    """D04 验收要的「30 页人工表」由 report 子命令生成。"""

    def test_report_writes_one_row_per_page(self):
        import csv
        import io
        from contextlib import redirect_stdout

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            pdf = build_pdf(base / "two.pdf", [[("第一页正文。", BODY_SIZE)], [("第二页正文。", BODY_SIZE)]])
            out_csv = base / "report.csv"
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli_main(["report", pdf, "--pages", "1-2", "--out", str(out_csv)])
            self.assertEqual(code, 0)
            with out_csv.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["page"] for r in rows], ["1", "2"])
            for column in ("page", "chars", "blocks", "headings", "sample"):
                self.assertIn(column, rows[0])
            self.assertIn("2 页", out.getvalue(), "应回报写入了多少页")

    def test_report_rejects_bad_page_range(self):
        import io
        from contextlib import redirect_stdout

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            pdf = build_pdf(base / "one.pdf", [[("正文。", BODY_SIZE)]])
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli_main(["report", pdf, "--pages", "5-1"])
            self.assertEqual(code, 2)
            self.assertIn("页码范围", out.getvalue())


@unittest.skipUnless(HAVE_PYMUPDF, "需要 pymupdf：pip install -r requirements-pdf.txt")
class RegistryTest(unittest.TestCase):
    def setUp(self):
        reset_registry()
        register_builtin_parsers()

    def tearDown(self):
        reset_registry()

    def test_pdf_suffix_is_registered(self):
        for suffix in (".pdf", ".PDF"):
            self.assertIsInstance(get_parser(f"x{suffix}"), PdfParser)


if __name__ == "__main__":
    unittest.main()
