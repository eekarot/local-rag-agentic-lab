"""D02 Markdown Parser 的单元测试。

D02 验收行点名的两个陷阱必须各有用例：
  1. 代码块内含 `#` —— 不能被当成标题；
  2. 表格紧跟标题 —— 不能被降级成 paragraph。

另外锁住两件下游依赖的事：
  - `section_path` 非空率 100%（D08 Parent Builder 靠它划章节边界）；
  - 标题块绝对不是 paragraph，且正文的 `section_path[-1]` 就是它所属的标题。
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag.ingestion.block import DocumentBlock
from src.rag.ingestion.parser.markdown_parser import MarkdownParser
from src.rag.ingestion.parser import get_parser, register_builtin_parsers, reset_registry

FENCE = "```"

SAMPLE = f"""# RAG 基础

第一段正文，讲 RAG 是什么。
这一行还在同一段里。

## 检索模型

| 模型 | 维度 | 语言 |
|---|---|---|
| nomic-embed-text | 768 | 英文 |
| bge-m3 | 1024 | 多语言 |

{FENCE}python
# 这是 Python 注释，不是 Markdown 标题
def score(x):
    return x * 2
{FENCE}

### 混合检索

- 向量检索
- BM25 检索

1. 第一步
2. 第二步

> 引用：切分粒度决定了召回上限。
> 第二行引用。

![流程图](img/pipeline.png)

***

收尾段落。
"""


def parse(text=SAMPLE, path="sample.md"):
    return MarkdownParser().parse(path, text=text)


def find(blocks, block_type):
    return [b for b in blocks if b.block_type == block_type]


class HeadingTest(unittest.TestCase):
    def test_atx_heading_levels_are_detected(self):
        blocks = parse()
        headings = [(b.heading_level, b.text) for b in find(blocks, "heading")]
        self.assertIn((1, "RAG 基础"), headings)
        self.assertIn((2, "检索模型"), headings)
        self.assertIn((3, "混合检索"), headings)

    def test_heading_text_strips_markers(self):
        blocks = parse()
        for block in find(blocks, "heading"):
            self.assertFalse(block.text.startswith("#"), f"标题文本残留井号: {block.text!r}")
            self.assertEqual(block.text, block.text.strip())

    def test_closing_hashes_are_stripped(self):
        blocks = parse("# 标题 ##\n\n正文\n")
        self.assertEqual(blocks[0].text, "标题")

    def test_setext_headings_are_supported(self):
        text = "Setext 一级\n=====\n\n正文一。\n\nSetext 二级\n-----\n\n正文二。\n"
        headings = find(parse(text), "heading")
        self.assertEqual([(b.heading_level, b.text) for b in headings], [(1, "Setext 一级"), (2, "Setext 二级")])


class CodeBlockTest(unittest.TestCase):
    def test_code_block_is_one_single_block(self):
        blocks = parse()
        code = find(blocks, "code")
        self.assertEqual(len(code), 1)
        self.assertIn("def score(x):", code[0].text)
        self.assertIn("return x * 2", code[0].text)

    def test_hash_inside_code_block_is_not_a_heading(self):
        """陷阱 1：代码注释里的 `#` 绝不能被解析成标题。"""
        blocks = parse()
        heading_texts = {b.text for b in find(blocks, "heading")}
        self.assertNotIn("这是 Python 注释，不是 Markdown 标题", heading_texts)
        self.assertFalse(any("Python 注释" in t for t in heading_texts))
        # 而且这行必须留在 code 块里
        self.assertTrue(any("这是 Python 注释" in b.text for b in find(blocks, "code")))

    def test_code_language_is_recorded(self):
        code = find(parse(), "code")[0]
        self.assertEqual(code.metadata.get("language"), "python")

    def test_unclosed_fence_still_consumes_to_end(self):
        text = f"# 标题\n\n{FENCE}js\nconst a = 1;\n"
        blocks = parse(text)
        self.assertEqual([b.block_type for b in blocks], ["heading", "code"])
        self.assertIn("const a = 1;", blocks[1].text)


class TableTest(unittest.TestCase):
    def test_table_directly_after_heading_stays_table(self):
        """陷阱 2：标题紧接表格时，表格不能被当作正文段落。"""
        text = "| a | b |\n|---|---|\n| 1 | 2 |\n"
        blocks = parse(text)
        self.assertEqual([b.block_type for b in blocks], ["table"], "表格被降级成了非 table 块")

    def test_table_after_heading_keeps_heading_above_it(self):
        text = "## 指标\n| a | b |\n|---|---|\n| 1 | 2 |\n"
        blocks = parse(text)
        self.assertEqual([b.block_type for b in blocks], ["heading", "table"])
        self.assertEqual(blocks[1].section_path[-1], "指标")

    def test_table_shape_is_recorded(self):
        table = find(parse(), "table")[0]
        self.assertEqual(table.metadata.get("n_rows"), 3)  # 表头 + 2 行数据
        self.assertEqual(table.metadata.get("n_cols"), 3)
        self.assertTrue(table.metadata.get("has_header"))
        self.assertEqual(table.text.count("\n"), 3)

    def test_paragraph_with_pipe_is_not_a_table(self):
        text = "这不是表格 | 只是正文里的竖线\n\n下一段。\n"
        blocks = parse(text)
        self.assertEqual([b.block_type for b in blocks], ["paragraph", "paragraph"])


class ListAndQuoteTest(unittest.TestCase):
    def test_unordered_list_is_one_block(self):
        lists = find(parse(), "list")
        self.assertTrue(any(b.metadata.get("ordered") is False for b in lists))
        unordered = next(b for b in lists if b.metadata.get("ordered") is False)
        self.assertIn("向量检索", unordered.text)
        self.assertIn("BM25 检索", unordered.text)

    def test_ordered_list_is_marked_ordered(self):
        ordered = [b for b in find(parse(), "list") if b.metadata.get("ordered")]
        self.assertEqual(len(ordered), 1)
        self.assertEqual(ordered[0].metadata.get("n_items"), 2)

    def test_blockquote_is_its_own_type(self):
        quotes = find(parse(), "quote")
        self.assertEqual(len(quotes), 1)
        self.assertIn("切分粒度决定了召回上限", quotes[0].text)
        self.assertFalse(quotes[0].text.lstrip().startswith(">"), "引用块文本应去掉 > 前缀")

    def test_horizontal_rule_is_skipped(self):
        blocks = parse()
        self.assertFalse(any(b.text.strip() in {"***", "---"} for b in blocks))
        self.assertFalse(any(b.block_type == "heading" and not b.text for b in blocks))

    def test_image_line_becomes_caption(self):
        captions = find(parse(), "caption")
        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0].metadata.get("alt"), "流程图")
        self.assertEqual(captions[0].metadata.get("src"), "img/pipeline.png")


class SectionPathTest(unittest.TestCase):
    def test_section_path_is_never_empty(self):
        """D02 验收：`section_path` 非空率 100%。"""
        blocks = parse()
        self.assertTrue(blocks)
        empty = [b for b in blocks if not b.section_path]
        self.assertEqual(empty, [], f"有 {len(empty)} 个 block 的 section_path 为空")

    def test_paragraph_inherits_its_heading(self):
        blocks = parse()
        paragraph = next(b for b in blocks if b.block_type == "paragraph" and "第一段正文" in b.text)
        self.assertEqual(paragraph.section_path, ["RAG 基础"])

    def test_nested_sections_stack_correctly(self):
        blocks = parse()
        table = find(blocks, "table")[0]
        self.assertEqual(table.section_path, ["RAG 基础", "检索模型"])
        tail = next(b for b in blocks if "收尾段落" in b.text)
        self.assertEqual(
            tail.section_path,
            ["RAG 基础", "检索模型", "混合检索"],
            "`***` 只是视觉分隔，不改变章节归属",
        )

    def test_content_before_first_heading_falls_back_to_document_title(self):
        blocks = MarkdownParser().parse("my-doc.md", text="开头没有标题的正文。\n\n# 后面才有标题\n")
        self.assertEqual(blocks[0].section_path, ["my-doc"])

    def test_heading_block_path_includes_itself(self):
        heading = next(b for b in find(parse(), "heading") if b.text == "混合检索")
        self.assertEqual(heading.section_path, ["RAG 基础", "检索模型", "混合检索"])


class BlockIntegrityTest(unittest.TestCase):
    def test_no_empty_blocks(self):
        for block in parse():
            self.assertTrue(block.text.strip(), f"出现空 block: {block.to_dict()}")

    def test_block_ids_are_unique(self):
        ids = [b.block_id for b in parse()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_heading_is_never_merged_into_paragraph(self):
        """「不要把标题和正文分离」的反面：标题也不能被吞进正文段落。"""
        blocks = parse()
        for index, block in enumerate(blocks):
            if block.block_type != "heading":
                continue
            self.assertNotIn("\n", block.text, "标题块里混进了正文")
            if index + 1 < len(blocks):
                self.assertNotIn(block.text, blocks[index + 1].text)

    def test_source_offsets_cover_the_raw_block(self):
        source = SAMPLE
        blocks = parse(source)
        for block in blocks:
            self.assertIsNotNone(block.source_start)
            self.assertIsNotNone(block.source_end)
            self.assertLessEqual(block.source_start, block.source_end)
            raw = source[block.source_start : block.source_end]
            self.assertTrue(raw.strip(), "偏移区间内没有内容")
        heading = next(b for b in blocks if b.text == "检索模型")
        self.assertTrue(source[heading.source_start : heading.source_end].startswith("## 检索模型"))

    def test_empty_document_yields_no_blocks(self):
        self.assertEqual(parse(""), [])
        self.assertEqual(parse("\n\n   \n"), [])

    def test_crlf_is_normalised(self):
        blocks = parse("# 标题\r\n\r\n正文行一。\r\n正文行二。\r\n")
        self.assertEqual([b.block_type for b in blocks], ["heading", "paragraph"])
        self.assertNotIn("\r", blocks[1].text)
        self.assertIn("正文行二", blocks[1].text)

    def test_utf8_bom_is_stripped(self):
        blocks = MarkdownParser().parse("bom.md", text="\ufeff# 带 BOM 的标题\n\n正文\n")
        self.assertEqual(blocks[0].text, "带 BOM 的标题")

    def test_parse_reads_file_when_text_not_given(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.md"
            path.write_text("# 磁盘标题\n\n磁盘正文。\n", encoding="utf-8")
            blocks = MarkdownParser().parse(str(path))
            self.assertEqual(blocks[0].text, "磁盘标题")
            self.assertIn("磁盘正文", blocks[1].text)


class RegistryTest(unittest.TestCase):
    def setUp(self):
        reset_registry()
        register_builtin_parsers()

    def tearDown(self):
        reset_registry()

    def test_markdown_suffixes_are_registered(self):
        for suffix in (".md", ".markdown", ".MD"):
            self.assertIsInstance(get_parser(f"x{suffix}"), MarkdownParser)

    def test_parser_declares_version(self):
        self.assertTrue(MarkdownParser().parser_version)


if __name__ == "__main__":
    unittest.main()
