"""D03 TXT Parser 的单元测试。

D03 验收：输出 block 数 ≥ 空行分隔段数；空 block 数 == 0。

TXT 没有结构标记，所以解析器只做**分段**，不做长度控制（长度是 D06 的事）。
额外提供句子数统计（方案 §2.2「段落、句子识别」），供 D09 Validator 用。
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag.ingestion.parser import get_parser, register_builtin_parsers, reset_registry
from src.rag.ingestion.parser.txt_parser import TxtParser

SAMPLE = """第一段第一行。
第一段第二行。

第二段只有一行。

第三段开头。

第四段。
"""


class ParagraphTest(unittest.TestCase):
    def test_block_count_is_at_least_blank_line_paragraphs(self):
        blocks = TxtParser().parse("a.txt", text=SAMPLE)
        paragraphs = [p for p in SAMPLE.split("\n\n") if p.strip()]
        self.assertGreaterEqual(len(blocks), len(paragraphs))

    def test_no_empty_blocks(self):
        blocks = TxtParser().parse("a.txt", text=SAMPLE)
        self.assertTrue(blocks)
        for block in blocks:
            self.assertTrue(block.text.strip(), f"出现空 block: {block.to_dict()}")

    def test_paragraph_text_is_preserved(self):
        blocks = TxtParser().parse("a.txt", text=SAMPLE)
        self.assertIn("第一段第一行", blocks[0].text)
        self.assertIn("第一段第二行", blocks[0].text, "同一空行段内的软换行应留在同一段")
        self.assertNotIn("第二段只有一行", blocks[0].text, "空行必须切断段落")
        joined = "\n\n".join(b.text for b in blocks)
        for piece in ("第一段第一行", "第二段只有一行", "第三段开头", "第四段"):
            self.assertIn(piece, joined)

    def test_whitespace_only_file_yields_nothing(self):
        for text in ("", "\n\n\n", "   \n\t\n"):
            self.assertEqual(TxtParser().parse("a.txt", text=text), [])

    def test_crlf_is_normalised(self):
        blocks = TxtParser().parse("a.txt", text="第一段。\r\n\r\n第二段。\r\n")
        self.assertEqual(len(blocks), 2)
        self.assertNotIn("\r", "".join(b.text for b in blocks))

    def test_bom_is_stripped(self):
        blocks = TxtParser().parse("a.txt", text="\ufeff第一段。\n\n第二段。\n")
        self.assertTrue(blocks[0].text.startswith("第一段"))

    def test_excess_blank_lines_do_not_create_empty_paragraphs(self):
        blocks = TxtParser().parse("a.txt", text="段落一\n\n\n\n\n段落二\n")
        self.assertEqual([b.text for b in blocks], ["段落一", "段落二"])


class SentenceMetadataTest(unittest.TestCase):
    def test_sentence_count_is_recorded(self):
        blocks = TxtParser().parse("a.txt", text="第一句。第二句！第三句？\n")
        self.assertEqual(blocks[0].metadata.get("n_sentences"), 3)

    def test_english_sentences_are_counted(self):
        blocks = TxtParser().parse("a.txt", text="First one. Second one! Done?\n")
        self.assertEqual(blocks[0].metadata.get("n_sentences"), 3)

    def test_text_without_punctuation_counts_as_one(self):
        blocks = TxtParser().parse("a.txt", text="没有任何标点的一段文字\n")
        self.assertEqual(blocks[0].metadata.get("n_sentences"), 1)


class StructureTest(unittest.TestCase):
    def test_every_block_has_section_path(self):
        blocks = TxtParser().parse("readme.txt", text=SAMPLE)
        for block in blocks:
            self.assertEqual(block.section_path, ["readme"], "TXT 无章节结构，应回退到文件名")

    def test_offsets_point_into_source(self):
        blocks = TxtParser().parse("a.txt", text=SAMPLE)
        for block in blocks:
            self.assertIsNotNone(block.source_start)
            self.assertLessEqual(block.source_start, block.source_end)
            self.assertTrue(SAMPLE[block.source_start : block.source_end].strip())

    def test_block_ids_are_unique(self):
        blocks = TxtParser().parse("a.txt", text=SAMPLE)
        self.assertEqual(len(blocks), len({b.block_id for b in blocks}))

    def test_all_blocks_are_paragraphs(self):
        blocks = TxtParser().parse("a.txt", text=SAMPLE)
        self.assertEqual({b.block_type for b in blocks}, {"paragraph"})

    def test_parse_reads_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.txt"
            path.write_text("磁盘段落一。\n\n磁盘段落二。\n", encoding="utf-8")
            blocks = TxtParser().parse(str(path))
            self.assertEqual([b.text for b in blocks], ["磁盘段落一。", "磁盘段落二。"])


class RegistryTest(unittest.TestCase):
    def setUp(self):
        reset_registry()
        register_builtin_parsers()

    def tearDown(self):
        reset_registry()

    def test_txt_suffix_is_registered(self):
        for suffix in (".txt", ".text", ".TXT"):
            self.assertIsInstance(get_parser(f"x{suffix}"), TxtParser)


if __name__ == "__main__":
    unittest.main()
