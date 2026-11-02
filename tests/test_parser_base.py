"""D01：Parser 基类与注册表契约（方案 §2.2「文档解析统一接口」）。"""

import unittest

from src.rag.ingestion.block import DocumentBlock
from src.rag.ingestion.errors import UnsupportedFormatError
from src.rag.ingestion.parsers import (
    BaseParser,
    available_suffixes,
    get_parser,
    register_parser,
    reset_registry,
)


class FakeMarkdownParser(BaseParser):
    parser_version = "fake_v1"
    supported_suffixes = (".md", ".markdown")

    def parse(self, path, text=None):
        self.received = (str(path), text)
        return [
            DocumentBlock(block_id="b1", block_type="heading", text="标题", heading_level=1),
            DocumentBlock(block_id="b2", block_type="paragraph", text="正文",
                          section_path=["标题"]),
        ]


class BaseParserContractTest(unittest.TestCase):
    def setUp(self):
        reset_registry()

    def tearDown(self):
        reset_registry()

    def test_base_parser_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            BaseParser()

    def test_subclass_without_parse_is_rejected(self):
        class Incomplete(BaseParser):
            supported_suffixes = (".txt",)

        with self.assertRaises(TypeError):
            Incomplete()

    def test_parser_version_is_required_metadata(self):
        self.assertTrue(FakeMarkdownParser.parser_version)

    def test_parse_returns_document_blocks(self):
        blocks = FakeMarkdownParser().parse("a.md", text="# 标题\n\n正文")
        self.assertEqual([b.block_type for b in blocks], ["heading", "paragraph"])
        self.assertTrue(all(isinstance(b, DocumentBlock) for b in blocks))


class RegistryTest(unittest.TestCase):
    def setUp(self):
        reset_registry()
        register_parser(FakeMarkdownParser)

    def tearDown(self):
        reset_registry()

    def test_get_parser_matches_suffix(self):
        self.assertIsInstance(get_parser("notes.md"), FakeMarkdownParser)

    def test_suffix_match_is_case_insensitive(self):
        self.assertIsInstance(get_parser("NOTES.MD"), FakeMarkdownParser)

    def test_nested_path_without_suffix_uses_suffix_of_path(self):
        self.assertIsInstance(get_parser("data/raw/deep/a.markdown"), FakeMarkdownParser)

    def test_unknown_suffix_raises_with_supported_list(self):
        with self.assertRaises(UnsupportedFormatError) as ctx:
            get_parser("a.docx")
        message = str(ctx.exception)
        self.assertIn(".docx", message)
        self.assertIn(".md", message)

    def test_available_suffixes_is_sorted_and_complete(self):
        self.assertEqual(available_suffixes(), [".markdown", ".md"])

    def test_duplicate_suffix_registration_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            register_parser(FakeMarkdownParser)
        self.assertIn(".md", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
