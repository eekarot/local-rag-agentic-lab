"""D01：DocumentBlock 数据结构契约（方案 §2.2）。"""

import unittest

from src.rag.ingestion.block import BLOCK_TYPES, DocumentBlock


def make(**overrides):
    kwargs = {"block_id": "b-001", "block_type": "paragraph", "text": "正文"}
    kwargs.update(overrides)
    return DocumentBlock(**kwargs)


class BlockTypeTest(unittest.TestCase):
    def test_accepts_every_declared_block_type(self):
        self.assertEqual(
            set(BLOCK_TYPES),
            {"heading", "paragraph", "list", "table", "code", "quote", "caption"},
        )
        for block_type in BLOCK_TYPES:
            block = make(
                block_type=block_type,
                heading_level=1 if block_type == "heading" else None,
            )
            self.assertEqual(block.block_type, block_type)

    def test_rejects_unknown_block_type(self):
        with self.assertRaises(ValueError) as ctx:
            make(block_type="paragraphs")
        # 错误信息必须列出合法取值，否则调用方无从修正
        self.assertIn("paragraph", str(ctx.exception))
        self.assertIn("heading", str(ctx.exception))


class FieldValidationTest(unittest.TestCase):
    def test_block_id_must_be_non_empty(self):
        for bad in ("", "   ", None):
            with self.subTest(block_id=bad), self.assertRaises(ValueError):
                make(block_id=bad)

    def test_text_must_be_string(self):
        with self.assertRaises(ValueError):
            make(text=None)

    def test_heading_requires_heading_level(self):
        with self.assertRaises(ValueError) as ctx:
            make(block_type="heading", text="1. 简介")
        self.assertIn("heading_level", str(ctx.exception))

    def test_heading_level_out_of_range_rejected(self):
        for bad in (0, 7, -1):
            with self.subTest(level=bad), self.assertRaises(ValueError):
                make(block_type="heading", text="1. 简介", heading_level=bad)

    def test_source_range_must_be_ordered(self):
        with self.assertRaises(ValueError):
            make(source_start=100, source_end=50)

    def test_source_range_allows_open_ends(self):
        self.assertEqual(make(source_start=10).source_start, 10)
        self.assertIsNone(make(source_start=10).source_end)

    def test_page_number_must_be_positive(self):
        for bad in (0, -3):
            with self.subTest(page=bad), self.assertRaises(ValueError):
                make(page_number=bad)

    def test_section_path_default_is_not_shared_between_instances(self):
        first, second = make(block_id="a"), make(block_id="b")
        first.section_path.append("Retrieval")
        self.assertEqual(second.section_path, [])

    def test_metadata_default_is_not_shared_between_instances(self):
        first, second = make(block_id="a"), make(block_id="b")
        first.metadata["k"] = 1
        self.assertEqual(second.metadata, {})


class SerializationTest(unittest.TestCase):
    def test_to_dict_from_dict_roundtrip(self):
        block = make(
            block_type="heading",
            text="2.1 HNSW",
            heading_level=2,
            page_number=3,
            section_path=["Retrieval", "HNSW"],
            source_start=120,
            source_end=131,
            metadata={"lang": "en"},
        )
        self.assertEqual(DocumentBlock.from_dict(block.to_dict()), block)

    def test_to_dict_keeps_all_spec_fields(self):
        self.assertEqual(
            set(make().to_dict()),
            {
                "block_id",
                "block_type",
                "text",
                "heading_level",
                "page_number",
                "section_path",
                "source_start",
                "source_end",
                "metadata",
            },
        )


if __name__ == "__main__":
    unittest.main()
