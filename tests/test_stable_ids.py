"""D10 稳定 ID 与内容哈希（方案 §3.4）。

D10 验收：
  - 同一输入连跑两次，`sha256sum chunks.jsonl` 完全一致；
  - 改 `chunker_version` 后全量 ID 变化（负向测试）。

ID 规则（方案 §3.4）：
  parent_id = hash(document_id + document_version + parent_index + chunker_version)
  chunk_id  = hash(parent_id + child_index + content_hash + chunker_version)
"""

import re
import unittest

from src.rag.ingestion.ids import (
    ID_PATTERN,
    content_hash,
    document_version,
    make_chunk_id,
    make_document_id,
    make_parent_id,
)

HEX32 = re.compile(r"^[0-9a-f]{32}$")


class FormatTest(unittest.TestCase):
    def test_id_prefixes_and_length(self):
        document_id = make_document_id("data/rag_basics.md")
        parent_id = make_parent_id(document_id, "sha256:abc", 0, "v1")
        chunk_id = make_chunk_id(parent_id, 0, "sha256:def", "v1")

        self.assertTrue(document_id.startswith("doc_"))
        self.assertTrue(parent_id.startswith("par_"))
        self.assertTrue(chunk_id.startswith("chk_"))
        for value in (document_id, parent_id, chunk_id):
            self.assertTrue(HEX32.match(value.split("_", 1)[1]), f"ID 主体应为 32 位小写十六进制: {value}")
            self.assertTrue(ID_PATTERN.match(value), value)

    def test_hash_fields_carry_sha256_prefix(self):
        self.assertTrue(content_hash("abc").startswith("sha256:"))
        self.assertEqual(len(content_hash("abc")), len("sha256:") + 64)
        self.assertTrue(document_version("abc").startswith("sha256:"))


class DeterminismTest(unittest.TestCase):
    def test_same_inputs_give_same_ids(self):
        args = ("doc_1", "sha256:v", 3, "chunker_v1")
        self.assertEqual(make_parent_id(*args), make_parent_id(*args))
        chunk_args = ("par_1", 2, "sha256:c", "chunker_v1")
        self.assertEqual(make_chunk_id(*chunk_args), make_chunk_id(*chunk_args))
        self.assertEqual(make_document_id("a/b.md"), make_document_id("a/b.md"))

    def test_each_input_component_changes_the_id(self):
        base_parent = make_parent_id("doc_1", "sha256:v", 0, "chunker_v1")
        variants = [
            make_parent_id("doc_2", "sha256:v", 0, "chunker_v1"),
            make_parent_id("doc_1", "sha256:v2", 0, "chunker_v1"),
            make_parent_id("doc_1", "sha256:v", 1, "chunker_v1"),
            make_parent_id("doc_1", "sha256:v", 0, "chunker_v2"),
        ]
        self.assertEqual(len(set(variants + [base_parent])), 5, "任一输入变化都必须改变 parent_id")

        base_chunk = make_chunk_id("par_1", 0, "sha256:c", "chunker_v1")
        chunk_variants = [
            make_chunk_id("par_2", 0, "sha256:c", "chunker_v1"),
            make_chunk_id("par_1", 1, "sha256:c", "chunker_v1"),
            make_chunk_id("par_1", 0, "sha256:c2", "chunker_v1"),
            make_chunk_id("par_1", 0, "sha256:c", "chunker_v2"),
        ]
        self.assertEqual(len(set(chunk_variants + [base_chunk])), 5)

    def test_document_id_is_stable_across_content_changes(self):
        """文档 ID 认来源（路径），不认内容；内容变化只改 document_version。"""
        self.assertEqual(make_document_id("a/b.md"), make_document_id("a/b.md"))
        self.assertNotEqual(document_version("v1"), document_version("v2"))

    def test_content_hash_is_sensitive_to_whitespace(self):
        self.assertNotEqual(content_hash("abc"), content_hash("abc "))
        self.assertNotEqual(content_hash("abc"), content_hash("ABC"))


class NegativeChunkerVersionTest(unittest.TestCase):
    """D10 的负向验收：改 chunker_version 后**全量** ID 必须变化。"""

    def test_all_ids_change_when_chunker_version_changes(self):
        corpus = [
            ("doc_1", "sha256:v1", ["第一段正文", "第二段正文"]),
            ("doc_2", "sha256:v2", ["另一篇正文"]),
        ]
        old, new = set(), set()
        for document_id, version, paragraphs in corpus:
            for index, text in enumerate(paragraphs):
                old.add(make_parent_id(document_id, version, index, "chunker_v1"))
                new.add(make_parent_id(document_id, version, index, "chunker_v2"))
        self.assertEqual(old & new, set(), "chunker_version 变化后不允许有任何 ID 复用")


if __name__ == "__main__":
    unittest.main()
