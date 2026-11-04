"""`section_path` 从 SQLite 出来是 **JSON 字符串**，不是 list —— 这条边界必须被钉住。

踩过的坑（N11，2026-11-04 追踪 query 时发现）：`parents.section_path` 落库时是
`'["深入理解AI Agent", "第3 章…"]'`，`load_parent_records()` 原样返回，
`assemble()` 里 `list(...)` 会把字符串拆成**单个字符**，于是 prompt 的每块头部变成

    section=[ > " > 深 > 入 > 理 > 解 > A > I ...

每个证据块白吃 100+ 字符，引用解析拿到的 section 也是乱的。
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.rag.context.assembler import assemble
from src.rag.ingestion.tokenizer import TokenCounter
from src.rag.retrieval.store_wiring import load_parent_records
from src.rag.store.document_store import DocumentStore


def _seed(db: Path, section_path=("深入理解AI Agent", "第3 章用户记忆和知识库")):
    store = DocumentStore(db)
    document = {
        "document_id": "doc_1", "source_uri": "a.md", "file_name": "a.md", "file_type": "md",
        "title": "深入理解AI Agent", "language": None, "document_version": "sha256:v1",
        "parser_version": "md_v1", "created_at": None, "status": "active",
    }
    parents = [{
        "parent_id": "par_1", "document_id": "doc_1", "document_version": "sha256:v1",
        # DocumentStore 内部会 json.dumps 成 JSON 文本落库（和真实入库路径一致）
        "language": None, "title": "深入理解AI Agent",
        "section_path": list(section_path) if section_path else None,
        "content": "智能体化 RAG 把检索变成 Agent 可调用的工具。", "token_count": 18,
        "source_start": 0, "source_end": 20, "chunker_version": "v1", "parent_index": 0,
    }]
    children = [{
        "chunk_id": "chk_1", "parent_id": "par_1", "document_id": "doc_1",
        "document_version": "sha256:v1", "content_hash": "sha256:c1", "token_count": 18,
        "source_start": 0, "source_end": 20, "embedding_version": "v1", "status": "active",
    }]
    store.replace_document(document, parents, children)
    return db


class LoadParentRecordsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _seed(Path(self.tmp.name) / "documents.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_section_path_is_a_list_of_strings(self):
        record = load_parent_records(self.db)[0]
        self.assertIsInstance(record["section_path"], list)
        self.assertEqual(record["section_path"], ["深入理解AI Agent", "第3 章用户记忆和知识库"])

    def test_parent_index_and_document_id_survive(self):
        """D32 的相邻合并只认 (document_id, parent_index)，这两个字段不能缺。"""
        record = load_parent_records(self.db)[0]
        self.assertEqual(record["document_id"], "doc_1")
        self.assertEqual(record["parent_index"], 0)

    def test_null_section_path_becomes_empty_list(self):
        db = _seed(Path(self.tmp.name) / "null.sqlite3", section_path=None)
        self.assertEqual(load_parent_records(db)[0]["section_path"], [])

    def test_invalid_json_does_not_raise(self):
        """库里存了坏 JSON（历史脏数据/手工改动）不能让整条链路炸。"""
        db = _seed(Path(self.tmp.name) / "bad.sqlite3")
        # Windows 上 `with sqlite3.connect()` 只提交不关闭 → 必须显式 close，否则删不掉临时库
        conn = sqlite3.connect(db)
        try:
            conn.execute("UPDATE parents SET section_path = 'not-json'")
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(load_parent_records(db)[0]["section_path"], [])


class AssemblerSectionPathTest(unittest.TestCase):
    """即使上游给的是 JSON 字符串，组装器也不许把它拆成字符（防御在消费侧再写一层）。"""

    def _parents(self, section_path):
        return {"par_1": {"document_id": "doc_1", "parent_index": 0, "title": "t",
                          "source_uri": "a.md", "section_path": section_path,
                          "text": "智能体化 RAG 把检索变成 Agent 可调用的工具。"}}

    def test_string_section_path_is_parsed_not_iterated(self):
        assembled = assemble([{"parent_id": "par_1"}],
                             self._parents('["深入理解AI Agent", "第3 章"]'),
                             counter=TokenCounter(), budget=6000)
        self.assertEqual(assembled.blocks[0].section_path, ["深入理解AI Agent", "第3 章"])
        self.assertNotIn(" > 深", assembled.render())

    def test_list_section_path_unchanged(self):
        assembled = assemble([{"parent_id": "par_1"}], self._parents(["A", "B"]),
                             counter=TokenCounter(), budget=6000)
        self.assertEqual(assembled.blocks[0].section_path, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
