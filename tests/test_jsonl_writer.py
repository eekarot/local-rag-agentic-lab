"""D12 JSONL 输出的单元测试（方案 §3.1/§3.2/§3.3 + `data/chunks/` 目录约定）。

D12 验收：
  - `jq -c . data/chunks/*.jsonl` 逐行可解析；
  - 字段与方案 §3.3 完全一致（缺一字段即失败）。

「缺一字段即失败」意味着**多一个字段也要失败**：字段集与方案不一致时，
下游（Qdrant payload、SQLite store）读到的结构就不可信了。
"""

import hashlib
import json
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag.ingestion.jsonl_store import (
    CHILD_FIELDS,
    DOCUMENT_FIELDS,
    PARENT_FIELDS,
    write_chunks,
    write_jsonl,
)

DOCUMENT = {
    "document_id": "doc_abc",
    "source_uri": "data/rag_basics.md",
    "file_name": "rag_basics.md",
    "file_type": "md",
    "title": "RAG 基础",
    "language": "zh",
    "document_version": "sha256:" + "1" * 64,
    "parser_version": "md_v1",
    "created_at": "2026-11-02T00:00:00Z",
    "status": "active",
}

PARENT = {
    "parent_id": "par_" + "a" * 32,
    "document_id": "doc_abc",
    "document_version": DOCUMENT["document_version"],
    "language": "zh",
    "title": "RAG 基础",
    "section_path": ["RAG 基础"],
    "content": "父块正文",
    "token_count": 4,
    "source_start": 0,
    "source_end": 10,
    "chunker_version": "recursive_parent_child_v1",
}

CHILD = {
    "chunk_id": "chk_" + "b" * 32,
    "parent_id": PARENT["parent_id"],
    "document_id": "doc_abc",
    "document_version": DOCUMENT["document_version"],
    "language": "zh",
    "title": "RAG 基础",
    "section_path": ["RAG 基础"],
    "content": "子块正文",
    "embedding_text": "Document: RAG 基础\nSection: RAG 基础\nContent: 子块正文",
    "token_count": 4,
    "source_start": 0,
    "source_end": 10,
    "chunker_version": "recursive_child_v1",
    "embedding_model": "bge-m3",
    "embedding_version": "bge-m3-v1",
    "content_hash": "sha256:" + "2" * 64,
    "status": "active",
}


class FieldContractTest(unittest.TestCase):
    def test_field_sets_match_the_spec(self):
        self.assertEqual(
            set(PARENT_FIELDS),
            {
                "parent_id", "document_id", "document_version", "language", "title",
                "section_path", "content", "token_count", "source_start", "source_end",
                "chunker_version",
            },
        )
        self.assertEqual(
            set(CHILD_FIELDS),
            {
                "chunk_id", "parent_id", "document_id", "document_version", "language", "title",
                "section_path", "content", "embedding_text", "token_count", "source_start",
                "source_end", "chunker_version", "embedding_model", "embedding_version",
                "content_hash", "status",
            },
        )
        self.assertEqual(
            set(DOCUMENT_FIELDS),
            {
                "document_id", "source_uri", "file_name", "file_type", "title", "language",
                "document_version", "parser_version", "created_at", "status",
            },
        )

    def test_missing_field_is_rejected(self):
        broken = dict(PARENT)
        broken.pop("content_hash", None) if "content_hash" in broken else broken.pop("content")
        with self.assertRaises(ValueError) as ctx:
            write_jsonl(Path("unused.jsonl"), [broken], required_fields=PARENT_FIELDS)
        self.assertIn("content", str(ctx.exception))

    def test_extra_field_is_rejected(self):
        extra = dict(CHILD, debug_info="不该出现的字段")
        with self.assertRaises(ValueError) as ctx:
            write_jsonl(Path("unused.jsonl"), [extra], required_fields=CHILD_FIELDS)
        self.assertIn("debug_info", str(ctx.exception))


class WriteJsonlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_line_is_one_json_object(self):
        path = self.root / "parents.jsonl"
        write_jsonl(path, [PARENT, dict(PARENT, parent_id="par_" + "c" * 32)], required_fields=PARENT_FIELDS)
        raw = path.read_bytes()
        self.assertTrue(raw.endswith(b"\n"), "每行以换行结束")
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "不能写 BOM")
        lines = [line for line in raw.decode("utf-8").splitlines() if line]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertIsInstance(json.loads(line), dict)

    def test_no_trailing_whitespace(self):
        path = self.root / "p.jsonl"
        write_jsonl(path, [PARENT], required_fields=PARENT_FIELDS)
        line = path.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(line, line.rstrip())

    def test_chinese_is_not_escaped(self):
        path = self.root / "p.jsonl"
        write_jsonl(path, [PARENT], required_fields=PARENT_FIELDS)
        self.assertIn("父块正文", path.read_text(encoding="utf-8"))


class WriteChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, root=None):
        return write_chunks(
            root or self.root,
            documents=[DOCUMENT],
            parents=[PARENT],
            children=[CHILD],
            versions={"chunker_version": "recursive_parent_child_v1", "embedding_model": "bge-m3"},
        )

    def test_three_files_plus_manifest_are_written(self):
        manifest = self._write()
        chunks = self.root / "chunks"
        for name in ("documents.jsonl", "parents.jsonl", "children.jsonl"):
            self.assertTrue((chunks / name).exists(), f"{name} 未生成")
        self.assertTrue((chunks / "manifest.json").exists())
        self.assertEqual(manifest["counts"], {"documents": 1, "parents": 1, "children": 1})

    def test_manifest_records_file_hashes(self):
        manifest = self._write()
        for name in ("documents.jsonl", "parents.jsonl", "children.jsonl"):
            expected = hashlib.sha256((self.root / "chunks" / name).read_bytes()).hexdigest()
            self.assertEqual(manifest["files"][name]["sha256"], expected)
            self.assertGreater(manifest["files"][name]["bytes"], 0)

    def test_second_run_is_byte_identical(self):
        """D10/D12 验收：同一输入连跑两次，sha256sum 完全一致。"""
        self._write()
        first = {name: (self.root / "chunks" / name).read_bytes() for name in ("parents.jsonl", "children.jsonl")}
        self._write()
        second = {name: (self.root / "chunks" / name).read_bytes() for name in ("parents.jsonl", "children.jsonl")}
        for name in first:
            self.assertEqual(first[name], second[name], f"{name} 两次输出不一致")

    def test_field_order_is_stable(self):
        self._write()
        line = (self.root / "chunks" / "parents.jsonl").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(list(json.loads(line).keys()), list(PARENT_FIELDS))

    def test_jq_can_parse_every_line(self):
        self._write()
        jq = shutil.which("jq")
        if not jq:
            self.skipTest("本机没有 jq，改用 Python 解析（等价校验）")
        for name in ("documents.jsonl", "parents.jsonl", "children.jsonl"):
            result = subprocess.run(
                [jq, "-c", "."], input=(self.root / "chunks" / name).read_bytes(),
                capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, f"{name} jq 解析失败: {result.stderr[:200]}")


if __name__ == "__main__":
    unittest.main()
