"""D17：SQLite Document Store（Parent 原文 + 文档元数据）。

验收（方案 §17.3）：
- 任意 `chunk_id` 可查到 parent 原文与来源路径；
- `SELECT count(*) FROM parents` == JSONL 中 parent 数。

设计取舍：
- **Parent 原文只在这里存一份**。Qdrant 的 payload 只放 Child 文本（`text`），
  检索命中后按 `parent_id` 回这儿取父块，避免每个 Child 重复存整段父文（§3.5）。
- `children` 表只存**定位用**的瘦字段（chunk_id → parent_id / document_id），
  不放向量也不放全文：向量在 Qdrant，全文在 `parents`。
- 外键强制打开：Child 指向不存在的 Parent 必须报错，不能让孤儿悄悄进库。
- 所有写操作在**一个事务**里：`replace_document` 要么整篇换掉，要么什么都不变。
"""

import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag.store.document_store import (
    DocumentStore,
    ParentNotFoundError,
    load_chunks_dir,
)

DOC = {
    "document_id": "doc_" + "a" * 32,
    "source_uri": "md/nvidia/hnsg.md",
    "file_name": "hnsg.md",
    "file_type": "md",
    "title": "HNSW Guide",
    "language": "en",
    "document_version": "sha256:v1",
    "parser_version": "md_v1",
    "created_at": "2026-11-02T00:00:00Z",
    "status": "active",
}


def _parent(index: int, *, document_id: str = DOC["document_id"], version: str = "sha256:v1") -> dict:
    return {
        "parent_id": "par_" + f"{index:032x}",
        "document_id": document_id,
        "document_version": version,
        "language": "en",
        "title": f"Section {index}",
        "section_path": ["Retrieval", f"Part {index}"],
        "content": f"父块正文 {index}。\n第二行带换行与中文。",
        "token_count": 100 + index,
        "source_start": index * 1000,
        "source_end": index * 1000 + 999,
        "chunker_version": "recursive_parent_child_v1",
    }


def _child(index: int, *, document_id: str = DOC["document_id"], version: str = "sha256:v1") -> dict:
    return {
        "chunk_id": "chk_" + f"{index:032x}",
        "parent_id": "par_" + f"{index // 2:032x}",
        "document_id": document_id,
        "document_version": version,
        "language": "en",
        "title": f"Section {index}",
        "section_path": ["Retrieval", f"Part {index}"],
        "content": f"子块正文 {index}",
        "embedding_text": f"Document: HNSW Guide\nSection: Retrieval / Part {index}\nContent: 子块正文 {index}",
        "token_count": 50 + index,
        "source_start": index * 100,
        "source_end": index * 100 + 99,
        "chunker_version": "recursive_child_v1",
        "embedding_model": "bge-m3",
        "embedding_version": "bge-m3-v1",
        "content_hash": f"sha256:c{index}",
        "status": "active",
    }


class StoreTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "documents.sqlite3"
        self.store = DocumentStore(self.path)
        self.addCleanup(self.store.close)

    def seed(self, n_parents=3, n_children=6, **kwargs):
        parents = [_parent(i, **kwargs) for i in range(n_parents)]
        children = [_child(i, **kwargs) for i in range(n_children)]
        self.store.replace_document(DOC, parents, children)
        return parents, children


class SchemaTest(StoreTestBase):
    def test_counts_after_replace(self):
        self.seed()
        stats = self.store.stats()
        self.assertEqual(stats["documents"], 1)
        self.assertEqual(stats["parents"], 3)
        self.assertEqual(stats["children"], 6)
        self.assertGreater(stats["bytes"], 0)

    def test_counts_match_jsonl_semantics(self):
        """验收：`SELECT count(*) FROM parents` == JSONL 行数。"""
        parents, children = self.seed(n_parents=5, n_children=9)
        with self.store._connect() as conn:  # noqa: SLF001 - 验收要求直接查表
            rows = conn.execute("SELECT count(*) FROM parents").fetchone()[0]
        self.assertEqual(rows, len(parents))
        self.assertEqual(self.store.count("parents"), len(parents))
        self.assertEqual(self.store.count("children"), len(children))

    def test_foreign_key_enforced(self):
        orphan = _child(0)
        orphan["parent_id"] = "par_" + "f" * 32  # 不存在的父块
        with self.assertRaises(ParentNotFoundError):
            self.store.replace_document(DOC, [_parent(0)], [orphan])

    def test_foreign_keys_are_on(self):
        with self.store._connect() as conn:  # noqa: SLF001
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)


class LookupTest(StoreTestBase):
    def test_any_chunk_id_resolves_parent_text_and_source(self):
        """验收核心：任意 chunk_id → parent 原文 + 来源路径。"""
        self.seed()
        row = self.store.get_parent_for_chunk("chk_" + f"{3:032x}")
        self.assertIsNotNone(row)
        self.assertEqual(row["parent_id"], "par_" + f"{1:032x}")   # 3 // 2 == 1
        self.assertEqual(row["content"], "父块正文 1。\n第二行带换行与中文。")
        self.assertEqual(row["source_uri"], "md/nvidia/hnsg.md")
        self.assertEqual(row["document_id"], DOC["document_id"])
        self.assertEqual(row["section_path"], ["Retrieval", "Part 1"])

    def test_unknown_chunk_returns_none(self):
        self.seed()
        self.assertIsNone(self.store.get_parent_for_chunk("chk_" + "f" * 32))

    def test_unicode_and_newlines_preserved_exactly(self):
        self.seed()
        parent = self.store.get_parent("par_" + f"{0:032x}")
        self.assertIn("\n", parent["content"])
        self.assertIn("中文", parent["content"])

    def test_get_document_roundtrip(self):
        self.seed()
        doc = self.store.get_document(DOC["document_id"])
        self.assertEqual(doc["source_uri"], DOC["source_uri"])
        self.assertEqual(doc["document_version"], "sha256:v1")

    def test_children_of_document(self):
        self.seed()
        ids = [c["chunk_id"] for c in self.store.get_children(DOC["document_id"])]
        self.assertEqual(len(ids), 6)
        self.assertIn("chk_" + f"{5:032x}", ids)


class IdempotencyTest(StoreTestBase):
    def test_reimport_same_document_does_not_duplicate(self):
        self.seed()
        before = self.store.stats()
        self.seed()  # 完全相同的第二次导入
        after = self.store.stats()
        self.assertEqual((after["documents"], after["parents"], after["children"]),
                         (before["documents"], before["parents"], before["children"]))

    def test_new_version_replaces_old_rows(self):
        self.seed()
        old_parent_ids = {p["parent_id"] for p in self.store.get_parents(DOC["document_id"])}
        # 真实场景里文档改了 → 内容变了 → parent_id 也变（D10 的 id 由内容派生），
        # 所以用另一组 id 模拟「新版本」，否则测的是「用同 id 覆盖」而不是「清旧版本」
        parents = [_parent(i + 100, version="sha256:v2") for i in range(2)]
        # child 的 parent 是 `index // 2`，所以 200..203 正好挂在 100/101 两个父块下
        children = [_child(i + 200, version="sha256:v2") for i in range(4)]
        self.store.replace_document({**DOC, "document_version": "sha256:v2"}, parents, children)
        stats = self.store.stats()
        self.assertEqual(stats["parents"], 2, "旧版本父块必须被清掉")
        self.assertEqual(stats["children"], 4)
        for parent_id in old_parent_ids:
            self.assertIsNone(self.store.get_parent(parent_id), "旧版本父块不该残留（否则成孤儿向量源）")
        self.assertEqual(self.store.get_document(DOC["document_id"])["document_version"], "sha256:v2")

    def test_failed_replace_keeps_previous_version(self):
        """事务性：坏数据导入失败，旧版本必须原样保留。"""
        self.seed()
        before = self.store.stats()
        bad = _child(0)
        bad["parent_id"] = "par_" + "e" * 32
        with self.assertRaises(ParentNotFoundError):
            self.store.replace_document({**DOC, "document_version": "sha256:v3"}, [_parent(0)], [bad])
        after = self.store.stats()
        self.assertEqual((after["parents"], after["children"]), (before["parents"], before["children"]))
        self.assertEqual(self.store.get_document(DOC["document_id"])["document_version"], "sha256:v1")


class DeleteTest(StoreTestBase):
    def test_delete_document_removes_everything(self):
        self.seed()
        removed = self.store.delete_document(DOC["document_id"])
        self.assertEqual(removed, 1)
        stats = self.store.stats()
        self.assertEqual((stats["documents"], stats["parents"], stats["children"]), (0, 0, 0))
        self.assertIsNone(self.store.get_parent_for_chunk("chk_" + f"{0:032x}"))

    def test_delete_unknown_is_noop(self):
        self.seed()
        self.assertEqual(self.store.delete_document("doc_" + "f" * 32), 0)


class PersistenceTest(unittest.TestCase):
    def test_survives_restart(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.sqlite3"
            store = DocumentStore(path)
            store.replace_document(DOC, [_parent(0)], [_child(0)])
            store.close()
            again = DocumentStore(path)
            try:
                self.assertEqual(again.stats()["parents"], 1)
                self.assertIsNotNone(again.get_parent_for_chunk("chk_" + f"{0:032x}"))
            finally:
                again.close()


class JsonlImportTest(unittest.TestCase):
    """从 `chunks/` 目录整体导入 + 与 JSONL 行数核对（D17 验收命令）。"""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.chunks_dir = Path(self._tmp.name) / "chunks"
        self.chunks_dir.mkdir()
        self._write("documents.jsonl", [DOC])
        self._write("parents.jsonl", [_parent(i) for i in range(3)])
        self._write("children.jsonl", [_child(i) for i in range(6)])

    def _write(self, name, rows):
        (self.chunks_dir / name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )

    def test_import_counts_match(self):
        result = load_chunks_dir(self.chunks_dir)
        self.assertEqual((result["documents"], result["parents"], result["children"]), (1, 3, 6))
        store = DocumentStore(Path(self._tmp.name) / "d.sqlite3")
        self.addCleanup(store.close)
        counts = store.import_chunks_dir(self.chunks_dir)
        self.assertEqual(counts["parents"], 3)
        self.assertEqual(store.count("parents"), 3)

    def test_verify_detects_missing_rows(self):
        store = DocumentStore(Path(self._tmp.name) / "d.sqlite3")
        self.addCleanup(store.close)
        store.import_chunks_dir(self.chunks_dir)
        report = store.verify_against_jsonl(self.chunks_dir)
        self.assertTrue(report["ok"], report)

        self._write("children.jsonl", [_child(i) for i in range(8)])  # JSONL 多出两行
        report = store.verify_against_jsonl(self.chunks_dir)
        self.assertFalse(report["ok"])
        self.assertEqual(report["children"], {"jsonl": 8, "db": 6, "ok": False})

    def test_missing_jsonl_file_is_reported(self):
        (self.chunks_dir / "children.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            load_chunks_dir(self.chunks_dir)


if __name__ == "__main__":
    unittest.main()
