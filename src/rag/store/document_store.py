"""D17：SQLite Document Store（Parent 原文 + 文档元数据）。

职责边界（§3.5 / §5.3）：

| 存在哪 | 存什么 |
|---|---|
| Qdrant payload | Child 文本（`text`）+ 过滤字段 |
| **本模块（SQLite）** | **Parent 原文全文** + 文档元数据 + chunk→parent 的定位表 |
| `chunks/*.jsonl` | 全量真源（可重建前三者） |

三条硬约束：

1. **Parent 原文只存一份**。Qdrant payload 里不放父块全文，命中 Child 后按
   `parent_id` 回这里取，否则同一段父文会被复制 N 次（N = 该父块的子块数）。
2. **外键打开**。Child 指向不存在的 Parent 必须当场报错，不能让孤儿进库——
   孤儿的后果是「检索命中却回溯不到原文」，属于静默故障。
3. **换版本是一个事务**。`replace_document` 要么整篇换成新版本，要么旧版本原样保留；
   中间失败留下半篇旧半篇新的数据，比失败本身更糟。
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..ingestion.errors import IngestionError
from .errors import VectorStoreError

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    source_uri       TEXT NOT NULL,
    file_name        TEXT,
    file_type        TEXT,
    title            TEXT,
    language         TEXT,
    document_version TEXT NOT NULL,
    parser_version   TEXT,
    created_at       TEXT,
    status           TEXT NOT NULL DEFAULT 'active',
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS parents (
    parent_id        TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    document_version TEXT NOT NULL,
    language         TEXT,
    title            TEXT,
    section_path     TEXT NOT NULL DEFAULT '[]',   -- JSON 数组
    content          TEXT NOT NULL,
    token_count      INTEGER NOT NULL,
    source_start     INTEGER,
    source_end       INTEGER,
    chunker_version  TEXT NOT NULL,
    parent_index     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_parents_document ON parents(document_id);

CREATE TABLE IF NOT EXISTS children (
    chunk_id         TEXT PRIMARY KEY,
    parent_id        TEXT NOT NULL REFERENCES parents(parent_id) ON DELETE CASCADE,
    document_id      TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    document_version TEXT NOT NULL,
    content_hash     TEXT,
    token_count      INTEGER,
    source_start     INTEGER,
    source_end       INTEGER,
    embedding_version TEXT,
    status           TEXT NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_children_document ON children(document_id);
CREATE INDEX IF NOT EXISTS idx_children_parent ON children(parent_id);
"""

#: 允许 COUNT 的表名白名单（避免把表名拼进 SQL 造成注入）
TABLES = ("documents", "parents", "children")

DATA_FILES = ("documents.jsonl", "parents.jsonl", "children.jsonl")


class DocumentStoreError(VectorStoreError):
    """Document Store 的通用错误。"""


class ParentNotFoundError(DocumentStoreError):
    """Child 指向的 Parent 不存在。"""


class DocumentStore:
    """单进程 SQLite Document Store。非线程安全（摄取是单进程顺序写的）。"""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # -- 基础设施 -------------------------------------------------------

    @contextmanager
    def _connect(self):
        """每次操作独立连接，用完显式关闭（Windows 上不关会锁住 db 文件）。"""
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")     # 默认是 OFF，必须显式打开
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        """兼容 `addCleanup(store.close)`；连接是按需建立的不需要长连接。"""

    # -- 写入 -----------------------------------------------------------

    def replace_document(
        self,
        document: Mapping[str, Any],
        parents: Sequence[Mapping[str, Any]],
        children: Sequence[Mapping[str, Any]],
    ) -> dict[str, int]:
        """整篇替换：先删旧版本的 parents/children，再写新的（单事务）。"""
        document_id = document["document_id"]
        parent_ids = {p["parent_id"] for p in parents}
        orphans = sorted({c["parent_id"] for c in children} - parent_ids)
        if orphans:
            raise ParentNotFoundError(
                f"{len(orphans)} 个 child 的 parent_id 不在本次导入的 parent 集合里"
                f"（例如 {orphans[0]}）；孤儿 Child 会导致「命中却回溯不到父块」"
            )

        with self._connect() as conn:
            conn.execute("DELETE FROM children WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM parents WHERE document_id = ?", (document_id,))
            conn.execute(
                "INSERT INTO documents (document_id, source_uri, file_name, file_type, title, language,"
                " document_version, parser_version, created_at, status)"
                " VALUES (:document_id, :source_uri, :file_name, :file_type, :title, :language,"
                " :document_version, :parser_version, :created_at, :status)"
                " ON CONFLICT(document_id) DO UPDATE SET"
                "   source_uri=excluded.source_uri, file_name=excluded.file_name,"
                "   file_type=excluded.file_type, title=excluded.title, language=excluded.language,"
                "   document_version=excluded.document_version, parser_version=excluded.parser_version,"
                "   created_at=excluded.created_at, status=excluded.status,"
                "   updated_at=datetime('now')",
                {
                    "document_id": document_id,
                    "source_uri": document.get("source_uri"),
                    "file_name": document.get("file_name"),
                    "file_type": document.get("file_type"),
                    "title": document.get("title"),
                    "language": document.get("language"),
                    "document_version": document["document_version"],
                    "parser_version": document.get("parser_version"),
                    "created_at": document.get("created_at"),
                    "status": document.get("status") or "active",
                },
            )
            conn.executemany(
                "INSERT INTO parents (parent_id, document_id, document_version, language, title,"
                " section_path, content, token_count, source_start, source_end, chunker_version, parent_index)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        p["parent_id"], p["document_id"], p["document_version"], p.get("language"),
                        p.get("title"), json.dumps(p.get("section_path") or [], ensure_ascii=False),
                        p["content"], int(p["token_count"]), p.get("source_start"), p.get("source_end"),
                        p["chunker_version"], p.get("parent_index"),
                    )
                    for p in parents
                ],
            )
            conn.executemany(
                "INSERT INTO children (chunk_id, parent_id, document_id, document_version, content_hash,"
                " token_count, source_start, source_end, embedding_version, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c["chunk_id"], c["parent_id"], c["document_id"], c["document_version"],
                        c.get("content_hash"), c.get("token_count"), c.get("source_start"),
                        c.get("source_end"), c.get("embedding_version"), c.get("status") or "active",
                    )
                    for c in children
                ],
            )
        return {"documents": 1, "parents": len(parents), "children": len(children)}

    def import_chunks_dir(self, chunks_dir: str | os.PathLike[str]) -> dict[str, int]:
        """把一个 `chunks/` 目录（D12 产物）整体导入。按 document 分组、逐篇整替。"""
        grouped = load_chunks_dir(chunks_dir)
        totals = {"documents": 0, "parents": 0, "children": 0}
        for document_id, bundle in grouped["by_document"].items():
            counts = self.replace_document(
                bundle["document"], bundle["parents"], bundle["children"]
            )
            for key in totals:
                totals[key] += counts[key]
        return totals

    # -- 读取 -----------------------------------------------------------

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def get_parent(self, parent_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM parents WHERE parent_id = ?", (parent_id,)).fetchone()
        return _parent_row(row) if row else None

    def get_parents(self, document_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM parents WHERE document_id = ? ORDER BY parent_index, parent_id", (document_id,)
            ).fetchall()
        return [_parent_row(r) for r in rows]

    def get_children(self, document_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, parent_id, document_id, token_count FROM children WHERE document_id = ?"
                " ORDER BY chunk_id",
                (document_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_parent_for_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """验收主路径：任意 `chunk_id` → parent 原文 + 来源路径。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT p.*, d.source_uri, d.file_name, d.title AS document_title"
                " FROM children c"
                " JOIN parents p ON p.parent_id = c.parent_id"
                " JOIN documents d ON d.document_id = c.document_id"
                " WHERE c.chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        parent = _parent_row(row)
        parent.update({
            "source_uri": row["source_uri"],
            "file_name": row["file_name"],
            "document_title": row["document_title"],
        })
        return parent

    # -- 统计与核对 -----------------------------------------------------

    def parents_containing(self, spans: Iterable[str]) -> set[str]:
        """哪些父块包含这些原文片段（D14 用 gold_spans 造判定金标）。"""
        found: set[str] = set()
        with self._connect() as conn:
            for span in spans:
                if not span:
                    continue
                rows = conn.execute(
                    "SELECT parent_id FROM parents WHERE content LIKE ?", (f"%{span}%",)
                ).fetchall()
                found.update(row[0] for row in rows)
        return found

    def documents_containing(self, spans: Iterable[str]) -> set[str]:
        """哪些文档（source_uri）包含这些原文片段。D14 用它按金标挑样本文件。"""
        found: set[str] = set()
        with self._connect() as conn:
            for span in spans:
                if not span:
                    continue
                rows = conn.execute(
                    "SELECT DISTINCT d.source_uri FROM parents p JOIN documents d"
                    " ON d.document_id = p.document_id WHERE p.content LIKE ?",
                    (f"%{span}%",),
                ).fetchall()
                found.update(row[0] for row in rows)
        return found

    def count(self, table: str) -> int:
        if table not in TABLES:
            raise DocumentStoreError(f"未知表名 {table!r}，只支持 {TABLES}")
        with self._connect() as conn:
            return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            page_count = conn.execute("PRAGMA page_count").fetchone()[0]
            page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        return {
            "documents": self.count("documents"),
            "parents": self.count("parents"),
            "children": self.count("children"),
            "bytes": int(page_count) * int(page_size),
        }

    def verify_against_jsonl(self, chunks_dir: str | os.PathLike[str]) -> dict[str, Any]:
        """逐表比对「JSONL 行数」与「库里行数」。D17 的验收命令用它。"""
        directory = Path(chunks_dir)
        report: dict[str, Any] = {"ok": True}
        for name in ("documents", "parents", "children"):
            path = directory / f"{name}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"缺少 {path}（不是 D12 产出的 chunks 目录？）")
            expected = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            actual = self.count(name)
            report[name] = {"jsonl": expected, "db": actual, "ok": expected == actual}
            report["ok"] = report["ok"] and expected == actual
        return report

    def delete_document(self, document_id: str) -> int:
        with self._connect() as conn:
            # children / parents 靠 ON DELETE CASCADE 级联删除
            cursor = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            return int(cursor.rowcount)


def _parent_row(row: sqlite3.Row) -> dict[str, Any]:
    parent = dict(row)
    parent["section_path"] = json.loads(parent.get("section_path") or "[]")
    return parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise IngestionError(f"{path.name}:{line_no} 不是合法 JSON：{exc}") from exc
    return rows


def load_chunks_dir(chunks_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """读 `chunks/` 下的三个 JSONL，按 document_id 分组返回。"""
    directory = Path(chunks_dir)
    for name in DATA_FILES:
        if not (directory / name).exists():
            raise FileNotFoundError(f"缺少 {directory / name}（不是 D12 产出的 chunks 目录？）")

    documents = _read_jsonl(directory / "documents.jsonl")
    parents = _read_jsonl(directory / "parents.jsonl")
    children = _read_jsonl(directory / "children.jsonl")

    by_document: dict[str, dict[str, list]] = {
        doc["document_id"]: {"document": doc, "parents": [], "children": []} for doc in documents
    }
    for parent in parents:
        bucket = by_document.get(parent["document_id"])
        if bucket is None:
            raise IngestionError(f"parent {parent['parent_id']} 的 document_id 不在 documents.jsonl 里")
        bucket["parents"].append(parent)
    for child in children:
        bucket = by_document.get(child["document_id"])
        if bucket is None:
            raise IngestionError(f"child {child['chunk_id']} 的 document_id 不在 documents.jsonl 里")
        bucket["children"].append(child)

    return {
        "documents": len(documents),
        "parents": len(parents),
        "children": len(children),
        "by_document": by_document,
    }


# --------------------------------------------------------------------- CLI


def cli_main(argv: Iterable[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="document_store", description="Document Store（D17）")
    parser.add_argument("--db", default=str(Path("storage") / "documents.sqlite3"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="导入 chunks/ 目录")
    imp.add_argument("chunks_dir")
    ver = sub.add_parser("verify", help="与 JSONL 行数核对")
    ver.add_argument("chunks_dir")
    show = sub.add_parser("show", help="按 chunk_id 查父块原文与来源")
    show.add_argument("chunk_id")
    rm = sub.add_parser("delete", help="删除一篇文档（级联）")
    rm.add_argument("document_id")
    sub.add_parser("stats", help="看行数与体积")

    args = parser.parse_args(list(argv) if argv is not None else None)
    store = DocumentStore(args.db)
    try:
        if args.cmd == "import":
            counts = store.import_chunks_dir(args.chunks_dir)
            print(f"[import] documents={counts['documents']} parents={counts['parents']} "
                  f"children={counts['children']} → {args.db}")
            report = store.verify_against_jsonl(args.chunks_dir)
            print(f"[verify] {'一致' if report['ok'] else '不一致'}："
                  + " ".join(f"{k}={v['jsonl']}/{v['db']}" for k, v in report.items() if isinstance(v, dict)))
            return 0 if report["ok"] else 1
        if args.cmd == "verify":
            report = store.verify_against_jsonl(args.chunks_dir)
            for name in ("documents", "parents", "children"):
                item = report[name]
                print(f"  {name:10s} jsonl={item['jsonl']:>6} db={item['db']:>6} {'OK' if item['ok'] else '不匹配'}")
            return 0 if report["ok"] else 1
        if args.cmd == "show":
            row = store.get_parent_for_chunk(args.chunk_id)
            if row is None:
                print(f"没有找到 chunk_id={args.chunk_id}", file=sys.stderr)
                return 1
            print(json.dumps(row, ensure_ascii=False, indent=2))
            return 0
        if args.cmd == "delete":
            print(f"[delete] 删除了 {store.delete_document(args.document_id)} 篇文档")
            return 0
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(cli_main())
