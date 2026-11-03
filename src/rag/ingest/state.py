"""D18 的状态机：每个文档一行，记阶段、时间、版本与错误。

为什么单独一个 SQLite 文件而不是复用 Document Store：

- Document Store 存的是**内容**（Parent 原文 / 元数据），删文档就该整篇消失；
- 状态表存的是**处理过程**（阶段、重试次数、失败历史），文档删了也该留痕，
  否则「这篇为什么没进来」永远查不到。
- 两者生命周期不同，混在一张库里会让 `delete_document` 的级联删到不该删的东西。

不进状态表的东西：进度百分比之类的派生值。它们随时能从 `stage_counts` 算出来，
存下来只会不一致。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class Stage(str, Enum):
    """§10.1 的状态序列与失败态（名字与文档一致，不要改）。"""

    PENDING = "PENDING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    COMPLETED = "COMPLETED"
    FAILED_PARSING = "FAILED_PARSING"
    FAILED_CHUNKING = "FAILED_CHUNKING"
    FAILED_EMBEDDING = "FAILED_EMBEDDING"
    FAILED_INDEXING = "FAILED_INDEXING"

    @classmethod
    def for_stage(cls, stage: "Stage") -> "Stage":
        """阶段 → 对应的失败态。"""
        return {
            Stage.PARSING: cls.FAILED_PARSING,
            Stage.CHUNKING: cls.FAILED_CHUNKING,
            Stage.EMBEDDING: cls.FAILED_EMBEDDING,
            Stage.INDEXING: cls.FAILED_INDEXING,
        }[stage]

    @property
    def is_failed(self) -> bool:
        return self.value.startswith("FAILED_")


SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_state (
    doc_key           TEXT PRIMARY KEY,
    document_id       TEXT,
    document_version  TEXT,
    stage             TEXT NOT NULL,
    chunker_version   TEXT,
    embedding_version TEXT,
    attempts          INTEGER NOT NULL DEFAULT 0,
    error             TEXT,
    started_at        TEXT,
    updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_state_stage ON ingest_state(stage);

CREATE TABLE IF NOT EXISTS ingest_failures (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_key    TEXT NOT NULL,
    stage      TEXT NOT NULL,
    error      TEXT NOT NULL,
    resolved   INTEGER NOT NULL DEFAULT 0,
    -- 'completed'（重试成功）/ 'source_removed'（源文件已删除，不再可重试）
    resolution TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_failures_doc ON ingest_failures(doc_key);
"""


class IngestState:
    """导入状态表。单进程顺序写（摄取不并发写同一个文档）。"""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(ingest_failures)")}
            if "resolution" not in columns:   # 老库补列（SQLite 不支持 IF NOT EXISTS）
                conn.execute("ALTER TABLE ingest_failures ADD COLUMN resolution TEXT")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        """兼容 addCleanup；连接是按需建立的。"""

    # -- 状态流转 -------------------------------------------------------

    def begin(self, doc_key: str) -> None:
        """开始（或重试）一个文档：置 PARSING，清掉上一次的 error。"""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ingest_state (doc_key, stage, attempts, started_at, updated_at)"
                " VALUES (?, ?, 1, datetime('now'), datetime('now'))"
                " ON CONFLICT(doc_key) DO UPDATE SET"
                "   stage=excluded.stage,"
                "   attempts=ingest_state.attempts + 1,"
                "   error=NULL, finished_at=NULL, updated_at=datetime('now')",
                (doc_key, Stage.PARSING.value),
            )

    def advance(self, doc_key: str, stage: Stage) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE ingest_state SET stage = ?, updated_at = datetime('now') WHERE doc_key = ?",
                (stage.value, doc_key),
            )

    def complete(
        self,
        doc_key: str,
        *,
        document_id: str | None = None,
        document_version: str | None = None,
        chunker_version: str | None = None,
        embedding_version: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE ingest_state SET stage = ?, document_id = coalesce(?, document_id),"
                " document_version = coalesce(?, document_version),"
                " chunker_version = coalesce(?, chunker_version),"
                " embedding_version = coalesce(?, embedding_version),"
                " error = NULL, finished_at = datetime('now'), updated_at = datetime('now')"
                " WHERE doc_key = ?",
                (Stage.COMPLETED.value, document_id, document_version,
                 chunker_version, embedding_version, doc_key),
            )
            # 该文档此前所有未解决的失败记录 → 标记已解决（留痕但不删）
            conn.execute(
                "UPDATE ingest_failures SET resolved = 1, resolution = 'completed'"
                " WHERE doc_key = ? AND resolved = 0",
                (doc_key,),
            )

    def fail(self, doc_key: str, stage: Stage, error: str) -> None:
        """记失败态 + 失败清单一行。`stage` 传**发生失败时**的阶段。"""
        failed_stage = Stage.for_stage(stage) if not stage.is_failed else stage
        with self._connect() as conn:
            conn.execute(
                "UPDATE ingest_state SET stage = ?, error = ?, updated_at = datetime('now'),"
                " finished_at = datetime('now') WHERE doc_key = ?",
                (failed_stage.value, error, doc_key),
            )
            conn.execute(
                "INSERT INTO ingest_failures (doc_key, stage, error) VALUES (?, ?, ?)",
                (doc_key, failed_stage.value, error),
            )

    # -- 查询 -----------------------------------------------------------

    def get(self, doc_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ingest_state WHERE doc_key = ?", (doc_key,)).fetchone()
        return dict(row) if row else None

    def is_done(
        self,
        doc_key: str,
        *,
        document_version: str | None = None,
        chunker_version: str | None = None,
        embedding_version: str | None = None,
    ) -> bool:
        """断点恢复的判据：COMPLETED **且**三处版本都没变。"""
        row = self.get(doc_key)
        if row is None or row["stage"] != Stage.COMPLETED.value:
            return False
        if document_version is not None and row["document_version"] != document_version:
            return False
        if chunker_version is not None and row["chunker_version"] != chunker_version:
            return False
        if embedding_version is not None and row["embedding_version"] != embedding_version:
            return False
        return True

    def stage_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT stage, count(*) FROM ingest_state GROUP BY stage").fetchall()
        return {stage: int(count) for stage, count in rows}

    def failures(self, *, unresolved_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM ingest_failures"
        if unresolved_only:
            sql += " WHERE resolved = 0"
        sql += " ORDER BY id"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql).fetchall()]

    def failed_documents(self) -> list[dict[str, Any]]:
        """当前处于任一 FAILED_* 状态的文档（按 doc_key 排序，稳定重试顺序）。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ingest_state WHERE stage LIKE 'FAILED_%' ORDER BY doc_key"
            ).fetchall()
        return [dict(r) for r in rows]

    def all_document_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT document_id FROM ingest_state WHERE document_id IS NOT NULL ORDER BY doc_key"
            ).fetchall()
        return [r["document_id"] for r in rows]

    def stats(self) -> dict[str, Any]:
        return {
            "documents": sum(self.stage_counts().values()),
            "stage_counts": self.stage_counts(),
            "failures": len(self.failures()),
            "unresolved_failures": len(self.failures(unresolved_only=True)),
        }

    # -- 运维 -----------------------------------------------------------

    def reset(self, doc_key: str) -> None:
        """彻底忘掉一个文档的处理记录（改了切分规则想强制重跑时用）。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM ingest_state WHERE doc_key = ?", (doc_key,))

    def prune_missing(self, root: str | os.PathLike[str]) -> int:
        """清掉「源文件已经不存在」的状态行。

        不做这一步的后果：文件改名/删除后，它的 FAILED_* 行会永远挂在
        `failed_documents()` 里，重试清单里出现一个再也修不好的幽灵项。
        注意**只删状态行**，`ingest_failures` 里的历史保留（那是审计用的）。
        """
        base = Path(root)
        removed = 0
        with self._connect() as conn:
            # 失败表也要看：状态行可能早就被删了，失败记录却还挂着（实测踩到）
            unresolved = [
                r["doc_key"] for r in conn.execute(
                    "SELECT DISTINCT doc_key FROM ingest_failures WHERE resolved = 0")
            ]
            for key in unresolved:
                if not (base / key).exists():
                    conn.execute(
                        "UPDATE ingest_failures SET resolved = 1, resolution = 'source_removed'"
                        " WHERE doc_key = ? AND resolved = 0",
                        (key,),
                    )
            rows = [r["doc_key"] for r in conn.execute("SELECT doc_key FROM ingest_state")]
            for key in rows:
                if not (base / key).exists():
                    conn.execute("DELETE FROM ingest_state WHERE doc_key = ?", (key,))
                    # 源文件没了，它的失败记录再也不会被重试；标记为已解决并注明原因，
                    # 否则重试清单里永远躺着一条修不好的幽灵项
                    conn.execute(
                        "UPDATE ingest_failures SET resolved = 1, resolution = 'source_removed'"
                        " WHERE doc_key = ? AND resolved = 0",
                        (key,),
                    )
                    removed += 1
        return removed

    def forget_document_id(self, document_id: str) -> int:
        """按 document_id 清状态行（删除文档时调用）。

        不清的后果：文档删了、索引里没了，状态表还标 `COMPLETED`，重跑会被
        `is_done()` 直接跳过——**永远重建不回来**（D20 实测踩到）。
        """
        with self._connect() as conn:
            return int(conn.execute(
                "DELETE FROM ingest_state WHERE document_id = ?", (document_id,)).rowcount)

    def forget_missing(self, existing_keys: Iterable[str]) -> int:
        """清掉源文件已不存在的记录（避免状态表无限膨胀）。"""
        keys = set(existing_keys)
        removed = 0
        with self._connect() as conn:
            rows = [r["doc_key"] for r in conn.execute("SELECT doc_key FROM ingest_state")]
            for key in rows:
                if key not in keys:
                    conn.execute("DELETE FROM ingest_state WHERE doc_key = ?", (key,))
                    removed += 1
        return removed
