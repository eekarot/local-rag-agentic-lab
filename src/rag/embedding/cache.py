"""内容 Hash 缓存：把「文本 → 向量」落到本地 SQLite。

为什么用 SQLite 而不是 JSONL / npy：

| 方案 | 每条 1024 维占用 | 随机查 | 增量写 |
|---|---|---|---|
| JSONL + 十进制数组 | ≈ 11 KB | 要全量载入 | 追加可以 |
| JSONL + base64 float32 | ≈ 5.5 KB | 要全量载入 | 追加可以 |
| **SQLite + float32 BLOB** | **4 KB** | 主键索引 | UPSERT |

全量 81k chunk 时三种方案的差距是 340 MB vs 450 MB vs 900 MB，且只有 SQLite
能按 key 随机取（D18 断点恢复要的就是这个）。

缓存键把 `model` / `version` / `kind` 一起算进 hash：
- 换模型或换预处理版本必须重新编码，否则复用旧向量等于静默降级；
- doc 与 query 分开命名空间，为将来「非对称模型」（query 需要前缀）留位。
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key       TEXT PRIMARY KEY,
    model     TEXT NOT NULL,
    version   TEXT NOT NULL,
    kind      TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    text_len  INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model, version);
"""


class EmbeddingCache:
    """`key → float32 向量` 的本地缓存。非线程安全的单进程用法。"""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # -- 连接与生命周期 ------------------------------------------------

    @contextmanager
    def _connect(self):
        """每次操作独立连接，用完**必须关**。

        坑：`with sqlite3.connect(...) as conn` 只提交事务、不关闭连接，
        Windows 上会一直锁住 .sqlite3 文件，临时目录清理直接 PermissionError，
        长跑脚本里则是句柄泄漏。所以这里用 contextmanager 显式 close。
        """
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *exc) -> None:
        pass

    # -- 键 ------------------------------------------------------------

    @staticmethod
    def key_for(*, kind: str, model: str, version: str, text: str) -> str:
        payload = "\x1f".join((kind, model, version, text))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- 读写 ----------------------------------------------------------

    def get_many(self, keys: Sequence[str], expected_dim: int) -> dict[str, np.ndarray]:
        """批量取。维度不符或 blob 损坏的行按 **miss** 处理（会被重算覆盖）。"""
        if not keys:
            return {}
        found: dict[str, np.ndarray] = {}
        with self._connect() as conn:
            # SQLite 变量上限 999，分批查
            for start in range(0, len(keys), 900):
                chunk = keys[start:start + 900]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT key, dim, vector FROM embeddings WHERE key IN ({placeholders})",
                    chunk,
                ).fetchall()
                for key, dim, blob in rows:
                    # 先按字节数校验再 frombuffer：blob 被截断时 frombuffer 会抛
                    # ValueError("buffer size must be a multiple of element size")，
                    # 缓存坏行不该让整条摄取链挂掉——按 miss 处理，重算覆盖。
                    if blob is None or dim != expected_dim or len(blob) != expected_dim * 4:
                        continue
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(
        self,
        items: Iterable[tuple[str, np.ndarray, str, str, str, int]],
    ) -> int:
        """写入 `(key, vector, model, version, kind, text_len)`，按 key UPSERT。"""
        rows = [
            (key, model, version, kind, int(dim), np.asarray(vector, dtype=np.float32).tobytes(), text_len)
            for key, vector, model, version, kind, text_len in items
            for dim in (np.asarray(vector, dtype=np.float32).size,)
        ]
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO embeddings (key, model, version, kind, dim, vector, text_len)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET"
                "   model=excluded.model, version=excluded.version, kind=excluded.kind,"
                "   dim=excluded.dim, vector=excluded.vector, text_len=excluded.text_len",
                rows,
            )
            conn.commit()
        return len(rows)

    # -- 运维 ----------------------------------------------------------

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            rows, total_bytes = conn.execute(
                "SELECT count(*), coalesce(sum(length(vector)), 0) FROM embeddings"
            ).fetchone()
            by_model = conn.execute(
                "SELECT model, version, count(*) FROM embeddings GROUP BY model, version"
            ).fetchall()
        return {
            "rows": int(rows),
            "bytes": int(total_bytes),
            "by_model": {f"{m}/{v}": int(c) for m, v, c in by_model},  # type: ignore[dict-item]
        }

    def purge(self, *, model: str | None = None) -> int:
        with self._connect() as conn:
            if model is None:
                cur = conn.execute("DELETE FROM embeddings")
            else:
                cur = conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
            conn.commit()
            return cur.rowcount
