"""D23：翻译缓存（SQLite）。

为什么要缓存：翻译是**逐条走 LLM** 的，一条 1~2 s；三路召回每轮都要翻，评估跑
101 题 × 多个消融格就是几百次调用。缓存命中是毫秒级，而且让「同一题重复评估」
的结果稳定（模型温度=0 也不敢保证绝对可复现，缓存可以）。

键里必须带 `model` 与 `version`：换了模型或改了提示词，旧译文不该再被当成新口径
的结果复用——否则 A/B 会分不清是翻译变了还是检索变了（同 D21 版本闸门的思路）。
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS translations (
    key        TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    translated TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class TranslationCache:
    def __init__(self, path: str | Path, *, model: str, version: str = "v1"):
        self.path = Path(path)
        self.model = model
        self.version = version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        self._hits = 0
        self._misses = 0

    @contextmanager
    def _conn(self):
        """`with sqlite3.connect(...)` 只提交、**不关闭**（Windows 上会把文件锁住），
        所以这里自己管开与关。"""
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def key(self, text: str) -> str:
        raw = f"{self.model}|{self.version}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, text: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT translated FROM translations WHERE key = ?", (self.key(text),)
            ).fetchone()
        if row is None:
            self._misses += 1
            return None
        self._hits += 1
        return row[0]

    def put(self, text: str, translated: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO translations (key, source, translated) VALUES (?, ?, ?)",
                (self.key(text), text, translated),
            )

    def stats(self) -> dict[str, int | float]:
        with self._conn() as conn:
            entries = conn.execute("SELECT count(*) FROM translations").fetchone()[0]
        total = self._hits + self._misses
        return {
            "entries": entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "model": self.model,
            "version": self.version,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
