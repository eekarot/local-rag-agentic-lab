"""D21：索引版本化——换 embedding 模型必须换 collection。

写入前的对账规则：

| 变化 | 要换 collection 吗 | 理由 |
|---|---|---|
| `embedding_model` | **要** | 向量空间不同，混在一起必然垃圾 |
| `embedding_version` | **要** | 预处理/口径变了，向量含义变了 |
| `vector_size` | **要** | 维度不同直接写不进去（但同维不同模型才致命） |
| `chunker_version` | 不用 | chunk_id 变了，向量空间没变 |
| `parser_version` | 不用 | 同上 |

反向用例（D21 验收要求）：拿不同的 embedding_version 调 `assert_compatible`，
必须抛 `CollectionVersionMismatch`，且消息里**直接给出应该新建的 collection 名**——
报错不给解法，等于把人卡在原地。
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS index_versions (
    collection        TEXT PRIMARY KEY,
    embedding_model   TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    vector_size       INTEGER NOT NULL,
    chunker_version   TEXT,
    status            TEXT NOT NULL DEFAULT 'active',
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class CollectionVersionMismatch(RuntimeError):
    """collection 与本次写入的模型口径不一致 / 已退役。"""


def collection_name_for(embedding_version: str, default: str = "rag_chunks_v1") -> str:
    """按 embedding 版本给出建议的 collection 名（全小写 + 下划线，`/` 也压平）。"""
    if embedding_version == "bge-m3-v1":
        return default          # 保持已有数据可用：v1 就是当前这个 collection
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", embedding_version).strip("_").lower()
    return f"rag_chunks_{slug}"


class IndexVersions:
    """collection ↔ 模型口径的登记表（SQLite）。"""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

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
        """兼容 addCleanup；连接按需建立。"""

    # -- 登记 -----------------------------------------------------------

    def register(
        self,
        *,
        collection: str,
        embedding_model: str,
        embedding_version: str,
        vector_size: int,
        chunker_version: str | None = None,
    ) -> dict[str, Any]:
        """登记（或原样重复登记）一个 collection。口径不同 → 抛错，不覆盖。"""
        existing = self.get(collection)
        if existing is not None:
            self._assert_same(collection, existing, embedding_model, embedding_version, vector_size)
            return existing
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO index_versions (collection, embedding_model, embedding_version,"
                " vector_size, chunker_version) VALUES (?, ?, ?, ?, ?)",
                (collection, embedding_model, embedding_version, int(vector_size), chunker_version),
            )
        return self.get(collection)  # type: ignore[return-value]

    def ensure_registered(self, **kwargs: Any) -> dict[str, Any]:
        """D15 先建了 collection、D21 后上线时用：没有记录就按当前口径补登记。"""
        return self.register(**kwargs)

    # -- 对账 -----------------------------------------------------------

    def assert_compatible(
        self,
        collection: str,
        *,
        embedding_model: str,
        embedding_version: str,
        vector_size: int,
        chunker_version: str | None = None,
    ) -> dict[str, Any]:
        """写入前的闸门。列不存在时自动登记（首次写入）。"""
        existing = self.get(collection)
        if existing is None:
            return self.register(
                collection=collection, embedding_model=embedding_model,
                embedding_version=embedding_version, vector_size=vector_size,
                chunker_version=chunker_version,
            )
        if existing["status"] == "retired":
            raise CollectionVersionMismatch(
                f"collection {collection} 已退役（status=retired），拒绝写入。"
                f"改用 active 的 collection：{[r['collection'] for r in self.active()] or '（无，请先建）'}"
            )
        self._assert_same(collection, existing, embedding_model, embedding_version, vector_size)
        return existing

    def check(
        self,
        collection: str,
        *,
        embedding_model: str,
        embedding_version: str,
        vector_size: int,
    ) -> dict[str, Any]:
        """**只读**对账（CLI 用）。未登记时不写记录——诊断命令不该改状态。"""
        existing = self.get(collection)
        if existing is None:
            return {"collection": collection, "registered": False,
                    "would_register": {"embedding_model": embedding_model,
                                       "embedding_version": embedding_version,
                                       "vector_size": int(vector_size)}}
        if existing["status"] == "retired":
            raise CollectionVersionMismatch(f"collection {collection} 已退役（status=retired）")
        self._assert_same(collection, existing, embedding_model, embedding_version, vector_size)
        return {**existing, "registered": True}

    def _assert_same(
        self,
        collection: str,
        existing: dict[str, Any],
        embedding_model: str,
        embedding_version: str,
        vector_size: int,
    ) -> None:
        problems = []
        if existing["embedding_model"] != embedding_model:
            problems.append(f"embedding_model {existing['embedding_model']} → {embedding_model}")
        if existing["embedding_version"] != embedding_version:
            problems.append(
                f"embedding_version {existing['embedding_version']} → {embedding_version}")
        if int(existing["vector_size"]) != int(vector_size):
            problems.append(f"vector_size {existing['vector_size']} → {vector_size}")
        if not problems:
            return
        suggested = collection_name_for(embedding_version, default=collection)
        if suggested == collection:
            suggested = f"{collection}_new"      # 同版本不能自指，给个明确的新名字候选
        raise CollectionVersionMismatch(
            f"collection {collection} 的模型口径与本次写入不一致（{'；'.join(problems)}）。"
            f"同维不同模型的向量混写不会报错，只会让检索静默变差，所以拒绝写入。"
            f"请新建 collection（建议名 {suggested}）并全量重建，"
            f"或用 python -m src.rag.store.index_versions list 查看现有登记"
        )

    # -- 查询与运维 -----------------------------------------------------

    def get(self, collection: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM index_versions WHERE collection = ?", (collection,)).fetchone()
        return dict(row) if row else None

    def list(self, *, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM index_versions"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY created_at, collection"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def active(self) -> list[dict[str, Any]]:
        return self.list(status="active")

    def retire(self, collection: str) -> None:
        self._set_status(collection, "retired")

    def activate(self, collection: str) -> None:
        self._set_status(collection, "active")

    def _set_status(self, collection: str, status: str) -> None:
        if self.get(collection) is None:
            raise CollectionVersionMismatch(f"collection {collection} 没有登记记录，无法改状态")
        with self._connect() as conn:
            conn.execute(
                "UPDATE index_versions SET status = ?, updated_at = datetime('now') WHERE collection = ?",
                (status, collection),
            )

    def forget(self, collection: str) -> int:
        with self._connect() as conn:
            return int(conn.execute("DELETE FROM index_versions WHERE collection = ?", (collection,)).rowcount)


# --------------------------------------------------------------------- CLI


def cli_main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(prog="index_versions", description="索引版本登记表（D21）")
    parser.add_argument("--db", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出所有 collection 及其口径")
    for name, help_text in (("retire", "标记退役（拒绝写入）"), ("activate", "重新启用"), ("forget", "删掉登记记录")):
        sub_parser = sub.add_parser(name, help=help_text)
        sub_parser.add_argument("collection")
    check = sub.add_parser("check", help="检查当前模型口径能否写进该 collection")
    check.add_argument("collection")
    check.add_argument("--embedding-model", required=True)
    check.add_argument("--embedding-version", required=True)
    check.add_argument("--vector-size", type=int, required=True)

    args = parser.parse_args(argv)
    path = args.db or (Path("storage") / "index_versions.sqlite3")
    versions = IndexVersions(path)
    try:
        if args.cmd == "list":
            rows = versions.list()
            if not rows:
                print("（还没有登记记录）")
            for row in rows:
                print(f"  {row['collection']:28s} {row['embedding_model']}/{row['embedding_version']:12s} "
                      f"dim={row['vector_size']:4d} chunker={row['chunker_version'] or '-':24s} {row['status']}")
        elif args.cmd == "retire":
            versions.retire(args.collection)
            print(f"[retire] {args.collection}")
        elif args.cmd == "activate":
            versions.activate(args.collection)
            print(f"[activate] {args.collection}")
        elif args.cmd == "forget":
            print(f"[forget] 删除 {versions.forget(args.collection)} 条")
        elif args.cmd == "check":
            row = versions.check(
                args.collection, embedding_model=args.embedding_model,
                embedding_version=args.embedding_version, vector_size=args.vector_size,
            )
            print(json.dumps(row, ensure_ascii=False, indent=2))
    except CollectionVersionMismatch as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    finally:
        versions.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
