"""回填 `parents.parent_index`（N6）。

**为什么要这一步**：`Parent.to_record()` 以前漏了 `parent_index`，于是 25,411 行父块的这一列
全是 NULL。受影响的地方有两处：

- D09 的父块关联键是 `(document_id, parent_index)`；
- D32 的「同文档相邻父块合并」靠它判断顺序。

**为什么能回填**：`python scripts/ingest.py` 是按文档、按父块顺序逐篇写入的，
所以 SQLite 的 `rowid` 在**同一 document_id 内**就是父块顺序。实测抽查
（admin-manual.md 1,903 个父块）rowid 顺序 = 正文顺序：Preface → 0.3 → 0.4 → 0.5 →
Introduction → 1.2.1 → 1.2.2 …，与 `section_path` 完全一致。

**幂等**：默认只填 NULL 行（`--force` 才重算全部），所以可以反复跑。

    python scripts/backfill_parent_index.py --dry-run
    python scripts/backfill_parent_index.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import STORAGE_DIR  # noqa: E402

DEFAULT_DB = STORAGE_DIR / "documents.sqlite3"

_CTE = """
WITH ordered AS (
    SELECT rowid AS rid,
           ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY rowid) - 1 AS idx
    FROM parents
)
UPDATE parents
   SET parent_index = (SELECT idx FROM ordered WHERE ordered.rid = parents.rowid)
 WHERE {where}
"""


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    total = conn.execute("SELECT count(*) FROM parents").fetchone()[0]
    filled = conn.execute("SELECT count(*) FROM parents WHERE parent_index IS NOT NULL").fetchone()[0]
    docs = conn.execute("SELECT count(DISTINCT document_id) FROM parents").fetchone()[0]
    return {"parents": total, "filled": filled, "missing": total - filled, "documents": docs}


def backfill(db_path: str | Path = DEFAULT_DB, *, force: bool = False, dry_run: bool = False) -> dict:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    before = stats(conn)
    where = "1=1" if force else "parent_index IS NULL"

    target = conn.execute(f"SELECT count(*) FROM parents WHERE {where}").fetchone()[0]
    sample = conn.execute(
        "WITH ordered AS (SELECT rowid AS rid, ROW_NUMBER() OVER"
        " (PARTITION BY document_id ORDER BY rowid) - 1 AS idx FROM parents)"
        " SELECT p.rowid, p.document_id, p.parent_index, o.idx FROM parents p"
        " JOIN ordered o ON o.rid = p.rowid WHERE p.parent_index IS NULL ORDER BY p.rowid LIMIT 5"
    ).fetchall()
    preview = [{"rowid": row["rowid"], "document_id": row["document_id"],
                "old": row["parent_index"], "new": row["idx"]} for row in sample]

    applied = 0
    if not dry_run and target:
        with conn:
            applied = conn.execute(_CTE.format(where=where)).rowcount
    after = stats(conn) if not dry_run else before
    conn.close()
    return {"db": str(db_path), "before": before, "target": target, "applied": applied,
            "after": after, "preview": preview, "dry_run": dry_run}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回填 parents.parent_index（N6）")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--force", action="store_true", help="重算已有值（默认只填 NULL）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = backfill(args.db, force=args.force, dry_run=args.dry_run)
    print(f"数据库：{result['db']}")
    print(f"回填前：父块 {result['before']['parents']}，已填 {result['before']['filled']}，"
          f"缺失 {result['before']['missing']}，文档 {result['before']['documents']}")
    print("抽样（rowid / document_id / 旧值 → 新值）：")
    for row in result["preview"]:
        print(f"  {row['rowid']:6d}  {row['document_id'][:18]:20s} {row['old']} → {row['new']}")
    if result["dry_run"]:
        print(f"（dry-run）将更新 {result['target']} 行，未写入")
    else:
        print(f"已更新 {result['applied']} 行；回填后仍缺失 {result['after']['missing']}")
    return 0 if result["dry_run"] or result["after"]["missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
