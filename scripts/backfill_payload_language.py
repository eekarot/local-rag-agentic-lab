"""回填 Qdrant payload 的 `language`（N61 的存量部分）。

**为什么需要**：`children.language` 曾经恒为 None（先建子块、后判语言），
而 `payload_from_child()` 会把 None 字段整个剔除 —— 于是**每个点的 payload 里都没有
`language`**，可那个字段的 payload 索引早就建好了（N1）：索引在，字段不在，
「只要英文块」这类过滤恒为 0 命中，而且不报错。

代码侧已修（`IngestPipeline._prepare()` 里算语言挪到 `build_children()` 之前），
新入库的文档会自动带上。**存量**两个集合不会自己变好，所以给出这个回填：

- 数据源只有一个：Document Store 的 `parents.language`（建库时判过一次，是权威值）；
- 只补 payload 里**缺** `language` 的点，已有的不覆盖（幂等，可反复跑）；
- Store 里查不到、或那一列是 NULL 的，**不猜、不写**，只统计并打印。

    python scripts/backfill_payload_language.py --collection bench_k8s_v1 --dry-run
    python scripts/backfill_payload_language.py --collection bench_k8s_v1

只改 payload，**不动向量、不动 id、不动 Store** —— 所以不需要重建库，也不该改变检索指标
（跑完请复核一遍基线，见 ROADMAP C10 的验收命令）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_COLLECTION = "bench_k8s_v1"
DEFAULT_STORE = ROOT / "storage" / "bench" / "documents.sqlite3"
SCROLL_PAGE = 500


def load_parent_languages(store_path: Path) -> dict[str, str | None]:
    """`parent_id → language`（Store 是权威值；NULL 也要收进来，好统计有多少不能回填）。"""
    from src.rag.store.document_store import DocumentStore

    store = DocumentStore(store_path)
    with store._connect() as conn:      # noqa: SLF001  同包内部
        # 与读侧 `load_parent_records()` 同一个口径：父块没有就退到文档级
        # （历史库里 parents.language 常是 NULL，硬读那一列会得出「全都没法补」的假结论）
        rows = conn.execute(
            "SELECT p.parent_id, COALESCE(p.language, d.language) AS language"
            " FROM parents p JOIN documents d ON d.document_id = p.document_id"
        ).fetchall()
    return {row["parent_id"]: row["language"] for row in rows}


def plan(client, collection: str, languages: dict[str, str | None], *, limit: int | None = None):
    """算出「哪些点要补什么」。**纯读**，不写库（`--dry-run` 与真跑共用这一段）。"""
    patch: list[tuple[str, str]] = []        # (point_id, language)
    stats = {"points": 0, "already_has_language": 0, "to_patch": 0,
             "unknown_parent": 0, "null_language": 0}
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=SCROLL_PAGE, offset=offset,
            with_payload=["parent_id", "language"], with_vectors=False,
        )
        for point in points:
            stats["points"] += 1
            payload = point.payload or {}
            if payload.get("language"):
                stats["already_has_language"] += 1
                continue
            parent_id = payload.get("parent_id")
            if parent_id not in languages:
                stats["unknown_parent"] += 1
                continue
            language = languages[parent_id]
            if not language:
                stats["null_language"] += 1
                continue
            patch.append((str(point.id), str(language)))
            stats["to_patch"] += 1
            if limit is not None and stats["to_patch"] >= limit:
                return patch, stats, offset
        if offset is None:
            return patch, stats, None


def apply_patch(client, collection: str, patch: list[tuple[str, str]], *, batch: int = 256) -> int:
    """按语言分组批量 `set_payload`（同值的点一次请求，减少往返）。"""
    from qdrant_client import models

    grouped: dict[str, list[str]] = {}
    for point_id, language in patch:
        grouped.setdefault(language, []).append(point_id)
    written = 0
    for language, ids in grouped.items():
        for start in range(0, len(ids), batch):
            chunk = ids[start:start + batch]
            client.set_payload(collection, payload={"language": language},
                               points=chunk, wait=True)
            written += len(chunk)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回填 Qdrant payload 的 language（N61）")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE,
                        help="Document Store（parents.language 是权威值）")
    parser.add_argument("--qdrant-url", default=None, help="默认读 QDRANT_URL")
    parser.add_argument("--dry-run", action="store_true", help="只算不写")
    parser.add_argument("--limit", type=int, default=None, help="最多补多少个点（抽查用）")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args(argv)

    from src.rag.store.qdrant_store import make_client

    if not args.store.exists():
        print(f"Document Store 不存在：{args.store}", file=sys.stderr)
        return 2
    client = make_client()
    if args.collection not in [c.name for c in client.get_collections().collections]:
        print(f"集合不存在：{args.collection}", file=sys.stderr)
        return 2

    languages = load_parent_languages(args.store)
    patch, stats, _ = plan(client, args.collection, languages, limit=args.limit)
    print(f"集合 {args.collection}｜Store {args.store}")
    print(f"  扫描点 {stats['points']}｜已有 language {stats['already_has_language']}"
          f"｜待补 {stats['to_patch']}")
    if stats["unknown_parent"]:
        print(f"  [跳过] Store 里查不到 parent_id 的点：{stats['unknown_parent']}")
    if stats["null_language"]:
        print(f"  [跳过] Store 里这条父块/它的文档 language 都是 NULL 的点：{stats['null_language']}"
              f"（这类 Store 不是 scripts/ingest.py 建的、或建于语言列落库之前 —— "
              f"要么重跑 ingest，要么按正文重判；本脚本不猜）")
    if args.dry_run:
        for point_id, language in patch[:3]:
            print(f"  [dry-run] {point_id} → language={language}")
        print("dry-run：没有写入任何东西")
        return 0
    if not patch:
        print("无事可做（幂等）")
        return 0
    written = apply_patch(client, args.collection, patch, batch=args.batch)
    print(f"已回填 {written} 个点的 language")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
