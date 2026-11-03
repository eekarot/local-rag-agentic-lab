"""一条命令把「一堆文件」变成「Qdrant + Document Store 可检索」。

    # 全量（可反复执行，已完成的不重算）
    python scripts/ingest.py --dir data

    # 小批量试跑（先 200 篇看看）
    python scripts/ingest.py --dir data/md --limit 200

    # 只看不做
    python scripts/ingest.py --dir data --dry-run

    # 进度与失败
    python scripts/ingest.py --status
    python scripts/ingest.py --failures
    python scripts/ingest.py --retry-failed

    # 单条重试（doc_key 就是相对 --dir 的路径）
    python scripts/ingest.py --dir data --only md/nvidia/hnsg.md

    # 强制重做（忽略已完成状态）
    python scripts/ingest.py --dir data/md --limit 50 --force

设计要点：
- 默认**接着上次跑**：状态表里有 `COMPLETED` 且三处版本（文档/切分/embedding）都没变的
  文档直接跳过，日志里是 `[skip]`；
- 中断随时安全：停在任何阶段，重跑都能收敛（状态表 + embedding 缓存 + point id 幂等）；
- 失败不打断整轮：进 `ingest_failures`，`--retry-failed` 或 `--only` 单条重试。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import STORAGE_DIR  # noqa: E402
from src.rag.embedding.service import EmbeddingService  # noqa: E402
from src.rag.ingest.pipeline import DEFAULT_PATTERNS, IngestPipeline  # noqa: E402
from src.rag.ingest.state import IngestState  # noqa: E402
from src.rag.store.document_store import DocumentStore  # noqa: E402
from src.rag.store.indexer import Indexer  # noqa: E402
from src.rag.store.qdrant_store import COLLECTION_NAME, ensure_collection, make_client  # noqa: E402

STATE_PATH = STORAGE_DIR / "ingest" / "state.sqlite3"
DOCUMENT_DB = STORAGE_DIR / "documents.sqlite3"


def build_pipeline(args, *, log) -> tuple[IngestPipeline, object, EmbeddingService, IngestState]:
    client = make_client()
    ensure_collection(client, args.collection)
    state = IngestState(args.state or STATE_PATH)
    store = DocumentStore(args.document_db or DOCUMENT_DB)
    embedder = EmbeddingService(cache_path=args.cache or (STORAGE_DIR / "cache" / "embeddings.sqlite3"))
    indexer = Indexer(
        client=client, collection=args.collection, store=store, embedder=embedder,
        batch_size=args.batch_size,
    )
    pipeline = IngestPipeline(
        data_dir=args.dir, state=state, store=store, indexer=indexer, embedder=embedder,
        log=log, patterns=tuple(args.patterns), window_docs=args.window_docs,
        window_children=args.window_children,
    )
    return pipeline, client, embedder, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest", description="端到端摄入：文件 → Qdrant + Document Store")
    parser.add_argument("--dir", type=Path, default=ROOT / "data", help="语料根目录")
    parser.add_argument("--patterns", nargs="*", default=list(DEFAULT_PATTERNS), help="文件通配符")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少篇（按路径排序）")
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些 doc_key（相对 --dir 的路径）")
    parser.add_argument("--force", action="store_true", help="忽略已完成状态，强制重做")
    parser.add_argument("--dry-run", action="store_true", help="只发现不写入")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--document-db", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256, help="单次 upsert 的点数")
    parser.add_argument("--window-docs", type=int, default=64)
    parser.add_argument("--window-children", type=int, default=256)
    parser.add_argument("--fail-on-error", action="store_true", help="任一文档失败就中止（默认继续）")
    parser.add_argument("--status", action="store_true", help="看进度")
    parser.add_argument("--failures", action="store_true", help="列失败清单")
    parser.add_argument("--retry-failed", action="store_true", help="只重跑失败文档")
    parser.add_argument("--json", action="store_true", help="输出 JSON（便于脚本消费）")
    args = parser.parse_args(argv)

    lines: list[str] = []

    def log(message: str) -> None:
        lines.append(message)
        if not args.json:
            print(message, flush=True)

    state = IngestState(args.state or STATE_PATH)
    if args.status:
        print(json.dumps(state.stats(), ensure_ascii=False, indent=2))
        counts = state.stage_counts()
        total = sum(counts.values())
        done = counts.get("COMPLETED", 0)
        if total:
            print(f"进度：{done}/{total} 完成（{done / total:.1%}）")
        return 0
    if args.failures:
        rows = state.failures(unresolved_only=not args.retry_failed)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    pipeline, client, embedder, state = build_pipeline(args, log=log)
    started = time.perf_counter()
    try:
        if args.retry_failed:
            summary = pipeline.retry_failed(dry_run=args.dry_run,
                                            continue_on_error=not args.fail_on_error)
        else:
            summary = pipeline.run(
                limit=args.limit, force=args.force, dry_run=args.dry_run,
                only_keys=args.only, continue_on_error=not args.fail_on_error,
            )
    finally:
        client.close()
    summary["elapsed_wall_s"] = round(time.perf_counter() - started, 1)
    summary["state"] = state.stats()
    summary["embedding_stats"] = embedder.stats()
    if args.json:
        print(json.dumps({"summary": summary, "log": lines}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
