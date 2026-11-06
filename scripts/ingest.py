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
from src.rag.ingest.pipeline import DEFAULT_PATTERNS, IngestPipeline, IngestSummary  # noqa: E402
from src.rag.ingestion.errors import TokenizerUnavailable  # noqa: E402
from src.rag.ingest.state import IngestState  # noqa: E402
from src.rag.store.document_store import DocumentStore  # noqa: E402
from src.rag.store.index_versions import IndexVersions  # noqa: E402
from src.rag.store.indexer import Indexer  # noqa: E402
from src.rag.store.qdrant_store import default_collection, ensure_collection, make_client  # noqa: E402

STATE_PATH = STORAGE_DIR / "ingest" / "state.sqlite3"



def default_state_path(collection: str) -> Path:
    """默认状态表**按集合分文件**。

    为什么要分：状态表原本只有一份，`_bind_collection()` 发现集合不符就 `DELETE` 清空进度
    （那是「换集合 = 换一整套库」的正确行为）。问题是**同一台机器上换集合作实验**时，
    正经库的 1,467 条进度会被一起清掉 —— `--status` 报 0 篇（很像「库是空的」），
    `--delete` 查不到 document_id 直接跳过（N33，v0.1.0 手测实测）。

    默认集合仍叫 `state.sqlite3`：老库在原位，升级后能被「认领」而不是白跑全量。
    """
    if collection == default_collection():
        return STATE_PATH
    return STORAGE_DIR / "ingest" / f"state-{collection}.sqlite3"

DOCUMENT_DB = STORAGE_DIR / "documents.sqlite3"
VERSIONS_DB = STORAGE_DIR / "index_versions.sqlite3"


def build_pipeline(args, *, log) -> tuple[IngestPipeline, object, EmbeddingService, IngestState]:
    client = make_client()
    ensure_collection(client, args.collection)
    state = IngestState(args.state or default_state_path(args.collection),
                        collection=args.collection)
    store = DocumentStore(args.document_db or DOCUMENT_DB)
    if getattr(state, "collection_switched", False):
        print(f"[ingest] 状态表原属于别的集合，已按「全新库」重建（当前集合 {args.collection}）",
              file=sys.stderr)
    embedder = EmbeddingService(cache_path=args.cache or (STORAGE_DIR / "cache" / "embeddings.sqlite3"))
    versions = IndexVersions(args.versions or VERSIONS_DB)
    indexer = Indexer(
        client=client, collection=args.collection, store=store, embedder=embedder,
        batch_size=args.batch_size, versions=versions, state=state,
    )
    pipeline = IngestPipeline(
        data_dir=args.dir, state=state, store=store, indexer=indexer, embedder=embedder,
        log=log, patterns=tuple(args.patterns), window_docs=args.window_docs,
        window_children=args.window_children,
    )
    return pipeline, client, embedder, state


def _hint_tokenizer_if_missing(lines: list[str]) -> None:
    """空目录指引之外再补一条：词表没下的话，下一条命令还会失败，一次说完。

    干净克隆的典型顺序是「空 data/ → 下词表 → 建库」。只讲前一半，用户会在
    第二条命令上再撞一次墙（A5 实测）。
    """
    from src.rag.ingestion.errors import TokenizerUnavailable
    from src.rag.ingestion.tokenizer import TokenCounter

    try:
        TokenCounter()
    except TokenizerUnavailable as exc:
        lines.extend(str(exc).splitlines())


def _no_work(args, lines: list[str]) -> bool:
    """空语料 / `--only` 没匹配：先给指引并返回 True，**不建管线**。

    为什么要前置：建管线会初始化 `TokenCounter`（17 MB 词表）。干净克隆里还没下词表，
    用户按 README 跑第一条命令时，会在「该放哪」的指引之前先吃到一屏
    `TokenizerUnavailable` 堆栈（A5 真·干净环境实测）。指引比堆栈有用得多。
    """
    from src.rag.ingest.pipeline import discover_corpus, empty_corpus_hint, missing_only_keys_hint

    discovered = discover_corpus(args.dir, args.patterns)
    if not discovered:
        lines.extend(empty_corpus_hint(args.dir, args.patterns))
        _hint_tokenizer_if_missing(lines)
        return True
    if args.only:
        present = {p.relative_to(args.dir).as_posix() for p in discovered}
        missing = [key for key in args.only if key not in present]
        if missing:
            lines.extend(missing_only_keys_hint(missing))
            return True
    return False


def _collection_points(collection: str) -> int | None:
    """当前集合有多少点；连不上返回 None（状态查询不该因为服务没起就失败）。"""
    try:
        client = make_client(timeout=5)
        try:
            return client.count(collection_name=collection, exact=True).count
        finally:
            client.close()
    except Exception:                                  # noqa: BLE001 - 只用来显示
        return None


def build_parser() -> argparse.ArgumentParser:
    """把参数定义抽出来，用例才能验证「默认值服从环境变量」（A5 撞库之后加的）。"""
    parser = argparse.ArgumentParser(prog="ingest", description="端到端摄入：文件 → Qdrant + Document Store")
    parser.add_argument("--dir", type=Path, default=ROOT / "data", help="语料根目录")
    parser.add_argument("--patterns", nargs="*", default=list(DEFAULT_PATTERNS), help="文件通配符")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少篇（按路径排序）")
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些 doc_key（相对 --dir 的路径）")
    parser.add_argument("--force", action="store_true", help="忽略已完成状态，强制重做")
    parser.add_argument("--dry-run", action="store_true", help="只发现不写入")
    parser.add_argument("--collection", default=default_collection())
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--document-db", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--versions", type=Path, default=None, help="索引版本登记表（D21）")
    parser.add_argument("--batch-size", type=int, default=256, help="单次 upsert 的点数")
    parser.add_argument("--window-docs", type=int, default=64)
    parser.add_argument("--window-children", type=int, default=256)
    parser.add_argument("--fail-on-error", action="store_true", help="任一文档失败就中止（默认继续）")
    parser.add_argument("--status", action="store_true", help="看进度")
    parser.add_argument("--failures", action="store_true", help="列失败清单")
    parser.add_argument("--retry-failed", action="store_true", help="只重跑失败文档")
    parser.add_argument("--delete", nargs="*", default=None,
                        help="删除这些 doc_key（Qdrant + Store + 状态表一起清），之后可重跑重建")
    parser.add_argument("--json", action="store_true", help="输出 JSON（便于脚本消费）")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    lines: list[str] = []

    def log(message: str) -> None:
        lines.append(message)
        if not args.json:
            print(message, flush=True)

    state = IngestState(args.state or default_state_path(args.collection),
                        collection=args.collection)
    if args.status:
        # 只说「0 篇」会让人以为库是空的。三样一起报才看得出问题在哪：
        # 哪个集合、哪本账、库里实际有多少点（N33 的排查就靠这个对照）。
        stats = state.stats()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        counts = state.stage_counts()
        total = sum(counts.values())
        done = counts.get("COMPLETED", 0)
        print(f"集合：{args.collection}")
        print(f"状态表：{state.path}（{total} 条记录）")
        points = _collection_points(args.collection)
        if points is not None:
            print(f"库内点数：{points}")
        if total:
            print(f"进度：{done}/{total} 完成（{done / total:.1%}）")
        elif points:
            # 有向量没进度 = 进度记录丢了（换集合共用一本账的老问题，见 N33）。
            # 别让人以为库是空的：重建是安全的（点 id 幂等），只是会重算一遍。
            print("[警告] 库里有向量，但状态表没有进度记录 —— 进度丢了，不是库空了")
            print("       重跑一次建库即可（点 id 幂等；embedding 有缓存，通常几分钟内）")
            print(f"       python scripts/ingest.py --dir data")
        return 0
    if args.failures:
        rows = state.failures(unresolved_only=not args.retry_failed)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    # 空语料/没匹配（O1）是**第一次运行的最常见情形**，必须在 build_pipeline 之前判定：
    # 建管线要初始化 TokenCounter（17 MB 词表），干净克隆里它还不存在，于是用户在
    # 「该放哪」的指引之前先吃到一屏 TokenizerUnavailable 堆栈（A5 真·干净环境实测）。
    if not args.status and not args.failures and not args.delete and not args.retry_failed:
        if _no_work(args, lines):
            if args.json:
                print(json.dumps({"discovered": 0, "skipped": 0, "indexed": 0, "failed": 0,
                                  "documents": 0, "parents": 0, "children": 0}, ensure_ascii=False,
                                 indent=2))
            else:
                print(chr(10).join(lines))
            return 0

    try:
        pipeline, client, embedder, state = build_pipeline(args, log=log)
    except TokenizerUnavailable as exc:
        # 缺词表是**新建库最常见的第二种失败**（第一种是空 data/）。异常文案已经写清
        # 下载命令，不该以堆栈形式出现 —— 用户会以为是自己装错了（A5 真·干净环境实测）。
        print(str(exc), file=sys.stderr)
        return 1
    if args.delete:
        indexer = pipeline.indexer
        for key in args.delete:
            row = state.get(key)
            document_id = row["document_id"] if row else None
            if document_id is None:
                print(f"[delete] {key}：状态表里没有 document_id，跳过", file=sys.stderr)
                continue
            result = indexer.delete_document(document_id)
            print(f"[delete] {key} → 删点 {result['points']}、删文档 {result['documents']}、"
                  f"清状态 {result.get('state_reset', 0)}")
        client.close()
        return 0
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
