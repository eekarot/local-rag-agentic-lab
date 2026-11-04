"""E07 产品入口：`python -m src.ask "你的问题"`。

它是"新链"的**唯一面向使用者的命令**（此前只有 `src/eval_answer.py` 能跑通整条链路，
而那是评估器：按题号取题、输出 JSON/CSV，不适合当产品入口）。

    python -m src.ask "智能体化 RAG 和传统 RAG 有什么区别？"
    python -m src.ask "What is agentic RAG?" --json
    python -m src.ask "问题" --retrieve-only --sources all     # 只看检索，不调模型
    python -m src.ask "问题" --no-rerank                       # 关掉重排（快 0.7 s，排序变差）

链路本身在 `src/rag/ask/pipeline.py`，这里只负责**装配**（真件：Qdrant / bge-m3 /
ONNX reranker / Ollama）与打印。装配全部收在 `build_components()` 里，测试可以整个替换。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT_DIR))

from src.config import OLLAMA_CHAT_MODEL, STORAGE_DIR  # noqa: E402
from src.rag.ask.pipeline import DEFAULT_BUDGET, DEFAULT_K, DEFAULT_POOL, answer_query, render_cli  # noqa: E402
from src.rag.retrieval.store_wiring import build_default_retriever, load_parent_records  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ask",
        description="中文/英文提问 → 三路召回 → 重排 → 组装 → 生成 → 引用体检",
    )
    parser.add_argument("question", nargs="*", help="问题（不给则读标准输入的整段文本）")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"进上下文的块数（默认 {DEFAULT_K}）")
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL,
                        help=f"每路检索深度 / 重排候选池（默认 {DEFAULT_POOL}）")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help=f"上下文 token 预算（默认 {DEFAULT_BUDGET}）")
    parser.add_argument("--no-rerank", action="store_true", help="关掉 cross-encoder 重排")
    parser.add_argument("--no-translate", action="store_true", help="不调翻译（英文路直接用原问题）")
    parser.add_argument("--zh-lexical", action="store_true", help="加第四路中文 BM25（B05：默认不开）")
    parser.add_argument("--retrieve-only", action="store_true", help="只检索不生成（不占 CPU 跑 9B）")
    parser.add_argument("--sources", default="5", help="来源表条数，或 all（默认 5）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供脚本/前端消费）")
    parser.add_argument("--quiet", action="store_true", help="不打印装配日志")
    return parser


def _question_text(args: argparse.Namespace) -> str:
    text = " ".join(args.question).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        raise SystemExit("用法：python -m src.ask \"你的问题\"（或把问题从 stdin 传入）")
    return text


def build_components(args: argparse.Namespace) -> dict[str, Any]:
    """真件装配。所有重量级对象都在这里创建，`main()` 之后只做编排。"""
    from src.ollama_client import chat
    from src.rag.ingestion.tokenizer import TokenCounter
    from src.rag.lang.detect import detect
    from src.rag.retrieval.rerank import CrossEncoderReranker, RerankConfig
    from src.rag.translate.cache import TranslationCache
    from src.rag.translate.service import TranslateService

    log: Callable[[str], None] = (lambda message: None) if args.quiet else print
    question = _question_text(args)
    started = time.perf_counter()

    language = detect(question)   # 返回 "zh"/"en"/"other"
    query_en = None
    if language == "zh" and not args.no_translate:
        translator = TranslateService(
            provider=lambda prompt: chat([{"role": "user", "content": prompt}],
                                         temperature=0.0, think=False),
            cache=TranslationCache(STORAGE_DIR / "cache" / "translations.sqlite3",
                                   model=OLLAMA_CHAT_MODEL, version="v1"),
            max_retries=1,
        )
        translated = translator.translate([question])[0]      # 五态出口，永不抛异常
        query_en = translated.text or None
        log(f"[translate] status={translated.status}｜{translated.latency_ms:.0f} ms")
    elif language != "zh":
        query_en = question           # 英文提问：英文路直接用原文，不翻译
    log(f"[setup] 语言={language}｜query_en={query_en or '（无）'}｜{time.perf_counter() - started:.1f} s")

    retriever = build_default_retriever(
        include_zh_lexical=args.zh_lexical,
        warmup_texts=[text for text in (question, query_en) if text],
        log=log,
    )
    parents = {row["id"]: row for row in load_parent_records()}
    counter = TokenCounter()
    reranker = CrossEncoderReranker(config=RerankConfig(enabled=not args.no_rerank))
    ask = None if args.retrieve_only else (
        lambda messages: chat(messages, temperature=0.0, think=False)
    )
    return {"retriever": retriever, "parents": parents, "counter": counter,
            "reranker": reranker, "ask": ask, "question_en": query_en}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question = _question_text(args)
    started = time.perf_counter()
    components = build_components(args)
    setup_ms = (time.perf_counter() - started) * 1000
    result = answer_query(
        question,
        retriever=components["retriever"],
        parents=components["parents"],
        counter=components["counter"],
        reranker=components["reranker"],
        ask=components["ask"],
        query_en=components.get("question_en"),
        k=args.k, pool=args.pool, budget=args.budget,
    )
    result.timings["setup_ms"] = setup_ms

    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return 0

    show = len(result.blocks) if str(args.sources).lower() == "all" else int(args.sources)
    print(render_cli(result, show_sources=show, list_blocks=args.retrieve_only))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
