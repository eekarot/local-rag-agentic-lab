"""D35 L2 答案层评估：把「检索到了」变成「答得对、引得住、该弃就弃」。

四个数（§17.5 D35 的验收口径）：

| 指标 | 自动/人工 | 含义 |
|---|---|---|
| `citation_precision` | **自动**（近似） | 引用到的证据块里，有多少块真的含 gold 父块 |
| `invalid_citation_rate` | **自动** | `[证据 n]` 中 n 越界 / 解析失败的比例 |
| `wrong_confidence` | **自动**（近似） | 负样本（库里无答案）**没有弃答**的比例 —— 模型最危险的失败模式 |
| `over_abstain` | **自动**（近似） | 正样本里明明检索到了 gold、却仍弃答的比例 |
| `faithfulness` | **人工** | 答案每句话是否被引用的证据支持（打分表见 CSV） |

**自动指标只是代理**：`citation_precision` 判的是「引用的块对不对」，不是「答案对不对」；
faithfulness 必须人工。所以这个脚本同时产出**人工打分表**（CSV，人工列留空），
自动数写 JSON，两边一起才是 D35 的交付物。

    python -m src.eval_answer run --positives 20 --negatives 20
    python -m src.eval_answer report
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT_DIR))

from src.config import OLLAMA_CHAT_MODEL  # noqa: E402
from src.rag.answer.citations import citation_summary, parse_citations, resolve_sources  # noqa: E402
from src.rag.answer.prompt import build_messages, detect_abstain  # noqa: E402
from src.rag.context.assembler import assemble  # noqa: E402
from src.rag.retrieval.fusion import RRF_K, rrf_fuse_routes  # noqa: E402
from src.rag.retrieval.rerank import (  # noqa: E402
    CrossEncoderReranker,
    RerankHit,
    best_window,
    rerank_hits,
)

DEFAULT_JSON = "experiments/reports/l2_auto.json"
DEFAULT_CSV = "experiments/reports/l2_scoring_sheet.csv"

HUMAN_COLUMNS = ("faithfulness_1_5", "citation_correct_y_n", "answer_correct_y_n", "human_note")


def gold_ids(question: Mapping[str, Any]) -> set[str]:
    return set((question.get("gold_points") or {}).get("parent_ids") or [])


def pick_questions(questions: Sequence[Mapping[str, Any]], positives: int, negatives: int) -> list[dict]:
    """正负样本都要，且**固定顺序**（可复现）：正样本取前 N 条有 gold 的，负样本取前 M 条。"""
    pos = [q for q in questions if gold_ids(q)][:positives]
    neg = [q for q in questions if q.get("type") == "negative"][:negatives]
    return [dict(q) for q in pos + neg]


def answer_one(question: Mapping[str, Any], retriever, parents, counter, reranker,
               ask: Callable[[list[dict[str, str]]], str], *, pool: int = 20, k: int = 10,
               budget: int = 6000) -> dict[str, Any]:
    """单题端到端：三路召回 → 重排 → 组装 → prompt → 模型 → 引用体检。"""
    started = time.perf_counter()
    result = retriever.retrieve(question["question"], query_en=question.get("query_en"), k=pool)
    routes = {name: route.ids[:pool] for name, route in result.routes.items()}
    fused = rrf_fuse_routes(routes, rrf_k=RRF_K, top_k=pool)
    hits = [RerankHit(chunk_id=hit.parent_id, parent_id=hit.parent_id,
                      text=(parents.get(hit.parent_id) or {}).get("text", ""),
                      score=hit.score, rank=rank)
            for rank, hit in enumerate(fused, start=1)]
    for hit in hits:
        hit.fields["window"] = best_window(hit.text, question["question"],
                                           max_tokens=reranker.config.max_length, counter=counter)
    ranked = rerank_hits(question["question"], hits, encoder=reranker.encoder,
                         config=reranker.config, top_n=k, stats=reranker.stats,
                         text_getter=lambda hit: hit.fields["window"])
    assembled = assemble([{"parent_id": hit.parent_id, "score": hit.score,
                           "rerank_score": hit.rerank_score} for hit in ranked],
                         parents, counter=counter, budget=budget)
    bundle = build_messages(question["question"], assembled)

    if bundle.abstained:                     # 无证据：不调模型
        answer, latency_ms, called = bundle.messages[0]["content"], 0.0, False
    else:
        started_llm = time.perf_counter()
        answer = ask(bundle.messages)
        latency_ms = (time.perf_counter() - started_llm) * 1000
        called = True

    report = parse_citations(answer, k=len(assembled.blocks) or None)
    cited_blocks = resolve_sources(report, [block.as_dict() for block in assembled.blocks])
    gold = gold_ids(question)
    cited_parents = {pid for block in cited_blocks for pid in block.get("parent_ids") or []}
    return {
        "qid": question.get("qid"),
        "type": question.get("type"),
        "question": question["question"],
        "answer": answer,
        "abstained": bool(bundle.abstained) or detect_abstain(answer),
        "abstain_reason": bundle.reason,
        "blocks": len(assembled.blocks),
        "context_tokens": assembled.tokens,
        "cited": report.cited,
        "out_of_range": report.out_of_range,
        "malformed": report.malformed,
        "parse_rate": round(report.parse_rate, 4),
        "cited_blocks": len(cited_blocks),
        "cited_hit_gold": len(cited_parents & gold),
        "gold_size": len(gold),
        "is_negative": question.get("type") == "negative",
        "llm_called": called,
        "latency_ms": round(latency_ms, 1),
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
        "report": report,
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """四个自动数 + 引用体检汇总。正负样本分开算，绝不混一个平均。"""
    positives = [item for item in records if not item["is_negative"]]
    negatives = [item for item in records if item["is_negative"]]

    def rate(items, predicate) -> float:
        return round(sum(1 for item in items if predicate(item)) / len(items), 4) if items else 0.0

    cited_total = sum(item["cited_blocks"] for item in positives)
    cited_hit = sum(item["cited_hit_gold"] for item in positives)
    occurrences = (sum(len(item["cited"]) + len(item["out_of_range"]) + len(item["malformed"])
                   for item in records))
    invalid = sum(len(item["out_of_range"]) for item in records) + sum(len(item["malformed"]) for item in records)

    return {
        "n": len(records),
        "positives": len(positives),
        "negatives": len(negatives),
        "citation_precision": round(cited_hit / cited_total, 4) if cited_total else 0.0,
        "invalid_citation_rate": round(invalid / occurrences, 4) if occurrences else 0.0,
        "wrong_confidence": rate(negatives, lambda item: not item["abstained"]),
        "over_abstain": rate(positives, lambda item: item["abstained"]),
        "negatives_abstained": rate(negatives, lambda item: item["abstained"]),
        "positives_answered_with_gold": rate(
            positives, lambda item: not item["abstained"] and item["cited_hit_gold"] > 0),
        "citations": citation_summary([item["report"] for item in records if item.get("report")]),
        "no_evidence_shortcut": rate(records, lambda item: not item["llm_called"]),
        "context_tokens_p50": round(statistics.fmean(item["context_tokens"] for item in records), 1) if records else 0.0,
        "latency_ms_p50": round(statistics.fmean([item["latency_ms"] for item in records if item["llm_called"]] or [0.0]), 1),
        "model": OLLAMA_CHAT_MODEL,
    }


def write_sheet(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """人工打分表：自动列填好，人工列留空。每行一道题，方便逐条打。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["qid", "type", "question", "answer", "cited", "blocks", "cited_blocks",
               "cited_hit_gold", "gold_size", "abstained", "parse_rate", *HUMAN_COLUMNS]
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in records:
            writer.writerow({
                "qid": item["qid"],
                "type": item["type"],
                "question": item["question"],
                "answer": (item["answer"] or "").replace("\r\n", " ").replace("\n", " "),
                "cited": ",".join(str(n) for n in item["cited"]),
                "blocks": item["blocks"],
                "cited_blocks": item["cited_blocks"],
                "cited_hit_gold": item["cited_hit_gold"],
                "gold_size": item["gold_size"],
                "abstained": int(item["abstained"]),
                "parse_rate": item["parse_rate"],
                **{column: "" for column in HUMAN_COLUMNS},
            })
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_answer")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="跑 L2（会真的调模型）")
    run.add_argument("--questions", default="experiments/questions.json")
    run.add_argument("--positives", type=int, default=20)
    run.add_argument("--negatives", type=int, default=20)
    run.add_argument("--pool", type=int, default=20)
    run.add_argument("--k", type=int, default=10)
    run.add_argument("--budget", type=int, default=6000)
    run.add_argument("--json", default=DEFAULT_JSON)
    run.add_argument("--csv", default=DEFAULT_CSV)
    run.add_argument("--dry-run", action="store_true", help="只用空答案跑链路，不调模型")
    run.add_argument("--partial", default=None,
                     help="逐题追加写 JSONL（默认 <json 同目录>/l2_auto.partial.jsonl），断了能续跑")
    run.add_argument("--resume", action="store_true", help="复用 --partial 里已完成的题，跳过重跑")
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 2

    from src.rag.ingestion.tokenizer import TokenCounter
    from src.rag.retrieval.store_wiring import build_default_retriever, load_parent_records

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    picked = pick_questions(questions, args.positives, args.negatives)
    parents = {row["id"]: row for row in load_parent_records()}
    counter = TokenCounter()
    reranker = CrossEncoderReranker()
    retriever = build_default_retriever(
        warmup_texts=[q["question"] for q in picked] + [q.get("query_en") or "" for q in picked],
    )

    if args.dry_run:
        ask = lambda messages: "[证据 1]（dry-run 占位答案）"
    else:
        from src.ollama_client import chat

        def ask(messages):
            return chat(messages, fmt=None)

    # 逐题落盘：一次 40 题要几十分钟，中途超时/中断不该白跑
    partial = Path(args.partial) if args.partial else Path(args.json).with_name("l2_auto.partial.jsonl")
    partial.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if args.resume and partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                done[item["qid"]] = item
        print(f"[resume] 已有 {len(done)} 题，跳过")

    records = []
    started = time.perf_counter()
    for index, question in enumerate(picked, start=1):
        if question.get("qid") in done:
            records.append(done[question["qid"]])
            print(f"[{index}/{len(picked)}] {question.get('qid')} [resume] 已存在")
            continue
        record = answer_one(question, retriever, parents, counter, reranker, ask,
                            pool=args.pool, k=args.k, budget=args.budget)
        records.append(record)
        stored = {key: value for key, value in record.items() if key != "report"}
        with partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stored, ensure_ascii=False) + chr(10))
        print(f"[{index}/{len(picked)}] {record['qid']} 弃答={int(record['abstained'])} "
              f"引用={record['cited']} 命中gold={record['cited_hit_gold']}/{record['gold_size']} "
              f"{record['latency_ms']:.0f} ms")
    summary = summarize(records)
    summary["elapsed_s"] = round(time.perf_counter() - started, 1)

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"summary": summary, "records": [
        {key: value for key, value in item.items() if key != "report"} for item in records
    ]}, ensure_ascii=False, indent=2), encoding="utf-8")
    sheet = write_sheet(records, args.csv)

    print("\n=== L2 自动指标 ===")
    for key in ("n", "positives", "negatives", "citation_precision", "invalid_citation_rate",
                "wrong_confidence", "over_abstain", "negatives_abstained",
                "positives_answered_with_gold", "no_evidence_shortcut",
                "context_tokens_p50", "latency_ms_p50"):
        print(f"  {key:28s} {summary[key]}")
    print(f"\nJSON：{args.json}\n人工打分表：{sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
