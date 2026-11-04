"""D34 消融 F / G：加 Reranker、加 Parent Expansion（上下文组装）。

F 与 G 是**两种不同的东西**，报告里必须分开看：

- **F = E + 重排**：改的是「列表顺序」，用 L1 指标（Recall / hit / MRR / nDCG）评；
- **G = F + 上下文组装**：不改列表顺序（顺序沿用 F），改的是「进 prompt 的东西」——
  块数、token 数、去重/合并掉多少、相邻重叠块占比。它的验收是 D32 的口径（≤5%）。

基线对照（G 的「改造前」）用 `legacy_context()`：一条命中一块、不去重不合并不控预算，
就是 `src/rag.py:format_context()` 的等价物。这样「重叠块 22.9%」是可复现的数字，不是引用的。

    python -m src.eval_rerank_ablation run --limit 202
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:  # pragma: no cover - 便于 `python src/eval_rerank_ablation.py`
    sys.path.insert(0, str(ROOT_DIR))

from src.eval_ablation import mrr_at_k, ndcg_at_k, percentile, recall_fraction  # noqa: E402
from src.eval_metrics import mcnemar_exact, paired_bootstrap  # noqa: E402
from src.rag.context.assembler import (  # noqa: E402
    adjacent_overlap_ratio,
    assemble,
    document_fragmentation,
)
from src.rag.retrieval.fusion import RRF_K, rrf_fuse_routes  # noqa: E402
from src.rag.retrieval.rerank import (  # noqa: E402
    CrossEncoderReranker,
    RerankConfig,
    RerankHit,
    best_window,
    rerank_hits,
)

DEFAULT_KS = (5, 10)
DEFAULT_POOL = 20


def gold_ids(question: Mapping[str, Any]) -> set[str]:
    return set((question.get("gold_points") or {}).get("parent_ids") or [])


def _metrics(ranked: Sequence[str], gold: set[str], k: int) -> dict[str, float]:
    return {
        "recall": recall_fraction(ranked, gold, k),
        "hit": 1.0 if set(ranked[:k]) & gold else 0.0,
        "mrr": mrr_at_k(ranked, gold, k),
        "ndcg": ndcg_at_k(ranked, gold, k),
    }


def legacy_context(hits: Sequence[Mapping[str, Any]], parents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """改造前的上下文口径：一条命中一块，不去重、不合并、不限预算（`format_context` 等价物）。"""
    blocks = []
    for index, hit in enumerate(hits, start=1):
        record = parents.get(hit["parent_id"]) or {}
        blocks.append({
            "index": index,
            "document_id": str(record.get("document_id") or ""),
            "parent_ids": [hit["parent_id"]],
            "parent_indexes": ([record["parent_index"]]
                               if isinstance(record.get("parent_index"), int) else []),
            "text": record.get("text") or "",
        })
    typed = _as_context_blocks(blocks)
    return {
        "blocks": blocks,
        "overlap_ratio": round(adjacent_overlap_ratio(typed), 4),
        "fragmentation": round(document_fragmentation(typed), 4),
    }


def _as_context_blocks(blocks: Sequence[Mapping[str, Any]]):
    from src.rag.context.assembler import ContextBlock

    return [
        ContextBlock(index=int(block["index"]), document_id=str(block.get("document_id") or ""),
                     parent_ids=list(block.get("parent_ids") or []),
                     parent_indexes=list(block.get("parent_indexes") or []),
                     chunk_ids=[], section_path=[],
                     source_uri="", title="", text=str(block.get("text") or ""), tokens=0)
        for block in blocks
    ]


def _summarize(values: Sequence[float]) -> dict[str, float]:
    return {"mean": round(statistics.fmean(values), 4) if values else 0.0}


def _ci(values: Sequence[float], n_boot: int, seed: int) -> dict[str, float]:
    """bootstrap 百分位区间（与 `eval_metrics.bootstrap_ci` 同口径，这里只要均值）。"""
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "half_width": 0.0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return {"mean": round(statistics.fmean(values), 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "half_width": round((hi - lo) / 2, 4)}


@dataclass
class CellRun:
    name: str
    metrics: dict[str, dict[str, float]]      # k → metric
    latencies: list[float]
    extra: dict[str, Any]


def run_matrix(
    questions: Sequence[Mapping[str, Any]],
    retriever,
    parents: Mapping[str, Mapping[str, Any]],
    *,
    counter,
    reranker: CrossEncoderReranker,
    ks: Sequence[int] = DEFAULT_KS,
    pool: int = DEFAULT_POOL,
    budget: int = 6000,
    n_boot: int = 1000,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    judged = [q for q in questions if gold_ids(q)]
    if not judged:
        raise ValueError("没有题目带 gold_points.parent_ids，无法评估")
    max_k = max(ks)

    per_question: list[dict[str, Any]] = []
    for index, question in enumerate(judged, start=1):
        result = retriever.retrieve(question["question"], query_en=question.get("query_en"), k=pool)
        routes = {name: route.ids[:pool] for name, route in result.routes.items()}
        fused = rrf_fuse_routes(routes, rrf_k=RRF_K, top_k=pool)
        hits = [
            RerankHit(chunk_id=hit.parent_id, parent_id=hit.parent_id,
                      text=(parents.get(hit.parent_id) or {}).get("text", ""),
                      score=hit.score, rank=rank)
            for rank, hit in enumerate(fused, start=1)
        ]
        for hit in hits:
            hit.fields["window"] = best_window(hit.text, question["question"],
                                               max_tokens=reranker.config.max_length, counter=counter)

        started = time.perf_counter()
        ranked = rerank_hits(question["question"], hits, encoder=reranker.encoder,
                             config=reranker.config, top_n=max_k, stats=reranker.stats,
                             text_getter=lambda hit: hit.fields["window"])
        rerank_ms = (time.perf_counter() - started) * 1000

        e_hits = [{"parent_id": hit.parent_id, "score": hit.score} for hit in hits]
        f_hits = [{"parent_id": hit.parent_id, "score": hit.score,
                   "rerank_score": hit.rerank_score} for hit in ranked]

        record: dict[str, Any] = {"qid": question.get("qid"), "gold": sorted(gold_ids(question)),
                                  "rerank_ms": rerank_ms, "cells": {}}
        for k in ks:
            record["cells"][str(k)] = {
                "E": _metrics([hit["parent_id"] for hit in e_hits[:pool]], gold_ids(question), k),
                "F": _metrics([hit["parent_id"] for hit in f_hits], gold_ids(question), k),
            }
        # G：上下文层指标（顺序沿用 F 的 top-k）
        f_ranked = [hit["parent_id"] for hit in f_hits]
        legacy = legacy_context([{"parent_id": pid} for pid in f_ranked[:max_k]], parents)
        assembled = assemble([{"parent_id": pid} for pid in f_ranked[:max_k]], parents,
                             counter=counter, budget=budget)
        record["context"] = {
            "legacy_blocks": len(legacy["blocks"]),
            "legacy_overlap": legacy["overlap_ratio"],
            "legacy_fragmentation": legacy["fragmentation"],
            "legacy_tokens": sum(counter.count(block["text"]) for block in legacy["blocks"]),
            "assembled_blocks": len(assembled.blocks),
            "assembled_overlap": assembled.stats.get("adjacent_overlap_ratio", 0.0),
            "assembled_fragmentation": assembled.stats.get("document_fragmentation", 0.0),
            "near_duplicate_text": assembled.stats.get("near_duplicate_text", 0),
            "assembled_tokens": assembled.tokens,
            "merged": assembled.stats.get("merged", 0),
            "duplicate_parent": assembled.stats.get("duplicate_parent", 0),
            "duplicate_text": assembled.stats.get("duplicate_text", 0),
        }
        per_question.append(record)
        if log and index % 20 == 0:
            log(f"[d34] {index}/{len(judged)}")

    cells: dict[str, CellRun] = {}
    for name in ("E", "F"):
        metrics = {str(k): {metric: [item["cells"][str(k)][name][metric] for item in per_question]
                            for metric in ("recall", "hit", "mrr", "ndcg")} for k in ks}
        cells[name] = CellRun(
            name=name,
            metrics={k: {metric: round(statistics.fmean(values), 4) for metric, values in bucket.items()}
                     for k, bucket in metrics.items()},
            latencies=[item["rerank_ms"] for item in per_question] if name == "F" else [0.0],
            extra={},
        )

    summary: dict[str, Any] = {"n": len(per_question), "pool": pool, "ks": list(ks),
                               "cells": {name: {"metrics": cell.metrics} for name, cell in cells.items()}}
    summary["cells"]["F"]["latency_ms"] = {"p50": round(percentile(cells["F"].latencies, 50), 1),
                                           "p95": round(percentile(cells["F"].latencies, 95), 1)}
    summary["reranker_stats"] = reranker.stats()

    # 配对检验：E vs F，逐题、逐指标
    pairs = []
    for k in ks:
        for metric in ("recall", "hit", "mrr", "ndcg"):
            a = [item["cells"][str(k)]["E"][metric] for item in per_question]
            b = [item["cells"][str(k)]["F"][metric] for item in per_question]
            # paired_bootstrap 的 diff = mean(第一个) - mean(第二个)，所以 F 放前面 → delta = F - E
            delta = paired_bootstrap(b, a, n_boot=n_boot, seed=seed)
            entry = {"k": k, "metric": metric, "delta": round(delta["diff"], 4),
                     "lo": round(delta["lo"], 4), "hi": round(delta["hi"], 4),
                     "a_mean": round(statistics.fmean(a), 4), "b_mean": round(statistics.fmean(b), 4)}
            if metric == "hit":
                a_only = sum(1 for x, y in zip(a, b) if x > y)
                b_only = sum(1 for x, y in zip(a, b) if y > x)
                test = mcnemar_exact(a_only, b_only)
                entry.update({"E_only": a_only, "F_only": b_only,
                              "p_value": round(test["p_value"], 4)})
            pairs.append(entry)
    summary["pairs"] = pairs

    # 上下文层（G）
    def _mean_of(key: str) -> float:
        return round(statistics.fmean(item["context"][key] for item in per_question), 4)

    summary["context"] = {key: _mean_of(key) for key in per_question[0]["context"]}
    summary["context"]["improved_overlap"] = round(
        summary["context"]["legacy_overlap"] - summary["context"]["assembled_overlap"], 4)
    legacy_tokens = summary["context"]["legacy_tokens"] or 0
    summary["context"]["token_saving"] = round(
        (legacy_tokens - summary["context"]["assembled_tokens"]) / legacy_tokens, 4
    ) if legacy_tokens else 0.0
    return summary


def render_report(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    ks = payload["ks"]
    lines.append(f"# D34 消融 F / G（{payload['n']} 题，池 {payload['pool']}）\n")
    lines.append("## F = E + Reranker（列表指标）\n")
    header = "| 方案 | k | Recall | hit | MRR | nDCG |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for name in ("E", "F"):
        for k in ks:
            metrics = payload["cells"][name]["metrics"][str(k)]
            lines.append(f"| {name} | {k} | {metrics['recall']:.4f} | {metrics['hit']:.4f} | "
                         f"{metrics['mrr']:.4f} | {metrics['ndcg']:.4f} |")
    latency = payload["cells"]["F"].get("latency_ms") or {}
    if latency:
        lines.append(f"\nF 的重排延迟：P50 {latency['p50']} ms / P95 {latency['p95']} ms\n")
    lines.append("## 配对检验（E vs F）\n")
    lines.append("| k | 指标 | E | F | delta | 95% CI | p |")
    lines.append("|---|---|---|---|---|---|---|")
    for pair in payload["pairs"]:
        p_value = f"{pair['p_value']:.4f}" if "p_value" in pair else "—"
        lines.append(f"| {pair['k']} | {pair['metric']} | {pair['a_mean']:.4f} | {pair['b_mean']:.4f} | "
                     f"{pair['delta']:+.4f} | [{pair['lo']:+.4f}, {pair['hi']:+.4f}] | {p_value} |")
    context = payload["context"]
    lines.append("\n## G = F + Context Assembler（上下文层）\n")
    lines.append("| 口径 | 块数 | token | 相邻重叠块占比 | 同文档碎片占比 |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| 改造前（format_context 等价物） | {context['legacy_blocks']:.2f} | "
                 f"{context['legacy_tokens']:.0f} | {context['legacy_overlap']:.2%} | "
                 f"{context['legacy_fragmentation']:.2%} |")
    lines.append(f"| 组装后 | {context['assembled_blocks']:.2f} | {context['assembled_tokens']:.0f} | "
                 f"{context['assembled_overlap']:.2%} | {context['assembled_fragmentation']:.2%} |")
    lines.append(f"\n- 合并相邻块：{context['merged']:.2f} 次/题；重复父块 {context['duplicate_parent']:.2f}、"
                 f"重复正文 {context['duplicate_text']:.2f}、近重复正文 {context['near_duplicate_text']:.2f}")
    lines.append(f"- 上下文 token：{context['legacy_tokens']:.0f} → {context['assembled_tokens']:.0f}"
                 f"（省 {context['token_saving']:.1%}）")
    lines.append(f"- 重叠块占比下降 **{context['improved_overlap']:.2%}**（D32 验收线 ≤5%）")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_rerank_ablation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="跑 E/F/G 矩阵")
    run.add_argument("--questions", default="experiments/questions.json")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--pool", type=int, default=DEFAULT_POOL)
    run.add_argument("--ks", default="5,10")
    run.add_argument("--budget", type=int, default=6000)
    run.add_argument("--model", default=None, help="覆盖默认 ONNX 模型 id")
    run.add_argument("--max-length", type=int, default=256)
    run.add_argument("--max-concurrency", type=int, default=1)
    run.add_argument("--boot", type=int, default=1000)
    run.add_argument("--out", default="experiments/reports/d34_rerank.md")
    run.add_argument("--json", default="experiments/reports/d34_rerank.json")
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 2

    from src.rag.ingestion.tokenizer import TokenCounter
    from src.rag.retrieval.store_wiring import build_default_retriever, load_parent_records

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    ks = [int(piece) for piece in str(args.ks).split(",") if piece.strip()]

    parents = {row["id"]: row for row in load_parent_records()}
    counter = TokenCounter()
    config = RerankConfig(max_length=args.max_length, max_concurrency=args.max_concurrency)
    if args.model:
        config.model_id = args.model
    reranker = CrossEncoderReranker(config=config)
    retriever = build_default_retriever(
        warmup_texts=[q["question"] for q in questions] + [q.get("query_en") or "" for q in questions],
    )

    started = time.perf_counter()
    payload = run_matrix(questions, retriever, parents, counter=counter, reranker=reranker,
                         ks=ks, pool=args.pool, budget=args.budget, n_boot=args.boot, log=print)
    payload["elapsed_s"] = round(time.perf_counter() - started, 1)
    payload["model"] = config.model_id

    report = render_report(payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print(f"报告：{out}\nJSON：{args.json}\n用时 {payload['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
