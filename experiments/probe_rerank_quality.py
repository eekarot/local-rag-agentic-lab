"""D31/D34 探针：重排到底有没有用，用哪个模型。

只回答两个问题：

1. 把 E（三路 RRF）的 top-20 送 cross-encoder 重排、取 top-10，比不重排强多少？
2. `mmarco-fp32`（快）和 `jina-int8`（慢一倍）哪个值？

判据是**逐题配对**（同一批题、同一份候选，只差有没有重排），不是比两个聚合区间。

    python experiments/probe_rerank_quality.py --limit 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_ablation import mrr_at_k, ndcg_at_k, percentile, recall_fraction  # noqa: E402
from src.rag.ingestion.tokenizer import TokenCounter  # noqa: E402
from src.rag.retrieval.fusion import RRF_K, rrf_fuse_routes  # noqa: E402
from src.rag.retrieval.rerank import (  # noqa: E402
    CrossEncoderReranker,
    RerankConfig,
    RerankHit,
    best_window,
    rerank_hits,
)
from src.rag.retrieval.store_wiring import build_default_retriever, load_parent_records  # noqa: E402

MODELS = {
    "mmarco-fp32": ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "onnx/model.onnx"),
    "jina-int8": ("jinaai/jina-reranker-v2-base-multilingual", "onnx/model_int8.onnx"),
}


def gold_of(question: dict) -> set[str]:
    return set((question.get("gold_points") or {}).get("parent_ids") or [])


def metrics(ranked: list[str], gold: set[str], k: int) -> dict[str, float]:
    return {
        "recall": recall_fraction(ranked, gold, k),
        "hit": 1.0 if set(ranked[:k]) & gold else 0.0,
        "mrr": mrr_at_k(ranked, gold, k),
        "ndcg": ndcg_at_k(ranked, gold, k),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="experiments/questions.json")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--models", default="mmarco-fp32,jina-int8")
    parser.add_argument("--out", default="experiments/probes/d31_rerank_quality.json")
    args = parser.parse_args()

    questions = [q for q in json.loads(Path(args.questions).read_text(encoding="utf-8")) if gold_of(q)]
    questions = questions[: args.limit]
    print(f"题数 {len(questions)}（有 gold 的前 {args.limit} 条）")

    parents = {row["id"]: row for row in load_parent_records()}
    counter = TokenCounter()
    retriever = build_default_retriever(
        warmup_texts=[q["question"] for q in questions] + [q.get("query_en") or "" for q in questions],
    )

    # 一次检索、多模型复用同一份候选（控制变量）
    prepared = []
    retrieve_ms = []
    for question in questions:
        started = time.perf_counter()
        result = retriever.retrieve(question["question"], query_en=question.get("query_en"), k=args.pool)
        retrieve_ms.append((time.perf_counter() - started) * 1000)
        routes = {name: route.ids[: args.pool] for name, route in result.routes.items()}
        fused = rrf_fuse_routes(routes, rrf_k=RRF_K, top_k=args.pool)
        hits = [
            RerankHit(chunk_id=hit.parent_id, parent_id=hit.parent_id,
                      text=parents.get(hit.parent_id, {}).get("text", ""),
                      score=hit.score, rank=rank)
            for rank, hit in enumerate(fused, start=1)
        ]
        prepared.append({"gold": gold_of(question), "hits": hits, "query": question["question"]})

    report: dict[str, object] = {"n": len(questions), "pool": args.pool, "k": args.k,
                                 "retrieve_ms": {"p50": round(percentile(retrieve_ms, 50), 1),
                                                 "p95": round(percentile(retrieve_ms, 95), 1)},
                                 "cells": {}}

    base = [metrics([h.parent_id for h in item["hits"]], item["gold"], args.k) for item in prepared]
    report["cells"]["E"] = {name: round(statistics.fmean(m[name] for m in base), 4) for name in base[0]}
    print(f"E（不重排，池 {args.pool}→top{args.k}）：{report['cells']['E']}")

    for label in args.models.split(","):
        model_id, onnx_file = MODELS[label]
        config = RerankConfig(model_id=model_id, onnx_file=onnx_file, max_length=256, top_k_pool=args.pool)
        reranker = CrossEncoderReranker(config=config)
        latencies, ranked_metrics = [], []
        for item in prepared:
            # 用**用户原句**挑窗口 + 重排（生产口径；英文 query 是内部召回用的）
            for hit in item["hits"]:
                hit.fields["window"] = best_window(hit.text, item["query"],
                                                   max_tokens=config.max_length, counter=counter)
            started = time.perf_counter()
            ranked = rerank_hits(item["query"], item["hits"], encoder=reranker.encoder,
                                 config=config, top_n=args.k, stats=reranker.stats,
                                 text_getter=lambda hit: hit.fields["window"])
            latencies.append((time.perf_counter() - started) * 1000)
            ranked_metrics.append(metrics([hit.parent_id for hit in ranked], item["gold"], args.k))
        summary = {name: round(statistics.fmean(m[name] for m in ranked_metrics), 4)
                   for name in ranked_metrics[0]}
        summary["latency_ms"] = {"p50": round(percentile(latencies, 50), 1),
                                 "p95": round(percentile(latencies, 95), 1)}
        summary["stats"] = reranker.stats()
        report["cells"][f"F-{label}"] = summary
        wins = sum(1 for a, b in zip(base, ranked_metrics) if b["hit"] > a["hit"])
        losses = sum(1 for a, b in zip(base, ranked_metrics) if b["hit"] < a["hit"])
        report["cells"][f"F-{label}"]["wins_vs_E"] = wins
        report["cells"][f"F-{label}"]["losses_vs_E"] = losses
        print(f"F-{label}：{summary}  胜/负={wins}/{losses}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
