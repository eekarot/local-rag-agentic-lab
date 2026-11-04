"""D28：消融矩阵 A–E。

| 方案 | 配置 | 在本项目的实现 |
|---|---|---|
| A | 中文 Query + Dense | `dense_zh` 的 top-k |
| B | 英文 Query + Dense | `dense_en` 的 top-k |
| C | 中文 Dense + 英文 Dense | 两路 top-k 的**并集候选池** |
| D | C + 英文 BM25 | 三路并集候选池 |
| E | D + RRF | 三路 RRF 融合后的 top-k |

关键口径：
- **C/D 用并集，不用融合**。RRF 是 E 的自变量；C/D 若也融合，E 就没有对照了。
- **每题只检索一次**，五格共用同一次结果 → 不靠 `disable` 造 A/B/C/D（那会把
  「关掉一路」的实现差异混进结论）。
- **MRR / nDCG 的排序用各路原始顺序**，C/D 的并集顺序 = 路由顺序拼接（保序去重）。
- 翻译耗时**单列**：D28 用人工 `query_en`，矩阵本身不含翻译开销；线上成本另算。

用法：

    python -m src.eval_ablation run --questions experiments/questions.json \
        --out experiments/reports/d28_ablation.md --json experiments/reports/d28_ablation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean as _mean
from typing import Any, Callable, Iterable, Mapping, Sequence

from .eval_metrics import bootstrap_ci, mcnemar_exact
from .rag.retrieval.fusion import RRF_K, rrf_fuse_routes, union_pool

CELL_NAMES = ("A", "B", "C", "D", "D2", "E", "E2")
_KNOWN_CELLS = set(CELL_NAMES)
DEFAULT_KS = (5, 10)
DEFAULT_TRANSLATION_MS = 3080.0      # D23 实测：冷路径 3.08 s/query（202 条 621.7 s）
DEFAULT_TRANSLATION_SOURCE = "D23 实测冷路径（3.08 s/query，缓存命中 0.2 s/202 条）"


# ----------------------------------------------------------------- 指标


def percentile(values: Sequence[float], pct: float) -> float:
    """线性插值分位数（与 `search.py` / 报表口径保持一致）。"""
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * (pct / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[int(position)]
    return values[low] + (values[high] - values[low]) * (position - low)


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """二值增益 nDCG@k。`ideal = Σ_{i=1..|G|} 1/log2(i+1)`。"""
    gold = set(gold or ())
    if not gold or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, item_id in enumerate(list(ranked)[:k], start=1):
        if item_id in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def recall_fraction(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold or ())
    if not gold or k <= 0:
        return 0.0
    found = len(gold & set(list(ranked)[:k]))
    return found / len(gold)


def mrr_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold or ())
    if not gold:
        return 0.0
    for rank, item_id in enumerate(list(ranked)[:k], start=1):
        if item_id in gold:
            return 1.0 / rank
    return 0.0


def parse_cells(text: str) -> list[str]:
    """`"ABCDED2"` / `"A,E2"` → `["A","B","C","D","E","D2"]`。`D2`/`E2` 是两字符 token。"""
    text = "".join(str(text or "").split()).replace("，", ",")
    tokens: list[str] = []
    for chunk in text.split(","):
        index = 0
        while index < len(chunk):
            two = chunk[index:index + 2]
            if two in _KNOWN_CELLS:
                tokens.append(two)
                index += 2
            else:
                tokens.append(chunk[index])
                index += 1
    unknown = [token for token in tokens if token not in _KNOWN_CELLS]
    if unknown:
        raise ValueError(f"未知方案：{unknown}；可选 {sorted(_KNOWN_CELLS)}")
    return tokens


# ----------------------------------------------------------------- 单题五格


def _gold_ids(question: Mapping[str, Any]) -> set[str]:
    return set((question.get("gold_points") or {}).get("parent_ids") or [])


def cell_rankings(result, k: int, *, rrf_k: int = RRF_K) -> dict[str, list[str]]:
    """把一次 `RetrievalResult` 摊成五格的排名列表（**按同一个 k 取**）。

    A/B/E 是「列表」，长度 ≤ k；C/D/D2 是「候选池」，长度 ≤ 路数×k。
    所以池的 Recall 是**池内覆盖率**（上限），列表的 Recall 是 top-k 命中率——
    报告里分列 `列表/池大小` 就是为了不让这两种口径混读。
    """
    route_ids = {name: route.ids for name, route in result.routes.items()}
    dense_zh = route_ids.get("dense_zh", [])[:k]
    dense_en = route_ids.get("dense_en", [])[:k]
    lexical_en = route_ids.get("lexical_en", [])[:k]

    three_routes = {"dense_zh": dense_zh, "dense_en": dense_en, "lexical_en": lexical_en}
    cells = {
        "A": list(dense_zh),
        "B": list(dense_en),
        "C": union_pool({"dense_zh": dense_zh, "dense_en": dense_en}),
        "D": union_pool(three_routes),
        # E 由三路列表自己算 RRF：**不抄 result.fused**，否则 `--include-zh-lexical`
        # 会把 E 也变成四路，E 与 E2 的差异就混进了「融合路数」之外的东西
        "E": [hit.parent_id for hit in
              rrf_fuse_routes(three_routes, rrf_k=rrf_k, top_k=k)],
    }
    if "lexical_zh" in route_ids:
        four_routes = dict(three_routes, lexical_zh=route_ids["lexical_zh"][:k])
        cells["D2"] = union_pool(four_routes)
        cells["E2"] = [hit.parent_id for hit in
                       rrf_fuse_routes(four_routes, rrf_k=rrf_k, top_k=k)]
    return cells


def _cell_latency(result, cell: str) -> float:
    debug = getattr(result, "debug", None) or {}
    per_route = debug.get("latency_ms") or {}
    if cell == "A":
        return float(per_route.get("dense_zh", 0.0))
    if cell == "B":
        return float(per_route.get("dense_en", 0.0))
    if cell == "C":
        return float(per_route.get("dense_zh", 0.0)) + float(per_route.get("dense_en", 0.0))
    if cell in ("D", "D2"):
        total = sum(float(per_route.get(name, 0.0))
                    for name in ("dense_zh", "dense_en", "lexical_en", "lexical_zh"))
        return total
    return float(debug.get("total_latency_ms") or
                 sum(float(v) for v in per_route.values()))


def run_matrix(
    questions: Sequence[Mapping[str, Any]],
    retriever,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    cells: str = "ABCDE",
    n_boot: int = 1000,
    seed: int = 0,
    translation_ms_per_query: float | None = DEFAULT_TRANSLATION_MS,
    translation_source: str = DEFAULT_TRANSLATION_SOURCE,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """跑一遍矩阵。**每题只检索一次**，五格共用。"""
    wanted = parse_cells(cells) if isinstance(cells, str) else list(cells)
    max_k = max(ks)
    judged = [q for q in questions if _gold_ids(q)]
    if not judged:
        raise ValueError("没有任何题带 gold_points.parent_ids，无法评估")

    per_cell: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {f"{k}": {"recall": [], "hit": [], "mrr": [], "ndcg": [],
                        "size": [], "latency": []} for k in ks}
        for name in wanted
    }

    started = time.perf_counter()
    for index, question in enumerate(judged, start=1):
        gold = _gold_ids(question)
        result = retriever.retrieve(
            question["question"],
            query_en=question.get("query_en"),
            k=max_k,
        )
        for k in ks:
            rankings = cell_rankings(result, k, rrf_k=RRF_K)
            for name in wanted:
                if name not in rankings:
                    continue
                ranked = rankings[name]
                latency = _cell_latency(result, name)
                bucket = per_cell[name][f"{k}"]
                pool_cell = name in ("C", "D", "D2")
                # 池子按「池内覆盖率」算（不再二次截断）；列表按 top-k 算
                limit = len(ranked) if pool_cell else k
                bucket["recall"].append(recall_fraction(ranked, gold, limit))
                bucket["hit"].append(1.0 if set(ranked[:limit]) & gold else 0.0)
                bucket["mrr"].append(mrr_at_k(ranked, gold, limit))
                bucket["ndcg"].append(ndcg_at_k(ranked, gold, limit))
                bucket["size"].append(float(len(ranked[:limit])))
                bucket["latency"].append(latency)
        if log and index % 20 == 0:
            log(f"[ablation] {index}/{len(judged)}")

    summary: dict[str, Any] = {}
    for name in wanted:
        cell_summary: dict[str, Any] = {"recall": {}, "hit": {}, "mrr": {}, "ndcg": {},
                                        "list_size": {}, "pool_size": {}, "latency_ms": {},
                                        "config": None}
        for k in ks:
            bucket = per_cell[name][f"{k}"]
            cell_summary["recall"][str(k)] = bootstrap_ci(bucket["recall"], n_boot=n_boot, seed=seed)
            cell_summary["hit"][str(k)] = {"mean": round(_mean(bucket["hit"]), 4)}
            cell_summary["mrr"][str(k)] = {"mean": round(_mean(bucket["mrr"]), 4)}
            cell_summary["ndcg"][str(k)] = {"mean": round(_mean(bucket["ndcg"]), 4)}
            size_key = "pool_size" if name in ("C", "D", "D2") else "list_size"
            cell_summary[size_key][str(k)] = {"mean": round(_mean(bucket["size"]), 2)}
            cell_summary["latency_ms"] = {
                "p50": round(percentile(bucket["latency"], 50), 1),
                "p95": round(percentile(bucket["latency"], 95), 1),
            }
        summary[name] = cell_summary

    # 配对检验：用 hit@k（二值）做 McNemar，跨方案两两比
    pairs = []
    for k in ks:
        for i, a in enumerate(wanted):
            for b in wanted[i + 1:]:
                hits_a = per_cell[a][f"{k}"]["hit"]
                hits_b = per_cell[b][f"{k}"]["hit"]
                a_only = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
                b_only = sum(1 for x, y in zip(hits_a, hits_b) if y and not x)
                test = mcnemar_exact(a_only, b_only)
                pairs.append({"a": a, "b": b, "k": k, "a_only": a_only, "b_only": b_only,
                              "p_value": round(test["p_value"], 4)})

    elapsed = time.perf_counter() - started
    return {
        "ks": list(ks),
        "cells": summary,
        "pairs": pairs,
        "n_questions": len(judged),
        "n_traces": len(judged),
        "translation_ms_per_query": translation_ms_per_query,
        "translation_source": translation_source,
        "elapsed_s": round(elapsed, 1),
    }


# ----------------------------------------------------------------- 报告


def render_report(payload: Mapping[str, Any]) -> str:
    ks = payload.get("ks") or [5, 10]
    cells = payload.get("cells") or {}
    names = [name for name in ("A", "B", "C", "D", "D2", "E", "E2") if name in cells]
    lines = [
        "# D28 消融矩阵 A–E",
        "",
        f"- 题数：**{payload.get('n_questions')}**（只统计带 `gold_points` 的题）",
        f"- 数据集：`{payload.get('dataset_sha256', 'n/a')}`",
        f"- 指标：Recall@{'/'.join(str(k) for k in ks)} + hit + MRR@10 + nDCG@10 + P50/P95",
        f"- 翻译耗时：**{payload.get('translation_ms_per_query')}** ms/query（{payload.get('translation_source')}）",
        "  —— 矩阵本身用人工 `query_en`，**不含翻译开销**；线上若走 D23 翻译，要把这一项加到 B/C/D/E 的延迟上",
        "",
    ]
    for k in ks:
        lines += [
            f"## Recall@{k}（父块粒度，含 95% CI 半宽）",
            "",
            "| 方案 | 配置 | Recall | ±半宽 | hit | MRR@" + str(k) + " | nDCG@" + str(k) + " | 列表/池大小 | P50 | P95 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        configs = {
            "A": "中文 Query + Dense", "B": "英文 Query + Dense",
            "C": "中英 Dense 并集", "D": "C + 英文 BM25",
            "D2": "D + 中文 BM25（B05 对照）", "E": "三路 RRF",
            "E2": "四路 RRF（+中文 BM25）",
        }
        for name in names:
            cell = cells[name]
            recall = cell["recall"].get(str(k), {})
            mrr = cell["mrr"].get(str(k), {}).get("mean", 0.0)
            ndcg = cell["ndcg"].get(str(k), {}).get("mean", 0.0)
            size = (cell.get("pool_size") or cell.get("list_size") or {}).get(str(k), {}).get("mean", 0.0)
            latency = cell.get("latency_ms") or {}
            lines.append(
                f"| {name} | {configs.get(name, '')} | {recall.get('mean', 0) * 100:.1f}% | "
                f"±{recall.get('half_width', 0) * 100:.1f} | "
                f"{(cell.get('hit') or {}).get(str(k), {}).get('mean', 0) * 100:.1f}% | "
                f"{mrr:.3f} | {ndcg:.3f} | {size:.1f} | {latency.get('p50', 0):.0f} ms | "
                f"{latency.get('p95', 0):.0f} ms |"
            )
        lines.append("")

    pairs = payload.get("pairs") or []
    if pairs:
        lines += ["## 配对检验（McNemar，hit@k）", "",
                  "| 对照 | k | A 独中 | B 独中 | p | 是否显著 |", "|---|---|---|---|---|---|"]
        for pair in pairs:
            significant = "**显著**" if pair["p_value"] < 0.05 else "不显著"
            lines.append(
                f"| {pair['a']} vs {pair['b']} | {pair['k']} | {pair['a_only']} | "
                f"{pair['b_only']} | {pair['p_value']:.4f} | {significant} |"
            )
        lines += ["", "> 只统计「一个中一个不中」的配对；双方都中/都不中的题不参与，"
                      "所以比两个独立区间更敏感。", ""]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- CLI


def _questions_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D28 消融矩阵 A–E")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="跑矩阵并出报告")
    run.add_argument("--questions", default="experiments/questions.json")
    run.add_argument("--ks", default="5,10")
    run.add_argument("--cells", default="ABCDE")
    run.add_argument("--limit", type=int, default=0, help="只用前 N 道可判定题（调试用）")
    run.add_argument("--no-warmup", action="store_true",
                     help="不做批量预编码（延迟会含冷启动，只有排查时才该用）")
    run.add_argument("--include-zh-lexical", action="store_true",
                     help="额外装配中文 BM25 路（B05 对照，会多出 D2 格）")
    run.add_argument("--collection", default=None)
    run.add_argument("--translation-ms", type=float, default=DEFAULT_TRANSLATION_MS)
    run.add_argument("--translation-source", default=DEFAULT_TRANSLATION_SOURCE)
    run.add_argument("--out", default="experiments/reports/d28_ablation.md")
    run.add_argument("--json", default="experiments/reports/d28_ablation.json")
    args = parser.parse_args(argv)

    from .rag.retrieval.store_wiring import build_default_retriever

    questions_path = Path(args.questions)
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if args.limit:
        seen = 0
        trimmed = []
        for question in questions:
            if (question.get("gold_points") or {}).get("parent_ids"):
                seen += 1
                if seen > args.limit:
                    break
            trimmed.append(question)
        questions = trimmed

    ks = tuple(int(x) for x in args.ks.replace(" ", "").split(",") if x)
    cells = args.cells
    if args.include_zh_lexical:
        existing = parse_cells(cells)
        for extra in ("D2", "E2"):
            if extra not in existing:
                cells += extra

    warmup: list[str] = []
    if not args.no_warmup:
        for question in questions:
            if (question.get("gold_points") or {}).get("parent_ids"):
                warmup.append(question["question"])
                if question.get("query_en"):
                    warmup.append(question["query_en"])

    retriever = build_default_retriever(
        collection=args.collection, pool_factor=4, include_zh_lexical=args.include_zh_lexical,
        warmup_texts=warmup,
    )
    payload = run_matrix(
        questions, retriever, ks=ks, cells=cells, log=print,
        translation_ms_per_query=args.translation_ms,
        translation_source=args.translation_source,
    )
    payload["dataset_sha256"] = _questions_sha256(questions_path)
    payload["routes"] = retriever.route_names

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(payload), encoding="utf-8")
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_report(payload))
    print(f"报告：{out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
