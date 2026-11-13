"""B05 机理探针：第四路 `lexical_zh`（jieba BM25）到底是在**补召回**还是在**改权重**。

区别很关键，因为端到端 +2.3 点 hit 可能来自两种完全不同的机制：

  A. 补召回：某些题的 gold 父块只有中文 BM25 能捞进 pool=40，其它三路都漏了
  B. 改权重：raging 全在融合层 —— 第四路和 `dense_zh` 吃同一段中文 query，
     等于给「按中文原文检索」这件事投了两票，把候选顺序搅了一遍

做法：同一批正样本、同一 retriever，只调 `retriever.retrieve()`（不带重排 / 不带窗口），
逐题记四路各自 top-40 里有没有 gold、gold 的名次，以及它是否**只**出现在第四路里。

跑法（仓库根目录，需 Qdrant + Ollama）：
    python experiments/probe_b05_route_contribution.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS = ROOT / "datasets" / "k8s-zh-en-v1" / "questions.jsonl"
OUT = ROOT / "docs" / "evidence" / "b05-lexical-zh-routes-k8s-v1.json"
COLLECTION = "bench_k8s_v1"
STORE = ROOT / "storage" / "bench" / "documents.sqlite3"
POOL = 40
ROUTES = ("dense_zh", "dense_en", "lexical_en", "lexical_zh")


def main() -> int:
    from scripts.eval_bench import build_translate_fn
    from src.rag.retrieval.store_wiring import build_default_retriever

    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    positives = [q for q in questions if q.get("gold_chunks")]

    translator = build_translate_fn()
    # 只热身一次：dense 路的 query 编码走 SQLite 缓存，第二轮起就快了
    retriever = build_default_retriever(
        collection=COLLECTION, store_path=STORE, routes=ROUTES,
        warmup_texts=[q["question"] for q in positives[:5]], log=print,
    )

    rows: list[dict] = []
    route_latency: dict[str, list[float]] = {name: [] for name in ROUTES}
    started = time.perf_counter()
    for index, question in enumerate(positives, start=1):
        gold = set(question["gold_chunks"])
        english = translator(question["question"])
        result = retriever.retrieve(question["question"], k=POOL, query_en=english)
        per_route: dict[str, dict] = {}
        for name, route in result.routes.items():
            route_latency[name].append(route.latency_ms)
            ids = list(route.ids)
            hits = [rank for rank, pid in enumerate(ids, start=1) if pid in gold]
            per_route[name] = {
                "enabled": route.enabled,
                "gold_rank": hits[0] if hits else None,
                "candidates": len(ids),
            }
        with_gold = [name for name, info in per_route.items() if info["gold_rank"]]
        rows.append({
            "qid": question["qid"],
            "type": question["type"],
            "expected_evidence_lang": question.get("expected_evidence_lang"),
            "per_route": per_route,
            "routes_with_gold": with_gold,
            "gold_only_in_lexical_zh": with_gold == ["lexical_zh"],
            "gold_without_lexical_zh": [name for name in with_gold if name != "lexical_zh"],
        })
        if index % 25 == 0:
            print(f"  {index}/{len(positives)} 题 / {time.perf_counter() - started:.0f} s")

    n = len(rows)
    only_zh = [r for r in rows if r["gold_only_in_lexical_zh"]]
    cover = {name: sum(1 for r in rows if r["per_route"][name]["gold_rank"]) for name in ROUTES}
    payload = {
        "note": "B05 机理：四路各自的 gold 覆盖率（top-40，不带重排/窗口），"
                "用来区分「补召回」与「只是改融合权重」",
        "params": {"collection": COLLECTION, "pool": POOL, "routes": list(ROUTES),
                   "positives": n, "query_en_mode": "translate"},
        "caveat": "探针直接调 retriever.retrieve()，与 eval_bench 的 retrieve_with_meta "
                  "不是同一条链路（少融合后的配额与重排）；只看路由级覆盖，不当端到端指标用",
        "coverage": {
            "per_route": cover,
            "per_route_ratio": {name: round(value / n, 4) for name, value in cover.items()},
            "at_least_one_route": sum(1 for r in rows if r["routes_with_gold"]),
            "gold_only_in_lexical_zh": len(only_zh),
            "gold_only_in_lexical_zh_qids": [r["qid"] for r in only_zh],
            "lexical_zh_displaces_nothing": n - sum(
                1 for r in rows if "lexical_zh" in r["routes_with_gold"]),
        },
        "route_latency_ms": {
            name: {"p50": round(statistics.median(values), 2),
                   "p90": round(sorted(values)[int(0.9 * len(values)) - 1], 2)}
            for name, values in route_latency.items() if values
        },
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n正样本 {n} 题；各路 top-{POOL} 里含 gold 的题数：")
    for name, value in cover.items():
        print(f"  {name:12} {value:3} / {n}  ({value / n:.1%})")
    print(f"  gold **只**在 lexical_zh 里出现：{len(only_zh)} 题 "
          f"{[r['qid'] for r in only_zh]}")
    print("路由级耗时（p50 / p90 ms）：")
    for name, stats in payload["route_latency_ms"].items():
        print(f"  {name:12} {stats['p50']:8.2f} / {stats['p90']:8.2f}")
    print(f"证据：{OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
