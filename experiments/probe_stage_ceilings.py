"""逐段拆账：中→英 / 英→中 的检索上限到底卡在哪一段。

链路的六段（每一段都能「完美」一次，看最终 hit@10 能到多少）：

  1. **翻译**：中文问 → 英文 query（C0c 已测：field vs translate 只差 1 题）
  2. **三路召回**：dense_zh(原句) / dense_en(英文 query) / lexical_en(BM25)
     每路 = Qdrant ANN 取 k*4 个子块 → collapse_to_parents 压到 k 个父块
  3. **子块→父块聚合**：同上，父块取「首次出现、最高分」
  4. **RRF 融合**（k=60）→ pool=40
  5. **窗口选择 + 重排**：256 token 窗口（按 query 词面重叠挑）→ cross-encoder
  6. **组装预算**：top-k 进上下文

测法：
  * route@40   生产口径（每路 40 个父块，实际搜 160 个子块）
  * route@200  深挖（每路 200 个父块，搜 800 个子块）—— 用来区分「没搜到」与「被排序挤掉」
  * fused@10   关掉重排时的融合 top-10
  * window     重排实际看到的那个 256-token 窗口里，到底有没有 gold 句子
  * final      生产口径的最终名次（基线 rows）
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 不许写本机绝对路径（身份守卫会红）
sys.path.insert(0, str(ROOT))

from scripts.eval_bench import (BENCH_STORE, QUESTIONS, _build_retriever, build_translate_fn,
                                load_parents, load_questions)
from src.rag.ask.pipeline import _parent_text
from src.rag.ingestion.tokenizer import TokenCounter
from src.rag.retrieval.rerank import best_window

CJK = re.compile(r"[\u4e00-\u9fff]")
POOL = 40
DEEP = 200

qs = load_questions(QUESTIONS)
parents = load_parents(BENCH_STORE)
counter = TokenCounter()
retriever = _build_retriever("bench_k8s_v1", BENCH_STORE, [], lambda _m: None)
translate = build_translate_fn()
baseline = {json.loads(line)["qid"]: json.loads(line)
            for line in (ROOT / "docs/baseline-k8s-v1.rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()}


def rank_in(ids, gold):
    for position, pid in enumerate(ids, start=1):
        if pid in gold:
            return position
    return None


def qlang(row):
    return "zh" if CJK.search(row["question"]) else "en"


rows = []
for question in qs:
    if question["type"] == "negative":
        continue
    gold = set(question["gold_chunks"])
    english = translate(question["question"])
    deep = retriever.retrieve(question["question"], query_en=english, k=DEEP, fuse=False)
    prod = retriever.retrieve(question["question"], query_en=english, k=POOL, fuse=False)
    record = {
        "qid": question["qid"],
        "qlang": qlang(question),
        "elang": "zh" if question["expected_evidence_lang"] == "zh-cn" else "en",
        "type": question["type"],
        "route": {}, "route_deep": {},
        "fused10": None, "final": baseline[question["qid"]]["first_hit_rank"],
    }
    for name in deep.routes:
        record["route"][name] = rank_in(list(prod.routes[name].ids), gold)
        record["route_deep"][name] = rank_in(list(deep.routes[name].ids), gold)

    fused = retriever.retrieve(question["question"], query_en=english, k=POOL, fuse=True).fused
    fused_ids = [hit.parent_id for hit in fused]
    record["fused10"] = rank_in(fused_ids[:10], gold)
    record["fused40"] = rank_in(fused_ids[:POOL], gold)

    # 窗口：生产口径（rerank_query=question）下，gold 所在父块被切出的那个窗口里有 gold 句子吗
    window_has_gold = None
    gold_text = " ".join(question.get("gold_spans") or [])
    for pid in gold:
        if pid not in fused_ids[:POOL]:
            continue
        text = _parent_text(parents.get(pid))
        window = best_window(text, question["question"], max_tokens=256, counter=counter)
        # 用 gold span 的第一句做判定（整段可能比窗口长，但答案句必须在窗口里）
        sentences = [s.strip() for s in re.split(r"(?<=[。！？；.!?])\s*", gold_text) if len(s.strip()) > 20]
        probe = sentences[0] if sentences else gold_text
        window_has_gold = bool(probe) and probe[:60] in window
        record["window_probe"] = probe[:60]
        break
    record["window_has_gold"] = window_has_gold
    rows.append(record)
    print(f"{question['qid']:12} {record['qlang']}→{record['elang']} "
          f"route@40={record['route']} route@200={record['route_deep']} "
          f"fused10={record['fused10']} final={record['final']} win={window_has_gold}", flush=True)

# ---------------- 汇总 ----------------
def cell_stats(picked, label):
    n = len(picked)
    def rate(fn):
        return round(sum(1 for r in picked if fn(r)) / n, 4)
    def any_route(slot, limit):
        return rate(lambda r: any(v is not None and v <= limit for v in r[slot].values()))
    out = {
        "label": label, "n": n,
        "final_hit@10": rate(lambda r: r["final"] is not None and r["final"] <= 10),
        "in_pool_route@40": any_route("route", POOL),
        "in_deep_route@200": any_route("route_deep", DEEP),
        "fused@10": rate(lambda r: r["fused10"] is not None),
        "fused@40": rate(lambda r: r["fused40"] is not None),
        "window_has_gold": round(
            sum(1 for r in picked if r["window_has_gold"]) / max(1, sum(1 for r in picked if r["window_has_gold"] is not None)), 4),
        "window_checked": sum(1 for r in picked if r["window_has_gold"] is not None),
        "per_route@40": {name: round(sum(1 for r in picked if r["route"].get(name) is not None) / n, 4)
                         for name in picked[0]["route"]},
        "per_route@200": {name: round(sum(1 for r in picked if r["route_deep"].get(name) is not None) / n, 4)
                          for name in picked[0]["route_deep"]},
    }
    return out


cells = {}
for ql, el, label in (("en", "en", "英→英"), ("zh", "zh", "中→中"),
                      ("en", "zh", "英→中"), ("zh", "en", "中→英")):
    picked = [r for r in rows if r["qlang"] == ql and r["elang"] == el]
    cells[label] = cell_stats(picked, label)

payload = {"k_pool": POOL, "k_deep": DEEP, "cells": cells, "rows": rows}
dest = ROOT / "docs/evidence/retrieval-stage-ceilings-k8s-v1.json"
dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

for label, stats in cells.items():
    print(f"\n== {label}（n={stats['n']}）")
    for key in ("final_hit@10", "in_pool_route@40", "in_deep_route@200", "fused@10", "fused@40",
                "window_has_gold"):
        print(f"   {key:20} {stats[key]}")
    print(f"   per_route@40    {stats['per_route@40']}")
    print(f"   per_route@200   {stats['per_route@200']}")
print("\n证据 →", dest.relative_to(ROOT))
