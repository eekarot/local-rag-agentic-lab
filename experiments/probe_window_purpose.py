"""窗口到底在做什么：三种窗口策略的对照（挑窗口的目的量化）。

`rerank` 只吃 256 token，候选块 p50 = 322 token、**63% 超过 256**（`docs/evidence` 里另有测算），
所以「看哪一段」是必答题，不是可选优化。三种答法：

  A `overlap`（生产）  按与 query 的词面重叠挑窗口（`best_window`）
  B `head`             直接取块首 256 token（等价于「不挑」）
  C `oracle`           gold 块换成「以答案句为中心」的窗口（上限参考）

同一批候选、同一个 cross-encoder、同一套题；A 必须与基线逐题一致（自检）。
另报一个诊断量：**窗口里到底有没有答案句**（A/B 各算一遍）。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_bench import (BENCH_STORE, QUESTIONS, build_translate_fn, load_parents,
                                load_questions)
from scripts.eval_bench import _build_retriever
from src.rag.ask.pipeline import _parent_text, retrieve_with_meta
from src.rag.ingestion.tokenizer import TokenCounter
from src.rag.retrieval.rerank import (CrossEncoderReranker, RerankConfig, RerankHit, best_window,
                                      rerank_hits)

CJK = re.compile(r"[\u4e00-\u9fff]")
POOL = 40
MODES = ("overlap", "head", "oracle")

questions = load_questions(QUESTIONS)
parents = load_parents(BENCH_STORE)
counter = TokenCounter()
reranker = CrossEncoderReranker(config=RerankConfig())
retriever = _build_retriever("bench_k8s_v1", BENCH_STORE, [], lambda _m: None)
translate = build_translate_fn()
baseline = {json.loads(line)["qid"]: json.loads(line)
            for line in (ROOT / "docs/baseline-k8s-v1.rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()}


def cell_of(question, row):
    zh_q = bool(CJK.search(question))
    zh_e = row.get("expected_evidence_lang") == "zh-cn"
    return ("中→中" if zh_q and zh_e else "中→英" if zh_q else "英→中" if zh_e else "英→英")


def head_window(text: str, max_tokens: int) -> str:
    """不挑：从块首开始攒到 256 token。"""
    out, used = [], 0
    for unit in re.split(r"(?<=[。！？；.!?])\s*", text):
        if not unit:
            continue
        n = counter.count(unit)
        if used + n > max_tokens:
            break
        out.append(unit)
        used += n
    return "".join(out) or text[: max_tokens * 2]


def oracle_window(text: str, span: str, max_tokens: int) -> str:
    """以答案句为中心：往前留一半预算，往后补满。"""
    start = text.find(span)
    if start < 0:
        return best_window(text, span, max_tokens=max_tokens, counter=counter)
    budget = max_tokens // 2
    kept, used = [], 0
    for unit in reversed(re.split(r"(?<=[。！？；.!?])\s*", text[:start])):
        if not unit:
            continue
        n = counter.count(unit)
        if used + n > budget:
            break
        kept.insert(0, unit)
        used += n
    prefix = "".join(kept)
    body = text[len(prefix):]
    out, used2 = [], 0
    for unit in re.split(r"(?<=[。！？；.!?])\s*", body):
        if not unit:
            continue
        n = counter.count(unit)
        if used2 + n > max_tokens - used:
            break
        out.append(unit)
        used2 += n
    return prefix + "".join(out)


rows = []
for question in questions:
    if question["type"] == "negative":
        continue
    gold = set(question["gold_chunks"])
    english = translate(question["question"])
    fused = retriever.retrieve(question["question"], query_en=english, k=POOL, fuse=True).fused
    hits = [RerankHit(chunk_id=hit.parent_id, parent_id=hit.parent_id,
                      text=_parent_text(parents.get(hit.parent_id)), score=hit.score,
                      rank=rank) for rank, hit in enumerate(fused, start=1)]
    span = (question.get("gold_spans") or [""])[0]

    record = {"qid": question["qid"], "cell": cell_of(question["question"], question)}
    for mode in MODES:
        for hit in hits:
            if mode == "overlap":
                hit.fields["window"] = best_window(hit.text, question["question"],
                                                   max_tokens=reranker.config.max_length,
                                                   counter=counter)
            elif mode == "head":
                hit.fields["window"] = head_window(hit.text, reranker.config.max_length)
            else:
                hit.fields["window"] = (oracle_window(hit.text, span, reranker.config.max_length)
                                        if hit.parent_id in gold and span else
                                        best_window(hit.text, question["question"],
                                                    max_tokens=reranker.config.max_length,
                                                    counter=counter))
        ranked = rerank_hits(question["question"], list(hits), encoder=reranker.encoder,
                             config=reranker.config, top_n=POOL, stats=reranker.stats,
                             text_getter=lambda h: h.fields["window"])
        order = [hit.parent_id for hit in ranked]
        position = next((i + 1 for i, pid in enumerate(order) if pid in gold), None)
        record[mode] = position
        if mode in ("overlap", "head") and span:
            gold_hit = next((h for h in hits if h.parent_id in gold), None)
            record[f"window_has_gold_{mode}"] = bool(gold_hit) and span[:60] in gold_hit.fields["window"]
    rows.append(record)
    same = record["overlap"] == (baseline[question["qid"]]["first_hit_rank"])
    print(f"{question['qid']:12} {record['cell']:6} overlap={record['overlap']} head={record['head']} "
          f"oracle={record['oracle']} 自检={same}", flush=True)

# ---- 汇总 ----
def rate(picked, key):
    return round(sum(1 for r in picked if r[key] and r[key] <= 10) / len(picked), 4)


def mrr(picked, key):
    return round(sum((1.0 / r[key]) if r[key] else 0.0 for r in picked) / len(picked), 4)


summary = {}
for cell in ("中→英", "英→中", "中→中", "英→英", "整体"):
    picked = rows if cell == "整体" else [r for r in rows if r["cell"] == cell]
    summary[cell] = {"n": len(picked), **{mode: {"hit@10": rate(picked, mode), "mrr": mrr(picked, mode)}
                                          for mode in MODES},
                     "window_has_gold": {
                         "overlap": round(sum(1 for r in picked if r.get("window_has_gold_overlap"))
                                          / max(1, sum(1 for r in picked if "window_has_gold_overlap" in r)), 4),
                         "head": round(sum(1 for r in picked if r.get("window_has_gold_head"))
                                       / max(1, sum(1 for r in picked if "window_has_gold_head" in r)), 4)}}

mismatch = [r["qid"] for r in rows if r["overlap"] != (baseline[r["qid"]]["first_hit_rank"])]
dest = ROOT / "docs/evidence/window-strategy-k8s-v1.json"
dest.write_text(json.dumps({"note": "三种窗口策略对照（A 生产 overlap / B 块首 / C 以答案句为中心）",
                            "self_check_mismatch": mismatch[:20],
                            "self_check_mismatch_count": len(mismatch),
                            "summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
               encoding="utf-8")

print(f"\n自检：与基线逐题不一致 {len(mismatch)} 题")
print(f"{'格':<8}{'n':>4}{'overlap hit/MRR':>22}{'head hit/MRR':>20}{'oracle hit/MRR':>22}"
      f"{'窗口含答案(A/B)':>18}")
for cell, stats in summary.items():
    print(f"{cell:<8}{stats['n']:>4}"
          f"{stats['overlap']['hit@10']:>12.4f}/{stats['overlap']['mrr']:.4f}"
          f"{stats['head']['hit@10']:>12.4f}/{stats['head']['mrr']:.4f}"
          f"{stats['oracle']['hit@10']:>12.4f}/{stats['oracle']['mrr']:.4f}"
          f"{stats['window_has_gold']['overlap']:>10.2f}/{stats['window_has_gold']['head']:.2f}")
print("\n证据 →", dest.relative_to(ROOT))
