"""窗口是不是那一刀：把 gold 块的窗口换成「一定含答案句」的，看重排能不能救回来。

背景（`docs/evidence/retrieval-stage-ceilings-k8s-v1.json`）：
  英→中：融合 top-10 有 77.8%，最终只剩 55.6%；而且**只有 42.9% 的窗口里含 gold 句子**。
  中→英：融合 77.3% → 最终 72.7%，窗口 90.5% 含 gold 句。

三个条件，同一批候选、同一个 cross-encoder、同一套题（只跑两个跨语言格）：
  A 生产口径      ：窗口按 query 词面重叠挑（`best_window(question)`）
  B oracle 窗口   ：gold 候选的窗口改成「以答案句为中心」的 256 token，其余不变
  C 完美重排     ：直接把融合 top-10 当结果（= 重排不损失）

B 与 C 的区别：B 只修窗口（重排打分仍然要自己判断），C 连打分都当成完美。
如果 B ≈ C，说明**卡的就是窗口**；如果 B 仍明显低于 C，说明 cross-encoder 本身跨语言不行。
"""
import json
import sys
from pathlib import Path

ROOT = r"C:/Users/zcl/Desktop/project/local-rag-agentic-lab"
sys.path.insert(0, ROOT)

from scripts.eval_bench import (BENCH_STORE, QUESTIONS, _build_retriever, build_translate_fn,
                                load_parents, load_questions)
from src.rag.ask.pipeline import _parent_text, retrieve_with_meta
from src.rag.ingestion.tokenizer import TokenCounter
from src.rag.retrieval.rerank import CrossEncoderReranker, RerankConfig, best_window, rerank_hits

import re

CJK = re.compile(r"[\u4e00-\u9fff]")
POOL = 40
qs = load_questions(QUESTIONS)
parents = load_parents(BENCH_STORE)
counter = TokenCounter()
reranker = CrossEncoderReranker(config=RerankConfig())
retriever = _build_retriever("bench_k8s_v1", BENCH_STORE, [], lambda _m: None)
translate = build_translate_fn()


def gold_window(parent_text: str, span: str, max_tokens: int) -> str:
    """以答案句为中心截一个窗口（模拟「窗口选择永远正确」）。"""
    start = parent_text.find(span)
    if start < 0:
        return best_window(parent_text, span, max_tokens=max_tokens, counter=counter)
    head = parent_text[:start]
    # 往前留一半预算，往后补满
    budget = max_tokens // 2
    lines = head.splitlines(keepends=True)
    kept, used = [], 0
    for line in reversed(lines):
        n = counter.count(line)
        if used + n > budget:
            break
        kept.insert(0, line)
        used += n
    prefix = "".join(kept)
    rest_tokens = max_tokens - used
    body = parent_text[len(prefix):]
    out, used2 = [], 0
    for line in body.splitlines(keepends=True):
        n = counter.count(line)
        if used2 + n > rest_tokens:
            break
        out.append(line)
        used2 += n
    return prefix + "".join(out)


def rank_of(hits, gold):
    for position, hit in enumerate(hits, start=1):
        if hit.parent_id in gold:
            return position
    return None


targets = [q for q in qs if q["type"] != "negative"
           and ((CJK.search(q["question"]) and q["expected_evidence_lang"] == "en")
                or (not CJK.search(q["question"]) and q["expected_evidence_lang"] == "zh-cn"))]
print(f"跨语言题 {len(targets)} 道")

results = []
for q in targets:
    gold = set(q["gold_chunks"])
    english = translate(q["question"])
    fused = retrieve_with_meta(q["question"], retriever=retriever, parents=parents, counter=counter,
                               reranker=reranker, pool=POOL, rerank=False, query_en=english).hits
    label = ("zh→en" if CJK.search(q["question"]) else "en→zh")

    # A：生产口径
    prod = retrieve_with_meta(q["question"], retriever=retriever, parents=parents, counter=counter,
                              reranker=reranker, pool=POOL, rerank=True, query_en=english)
    a = rank_of(prod.hits, gold)

    # B：gold 候选换成以答案句为中心的窗口
    for hit in fused:
        hit.fields["window"] = best_window(hit.text, q["question"],
                                           max_tokens=reranker.config.max_length, counter=counter)
        if hit.parent_id in gold and q["gold_spans"]:
            hit.fields["window"] = gold_window(hit.text, q["gold_spans"][0],
                                               reranker.config.max_length)
    b_hits = rerank_hits(q["question"], list(fused), encoder=reranker.encoder,
                         config=reranker.config, top_n=POOL, stats=reranker.stats,
                         text_getter=lambda h: h.fields["window"])
    b = rank_of(b_hits, gold)

    # C：完美重排 = 融合 top-10
    c = rank_of(fused[:10], gold)
    results.append({"qid": q["qid"], "cell": label, "A": a, "B": b, "C": c})
    print(f"{q['qid']:12} {label:6} 生产={a}  oracle窗口={b}  完美重排={c}", flush=True)


def rate(picked, key):
    return round(sum(1 for r in picked if r[key] and r[key] <= 10) / len(picked), 4)


summary = {}
for cell in ("zh→en", "en→zh"):
    picked = [r for r in results if r["cell"] == cell]
    summary[cell] = {"n": len(picked), "A_production": rate(picked, "A"),
                     "B_oracle_window": rate(picked, "B"), "C_perfect_rerank": rate(picked, "C")}

dest = Path(ROOT) / "docs/evidence/rerank-window-ceiling-k8s-v1.json"
dest.write_text(json.dumps({"note": "跨语言格的窗口上限实验（A 生产 / B oracle 窗口 / C 完美重排）",
                            "summary": summary, "rows": results}, ensure_ascii=False, indent=2) + "\n",
               encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("证据 →", dest.relative_to(Path(ROOT)))
