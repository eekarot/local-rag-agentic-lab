"""逐路消融：三路等权进 RRF，但它们对跨语言题的能力差得很远。

从阶段账（`docs/evidence/retrieval-stage-ceilings-k8s-v1.json`）里看到的中位名次：

    中→英：dense_en=4  lexical_en=2  dense_zh=11
    英→中：dense_zh=5  dense_en=5  lexical_en=27

`lexical_en` 是「英文 BM25 over 父块正文」。对英→中（英文问、中文块）它的中位名次是 27 ——
也就是说这一路基本没在干活，但它和另外两路**等权**进 RRF（每路 1/(60+rank)，不看路的质量）。
本探针就是把这句判断变成数字：关掉某一路，融合 + 重排后的 hit@10 会怎么变。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 不许写本机绝对路径（身份守卫会红）
sys.path.insert(0, str(ROOT))

from scripts.eval_bench import (BENCH_STORE, QUESTIONS, _build_retriever, build_translate_fn,
                                load_parents, load_questions)
from src.rag.ask.pipeline import _parent_text
from src.rag.ingestion.tokenizer import TokenCounter
from src.rag.retrieval.fusion import RRF_K
from src.rag.retrieval.rerank import (CrossEncoderReranker, RerankConfig, RerankHit, best_window,
                                      rerank_hits)

CJK = re.compile(r"[\u4e00-\u9fff]")
POOL = 40
qs = load_questions(QUESTIONS)
parents = load_parents(BENCH_STORE)
counter = TokenCounter()
reranker = CrossEncoderReranker(config=RerankConfig())
retriever = _build_retriever("bench_k8s_v1", BENCH_STORE, [], lambda _m: None)
translate = build_translate_fn()

VARIANTS = {
    "三路（生产）": (),
    "去掉 dense_zh": ("dense_zh",),
    "去掉 dense_en": ("dense_en",),
    "去掉 lexical_en": ("lexical_en",),
}

targets = [q for q in qs if q["type"] != "negative"
           and ((CJK.search(q["question"]) and q["expected_evidence_lang"] == "en")
                or (not CJK.search(q["question"]) and q["expected_evidence_lang"] == "zh-cn"))]

results = []
for q in targets:
    gold = set(q["gold_chunks"])
    english = translate(q["question"])
    cell = "zh→en" if CJK.search(q["question"]) else "en→zh"
    row = {"qid": q["qid"], "cell": cell}
    for label, disable in VARIANTS.items():
        result = retriever.retrieve(q["question"], query_en=english, k=POOL, disable=disable)
        fused = result.fused
        hits = [RerankHit(chunk_id=hit.parent_id, parent_id=hit.parent_id,
                          text=_parent_text(parents.get(hit.parent_id)), score=hit.score, rank=rank)
                for rank, hit in enumerate(fused, start=1)]
        for hit in hits:
            hit.fields["window"] = best_window(hit.text, q["question"],
                                               max_tokens=reranker.config.max_length, counter=counter)
        ranked = rerank_hits(q["question"], hits, encoder=reranker.encoder,
                             config=reranker.config, top_n=POOL, stats=reranker.stats,
                             text_getter=lambda h: h.fields["window"])
        order = [hit.parent_id for hit in ranked]
        pos = [i + 1 for i, pid in enumerate(order) if pid in gold]
        row[label] = pos[0] if pos else None
    results.append(row)
    print(f"{q['qid']:12} {cell:6} " + "  ".join(f"{k[4:] if k != '三路（生产）' else '全部'}={row[k]}"
                                                for k in VARIANTS), flush=True)

summary = {}
for cell in ("zh→en", "en→zh"):
    picked = [r for r in results if r["cell"] == cell]
    summary[cell] = {label: round(sum(1 for r in picked if r[label] and r[label] <= 10) / len(picked), 4)
                     for label in VARIANTS}
    summary[cell]["n"] = len(picked)

dest = ROOT / "docs/evidence/route-ablation-cross-lingual-k8s-v1.json"
dest.write_text(json.dumps({"note": "跨语言格的逐路消融（关一路 → RRF → 重排 → hit@10）",
                            "rrf_k": RRF_K, "pool": POOL, "summary": summary, "rows": results},
                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("证据 →", dest.relative_to(ROOT))
