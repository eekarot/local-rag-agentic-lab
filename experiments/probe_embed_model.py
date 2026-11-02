"""探针：对比两个 embedding 模型的跨语言检索区分度（不重建索引，只测相似度分布）。

用法：
    OLLAMA_EMBED_MODEL=nomic-embed-text python -m experiments.probe_embed_model
    OLLAMA_EMBED_MODEL=bge-m3           python -m experiments.probe_embed_model

判据：中文 query 找英文 gold chunk 时，gold 在同等规模候选池里的排名与 z 分数。
z 越高 = 跨语言对齐越强。当前基线（nomic）：z≈+3，真实 81k 池子中位排名 158。
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from src.config import OLLAMA_EMBED_MODEL, ROOT_DIR
from src.ollama_client import embed_texts
from src.retrieve import _load_embeddings, load_index

POOL_EN = 3000
POOL_ZH = 500
BATCH = 64
OUT_DIR = Path(ROOT_DIR) / "experiments" / "probes"


def is_chinese(text, limit=400):
    return any("\u4e00" <= ch <= "\u9fff" for ch in text[:limit])


def embed_in_batches(texts, batch=BATCH):
    vectors, started = [], time.perf_counter()
    for start in range(0, len(texts), batch):
        vectors.extend(embed_texts(texts[start:start + batch]))
        print(f"  embed {min(start + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    elapsed = time.perf_counter() - started
    arr = np.asarray(vectors, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    print(f"  embed 完成：{len(texts)} 条 / {elapsed:.1f} s（{elapsed / len(texts) * 1000:.1f} ms/条）")
    return arr, elapsed


def summarize(cos_gold, pool_cos):
    ranks = [int((pool > g).sum()) + 1 for g, pool in zip(cos_gold, pool_cos)]
    zs = [(g - p.mean()) / p.std() for g, p in zip(cos_gold, pool_cos)]
    ranks, zs = np.array(ranks), np.array(zs)
    return {
        "n": len(ranks),
        "rank_p50": float(np.median(ranks)),
        "rank_p90": float(np.percentile(ranks, 90)),
        "top4": float(np.mean(ranks <= 4)),
        "top10": float(np.mean(ranks <= 10)),
        "z_p50": float(np.median(zs)),
        "cos_gold_p50": float(np.median(cos_gold)),
        "cos_pool_mean": float(np.mean([p.mean() for p in pool_cos])),
    }


def main():
    index = load_index()
    records = index["records"]
    print(f"模型={OLLAMA_EMBED_MODEL}（当前索引里存的是 {index.get("embedding_model")}）")

    raw = json.load(open(Path(ROOT_DIR) / "experiments" / "questions.json", encoding="utf-8"))
    questions = raw["questions"] if isinstance(raw, dict) else raw
    positives = [q for q in questions if q["type"] != "negative"]

    rng = np.random.default_rng(0)
    en_rows = [i for i, r in enumerate(records) if not is_chinese(r["text"])]
    zh_rows = [i for i, r in enumerate(records) if is_chinese(r["text"])]
    en_idx = rng.choice(en_rows, POOL_EN, replace=False)
    zh_idx = rng.choice(zh_rows, POOL_ZH, replace=False)
    print(f"候选池：英文 {len(en_idx)} 条（库里共 {len(en_rows)}）、中文 {len(zh_idx)} 条（共 {len(zh_rows)}）")

    text_of = {r["id"]: r["text"] for r in records}
    gold_texts, gold_slices, cursor = [], [], 0
    for q in positives:
        texts = [text_of[g["id"]] for g in q["gold_chunks"]]
        gold_texts.extend(texts)
        gold_slices.append(slice(cursor, cursor + len(texts)))
        cursor += len(texts)
    print(f"题目 {len(positives)} 道，gold chunk 文本 {len(gold_texts)} 段")

    probe_questions = ["今天中午吃什么比较好？", "量子计算的基本原理是什么？", "请帮我写一首关于春天的诗"]
    q_texts = [q["question"] for q in positives] + probe_questions
    pool_texts = [records[i]["text"] for i in en_idx] + [records[i]["text"] for i in zh_idx]

    print("\n[1/2] 向量化问题 + gold")
    qvec, t_q = embed_in_batches(q_texts)
    gvec, t_g = embed_in_batches(gold_texts)
    print(f"[2/2] 向量化候选池 {len(pool_texts)} 条")
    pvec, t_p = embed_in_batches(pool_texts)
    pvec_en, pvec_zh = pvec[:POOL_EN], pvec[POOL_EN:]

    result = {
        "model": OLLAMA_EMBED_MODEL,
        "index_model": index.get("embedding_model"),
        "pool_en": POOL_EN,
        "pool_zh": POOL_ZH,
        "ms_per_text": (t_q + t_g + t_p) / (len(q_texts) + len(gold_texts) + len(pool_texts)) * 1000,
        "by_corpus": {},
        "sanity": {},
    }

    for lang, label, pool in (("zh", "中文问中文", pvec_zh), ("en", "中文问英文", pvec_en)):
        rows = [(i, q) for i, q in enumerate(positives) if q.get("corpus_lang") == lang]
        cos_gold, pool_cos = [], []
        for i, q in rows:
            g = gvec[gold_slices[i]]
            cos_gold.append(float((g @ qvec[i]).max()))
            pool_cos.append(pool @ qvec[i])
        stats = summarize(cos_gold, pool_cos)
        result["by_corpus"][lang] = stats
        print(f"\n{label}（pool={POOL_ZH if lang == 'zh' else POOL_EN}）")
        print(f"  排名 p50={stats['rank_p50']:.0f}  p90={stats['rank_p90']:.0f}  "
              f"top-4={stats['top4']:.0%}  top-10={stats['top10']:.0%}")
        print(f"  z p50={stats['z_p50']:+.2f}  cos(gold)={stats['cos_gold_p50']:.3f}  "
              f"cos(池内均值)={stats['cos_pool_mean']:.3f}")

    print("\n无关问题对照（应与 gold 相似度很低）")
    for j, text in enumerate(probe_questions):
        vec = qvec[len(positives) + j]
        sims = [float((gvec[s] @ vec).max()) for s in gold_slices]
        result["sanity"][text] = {"gold_max": float(np.max(sims)), "pool_en_mean": float((pvec_en @ vec).mean())}
        print(f"  «{text}»  gold 最高 cos={np.max(sims):.3f}  池内均值={float((pvec_en @ vec).mean()):.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"embed-model-{OLLAMA_EMBED_MODEL.replace(':', '-')}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
