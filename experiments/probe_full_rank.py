"""全量索引上的 gold 排名对比：同一批题、同一个 81,027 条语料，只换 embedding 模型。

用法：
    OLLAMA_EMBED_MODEL=nomic-embed-text python -m experiments.probe_full_rank --matrix <旧矩阵>
    OLLAMA_EMBED_MODEL=bge-m3           python -m experiments.probe_full_rank

同时给两个口径：
  精确 gold —— gold_chunks 里的 id 本身
  邻近族   —— gold 所在文档里 chunk_id 相差 ≤2 的块（CHUNK_OVERLAP=80 导致同一句常落在相邻块）
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.config import OLLAMA_EMBED_MODEL, ROOT_DIR
from src.ollama_client import embed_texts
from src.retrieve import load_index


def load_matrix(path, records):
    mat = np.load(path, mmap_mode="r")
    if mat.shape[0] != len(records):
        raise SystemExit(f"矩阵行数 {mat.shape[0]} 与记录数 {len(records)} 不匹配")
    mat = np.array(mat, dtype=np.float32)  # 拷贝一份，备份矩阵是只读 mmap
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(Path(ROOT_DIR) / "storage" / "embeddings.npy"))
    args = ap.parse_args()

    index = load_index()
    records = index["records"]
    mat = load_matrix(args.matrix, records)
    qs = json.load(open(Path(ROOT_DIR) / "experiments" / "questions.json", encoding="utf-8"))
    positives = [q for q in (qs["questions"] if isinstance(qs, dict) else qs) if q["type"] != "negative"]
    row_of = {r["id"]: i for i, r in enumerate(records)}

    qvec = np.array(embed_texts([q["question"] for q in positives]), dtype=np.float32)
    qvec /= np.linalg.norm(qvec, axis=1, keepdims=True)

    exact, family = [], []
    for i, q in enumerate(positives):
        sims = mat @ qvec[i]
        gold_rows = [row_of[c["id"]] for c in q["gold_chunks"]]
        best = sims[gold_rows].max()
        exact.append(int((sims > best).sum()) + 1)

        # 邻近族：同文档、chunk 序号相差 ≤2
        fam_rows = []
        for c in q["gold_chunks"]:
            base, _, num = c["id"].rpartition("::chunk-")
            for delta in (-2, -1, 0, 1, 2):
                cand = f"{base}::chunk-{int(num) + delta:04d}"
                if cand in row_of:
                    fam_rows.append(row_of[cand])
        family.append(int((sims > sims[fam_rows].max()).sum()) + 1)

    out = {"model": OLLAMA_EMBED_MODEL, "matrix": str(args.matrix), "by_corpus": {}}
    print(f"模型={OLLAMA_EMBED_MODEL}  矩阵={Path(args.matrix).name}  {mat.shape}")
    for lang, label in (("zh", "中文问中文"), ("en", "中文问英文")):
        idx = [i for i, q in enumerate(positives) if q.get("corpus_lang") == lang]
        e = np.array([exact[i] for i in idx])
        fam = np.array([family[i] for i in idx])
        block = {
            "n": len(idx),
            "exact": {"p50": float(np.median(e)), "p90": float(np.percentile(e, 90)),
                      "top4": float(np.mean(e <= 4)), "top10": float(np.mean(e <= 10))},
            "family": {"p50": float(np.median(fam)), "p90": float(np.percentile(fam, 90)),
                       "top4": float(np.mean(fam <= 4)), "top10": float(np.mean(fam <= 10))},
        }
        out["by_corpus"][lang] = block
        print(f"\n{label}（n={len(idx)}）")
        print(f"  精确 gold：排名 p50={block['exact']['p50']:.0f} p90={block['exact']['p90']:.0f} "
              f"top-4={block['exact']['top4']:.0%} top-10={block['exact']['top10']:.0%}")
        print(f"  邻近族   ：排名 p50={block['family']['p50']:.0f} p90={block['family']['p90']:.0f} "
              f"top-4={block['family']['top4']:.0%} top-10={block['family']['top10']:.0%}")

    out_dir = Path(ROOT_DIR) / "experiments" / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"full-rank-{OLLAMA_EMBED_MODEL.replace(':', '-')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
