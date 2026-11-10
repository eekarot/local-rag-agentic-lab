"""在 **SciFact**（BEIR 版 scifact）上跑检索评测。

与 `eval_bench.py` 的区别（不能混用，两个口径）：

| | `eval_bench.py`（自建 k8s 集） | 本脚本（SciFact） |
|---|---|---|
| gold 粒度 | `gold_chunks` = **父块 id** | qrels = **文档 id**（一篇摘要一个文档） |
| 汇总口径 | recall / hit / full / MRR @10 | **nDCG@10**（BEIR 论文口径）+ R@10 + MRR@10 |
| 题集 | 中英混排、我们自己改写 | 官方 claim，**不改写**（改了就没法和公开数字对账） |

因为 gold 在文档级、而检索链返回父块级，这里多做一步 **父块 → 文档** 的去重收敛
（同一文档命中多个父块只算一次，取最好名次），否则 nDCG 会被同文档的多个块虚高。

用法：

    RAG_COLLECTION=scifact_v1 python scripts/eval_beir.py \
        --store storage/bench-scifact --out docs/evidence/beir-scifact-v1-baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_metrics import ndcg_at_k  # noqa: E402

DATASET = ROOT / "datasets" / "scifact-v1"
QRELS = DATASET / "qrels" / "test.tsv"
QUERIES = DATASET / "queries.jsonl"
STORE = ROOT / "storage" / "bench-scifact" / "documents.sqlite3"

#: 对账锚点：BEIR 论文（Thakur et al., NeurIPS 2021）里 BM25 在 SciFact 上的 nDCG@10。
#: 别拿它当「必须达到的线」——它是**单语言、纯词面**的经典基线，
#: 我们用整条链（三路 + RRF + cross-encoder 重排）跑，高于它才算方案在这里站得住。
BM25_NDCG10_SCIFACT = 0.665
COLLECTION = "scifact_v1"


def read_qrels(path: Path = QRELS) -> dict[str, set[str]]:
    """BEIR qrels：`query-id<TAB>corpus-id<TAB>score`，score ≥ 1 算相关。"""
    qrels: dict[str, set[str]] = {}
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if index == 0 and parts[0].lower().startswith("query"):
            continue
        query_id, doc_id, score = parts[0], parts[1], float(parts[2] if len(parts) > 2 else 1.0)
        if score >= 1.0:
            qrels.setdefault(query_id, set()).add(doc_id)
    return qrels


def read_queries(path: Path = QUERIES) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            out[str(row["id"])] = row
    return out


def load_doc_map(store_path: Path, *, suffix: str = ".md") -> dict[str, str]:
    """从 Document Store 读「父块 → 文档 id」。

    不能复用 `eval_bench.load_parents()`：它刻意只读 `parents` 表（为了别为一个全 NULL 的
    `language` 列去 JOIN），而这里的文档 id 落在 `documents.file_name` / `source_uri` 上。
    """
    import sqlite3

    path = Path(store_path)
    if not path.is_file():
        raise SystemExit(f"找不到 Document Store：{path}（先按 NOTICE 里的命令建库）")
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute("SELECT p.parent_id, d.file_name, d.source_uri FROM parents p "
                            "JOIN documents d ON d.document_id = p.document_id").fetchall()
    finally:
        conn.close()
    return parent_to_doc({str(row[0]): {"file_name": row[1], "source_uri": row[2]} for row in rows},
                         suffix=suffix)


def parent_to_doc(parents: Mapping[str, Mapping], *, suffix: str = ".md") -> dict[str, str]:
    """父块 id → SciFact 文档 id（文档 id 落在 `file_name` / `source_uri` 上）。

    SciFact 的 `doc_id` 是数字，而库里文件名是 `<doc_id>.md`，所以取 basename 去掉扩展名。
    """
    mapping: dict[str, str] = {}
    for parent_id, record in parents.items():
        name = str(record.get("file_name") or record.get("source_uri") or "").replace("\\", "/")
        base = name.rsplit("/", 1)[-1]
        if base.endswith(suffix):
            base = base[: -len(suffix)]
        if base:
            mapping[parent_id] = base
    return mapping


def doc_ranking(parent_ids: Iterable[str], mapping: Mapping[str, str],
                *, limit: int | None = None) -> list[str]:
    """父块名次 → 文档名次（同文档只保留首次出现，即最好名次）。"""
    out: list[str] = []
    seen: set[str] = set()
    for parent_id in parent_ids:
        doc_id = mapping.get(parent_id)
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
        if limit is not None and len(out) >= limit:
            break
    return out


def score_query(ranking: Sequence[str], gold: Iterable[str], *, k: int = 10) -> dict:
    gold = set(gold)
    top = list(ranking)[:k]
    hits = [doc for doc in top if doc in gold]
    first = next((i + 1 for i, doc in enumerate(top) if doc in gold), None)
    return {
        "ndcg@%d" % k: round(ndcg_at_k(top, gold, k), 4),
        "recall@%d" % k: round(len(hits) / len(gold), 4) if gold else 0.0,
        "hit@%d" % k: 1 if hits else 0,
        "full@%d" % k: 1 if gold and gold.issubset(set(top)) else 0,
        "mrr@%d" % k: round(1.0 / first, 4) if first else 0.0,
        "first_hit_rank": first,
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def evaluate(*, collection: str = COLLECTION, store: Path = STORE, k: int = 10,
             pool: int = 40, rerank: bool = True, warmup: bool = True, disable: Sequence[str] = (),
             limit: int | None = None, qrels_path: Path = QRELS,
             queries_path: Path = QUERIES, log=None) -> dict:
    from scripts.eval_bench import _build_retriever, load_parents
    from src.rag.ask.pipeline import retrieve_with_meta
    from src.rag.ingestion.tokenizer import TokenCounter
    from src.rag.retrieval.rerank import CrossEncoderReranker, RerankConfig

    qrels = read_qrels(qrels_path)
    queries = read_queries(queries_path)
    items = [(qid, queries[qid]["claim"]) for qid in sorted(qrels) if qid in queries]
    if limit:
        items = items[:limit]
    parents = load_parents(store)
    mapping = load_doc_map(store)
    counter = TokenCounter()
    reranker = CrossEncoderReranker(config=RerankConfig(enabled=rerank))
    retriever = _build_retriever(collection, store, [text for _qid, text in items[:5]] if warmup else (),
                                 log or (lambda _m: None))

    rows, latencies, unmapped = [], [], 0
    for query_id, claim in items:
        started = time.perf_counter()
        outcome = retrieve_with_meta(claim, retriever=retriever, parents=parents, counter=counter,
                                    reranker=reranker, pool=pool, rerank=rerank, k=pool,
                                    disable=tuple(disable))
        latencies.append((time.perf_counter() - started) * 1000)
        ranked_parents = [hit.parent_id for hit in outcome.hits]
        unmapped += sum(1 for pid in ranked_parents if pid not in mapping)
        ranking = doc_ranking(ranked_parents, mapping, limit=k)
        row = {"qid": query_id, "claim": claim, "gold": sorted(qrels[query_id]),
               "retrieved": ranking, **score_query(ranking, qrels[query_id], k=k)}
        rows.append(row)

    positives = [row for row in rows if row["gold"]]
    summary = {key: round(statistics.mean(row[key] for row in positives), 4)
               for key in ("ndcg@%d" % k, "recall@%d" % k, "hit@%d" % k, "full@%d" % k,
                           "mrr@%d" % k)}
    return {
        "dataset": "scifact-v1（BEIR 版 scifact 的 dev 切分，即 BEIR 的 qrels/test.tsv）",
        "collection": collection,
        "store": str(store.relative_to(ROOT)) if store.is_relative_to(ROOT) else str(store),
        "k": k, "pool": pool, "rerank": rerank, "disable": list(disable),
        "queries": len(rows), "positives": len(positives),
        "unmapped_parent_hits": unmapped,
        "overall": summary,
        "latency_ms": {"p50": round(percentile(latencies, 0.5), 1),
                       "p90": round(percentile(latencies, 0.9), 1), "warmup": bool(warmup)},
        "note": "nDCG@10 是 BEIR 论文口径（二元相关、log2 折扣）；同一文档命中多个父块只算一次",
        "rows": rows,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SciFact（BEIR scifact）检索评测")
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pool", type=int, default=40)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--disable", nargs="*", default=[],
                        help="关掉某些检索路（消融用，例如 --disable dense_zh dense_en 只剩 BM25）")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rows-out", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None,
                        help="与既有结果 JSON 比 nDCG@10（逐题更好/更差/持平）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = evaluate(collection=args.collection, store=args.store, k=args.k, pool=args.pool,
                      rerank=not args.no_rerank, warmup=not args.no_warmup, limit=args.limit,
                      disable=tuple(args.disable))
    overall = result["overall"]
    print(f"题数 {result['queries']}（有相关文档 {result['positives']}）｜k={args.k} pool={args.pool} "
          f"rerank={not args.no_rerank} disable={list(args.disable) or '无'}")
    for key, value in overall.items():
        print(f"  {key:<12} {value}")
    print(f"  latency      p50={result['latency_ms']['p50']} ms  p90={result['latency_ms']['p90']} ms")
    if result["unmapped_parent_hits"]:
        print(f"  警告：{result['unmapped_parent_hits']} 个父块无法映射到文档 id（qrels 对不上）",
              file=sys.stderr)

    if args.baseline and args.baseline.exists():
        base = json.loads(args.baseline.read_text(encoding="utf-8"))
        base_rows = {row["qid"]: row for row in base.get("rows", [])}
        better = worse = same = 0
        for row in result["rows"]:
            old = base_rows.get(row["qid"])
            if not old:
                continue
            delta = row["ndcg@%d" % args.k] - old["ndcg@%d" % args.k]
            better += delta > 1e-9
            worse += delta < -1e-9
            same += abs(delta) <= 1e-9
        print(f"\\n## 与 {args.baseline.name} 比（nDCG@{args.k}）")
        print(f"  基线 {base['overall']['ndcg@%d' % args.k]:.4f} → 现在 {overall['ndcg@%d' % args.k]:.4f}"
              f"  ({overall['ndcg@%d' % args.k] - base['overall']['ndcg@%d' % args.k]:+.4f})")
        print(f"  逐题：更好 {better} · 更差 {worse} · 持平 {same}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\\n结果 → {args.out}")
    if args.rows_out:
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        args.rows_out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                        for row in result["rows"]), encoding="utf-8")
        print(f"逐题 → {args.rows_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
