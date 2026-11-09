"""在基准数据集上跑检索评估（`datasets/k8s-zh-en-v1/questions.jsonl`）。

指标口径按既定纪律：

- **同时报** `Recall@k` / `hit@k` / `full@k`（三者分母不同，不许只报一个综合分）；
- 分切片看：**证据语言**（zh / en / 跨语言）、**题型**（single_hop / multi_hop / numeric）；
- 负样本单独统计：它们的正确行为是**弃答**（没有 gold，不进 recall 分母），
  所以 `negatives` 只报「检索到的 top-1 相似度」而不混进 recall；
- 报延迟前先 **warmup**（每条 query 的第一次都是冷调用，会把 p50 带偏）。

`gold_chunks` 来自建库时的 child id；检索结果与它比对。命中判定看 **chunk_id**，
不看文本相似度 —— 否则改一次切分参数，历史指标就不可比了。

    python scripts/eval_bench.py --k 10 --out docs/baseline-k8s-v1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCH_STORE = ROOT / "storage" / "bench" / "documents.sqlite3"
QUESTIONS = ROOT / "datasets" / "k8s-zh-en-v1" / "questions.jsonl"
COLLECTION = "bench_k8s_v1"


def load_questions(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"找不到题集：{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_parents(path: Path) -> dict[str, dict]:
    import sqlite3

    if not path.is_file():
        raise SystemExit(f"找不到基准库的文档表：{path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    parents: dict[str, dict] = {}
    for row in conn.execute("SELECT parent_id, document_id, content, title, section_path, language "
                            "FROM parents"):
        parents[row["parent_id"]] = dict(row)
    conn.close()
    return parents


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def summarize(rows: list[dict], k: int) -> dict:
    """按题汇总：`hit@k` / `full@k` / `recall@k` / `mrr` **四个一起报**。

    三者分母不同：`hit` 看「至少命中一条」，`full` 看「gold 全中」，`recall` 是逐题比例均值。
    只报一个综合分，等于把「差一点」和「全跑偏」混在一起。
    """
    positives = [r for r in rows if r["gold"]]
    if not positives:
        return {"n": len(rows), "positives": 0}
    # MRR 的分母是**全部正样本**，未命中记 0。
    # 只对「命中的那些」求平均会把分数抬高一截（实测 0.7098 → 真实 0.6105），
    # 这类偏差正好让「改坏了」看起来像「改好了」。
    ranks = [r["first_hit_rank"] for r in positives]
    return {
        "n": len(rows),
        "positives": len(positives),
        f"hit@{k}": round(sum(1 for r in positives if r["hit"]) / len(positives), 4),
        f"full@{k}": round(sum(1 for r in positives if r["full"]) / len(positives), 4),
        f"recall@{k}": round(statistics.mean(r["recall"] for r in positives), 4),
        "mrr": round(statistics.mean((1.0 / rank) if rank else 0.0 for rank in ranks), 4),
    }


def _build_retriever(collection: str, store_path: Path, warmup_texts, log):
    """单独提出来是为了让用例能打桩验证「集合名真的传下去了」。"""
    from src.rag.retrieval.store_wiring import build_default_retriever

    return build_default_retriever(
        collection=collection,
        store_path=store_path,
        warmup_texts=warmup_texts,
        log=log,
    )


def evaluate(
    questions: list[dict],
    *,
    k: int,
    pool: int,
    rerank: bool,
    warmup: bool,
    collection: str = COLLECTION,
    store: Path = BENCH_STORE,
    parents: dict[str, dict] | None = None,
    counter=None,
    reranker=None,
) -> dict:
    from src.rag.ask.pipeline import retrieve_with_meta
    from src.rag.ingestion.tokenizer import TokenCounter
    from src.rag.retrieval.rerank import CrossEncoderReranker, RerankConfig

    if parents is None:
        parents = load_parents(store)
    if counter is None:
        counter = TokenCounter()
    if reranker is None:
        reranker = CrossEncoderReranker(config=RerankConfig(enabled=rerank))
    retriever = _build_retriever(
        collection,
        store,
        [q["question"] for q in questions[:5]] if warmup else (),
        lambda _msg: None,
    )

    rows: list[dict] = []
    latencies: list[float] = []
    for question in questions:
        query = question["question"]
        started = time.perf_counter()
        outcome = retrieve_with_meta(query, retriever=retriever, parents=parents, counter=counter,
                                     reranker=reranker, pool=pool, rerank=rerank)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        ranked = [hit.chunk_id for hit in outcome.hits][:k]   # 检索返回的是 parent id
        gold = set(question.get("gold_chunks") or [])
        hit_at = [rank for rank, cid in enumerate(ranked, start=1) if cid in gold]
        rows.append({
            "qid": question["qid"],
            "question": query,
            "type": question["type"],
            "expected_evidence_lang": question.get("expected_evidence_lang"),
            "gold": sorted(gold),
            "retrieved": ranked,
            "first_hit_rank": hit_at[0] if hit_at else None,
            "hit": bool(hit_at),
            "full": bool(gold) and gold.issubset(set(ranked)),
            "recall": (len(gold & set(ranked)) / len(gold)) if gold else None,
            "latency_ms": round(elapsed_ms, 1),
        })

    by_lang: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_lang[row["expected_evidence_lang"] or "negative"].append(row)
        by_type[row["type"]].append(row)

    negatives = [r for r in rows if r["type"] == "negative"]
    return {
        "collection": collection,
        "questions": str(QUESTIONS.relative_to(ROOT)),
        "k": k,
        "pool": pool,
        "rerank": rerank,
        "overall": summarize(rows, k),
        "by_evidence_lang": {name: summarize(items, k) for name, items in sorted(by_lang.items())},
        "by_type": {name: summarize(items, k) for name, items in sorted(by_type.items())},
        "negatives": {
            "n": len(negatives),
            "note": "负样本的正确行为是弃答；它们没有 gold，不进 recall 分母",
        },
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 1),
            "p90": round(percentile(latencies, 0.90), 1),
            "warmup": bool(warmup),
        },
        "rows": rows,
    }


def _display_path(path: Path) -> str:
    """仓库内给相对路径，仓库外给原样 —— **打印不能因为路径不在仓库里就崩**。"""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _resolve_paths(args):
    """相对路径按当前工作目录解析成绝对路径（`--out`/`--store`/`--questions`）。"""
    args.out = Path(args.out).resolve()
    args.store = Path(args.store).resolve()
    args.questions = Path(args.questions).resolve()
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval_bench", description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pool", type=int, default=40)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--collection", default=COLLECTION,
                        help=f"要评估的集合（默认 {COLLECTION}；做实验请用带随机后缀的名字）")
    parser.add_argument("--store", type=Path, default=BENCH_STORE,
                        help="该集合对应的文档表（parents 从这里取）")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "baseline-k8s-v1.json")
    parser.add_argument("--rows-out", type=Path, default=None, help="逐题 JSONL（默认与 --out 同名 .rows.jsonl）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _resolve_paths(build_parser().parse_args(argv))

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    result = evaluate(questions, k=args.k, pool=args.pool, rerank=not args.no_rerank,
                      warmup=not args.no_warmup, collection=args.collection, store=args.store)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_out = args.rows_out or args.out.with_suffix(".rows.jsonl")
    args.out.write_text(json.dumps({k: v for k, v in result.items() if k != "rows"},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in result["rows"]) + "\n",
                        encoding="utf-8")

    overall = result["overall"]
    print(f"题数 {overall.get('n')}（正样本 {overall.get('positives')}）｜k={result['k']} pool={result['pool']} "
          f"rerank={result['rerank']}")
    for key in (f"recall@{result['k']}", f"hit@{result['k']}", f"full@{result['k']}", "mrr"):
        if key in overall:
            print(f"  {key:12} {overall[key]}")
    print("按证据语言：")
    for name, stats in result["by_evidence_lang"].items():
        if stats.get("positives"):
            print(f"  {name:9} n={stats['positives']:3}  "
                  f"recall@{result['k']}={stats[f'recall@{result['k']}']}  "
                  f"hit@{result['k']}={stats[f'hit@{result['k']}']}  "
                  f"full@{result['k']}={stats[f'full@{result['k']}']}  mrr={stats['mrr']}")
    print(f"延迟（{'已热身' if result['latency_ms']['warmup'] else '未热身'}）："
          f" p50={result['latency_ms']['p50']} ms  p90={result['latency_ms']['p90']} ms")
    print(f"结果：{_display_path(args.out)} ｜逐题：{_display_path(rows_out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
