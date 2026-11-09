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

NL = chr(10)


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
    # 只读 parents 表：`language` 在历史库里全是 NULL（N49），真正的兜底在 `_block_lang()`
    # 按正文现判；为了一个还是 NULL 的列去 JOIN documents 只会把最小可用 schema 变复杂。
    for row in conn.execute("SELECT parent_id, document_id, content, title, section_path, language "
                            "FROM parents"):
        item = dict(row)
        # `text` 是检索链路认的键（与 store_wiring.load_parent_records 一致）；
        # 少了它，重排的窗口全是空串 —— 实测基线因此一直「含重排」却从没真重排过（N45）。
        item["text"] = item.get("content") or ""
        parents[row["parent_id"]] = item
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


#: 指标 -> (JSON 里的键模板, 表格里显示的名字)。键名必须和 `summarize()` 完全一致，
#: 否则门禁会拿 `MRR` 去查 `mrr` 然后 KeyError（第一次写就踩了）。
METRIC_LABELS = {
    "hit": ("hit@{k}", "hit@k"),
    "full": ("full@{k}", "full@k"),
    "recall": ("recall@{k}", "recall@k"),
    "mrr": ("mrr", "MRR"),
}


def questions_fingerprint(path: Path) -> str:
    """题集的 sha256 —— 门禁判断「比的还是不是同一套题」的唯一依据。

    不认文件名、不认行数：改一个字、换一题，行数可能一样。
    """
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_half_width(values: list[float], *, n_boot: int = 2000, alpha: float = 0.05,
                         seed: int = 0) -> float:
    """基线一侧的 95% bootstrap CI 半宽，用作门禁容差。

    为什么不用固定百分比（比如 1%）：我们这些指标在 95 题上本身就抖 ±0.04 量级，
    固定 1% 会把正常抖动全报成回归，最后没人看门禁。为什么不配对 bootstrap：
    容差要描述「基线自己能抖多少」，配对差会把两边的抖动互相抵消、容差变窄，反而更容易误报。
    """
    import random

    if not values:
        return 0.0
    rng = random.Random(seed)                    # 固定种子：同一份基线每次算出的容差一致
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low = means[int((alpha / 2) * n_boot)]
    high = means[max(0, int((1 - alpha / 2) * n_boot) - 1)]
    return round((high - low) / 2, 4)


def _metric_series(rows: list[dict]) -> dict[str, list[float]]:
    """逐题序列（只含正样本）：用来算 CI，也用来做逐题对比。"""
    positives = [r for r in rows if r["gold"]]
    return {
        "hit": [1.0 if r["hit"] else 0.0 for r in positives],
        "full": [1.0 if r["full"] else 0.0 for r in positives],
        "recall": [float(r["recall"] or 0.0) for r in positives],
        "mrr": [(1.0 / r["first_hit_rank"]) if r["first_hit_rank"] else 0.0 for r in positives],
    }


def _comparability_reasons(current: dict, baseline: dict) -> list[str]:
    reasons: list[str] = []
    if not baseline.get("questions_sha256"):
        reasons.append(
            "基线里没有 questions_sha256（旧基线）：无法确认题集是否一致，先重跑一次基线"
        )
    elif current.get("questions_sha256") != baseline["questions_sha256"]:
        reasons.append(
            f"题集不同：基线 {baseline['questions_sha256'][:12]}… vs 现在 "
            f"{str(current.get('questions_sha256'))[:12]}…"
        )
    for key in ("k", "pool", "rerank"):
        if current.get(key) != baseline.get(key):
            reasons.append(f"{key} 不同：基线 {baseline.get(key)} vs 现在 {current.get(key)}")
    base_ids = {r["qid"] for r in baseline.get("rows", [])}
    cur_ids = {r["qid"] for r in current.get("rows", [])}
    if base_ids and cur_ids and base_ids != cur_ids:
        reasons.append(
            f"逐题结果对不上：基线 {len(base_ids)} 题、现在 {len(cur_ids)} 题，"
            f"只有 {len(base_ids & cur_ids)} 题相同"
        )
    return reasons


def compare_results(current: dict, baseline: dict, baseline_rows: list[dict],
                    *, tolerance: float | None = None) -> dict:
    """把「现在」和「基线」比成一张表 + 一个退出码。

    退出码纪律：**不可比一律退 0**。CI 里非零退出等于「代码坏了」，
    而这里真正的情况是「你换了题/换了参数」—— 那种事要人看，不该把流水线卡死。
    """
    reasons = _comparability_reasons(current, baseline)
    if reasons:
        return {
            "comparable": False, "reasons": reasons, "metrics": {},
            "per_question": {}, "verdict": "not_comparable", "exit_code": 0,
            "tolerance": None,
        }

    k = current["k"]
    base_series = _metric_series(baseline_rows)
    cur_series = _metric_series(current["rows"])
    base_overall = baseline["overall"]
    cur_overall = current["overall"]

    metrics: dict[str, dict] = {}
    for name, (template, display) in METRIC_LABELS.items():
        key = template.format(k=k)
        base_value = base_overall[key]
        cur_value = cur_overall[key]
        tol = bootstrap_half_width(base_series[name]) if tolerance is None else float(tolerance)
        delta = round(cur_value - base_value, 4)
        if cur_value >= base_value + tol:
            verdict = "improved"
        elif cur_value < base_value - tol:
            verdict = "regression"
        else:
            verdict = "ok"
        metrics[name] = {
            "label": display, "key": key, "baseline": base_value, "current": cur_value, "delta": delta,
            "tolerance": round(tol, 4), "verdict": verdict,
        }

    base_rank = {r["qid"]: (r["first_hit_rank"] or 999) for r in baseline_rows if r["gold"]}
    cur_rank = {r["qid"]: (r["first_hit_rank"] or 999) for r in current["rows"] if r["gold"]}
    better = worse = same = 0
    moved = []
    for qid in sorted(base_rank):
        before, after = base_rank[qid], cur_rank[qid]
        if after < before:
            better += 1
        elif after > before:
            worse += 1
        else:
            same += 1
        if before != after:
            moved.append({"qid": qid, "before": base_rank[qid], "after": cur_rank[qid]})

    verdict = "regression" if any(m["verdict"] == "regression" for m in metrics.values()) else "ok"
    return {
        "comparable": True, "reasons": [], "metrics": metrics,
        "per_question": {"better": better, "worse": worse, "same": same, "moved": moved},
        "verdict": verdict,
        "exit_code": 1 if verdict == "regression" else 0,
        "tolerance": round(metrics["mrr"]["tolerance"] if tolerance is None else float(tolerance), 4),
    }


def format_comparison(cmp: dict) -> list[str]:
    """人看的 Markdown。不可比时**先讲原因**，再讲结论 —— 别让人盯着空表猜。"""
    lines: list[str] = []
    if not cmp["comparable"]:
        lines.append("**not_comparable**：这次结果与基线不可比，所以不给 delta、也不判红。原因：")
        lines.extend(f"- {reason}" for reason in cmp["reasons"])
        lines.append("（换题/换参数后请重立基线；同一套题同一套参数才能比。）")
        return lines

    lines.append(f"容差 = 基线一侧 95% bootstrap CI 半宽（MRR ±{cmp['tolerance']}）；掉出容差才判红。")
    lines.append("")
    lines.append("| 指标 | 基线 | 现在 | delta | 容差 | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    for metric in cmp["metrics"].values():
        lines.append(
            f"| {metric['label']} | {metric['baseline']:.4f} | {metric['current']:.4f} | "
            f"{metric['delta']:+.4f} | ±{metric['tolerance']} | {metric['verdict']} |"
        )
    counts = cmp["per_question"]
    lines.append("")
    lines.append(
        f"逐题（首命中排名）：更好 **{counts['better']}** · 更差 **{counts['worse']}** · "
        f"持平 **{counts['same']}**"
    )
    return lines


def query_for(question: dict, *, mode: str, translate=None) -> tuple[str, str | None]:
    """这次检索到底喂什么给 `dense_en` / `lexical_en`。

    三档，缺一档就会含糊：
    - `none`      ：只给可见问题。**这不是生产口径**，是「翻译缺失」的降级态，
                    也是修 N47 之前所有基线的真实口径（英文索引一直在吃中文原句）。
    - `field`     ：再传手写的 `query_en` = 翻译**上界**（翻译永远正确时能到哪），用来判断
                    「翻译有没有拖后腿」。
    - `translate` ：走和 `/ask` 同一套翻译服务（同模型同缓存）= 生产口径。
                    翻译失败时退化成 `none`（返回 None），绝不用机翻内容顶替。
    """
    if mode not in ("none", "field", "translate"):
        raise ValueError(f"未知的 query-en 口径：{mode}（可选 none / field / translate）")
    question_text = question["question"]
    if mode == "none":
        return question_text, None
    if mode == "field":
        english = (question.get("query_en") or "").strip()
        return question_text, (english or None)
    if translate is None:
        raise ValueError("translate 口径必须给 translate 函数（否则等于偷偷跑成 none）")
    english = (translate(question_text) or "").strip()
    return question_text, (english or None)


class TranslateAdapter:
    """`/ask` 的翻译口径，外加把「翻了几条」暴露出来给基线归档。"""

    def __init__(self, service):
        self.service = service

    def __call__(self, question: str) -> str | None:
        from src.rag.lang.detect import detect

        if detect(question) != "zh":
            return question
        result = self.service.translate([question])[0]
        # 只有真译文才算数：passthrough_* 的 text 是**中文原文回填**，
        # 当成英文喂下去等于悄悄退化成 none 口径（N48）。
        if result.status not in ("ok", "cached"):
            return None
        return result.text or None

    def stats(self) -> dict:
        return self.service.stats()


def build_translate_fn(chat=None):
    """和 `/ask` 完全同一个翻译服务：同模型、同缓存文件、同 provider 工厂。

    刻意不另写一套「评测专用翻译」—— 那样测出来的就不是生产行为。
    `chat` 只给用例注入假模型用。
    """
    from src.config import OLLAMA_CHAT_MODEL, STORAGE_DIR
    from src.rag.translate.cache import TranslationCache
    from src.rag.translate.service import TranslateService, provider_from_chat

    if chat is None:
        from src.ollama_client import chat as ollama_chat
        chat = ollama_chat

    service = TranslateService(
        provider=provider_from_chat(chat),
        cache=TranslationCache(STORAGE_DIR / "cache" / "translations.sqlite3",
                               model=OLLAMA_CHAT_MODEL, version="v1"),
        max_retries=1,
    )
    return TranslateAdapter(service)


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
    query_en: str = "none",
    translator=None,
    rerank_query: str = "question",
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
        query, english = query_for(question, mode=query_en, translate=translator)
        started = time.perf_counter()
        outcome = retrieve_with_meta(query, retriever=retriever, parents=parents, counter=counter,
                                     reranker=reranker, pool=pool, rerank=rerank,
                                     query_en=english, rerank_query=rerank_query)
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
        "query_en_mode": query_en,
        "rerank_query_mode": rerank_query,
        **({"translate": translator.stats()} if translator is not None
           and hasattr(translator, "stats") else {}),
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
    """相对路径按当前工作目录解析成绝对路径。

    可选参数（`--out` / `--compare` / `--report` / `--baseline-rows`）保持 `None`，
    否则 `Path(None)` 直接 TypeError（带 `--compare` 时 `--out` 默认就是 None）。
    """
    for name in ("out", "compare", "report", "baseline_rows", "rows_out"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).resolve())
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
    parser.add_argument("--out", type=Path, default=None,
                        help="汇总 JSON（默认 docs/baseline-k8s-v1.json；带 --compare 时不写，免得覆盖基线）")
    parser.add_argument("--rows-out", type=Path, default=None, help="逐题 JSONL（默认与 --out 同名 .rows.jsonl）")
    parser.add_argument("--compare", type=Path, default=None,
                        help="与基线对比并做回归门禁（基线汇总 JSON；逐题默认取同名 .rows.jsonl）")
    parser.add_argument("--baseline-rows", type=Path, default=None, help="基线逐题 JSONL（默认自动推断）")
    parser.add_argument("--query-en", choices=("none", "field", "translate"), default="none",
                        help="dense_en/lexical_en 吃什么：none = 只给原问题（旧口径）；"
                             "field = 再传手写 query_en（翻译上界）")
    parser.add_argument("--rerank-query", choices=("question", "by_lang"), default="question",
                        help="重排用哪个 query：question = 只用提问原句（默认）；"
                             "by_lang = 按候选块语言选（中文块用中文问、英文块用英文问）")
    parser.add_argument("--compare-tolerance", type=float, default=None,
                        help="覆盖容差（默认 = 基线一侧 95%% bootstrap CI 半宽）")
    parser.add_argument("--report", type=Path, default=None, help="把对比表另存为 Markdown")
    return parser


def _default_out(args) -> Path:
    return args.out or (ROOT / "docs" / "baseline-k8s-v1.json")


def main(argv: list[str] | None = None) -> int:
    args = _resolve_paths(build_parser().parse_args(argv))

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    result = evaluate(questions, k=args.k, pool=args.pool, rerank=not args.no_rerank,
                      warmup=not args.no_warmup, collection=args.collection, store=args.store,
                      query_en=args.query_en,
                      translator=build_translate_fn() if args.query_en == "translate" else None,
                      rerank_query=args.rerank_query)
    result["questions_sha256"] = questions_fingerprint(args.questions)

    out_path = None if args.compare and args.out is None else _default_out(args)
    rows_out = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows_out = args.rows_out or out_path.with_suffix(".rows.jsonl")
        out_path.write_text(json.dumps({k: v for k, v in result.items() if k != "rows"},
                                       ensure_ascii=False, indent=2) + NL, encoding="utf-8")
        rows_out.write_text(NL.join(json.dumps(r, ensure_ascii=False) for r in result["rows"]) + NL,
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
    if out_path is not None:
        print(f"结果：{_display_path(out_path)} ｜逐题：{_display_path(rows_out)}")

    if args.compare is None:
        return 0

    baseline_path = args.compare
    if not baseline_path.is_file():
        print(f"[compare] 找不到基线：{baseline_path}", file=sys.stderr)
        return 2
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows_path = args.baseline_rows or baseline_path.with_suffix(".rows.jsonl")
    if not baseline_rows_path.is_file():
        print(f"[compare] 找不到基线的逐题文件：{baseline_rows_path}", file=sys.stderr)
        return 2
    baseline_rows = [json.loads(line) for line in
                     baseline_rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    comparison = compare_results(result, baseline, baseline_rows,
                                tolerance=args.compare_tolerance)
    print("")
    print("## 与基线对比")
    lines_out = format_comparison(comparison)
    print(NL.join(lines_out))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("# 基线对比" + NL + NL + NL.join(lines_out) + NL, encoding="utf-8")
        print(f"[compare] 已写入 {_display_path(args.report)}")
    return int(comparison["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
