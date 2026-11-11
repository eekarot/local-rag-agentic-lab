"""C0f 第二步：四种「挑窗口」策略的端到端对照（真实交叉编码器，同一批候选同一套题）。

前置测量（`probe_c0f_window_signal.py`）已否掉 C0f 的原假设：词面重叠为 0 的块只占 4.5%
（英→中 **0%**），所以「无有效重叠就退回块首」几乎不触发；但它量出**上限还差 21–39 个点**
（gold 块里「随便哪一段都含答案句」= 1.0，现状 overlap 挑中的只有 0.79），
说明病在**挑错**，不在挑不到。

四种策略（只改送进 cross-encoder 的那一段，不改检索）：
  A `overlap`  生产口径：按词面重叠挑（`best_window`，候选窗口 = 全部可滑位置）
  B `head`     恒定块首（≈「不挑」，N54 里已证明 −13.7 点，作对照）
  C `last`     恒定块尾（中文正文/结论常在后半段，最省事的「反块首」）
  D `tiles`    让 cross-encoder 在**粗粒度分段**上各打一分、取最高

为什么 D 用「粗粒度分段」而不是「全部可滑窗口」：可滑窗口每块有 ~24 个（40 个候选 = 941 个），
一条 query 光编码就要 31 s —— 不可交付。粗粒度分段按预算把正文**切完不重叠**，
块内平均只有 1.6 段（p50 父块 322 token / 256），代价与基线同量级。

实现要点：**每题只编码一堆**（各候选的粗分段 + 生产口径挑出的那一段），
四个口径都从这份分数缓存里取 —— 否则要跑四遍编码器。
所以「A 与基线逐题一致」只作**参考**：探针走的是 `retriever.retrieve`，
与 `eval_bench` 的 `retrieve_with_meta` 在去重/并列排序上并不逐位相同
（N54 那次自检就有 10 题不一致），结论一律以 `eval_bench --compare` 为准。

报三件事，缺一不可（否则「均值涨了」可能只是三道题的抖动）：
逐题更好/更差/持平、每格 hit@10 / MRR、相对 A 的编码耗时倍率。

用法（长任务，逐题落 JSONL + 断点续跑）：
    python experiments/probe_c0f_window_strategy.py            # 从头跑
    python experiments/probe_c0f_window_strategy.py --resume   # 接着跑
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.eval_bench import (BENCH_STORE, QUESTIONS, _build_retriever, build_translate_fn,
                                load_parents, load_questions)
from src.rag.ask.pipeline import _parent_text
from src.rag.ingestion.tokenizer import TokenCounter
from src.rag.retrieval.rerank import (CrossEncoderReranker, RerankConfig, RerankHit, best_window,
                                      window_candidates)

POOL = 40
K = 10
MODES = ("overlap", "head", "last", "tiles")
ROWS_OUT = ROOT / "storage" / "expand" / "c0f-rows.jsonl"
DEST = ROOT / "docs" / "evidence" / "c0f-window-probe-k8s-v1.json"


def cell_of(question: dict) -> str:
    zh_q = bool(re.search(r"[\u4e00-\u9fff]", question["question"]))
    zh_e = question.get("expected_evidence_lang") == "zh-cn"
    return "中→中" if zh_q and zh_e else "中→英" if zh_q else "英→中" if zh_e else "英→英"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="C0f：四种窗口策略端到端对照")
    parser.add_argument("--resume", action="store_true", help="跳过已写进 --rows-out 的题")
    parser.add_argument("--rows-out", default=str(ROWS_OUT), help="逐题结果（断点用，默认 storage/）")
    parser.add_argument("--out", default=str(DEST), help="汇总证据 JSON")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 道正样本（调试用）")
    return parser.parse_args(argv)


def tile_windows(text: str, *, max_tokens: int, counter) -> list[str]:
    """把正文按预算**切完不重叠**（粗粒度分段）。

    与 `window_candidates()` 的区别：那个返回**全部可滑位置**（每块 ~24 个），
    给「按词面重叠挑」用很划算，但让 cross-encoder 逐段打分就要 40×24 次编码。
    这里只按预算切块，块内段数 = ceil(正文 token / 预算)，实测平均 1.6。
    """
    if not text:
        return [""]
    units = [unit for unit in re.split(r"(?<=[。！？；.!?])\s*", text) if unit]
    tiles: list[str] = []
    current: list[str] = []
    used = 0
    for unit in units:
        count = max(1, counter.count(unit))
        if current and used + count > max_tokens:
            tiles.append("".join(current))
            current, used = [], 0
        current.append(unit)
        used += count
    if current:
        tiles.append("".join(current))
    return tiles or [text[: max(4, max_tokens * 2)]]


def rank_of(order: list[str], gold: set[str], k: int = K) -> int | None:
    return next((i + 1 for i, pid in enumerate(order[:k]) if pid in gold), None)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows_path = Path(args.rows_out)
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    questions = [q for q in load_questions(QUESTIONS) if q["type"] != "negative"]
    if args.limit:
        questions = questions[: args.limit]
    done = set()
    if args.resume and rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["qid"])

    parents = load_parents(BENCH_STORE)
    counter = TokenCounter()
    reranker = CrossEncoderReranker(config=RerankConfig())
    retriever = _build_retriever("bench_k8s_v1", BENCH_STORE, [], lambda _m: None)
    translate = build_translate_fn()
    baseline = {json.loads(line)["qid"]: json.loads(line)
                for line in (ROOT / "docs" / "baseline-k8s-v1.rows.jsonl").read_text(
                    encoding="utf-8").splitlines() if line.strip()}

    with rows_path.open("a", encoding="utf-8") as sink:
        for index, question in enumerate(questions, start=1):
            qid = question["qid"]
            if qid in done:
                continue
            cell = cell_of(question)
            gold = set(question["gold_chunks"])
            started = time.perf_counter()
            english = translate(question["question"]) if cell.startswith("中") else None
            fused = retriever.retrieve(question["question"], query_en=english, k=POOL, fuse=True).fused
            hits = [RerankHit(chunk_id=hit.parent_id, parent_id=hit.parent_id,
                              text=_parent_text(parents.get(hit.parent_id)),
                              score=hit.score, rank=rank)
                    for rank, hit in enumerate(fused, start=1)]
            max_tokens = reranker.config.max_length
            tiles = [tile_windows(hit.text, max_tokens=max_tokens, counter=counter) for hit in hits]
            # 生产口径挑出来的那一段也要打分（它不在粗分段里，必须单独编码）
            lexical = [best_window(hit.text, question["question"], max_tokens=max_tokens,
                                  counter=counter) for hit in hits]
            flat = [("tile", i, j, window) for i, item in enumerate(tiles)
                    for j, window in enumerate(item)]
            flat += [("lex", i, 0, window) for i, window in enumerate(lexical)]
            encoded = time.perf_counter()
            scores = np.asarray(reranker.encoder.encode_pairs(
                question["question"], [window for *_, window in flat]), dtype=np.float32)
            encode_s = time.perf_counter() - encoded
            table: dict[int, list[float]] = {i: [0.0] * len(item) for i, item in enumerate(tiles)}
            lex_score = [0.0] * len(hits)
            for (kind, i, j, _window), score in zip(flat, scores):
                if kind == "tile":
                    table[i][j] = float(score)
                else:
                    lex_score[i] = float(score)

            record = {"qid": qid, "cell": cell, "encode_s": round(encode_s, 3),
                      "total_s": None, "self_check": None}
            for mode in MODES:
                if mode == "tiles":
                    scores_for_mode = [max(item) for item in table.values()]
                elif mode == "head":
                    scores_for_mode = [item[0] for item in table.values()]
                elif mode == "last":
                    scores_for_mode = [item[-1] for item in table.values()]
                else:
                    scores_for_mode = lex_score
                for hit, score in zip(hits, scores_for_mode):
                    hit.rerank_score = score
                order = [hit.parent_id for hit in sorted(
                    hits, key=lambda hit: (-(hit.rerank_score or 0.0), hit.rank, hit.chunk_id))]
                record[mode] = rank_of(order, gold)
            record["tiles_per_hit"] = round(sum(len(item) for item in tiles) / len(hits), 2)

            record["total_s"] = round(time.perf_counter() - started, 3)
            base_rank = baseline[qid]["first_hit_rank"] if baseline[qid]["hit"] else None
            record["self_check"] = record["overlap"] == base_rank
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"[{index}/{len(questions)}] {qid:12} {cell:6} " +
                  " ".join(f"{mode}={record[mode]}" for mode in MODES) +
                  f" enc={record['encode_s']:.1f}s 自检={record['self_check']}", flush=True)

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]

    def stats(picked, mode):
        hit = sum(1 for r in picked if r[mode]) / len(picked)
        mrr = sum((1.0 / r[mode]) if r[mode] else 0.0 for r in picked) / len(picked)
        return round(hit, 4), round(mrr, 4)

    def key(value):
        return value if value else 999

    summary = {}
    for cell in ("中→中", "中→英", "英→中", "英→英", "整体"):
        picked = rows if cell == "整体" else [r for r in rows if r["cell"] == cell]
        if not picked:
            continue
        entry = {"n": len(picked)}
        for mode in MODES:
            hit, mrr = stats(picked, mode)
            entry[mode] = {"hit@10": hit, "mrr": mrr}
        entry["vs_overlap"] = {
            mode: {"better": sum(1 for r in picked if key(r[mode]) < key(r["overlap"])),
                   "worse": sum(1 for r in picked if key(r[mode]) > key(r["overlap"])),
                   "same": sum(1 for r in picked if key(r[mode]) == key(r["overlap"]))}
            for mode in MODES if mode != "overlap"}
        summary[cell] = entry

    payload = {
        "note": "C0f 端到端对照：四种窗口策略，同一批候选 / 同一 cross-encoder / 同一题集",
        "params": {"pool": POOL, "k": K, "max_tokens": 256, "collection": "bench_k8s_v1",
                   "questions": len(rows), "reranker": "mmarco-mMiniLMv2"},
        "caveat": ("探针走 retriever.retrieve，与 eval_bench 的 retrieve_with_meta 在去重/并列排序上"
                   "不逐位相同，`self_check` 只作参考；结论以 eval_bench --compare 为准"),
        "self_check_mismatch_count": sum(1 for r in rows if not r["self_check"]),
        "self_check_mismatch": [r["qid"] for r in rows if not r["self_check"]][:20],
        "encode_p50_s": round(float(np.percentile([r["encode_s"] for r in rows], 50)), 3),
        "summary": summary,
        "rows": rows,
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n自检（仅参考）：与基线首命中排名不同 {payload['self_check_mismatch_count']}/{len(rows)} 题")
    print(f"{'格':<8}{'n':>4}" + "".join(f"{mode:>16}" for mode in MODES))
    for cell, s in summary.items():
        print(f"{cell:<8}{s['n']:>4}" + "".join(
            f"{s[m]['hit@10']:>8.4f}/{s[m]['mrr']:.4f}" for m in MODES))
    print("\n相对 overlap 的逐题配对（更好/更差/持平）：")
    for cell, s in summary.items():
        print(f"  {cell:<8}" + "  ".join(
            f"{m}: {v['better']}/{v['worse']}/{v['same']}" for m, v in s["vs_overlap"].items()))
    print(f"\n编码耗时 p50（秒/题，全部窗口各打一分）：{payload['encode_p50_s']}")
    print("证据 →", dest.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
