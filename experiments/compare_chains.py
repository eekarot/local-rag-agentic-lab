"""新旧两条检索链路的**严格配对对比**（同一批题、逐题配对）。

为什么需要单独一个脚本：`python -m src.evaluate compare` 比的是**聚合指标**，
而且（设计如此）拒绝跨 `content_hash` 的比较——换索引后它只会说 `not_comparable`。
要回答「换成 Qdrant parent/child 到底值不值」，唯一站得住的比法是：

> 同一批题 → 两条链路各跑一遍 → 只比**同一道题**的命中差异（McNemar 配对）。

用法：

    python experiments/compare_chains.py \\
        --old-log logs/2026-11-02/20261102T143004-2188.jsonl \\
        --new-log logs/2026-11-04/20261104T113427-12fd.jsonl \\
        --map vector=dense,bm25=lexical,hybrid=hybrid

口径提醒（报告里也要写）：
- 旧链路金标是 500-token chunk（`gold_chunks`），新链路是 ~2k-token 父块（`gold_points`），
  **父块更大 ⇒ 命中更容易**，这个方向对新链路有利，所以差距不能全部记成「检索变好了」。
- 交集通常比各自题数少（旧集 48 题可判定），配对样本变小，CI 变宽。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval_logs import detect_gold_space, load_traces, score_traces  # noqa: E402
from src.eval_metrics import mcnemar_exact  # noqa: E402


def parse_map(text: str) -> list[tuple[str, str]]:
    pairs = []
    for chunk in (text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"--map 每项要写成 old=new：{chunk!r}")
        old, new = chunk.split("=", 1)
        pairs.append((old.strip(), new.strip()))
    return pairs


def compare(old_log, new_log, questions_path, pairs, ks=(4, 10)) -> dict:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    old_traces, new_traces = load_traces(old_log), load_traces(new_log)
    old_space, new_space = detect_gold_space(old_traces), detect_gold_space(new_traces)
    old = score_traces(old_traces, questions, ks=ks, gold_space=old_space)
    new = score_traces(new_traces, questions, ks=ks, gold_space=new_space)

    old_pos = {(row["qid"], row["mode"]): row for row in old["positives"]}
    new_pos = {(row["qid"], row["mode"]): row for row in new["positives"]}
    common = sorted({qid for qid, _ in old_pos} & {qid for qid, _ in new_pos})

    rows = []
    for old_mode, new_mode in pairs:
        pair_rows = [(qid, old_pos[(qid, old_mode)], new_pos[(qid, new_mode)])
                     for qid in common if (qid, old_mode) in old_pos and (qid, new_mode) in new_pos]
        if not pair_rows:
            continue
        entry = {
            "old_mode": old_mode,
            "new_mode": new_mode,
            "n": len(pair_rows),
            "by_k": {},
        }
        for k in ks:
            a = [float(r1["hit"][k]) for _, r1, _ in pair_rows]
            b = [float(r2["hit"][k]) for _, _, r2 in pair_rows]
            old_only = sum(1 for x, y in zip(a, b) if x and not y)
            new_only = sum(1 for x, y in zip(a, b) if y and not x)
            both = sum(1 for x, y in zip(a, b) if x and y)
            neither = len(pair_rows) - old_only - new_only - both
            entry["by_k"][k] = {
                "old_rate": round(sum(a) / len(a), 4),
                "new_rate": round(sum(b) / len(b), 4),
                "old_only": old_only,
                "new_only": new_only,
                "both": both,
                "neither": neither,
                "p_value": round(mcnemar_exact(old_only, new_only)["p_value"], 4),
                "old_recall": round(sum(r1["recall"][k] for _, r1, _ in pair_rows) / len(pair_rows), 4),
                "new_recall": round(sum(r2["recall"][k] for _, _, r2 in pair_rows) / len(pair_rows), 4),
                "old_mrr": round(sum(r1["mrr"][k] for _, r1, _ in pair_rows) / len(pair_rows), 4),
                "new_mrr": round(sum(r2["mrr"][k] for _, _, r2 in pair_rows) / len(pair_rows), 4),
            }
        rows.append(entry)

    by_qid = {item["qid"]: item for item in questions}
    gold_chunks = [len(by_qid[qid].get("gold_chunks") or []) for qid in common]
    gold_parents = [len((by_qid[qid].get("gold_points") or {}).get("parent_ids") or []) for qid in common]
    return {
        "old_log": str(old_log),
        "new_log": str(new_log),
        "gold_space": {"old": old_space, "new": new_space},
        "n_judged": {"old": len({q for q, _ in old_pos}), "new": len({q for q, _ in new_pos}), "common": len(common)},
        "gold_granularity": {
            "old_chunks_mean": round(sum(gold_chunks) / len(gold_chunks), 2) if gold_chunks else None,
            "new_parents_mean": round(sum(gold_parents) / len(gold_parents), 2) if gold_parents else None,
        },
        "rows": rows,
    }


def render(result) -> str:
    lines = [
        f"旧链路日志：`{result['old_log']}`（金标空间 {result['gold_space']['old']}）",
        f"新链路日志：`{result['new_log']}`（金标空间 {result['gold_space']['new']}）",
        f"可判定题数：旧 {result['n_judged']['old']} / 新 {result['n_judged']['new']} / **交集 {result['n_judged']['common']}**",
        f"gold 粒度（交集题）：旧 {result['gold_granularity']['old_chunks_mean']} chunk/题，"
        f"新 {result['gold_granularity']['new_parents_mean']} parent/题 —— **父块更大，命中更容易**，方向对新链路有利",
        "",
        "| 旧模式 → 新模式 | n | k | 旧 hit | 新 hit | 旧独中 | 新独中 | 都中 | 都不中 | p | 旧 Recall | 新 Recall |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        for k, block in row["by_k"].items():
            flag = "**显著**" if block["p_value"] < 0.05 else "不显著"
            lines.append(
                f"| {row['old_mode']} → {row['new_mode']} | {row['n']} | {k} | "
                f"{block['old_rate']:.1%} | {block['new_rate']:.1%} | {block['old_only']} | {block['new_only']} | "
                f"{block['both']} | {block['neither']} | {block['p_value']:.4f} {flag} | "
                f"{block['old_recall']:.3f} | {block['new_recall']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="新旧检索链路的严格配对对比（同一批题）")
    parser.add_argument("--old-log", required=True)
    parser.add_argument("--new-log", required=True)
    parser.add_argument("--questions", default="experiments/questions.json")
    parser.add_argument("--map", default="vector=dense,bm25=lexical,hybrid=hybrid",
                        help="旧模式=新模式，逗号分隔")
    parser.add_argument("--ks", default="4,10")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    ks = tuple(int(x) for x in args.ks.replace(" ", "").split(",") if x)
    result = compare(args.old_log, args.new_log, args.questions, parse_map(args.map), ks=ks)
    print(render(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细：{args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
