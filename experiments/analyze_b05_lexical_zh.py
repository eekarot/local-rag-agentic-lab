"""B05 消融的后处理：把「控制组（三路）/ 对照组（四路）」两份逐题 JSONL 比成一张表。

为什么单独一个脚本而不是用 `eval_bench --compare`：`--compare` 是**回归门禁**，
它的职责是「换了口径就拒绝比较」（`routes` 就在可比性指纹里）。消融的意义正好相反
—— 我们**故意**只改一个口径键，然后逐题看谁变好谁变坏。

所以这里做三件事（都是门禁不提供的）：
1. 均值 delta：overall + 按语言格 + 按题型
2. 逐题更好 / 更差 / 持平计数（repo 规矩：只报均值 delta 等于藏起抖动）
3. 配对 bootstrap 的 95% CI，对照 A/A 噪声底（前轮实测 3 好 / 5 差 / 167 平 ≈ ±0.6 点 hit）

跑法（仓库根目录）：
    python experiments/analyze_b05_lexical_zh.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONTROL = ROOT / "storage" / "b05" / "b05-control-3routes.json"
TREAT = ROOT / "storage" / "b05" / "b05-treat-4routes.json"
OUT = ROOT / "docs" / "evidence" / "b05-lexical-zh-k8s-v1.json"
PROBE = ROOT / "docs" / "evidence" / "b05-lexical-zh-routes-k8s-v1.json"

#: 前轮 A/A（同代码重跑两次）实测的噪声底，写死在证据里当判据
AA_NOISE = {"better": 3, "worse": 5, "same": 167, "hit_delta_points": 0.6}

INF = 10 ** 9


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.with_suffix(".rows.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["qid"]] = row
    return rows


def rank_of(row: dict) -> int:
    return row["first_hit_rank"] or INF


def compare(control: dict[str, dict], treat: dict[str, dict], qids: list[str]) -> dict:
    better = worse = same = 0
    for qid in qids:
        before, after = rank_of(control[qid]), rank_of(treat[qid])
        if after < before:
            better += 1
        elif after > before:
            worse += 1
        else:
            same += 1
    return {"n": len(qids), "better": better, "worse": worse, "same": same}


def paired_bootstrap(control: dict[str, dict], treat: dict[str, dict], qids: list[str],
                     *, n_boot: int = 2000, seed: int = 0) -> dict:
    """配对 bootstrap：把「逐题 hit 差」重采样，看均值差有多少概率翻号。"""
    diffs = [(1.0 if treat[q]["hit"] else 0.0) - (1.0 if control[q]["hit"] else 0.0)
             for q in qids]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        means.append(statistics.fmean(sample))
    means.sort()
    observed = statistics.fmean(diffs)
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return {
        "mean": round(observed, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "p_non_positive": round(sum(1 for m in means if m <= 0) / n_boot, 4),
    }


def _mechanism(control: dict[str, dict], treat: dict[str, dict],
               qids: list[str]) -> dict:
    """把「变好 / 变差」的题对到路由覆盖率上：看涨跌到底来自补召回还是融合重排。

    这一步是 N50 的判据：只看均值会以为「第四路补了 10 道题」，逐题一对就发现
    变好的题里 **没有一道**是「gold 只出现在第四路」的那 5 道。
    """
    better = [q for q in qids if rank_of(treat[q]) < rank_of(control[q])]
    worse = [q for q in qids if rank_of(treat[q]) > rank_of(control[q])]
    block: dict = {"better_qids": better, "worse_qids": worse,
                   "better_rank_changes": {q: [control[q]["first_hit_rank"],
                                               treat[q]["first_hit_rank"]] for q in better},
                   "worse_rank_changes": {q: [control[q]["first_hit_rank"],
                                              treat[q]["first_hit_rank"]] for q in worse}}
    if not PROBE.is_file():
        block["probe"] = "缺 probes 证据，跳过交叉核对"
        return block

    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    rows = {row["qid"]: row for row in probe["rows"]}
    only_zh = set(probe["coverage"]["gold_only_in_lexical_zh_qids"])
    block.update({
        "gold_only_in_lexical_zh_qids": sorted(only_zh),
        "better_that_are_only_zh": sorted(set(better) & only_zh),
        "worse_that_are_only_zh": sorted(set(worse) & only_zh),
        "better_with_gold_already_in_dense_zh": sorted(
            q for q in better
            if "dense_zh" in (rows.get(q) or {}).get("routes_with_gold", [])),
        "worse_with_gold_already_in_dense_zh": sorted(
            q for q in worse
            if "dense_zh" in (rows.get(q) or {}).get("routes_with_gold", [])),
        "route_coverage": probe["coverage"]["per_route"],
        "route_latency_ms": probe["route_latency_ms"],
    })
    return block


def main() -> int:
    control_payload = json.loads(CONTROL.read_text(encoding="utf-8"))
    treat_payload = json.loads(TREAT.read_text(encoding="utf-8"))
    control, treat = load_rows(CONTROL), load_rows(TREAT)

    qids = [qid for qid in control if qid in treat]
    positives = [qid for qid in qids if control[qid]["gold"]]
    negatives = [qid for qid in qids if not control[qid]["gold"]]

    cells: dict[str, list[str]] = {}
    for qid in qids:
        cells.setdefault(control[qid]["expected_evidence_lang"] or "negative", []).append(qid)
    types: dict[str, list[str]] = {}
    for qid in qids:
        types.setdefault(control[qid]["type"], []).append(qid)

    metrics = {"overall": {"control": control_payload["overall"],
                           "treat": treat_payload["overall"]}}

    payload = {
        "note": "B05：给三路（dense_zh / dense_en / lexical_en）再加一路中文 BM25"
                "（jieba over 父块正文）—— 同一题集 / 同一参数 / 同一重排，只差 --routes",
        "params": {
            "collection": control_payload["collection"],
            "k": control_payload["k"],
            "pool": control_payload["pool"],
            "rerank": control_payload["rerank"],
            "query_en_mode": control_payload["query_en_mode"],
            "window": control_payload["window"],
            "questions_sha256": control_payload["questions_sha256"],
                "control_routes": control_payload["routes"],
            "treat_routes": treat_payload["routes"],
        },
        "caveat": "两次运行同一天、同一台机器、同一进程参数；控制组已复现正式基线"
                  "（四条指标逐位相同、逐题 0 好 0 差 175 平）",
        "aa_noise_floor": AA_NOISE,
        "overall": {
            "n": metrics["overall"]["control"].get("n"),
            "positives": metrics["overall"]["control"].get("positives"),
            "metrics": {key: {"control": metrics["overall"]["control"][key],
                              "treat": metrics["overall"]["treat"][key],
                              "delta": metrics["overall"]["treat"][key]
                                       - metrics["overall"]["control"][key]}
                        for key in ("hit@10", "full@10", "recall@10", "mrr")},
            "per_question": compare(control, treat, positives),
            "bootstrap_hit": paired_bootstrap(control, treat, positives),
            "negatives": len(negatives),
        },
        "by_evidence_lang": {
            name: {
                "n": len(qids_in_cell),
                "control": {k: v for k, v in control_payload["by_evidence_lang"]
                            .get(name, {}).items() if k != "n"},
                "treat": {k: v for k, v in treat_payload["by_evidence_lang"]
                          .get(name, {}).items() if k != "n"},
                "per_question": compare(control, treat, qids_in_cell),
            }
            for name, qids_in_cell in sorted(cells.items()) if name != "negative"
        },
        "by_type": {
            name: {
                "n": len(qids_in_cell),
                "control_hit": control_payload["by_type"].get(name, {}).get("hit@10"),
                "treat_hit": treat_payload["by_type"].get(name, {}).get("hit@10"),
                "per_question": compare(control, treat, qids_in_cell),
            }
            for name, qids_in_cell in sorted(types.items()) if name != "negative"
        },
        "latency_ms": {
            "control": {k: v for k, v in control_payload["latency_ms"].items()},
            "treat": {k: v for k, v in treat_payload["latency_ms"].items()},
        },
    }
    payload["mechanism"] = _mechanism(control, treat, positives)

    payload["latency_ms"]["note"] = (
        "端到端 p50 受重排与翻译主导，单看它判不出「多一路要多少钱」；"
        "路由级耗时见 probe_b05_route_contribution.py 的 route_latency_ms"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall = payload["overall"]
    print(f"正样本 {overall['positives']} ｜控制组 {control_payload['routes']} → "
          f"对照组 {treat_payload['routes']}")
    for key, value in overall["metrics"].items():
        print(f"  {key:10} {value['control']:.4f} → {value['treat']:.4f}  "
              f"delta {value['delta']:+.4f}")
    per = overall["per_question"]
    print(f"  逐题：更好 {per['better']} · 更差 {per['worse']} · 持平 {per['same']}")
    boot = overall["bootstrap_hit"]
    print(f"  配对 bootstrap（hit）：均值 {boot['mean']:+.4f} "
          f"CI95 [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}] "
          f"P(≤0)={boot['p_non_positive']}")
    print(f"  A/A 噪声底：{AA_NOISE['better']} 好 / {AA_NOISE['worse']} 差 / "
          f"{AA_NOISE['same']} 平（≈±{AA_NOISE['hit_delta_points']} 点 hit）")
    for name, cell in payload["by_evidence_lang"].items():
        c = cell["per_question"]
        print(f"  [{name}] n={cell['n']} hit {cell['control'].get('hit@10')} → "
              f"{cell['treat'].get('hit@10')}｜{c['better']} 好 {c['worse']} 差 {c['same']} 平")
    print(f"证据：{OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
