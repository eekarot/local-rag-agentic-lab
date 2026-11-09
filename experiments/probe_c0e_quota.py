"""C0e / C0h 的结果汇总器：把 `eval_bench --out` 的多次运行拼成一张证据表。

先跑出各配置（每条约 4 分钟，结果落 `storage/c0e/`，不入库）：

    python scripts/eval_bench.py --k 10 --pool 40 --query-en translate --lang-quota 3 \
        --out storage/c0e/q3.json
    python scripts/eval_bench.py --k 10 --pool 40 --query-en translate \
        --rerank-query by_lang --out storage/c0e/by_lang_raw.json
    python scripts/eval_bench.py --k 10 --pool 40 --query-en translate \
        --rerank-query by_lang --lang-quota 3 --out storage/c0e/bl_raw_q3.json
    python scripts/eval_bench.py --k 10 --pool 40 --query-en translate \
        --rerank-query by_lang --rerank-merge zscore --out storage/c0e/by_lang_z.json
    python scripts/eval_bench.py --k 10 --pool 40 --query-en translate \
        --rerank-query by_lang --rerank-merge zscore --lang-quota 3 --out storage/c0e/by_lang_z_q3.json
    python scripts/eval_bench.py --k 10 --pool 40 --query-en translate \
        --rerank-query by_lang --rerank-merge rank --out storage/c0e/by_lang_r.json
    python scripts/eval_bench.py --k 10 --pool 40 --query-en translate \
        --rerank-query by_lang --rerank-merge rank --lang-quota 3 --out storage/c0e/bl_rank_q3.json
    python scripts/eval_bench.py --k 10 --pool 40 --query-en translate \
        --rerank-query by_lang --rerank-merge zscore --max-other-lang 4 --out storage/c0e/z_cap4.json

再汇总：

    python experiments/probe_c0e_quota.py

结论（写死在 `findings` 里，防止以后有人只看总均值）：配额在这份题集上**从不触发**，
by_lang 这一族 7 种组合全部低于基线 —— 详见 `docs/evidence/c0e-language-quota-k8s-v1.json`。
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS = {
    "基线（question/raw，quota 0）": "docs/baseline-k8s-v1.json",
    "question/quota3": "storage/c0e/q3.json",
    "by_lang/raw": "storage/c0e/by_lang_raw.json",
    "by_lang/raw + quota3": "storage/c0e/bl_raw_q3.json",
    "by_lang/zscore": "storage/c0e/by_lang_z.json",
    "by_lang/zscore + quota3": "storage/c0e/by_lang_z_q3.json",
    "by_lang/rank": "storage/c0e/by_lang_r.json",
    "by_lang/rank + quota3": "storage/c0e/bl_rank_q3.json",
    "by_lang/zscore + 对方语言上限4（C0h）": "storage/c0e/z_cap4.json",
    "by_lang/zscore + 对方语言上限3（C0h）": "storage/c0e/z_cap3.json",
}
CJK = re.compile(r"[\u4e00-\u9fff]")


def rows_of(path: Path) -> dict[str, dict]:
    rows = path.with_suffix(".rows.jsonl")
    if not rows.exists():
        return {}
    return {json.loads(line)["qid"]: json.loads(line)
            for line in rows.read_text(encoding="utf-8").splitlines() if line.strip()}


def cell_of(row: dict) -> str:
    zh_q = bool(CJK.search(row["question"]))
    zh_e = row.get("expected_evidence_lang") == "zh-cn"
    return ("中→中" if zh_q and zh_e else "中→英" if zh_q else "英→中" if zh_e else "英→英")


def main() -> int:
    base_path = ROOT / "docs/baseline-k8s-v1.json"
    base_rows = rows_of(base_path)
    summary: dict[str, dict] = {}
    for label, relative in RUNS.items():
        path = ROOT / relative
        if not path.exists():
            print(f"跳过 {label}（缺 {relative}，先按模块文档里的命令跑一遍）")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = rows_of(path) or {row["qid"]: row for row in data.get("rows", [])}
        if not rows:
            print(f"跳过 {label}（{relative} 里没有逐题数据）")
            continue
        cells = {}
        for cell in ("中→英", "英→中", "中→中", "英→英"):
            picked = [row for row in rows.values() if cell_of(row) == cell]
            if not picked:
                continue
            hit = sum(1 for row in picked if row["first_hit_rank"] and row["first_hit_rank"] <= 10) / len(picked)
            mrr = sum((1.0 / row["first_hit_rank"]) if row["first_hit_rank"] else 0.0
                      for row in picked) / len(picked)
            cells[cell] = {"n": len(picked), "hit@10": round(hit, 4), "mrr": round(mrr, 4)}
        better = worse = same = 0
        for qid, row in rows.items():
            if qid not in base_rows:
                continue
            old = base_rows[qid]["first_hit_rank"] or 10 ** 6
            new = row["first_hit_rank"] or 10 ** 6
            better += new < old
            worse += new > old
            same += new == old
        summary[label] = {
            "file": relative,
            "overall": {"hit@10": data["overall"]["hit@10"], "mrr": data["overall"]["mrr"]},
            "cells": cells,
            "vs_baseline": {"更好": better, "更差": worse, "持平": same},
            "latency_p50_ms": (data.get("latency_ms") or {}).get("p50"),
        }

    print(f"{'配置':<30}{'hit@10':>8}{'MRR':>8}  {'中→英':>13}{'英→中':>13}{'中→中':>13}"
          f"{'英→英':>13}  好/差/平")
    for label, stats in summary.items():
        def show(cell):
            item = stats["cells"].get(cell)
            return f"{item['hit@10']:.3f}/{item['mrr']:.3f}" if item else "—"

        print(f"{label:<30}{stats['overall']['hit@10']:>8.4f}{stats['overall']['mrr']:>8.4f}  "
              f"{show('中→英'):>13}{show('英→中'):>13}{show('中→中'):>13}{show('英→英'):>13}  "
              f"{stats['vs_baseline']['更好']}/{stats['vs_baseline']['更差']}/{stats['vs_baseline']['持平']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
