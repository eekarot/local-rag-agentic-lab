"""基线对比（回归门禁）与模式间配对比较。

两条原则：

1. **不同 `content_hash` 不做门禁**：索引/模型/参数变了，指标本来就会漂，此时报警是噪声。
2. **判回归用 CI 半宽而不是固定阈值**：60 题上 2 个百分点的跌幅完全可能是抽样抖动，
   固定阈值（比如「跌 1% 就报警」）只会让人习惯性忽略告警。
"""

from __future__ import annotations

from .eval_logs import per_question_matrix
from .eval_metrics import mcnemar_exact

METRIC_NAMES = ("recall", "hit", "full", "mrr", "ndcg")


def runs_comparable(prev_meta, now_meta) -> tuple[bool, str]:
    """两次 run 的逐题配对是否成立。返回 (可比, 原因)。

    只看题集还不够：**换索引/换检索链路**后「上一份日志」和「这一份」根本不是同一个
    检索器，逐题差值（`+0.578`）会被误读成「改进了 57.8 个点」。实测就是踩到这条：
    新库基线 vs 旧 81k 索引的日志，hybrid 差值 +0.578、CI 全在 0 以上。
    """
    if prev_meta.get("content_hash") != now_meta.get("content_hash"):
        return False, "content_hash 不同（索引/模型/参数变了）"
    if prev_meta.get("dataset_sha256") != now_meta.get("dataset_sha256"):
        return False, "题目集不同"
    return True, ""


def _iter_metrics(metrics, metric_names=METRIC_NAMES):
    """展开 `{slice_kind: {slice_name: block}}` 里的标量指标。"""
    for slice_kind in ("by_mode", "by_type", "by_lang", "by_mode_type", "by_mode_lang"):
        block = metrics.get(slice_kind) or {}
        for slice_name, values in block.items():
            if slice_kind in ("by_mode_type", "by_mode_lang"):
                for inner_name, inner in values.items():
                    yield slice_kind, f"{slice_name}|{inner_name}", inner
            else:
                yield slice_kind, slice_name, values


def diff_baseline(baseline, current, metric_names=METRIC_NAMES) -> dict:
    """对比两份指标快照，返回 `{verdict, deltas, notes}`。

    verdict 取值：`ok` / `regression` / `improved` / `not_comparable`。
    """
    notes = []
    same_content = baseline.get("content_hash") == current.get("content_hash")
    same_dataset = baseline.get("dataset_sha256") == current.get("dataset_sha256")
    if not same_content:
        notes.append(
            f"content_hash 不同（{baseline.get('content_hash')} → {current.get('content_hash')}）："
            "索引/模型/参数已变，不做回归判定"
        )
    if not same_dataset:
        notes.append(
            f"题目集不同（{str(baseline.get('dataset_sha256'))[:12]} → "
            f"{str(current.get('dataset_sha256'))[:12]}）：不做回归判定"
        )

    deltas = []
    missing = []
    for (kind, name, base_block), (_, _, now_block) in zip(
        _iter_metrics(baseline.get("metrics") or {}, metric_names),
        _iter_metrics(current.get("metrics") or {}, metric_names),
    ):
        for metric in metric_names:
            # JSON 往返会把 int 键变成 str，两边先归一到 str 再配对
            base_ks = {str(key): value for key, value in (base_block.get(metric) or {}).items()}
            now_ks = {str(key): value for key, value in (now_block.get(metric) or {}).items()}
            for k in sorted(set(base_ks) | set(now_ks), key=lambda value: float(value)):
                base_value = base_ks.get(k)
                now_value = now_ks.get(k)
                if base_value is None or now_value is None:
                    missing.append(f"{kind}.{name}.{metric}@{k}")
                    continue
                delta = now_value["mean"] - base_value["mean"]
                tolerance = base_value.get("half_width") or 0.0
                comparable = same_content and same_dataset
                deltas.append({
                    "slice": f"{kind}.{name}",
                    "metric": metric,
                    "k": k,
                    "base": base_value["mean"],
                    "now": now_value["mean"],
                    "delta": delta,
                    "tolerance": tolerance,
                    "regressed": comparable and delta < -tolerance,
                    "improved": comparable and delta > tolerance,
                    "ci_overlap": not (now_value["hi"] < base_value["lo"] or now_value["lo"] > base_value["hi"]),
                })

    if missing:
        notes.append(f"缺失 {len(missing)} 项（{', '.join(missing[:3])}…），无法对比")

    if not same_content or not same_dataset or missing:
        verdict = "not_comparable"
    elif any(row["regressed"] for row in deltas):
        verdict = "regression"
    elif any(row["improved"] for row in deltas):
        verdict = "improved"
    else:
        verdict = "ok"

    return {"verdict": verdict, "deltas": deltas, "notes": notes, "missing": missing}


def compare_modes(scored, k=4, metric="hit") -> dict:
    """同一批题下两两模式的命中差异（McNemar 精确检验）。

    不看「各自区间是否重叠」——那在配对数少时几乎永远重叠，会得出「全都一样」的错觉。
    """
    matrix = per_question_matrix(scored, metric=metric, k=k)
    order = []
    for row in scored["positives"]:
        if row["mode"] not in order:
            order.append(row["mode"])
    modes = [mode for mode in order if any(mode in row for row in matrix.values())]
    results = {}
    for index, mode_a in enumerate(modes):
        for mode_b in modes[index + 1:]:
            a_only = b_only = both = neither = 0
            for row in matrix.values():
                if mode_a not in row or mode_b not in row:
                    continue
                got_a = bool(row[mode_a])
                got_b = bool(row[mode_b])
                if got_a and got_b:
                    both += 1
                elif got_a:
                    a_only += 1
                elif got_b:
                    b_only += 1
                else:
                    neither += 1
            n = a_only + b_only + both + neither
            stats = mcnemar_exact(a_only, b_only)
            results[(mode_a, mode_b)] = {
                "n": n,
                "a_only": a_only,
                "b_only": b_only,
                "both": both,
                "neither": neither,
                "p_value": stats["p_value"],
                "mean_a": (a_only + both) / n if n else 0.0,
                "mean_b": (b_only + both) / n if n else 0.0,
            }
    return results
