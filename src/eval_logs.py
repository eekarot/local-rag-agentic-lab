"""从 raglog 的 JSONL 里读回检索结果，算 L1 指标。

为什么不直接接受「内存里的 hits」：设计文档第 8 章的契约是**指标只从日志算**。
这样同一份日志可以反复用不同 gold / 不同切片 / 不同公式重算，不需要重跑 embedding，
也让「报告里的数字」和「review 时看到的日志」永远是同一份证据。

日志形态（见 src/raglog.py）：一个 trace 一行 `retrieve` 事件，`hits[]` 带 rank 与 chunk id。
本模块只依赖这些字段，不关心上层怎么调检索。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .eval_metrics import (
    DEFAULT_BOOT,
    DEFAULT_SEED,
    _percentile,
    bootstrap_ci,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    recall_fraction,
)


# ---------------------------------------------------------------- 读取

def latest_run(log_dir) -> str | None:
    """从 runs.jsonl 取最后一个 run_id。"""
    manifest = Path(log_dir) / "runs.jsonl"
    if not manifest.exists():
        return None
    last = None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "run" and payload.get("run_id"):
            last = payload["run_id"]
    return last


def load_traces(path) -> list[dict]:
    """把一个 run 的 JSONL 解析成 trace 列表（只保留有检索事件的 trace）。"""
    path = Path(path)
    traces: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        trace_id = event.get("trace_id")
        if not trace_id:
            continue
        trace = traces.setdefault(trace_id, {"trace_id": trace_id, "qid": None, "mode": None, "k": None})

        if event.get("qid") and not trace["qid"]:
            trace["qid"] = event["qid"]
        if event.get("mode") and not trace["mode"]:
            trace["mode"] = event["mode"]

        kind = event.get("event")
        if kind == "retrieve":
            hits = event.get("hits")
            if hits:
                ordered = sorted(hits, key=lambda item: item.get("rank") or 0)
                trace["hits"] = [item.get("id") for item in ordered]
                trace["sources"] = [item.get("source") for item in ordered]
            else:
                trace["hits"] = list(event.get("hit_ids") or [])
            trace["k"] = event.get("k") or trace["k"]
            trace["latency_ms"] = event.get("latency_ms")
            trace["scores"] = event.get("scores") or {}
            trace["branch"] = event.get("branch") or {}
            trace["n_unique_sources"] = event.get("n_unique_sources")
            trace["retrieve_events"] = trace.get("retrieve_events", 0) + 1
        elif kind == "trace":
            trace["status"] = event.get("status") or "ok"
            trace["latency_ms_total"] = event.get("latency_ms")
            trace["mode"] = event.get("mode") or trace["mode"]
            trace["k"] = event.get("k") or trace["k"]
        elif kind == "generate":
            trace["answer"] = {
                "cited": event.get("cited"),
                "abstained": event.get("abstained"),
            }

    return [trace for trace in traces.values() if trace.get("hits") and trace.get("qid")]


def _cos_z(trace):
    """把 top1 余弦换成「相对全库的 z 分数」。

    绝对余弦不可跨 embedding 模型比较（换 bge-m3 后负样本 top1 从 0.682 降到 0.619，
    但全库均值和 std 也同时变了）；z 分数才说得清「这一条到底有多离群」。
    旧日志没有 cos_mean / cos_std，返回 None，由上层降级处理。
    """
    vector = (trace.get("branch") or {}).get("vector") or {}
    top1, mean, std = vector.get("top1_cos"), vector.get("cos_mean"), vector.get("cos_std")
    if top1 is None or mean is None or not std:
        return None
    return (top1 - mean) / std


# ---------------------------------------------------------------- 打分

def score_traces(traces, questions, ks=(4, 10)) -> dict:
    """逐 trace 算指标，并按 positive / negative / unmatched 分组。

    负样本没有 gold，不参与 Recall；它们单独返回，供 L2 的弃答判定使用，
    这里只保留 `first_rank=None` 与检索耗时，避免在 L1 里混入无法解释的数字。
    """
    by_qid = {item["qid"]: item for item in questions}
    positives, negatives, unmatched = [], [], []

    for trace in traces:
        item = by_qid.get(trace.get("qid"))
        if item is None:
            unmatched.append(trace.get("qid"))
            continue
        gold = {chunk["id"] for chunk in item.get("gold_chunks") or []}
        hits = trace.get("hits") or []
        row = {
            "qid": item["qid"],
            "mode": trace.get("mode"),
            "type": item.get("type"),
            "lang": item.get("corpus_lang") or item.get("lang"),
            "k_logged": trace.get("k"),
            "first_rank": next((r for r, doc in enumerate(hits, start=1) if doc in gold), None),
            "latency_ms": trace.get("latency_ms"),
            "status": trace.get("status", "ok"),
            "n_unique_sources": trace.get("n_unique_sources"),
            # branch.vector 在纯 bm25 模式下是 null（键存在、值为 None），
            # 所以不能写 .get("vector", {})，必须再 or {}
            "top1_cos": ((trace.get("branch") or {}).get("vector") or {}).get("top1_cos"),
            "top1_cos_z": _cos_z(trace),
        }
        if item.get("type") == "negative" or not gold:
            row["hit"] = {k: False for k in ks}
            row["full"] = {k: False for k in ks}
            row["recall"] = {k: 0.0 for k in ks}
            row["mrr"] = {k: 0.0 for k in ks}
            row["ndcg"] = {k: 0.0 for k in ks}
            negatives.append(row)
            continue
        row["hit"] = {k: recall_at_k(hits, gold, k)[0] for k in ks}
        row["full"] = {k: recall_at_k(hits, gold, k)[1] for k in ks}
        row["recall"] = {k: recall_fraction(hits, gold, k) for k in ks}
        row["mrr"] = {k: mrr_at_k(hits, gold, k) for k in ks}
        row["ndcg"] = {k: ndcg_at_k(hits, gold, k) for k in ks}
        positives.append(row)

    return {
        "positives": positives,
        "negatives": negatives,
        "unmatched": sorted({qid for qid in unmatched if qid}),
    }


# ---------------------------------------------------------------- 聚合

def _metric_block(rows, ks, n_boot, seed) -> dict:
    block = {"n": len(rows)}
    for name in ("hit", "full"):
        block[name] = {
            k: bootstrap_ci([1.0 if row[name][k] else 0.0 for row in rows], n_boot=n_boot, seed=seed)
            for k in ks
        }
    for name in ("recall", "mrr", "ndcg"):
        block[name] = {
            k: bootstrap_ci([row[name][k] for row in rows], n_boot=n_boot, seed=seed) for k in ks
        }
    latencies = sorted(row["latency_ms"] for row in rows if row.get("latency_ms") is not None)
    block["latency_ms"] = {
        "n": len(latencies),
        "p50": _percentile(latencies, 0.5),
        "p95": _percentile(latencies, 0.95),
        "max": latencies[-1] if latencies else 0.0,
    }
    return block


def _slice(rows, ks, n_boot, seed) -> dict:
    return _metric_block(rows, ks, n_boot, seed) if rows else {"n": 0}


def aggregate(scored, ks=(4, 10), n_boot=DEFAULT_BOOT, seed=DEFAULT_SEED) -> dict:
    """按模式 / 题型 / 语种切片汇总。by_type 与 by_lang 是跨模式汇总（看难度分布）。"""
    rows = scored["positives"]
    by_mode = defaultdict(list)
    by_type = defaultdict(list)
    by_lang = defaultdict(list)
    by_mode_type = defaultdict(lambda: defaultdict(list))
    by_mode_lang = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_mode[row["mode"]].append(row)
        by_type[row["type"]].append(row)
        by_lang[row["lang"]].append(row)
        by_mode_type[row["mode"]][row["type"]].append(row)
        by_mode_lang[row["mode"]][row["lang"]].append(row)

    return {
        "n_questions": len({row["qid"] for row in rows}),
        "n_traces": len(rows),
        "n_negatives": len(scored["negatives"]),
        "n_negative_questions": len({row["qid"] for row in scored["negatives"]}),
        "by_mode": {mode: _slice(group, ks, n_boot, seed) for mode, group in by_mode.items()},
        "by_type": {name: _slice(group, ks, n_boot, seed) for name, group in by_type.items()},
        "by_lang": {name: _slice(group, ks, n_boot, seed) for name, group in by_lang.items()},
        "by_mode_type": {
            mode: {name: _slice(group, ks, n_boot, seed) for name, group in groups.items()}
            for mode, groups in by_mode_type.items()
        },
        "by_mode_lang": {
            mode: {name: _slice(group, ks, n_boot, seed) for name, group in groups.items()}
            for mode, groups in by_mode_lang.items()
        },
    }


def per_question_matrix(scored, metric="hit", k=4) -> dict[str, dict[str, float]]:
    """{qid: {mode: 0/1}}：配对比较（McNemar）与逐题 diff 用。"""
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for row in scored["positives"]:
        value = row[metric][k]
        matrix[row["qid"]][row["mode"]] = float(value) if not isinstance(value, bool) else float(value)
    return dict(matrix)
