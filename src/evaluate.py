"""L1（检索层）评估：run 落日志 → measure 从日志算 → compare 门禁。

契约见 docs/eval-and-logging-design.md 第 8 章：**指标只从日志算**。
所以本模块分成两段，`measure` 完全不调用模型：

    python -m src.evaluate run                      # 60 题 × 4 模式，落 raglog
    python -m src.evaluate measure latest           # 从日志算指标 + 生成报告
    python -m src.evaluate measure latest --write-baseline
    python -m src.evaluate compare latest           # 回归判定（跌出 CI 半宽 → exit 1）
    python -m src.evaluate list

`run` 会把 60 条 query 的向量一次性批量预计算（1 次 /api/embed 调用），
否则 240 次检索要重复 240 次 embedding（实测 290 s → 2 s）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import src.retrieve as retrieve_module
from . import raglog
from .config import QUESTIONS_PATH, TOP_K
from .eval_compare import compare_modes, diff_baseline
from .eval_logs import aggregate, latest_run, load_traces, score_traces
from .eval_metrics import DEFAULT_BOOT, paired_bootstrap
from .ollama_client import embed_texts
from .raglog_schema import NEG_SEPARATION_MIN, NEG_Z_ALERT
from .retrieve import load_index, retrieve

MODES = ["vector", "bm25", "hybrid", "hybrid_score"]
BASELINE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "baseline.json"
REPORT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "reports"


# ---------------------------------------------------------------- 工具

def load_questions(path=QUESTIONS_PATH):
    if not Path(path).exists():
        raise SystemExit(f"找不到评估集：{path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_hash(path=QUESTIONS_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_ks(text):
    return tuple(int(x) for x in str(text).replace(" ", "").split(",") if x)


@contextmanager
def batched_embedder(queries):
    """把逐条 embedding 换成查表：`retrieve()` 内部调用的是模块级 embed_texts。"""
    unique = list(dict.fromkeys(queries))
    started = time.perf_counter()
    vectors = embed_texts(unique)
    elapsed = (time.perf_counter() - started) * 1000
    cache = dict(zip(unique, vectors))
    print(f"  预热 query 向量：{len(unique)} 条 / {elapsed:.0f} ms（1 次 /api/embed）")

    original = retrieve_module.embed_texts

    def cached(texts):
        return [cache[text] for text in texts]

    retrieve_module.embed_texts = cached
    try:
        yield
    finally:
        retrieve_module.embed_texts = original


def _resolve_log_path(target) -> Path:
    """支持 `latest` / run_id / 完整路径。"""
    if target and Path(target).exists():
        return Path(target)
    log_dir = raglog.log_dir()
    if not target or target in ("latest", "auto"):
        run_id = latest_run(log_dir)
        if not run_id:
            raise SystemExit(f"{log_dir} 下没有 run 记录，先跑 `python -m src.evaluate run`")
        target = run_id
    run_id = target
    day = f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
    path = log_dir / day / f"{run_id}.jsonl"
    if not path.exists():
        raise SystemExit(f"找不到日志：{path}")
    return path


# ---------------------------------------------------------------- run

def cmd_run(args):
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    records = load_index()["records"]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"未知模式：{unknown}（可用：{MODES}）")

    retrieve_module.warmup()  # jieba 首次分词约 800 ms，别让它落进延迟统计
    print(f"评估集 {len(questions)} 题 × {len(modes)} 模式，k={args.k}；日志目录 {raglog.log_dir()}")
    raglog.reset_run()
    started = time.perf_counter()
    done = 0
    written = 0
    with batched_embedder([item["question"] for item in questions]):
        for item in questions:
            for mode in modes:
                for _ in range(args.repeat):
                    stats = {}
                    with raglog.trace(
                        query=item["question"],
                        qid=item["qid"],
                        mode=mode,
                        k=args.k,
                        meta={"stage": "l1", "qtype": item.get("type")},
                    ) as tracer:
                        t0 = time.perf_counter()
                        hits = retrieve(item["question"], mode=mode, k=args.k, records=records, stats=stats)
                        latency = (time.perf_counter() - t0) * 1000
                        tracer.retrieve(hits, query=item["question"], stats=stats, latency_ms=latency)
                        written += 1
            done += 1
            if done % 10 == 0 or done == len(questions):
                print(f"  {done}/{len(questions)} 题完成（{time.perf_counter() - started:.1f} s）")
    print(f"完成：{done} 题 / {written} 条 trace，用时 {time.perf_counter() - started:.1f} s")
    print(f"日志：{raglog.relative_path(raglog.run_path())}")
    print("下一步：python -m src.evaluate measure latest")
    return 0


# ---------------------------------------------------------------- measure

def _fmt(block, key, k):
    value = (block.get(key) or {}).get(k)
    if not value:
        return "—"
    return f"{value['mean']:.3f} [{value['lo']:.3f}, {value['hi']:.3f}]"


def _pct(block, key, k):
    value = (block.get(key) or {}).get(k)
    return f"{value['mean']:.1%}" if value else "—"


def _pct_ci(block, key, k):
    """点估计 ± 95% CI 半宽（百分点），一眼看出这点差异是不是噪声。"""
    value = (block.get(key) or {}).get(k)
    if not value:
        return "—"
    return f"{value['mean']:.1%}±{value['half_width'] * 100:.1f}"


def render_markdown(payload) -> str:
    metrics = payload["metrics"]
    ks = payload["ks"]
    lines = []
    lines.append(f"# L1 检索评估：`{payload['run_id']}`\n")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 日志 | `{payload['log_file']}` |")
    lines.append(f"| content_hash | `{payload['content_hash']}` |")
    lines.append(f"| 题目集 sha256 | `{payload['dataset_sha256'][:16]}` |")
    lines.append(
        f"| 题目数 | 正样本 {metrics['n_questions']} 题（算 Recall）"
        f" + 负样本 {metrics.get('n_negative_questions', 0)} 题（不计入） |"
    )
    lines.append(f"| trace 数 | {metrics['n_traces']} |")
    if payload.get("unmatched"):
        lines.append(f"| 未匹配 qid | {', '.join(payload['unmatched'][:6])}{' …' if len(payload['unmatched']) > 6 else ''} |")
    lines.append("")

    lines.append("## 按模式\n")
    lines.append(
        "| 模式 | n | "
        + " | ".join(f"Recall@{k}" for k in ks)
        + " | "
        + " | ".join(f"hit@{k}" for k in ks)
        + " | "
        + " | ".join(f"full@{k}" for k in ks)
        + " | MRR@10 | nDCG@10 | 检索 p50 / p95 (ms) |"
    )
    lines.append("|---" * (4 + 3 * len(ks)) + "|")
    for mode, block in metrics["by_mode"].items():
        lat = block.get("latency_ms") or {}
        lines.append(
            f"| {mode} | {block['n']} | "
            + " | ".join(_pct_ci(block, "recall", k) for k in ks)
            + " | "
            + " | ".join(_pct_ci(block, "hit", k) for k in ks)
            + " | "
            + " | ".join(_pct_ci(block, "full", k) for k in ks)
            + f" | {_fmt(block, 'mrr', 10)} | {_fmt(block, 'ndcg', 10)} | "
            + f"{lat.get('p50', 0):.0f} / {lat.get('p95', 0):.0f} |"
        )
    lines.append("")
    lines.append("> `Recall@k` = |G∩R_k|/|G|（主指标）；`hit` = 命中任一 gold（宽松）；`full` = 覆盖全部 gold（严格）。")
    lines.append("> chunk 重叠会让同一句原文落在 2–3 个 chunk 里，所以 `full` 天然低于 `hit`，两者不能混着比。\n")

    main_mode = payload.get("main_mode")
    if main_mode and metrics.get("by_mode_type", {}).get(main_mode):
        lines.append(f"## `{main_mode}` 按题型 / 语料\n")
        lines.append("| 切片 | n | " + " | ".join(f"Recall@{k}" for k in ks) + " | MRR@10 |")
        lines.append("|---" * (3 + len(ks)) + "|")
        for qtype, block in metrics["by_mode_type"][main_mode].items():
            lines.append(
                f"| type={qtype} | {block['n']} | "
                + " | ".join(_pct_ci(block, "recall", k) for k in ks)
                + f" | {_fmt(block, 'mrr', 10)} |"
            )
        for lang, block in (metrics.get("by_mode_lang", {}).get(main_mode) or {}).items():
            label = {"zh": "中文语料", "en": "英文语料（中文提问）"}.get(lang, lang)
            lines.append(
                f"| corpus={label} | {block['n']} | "
                + " | ".join(_pct_ci(block, "recall", k) for k in ks)
                + f" | {_fmt(block, 'mrr', 10)} |"
            )
        lines.append("")

    paired = payload.get("paired") or {}
    if paired:
        lines.append(f"## 模式配对比较（McNemar，hit@{payload.get('paired_k')}）\n")
        lines.append("| A | B | n | A 独中 | B 独中 | 都中 | 都不中 | A | B | p |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for pair_key, row in paired.items():
            mode_a, mode_b = pair_key.split("~")
            flag = "**显著**" if row["p_value"] < 0.05 else "不显著"
            lines.append(
                f"| {mode_a} | {mode_b} | {row['n']} | {row['a_only']} | {row['b_only']} | "
                f"{row['both']} | {row['neither']} | {row['mean_a']:.1%} | {row['mean_b']:.1%} | "
                f"{row['p_value']:.4f} {flag} |"
            )
        lines.append("")
        lines.append("> 配对检验只看「一成一败」的题：都不中说明这题当前检索全军覆没，不计入模式差异。\n")

    prev = payload.get("deltas_vs_prev") or {}
    if prev.get("by_mode"):
        lines.append(f"## 与上一份日志的逐题配对（hit@{payload.get('paired_k')}）\n")
        lines.append(f"上一份 run：`{prev['previous_run']}`\n")
        lines.append("| 模式 | n | 本次 − 上次 | 95% CI | p |")
        lines.append("|---|---|---|---|---|")
        for mode, row in prev["by_mode"].items():
            lines.append(
                f"| {mode} | {row['n']} | {row['diff']:+.3f} | "
                f"[{row['lo']:+.3f}, {row['hi']:+.3f}] | {row['p_value']:.4f} |"
            )
        lines.append("")
        lines.append("> 只有同一批题、同一模式的逐题差值才有意义；CI 跨 0 表示这次改动没测出效果。\n")


    negatives = payload.get("negatives") or {}
    if negatives:
        lines.append("## 负样本（不计 Recall，供 L2 用）\n")
        if negatives.get("n_with_z"):
            lines.append(
                f"- top1 相对全库的 z 分数 p50 / max："
                f"**{negatives['top1_z_p50']:+.2f} / {negatives['top1_z_max']:+.2f}**"
                f"（≥{negatives['z_alert']:.0f} 的占 {negatives['share_z_ge_alert']:.1%}）"
            )
        else:
            lines.append(
                "- ⚠️ 日志缺少 `cos_mean` / `cos_std`（旧版日志），无法计算 z 分数；"
                "请重跑 `python -m src.evaluate run`。"
            )
        if negatives.get("top1_cos_p50") is not None:
            lines.append(
                f"- top1 绝对余弦 p50 / max：{negatives['top1_cos_p50']:.3f} / "
                f"{negatives['top1_cos_max']:.3f}（**仅同一模型内可比**，换模型后尺度会变）"
            )
        if negatives.get("separation") is not None:
            verdict = "可以" if negatives["separable"] else "不可以"
            lines.append(
                f"- 正样本 top1 z 中位数 {negatives['positives_top1_z_p50']:+.2f} − 负样本 "
                f"{negatives['top1_z_p50']:+.2f} = **{negatives['separation']:+.2f}**"
                f" → {verdict}用单一向量阈值做弃答判定"
            )
        lines.append("")
        lines.append("> 负样本的判定属于 L2（弃答三角），这里只记录检索层「有没有硬凑结果」。\n")

    lines.append("## 复现\n")
    lines.append("```powershell")
    lines.append(f"python -m src.evaluate run --modes {','.join(metrics['by_mode'])} --k {max(ks)}")
    lines.append(f"python -m src.evaluate measure {payload['run_id']}")
    lines.append("```")
    return "\n".join(lines) + "\n"


def _negative_summary(rows, positives=None):
    """负样本在 L1 的唯一有意义信号：检索器有多「自信」地硬凑了结果。

    检索层永远返回 k 条，所以「有没有结果」恒为 100%，不能当指标。
    关键口径是 **相对全库的 z 分数**：绝对余弦不可跨 embedding 模型比。
    实测：换 bge-m3 后负样本 top1 绝对余弦 0.682→0.619（看上去像变好），
    但 z 分数 +5.74→+5.08（实际略变差），因为全库均值也同时下移。

    `separation = 正样本 top1 z 中位数 - 负样本 top1 z 中位数`：
    只有它明显为正，才谈得上「用一个向量相似度阈值来弃答」。
    """
    if not rows:
        return {}

    cos = sorted(row["top1_cos"] for row in rows if row.get("top1_cos") is not None)
    zs = sorted(row["top1_cos_z"] for row in rows if row.get("top1_cos_z") is not None)
    pos_zs = sorted(
        row["top1_cos_z"] for row in (positives or []) if row.get("top1_cos_z") is not None
    )
    out = {
        "n": len(rows),
        "n_with_z": len(zs),
        # 绝对余弦保留，但只作为「同一模型内」的参考
        "top1_cos_p50": cos[len(cos) // 2] if cos else None,
        "top1_cos_max": cos[-1] if cos else None,
        "top1_z_p50": zs[len(zs) // 2] if zs else None,
        "top1_z_max": zs[-1] if zs else None,
        "share_z_ge_alert": (
            sum(1 for z in zs if z >= NEG_Z_ALERT) / len(zs) if zs else None
        ),
        "z_alert": NEG_Z_ALERT,
        "positives_top1_z_p50": pos_zs[len(pos_zs) // 2] if pos_zs else None,
        "separation": None,
        "separable": None,
    }
    if zs and pos_zs:
        separation = out["positives_top1_z_p50"] - out["top1_z_p50"]
        out["separation"] = separation
        out["separable"] = separation >= NEG_SEPARATION_MIN
    return out


def build_payload(args):
    questions = load_questions(args.questions)
    path = _resolve_log_path(args.run)
    traces = load_traces(path)
    if not traces:
        raise SystemExit(f"{path} 里没有可用的检索事件")

    scored = score_traces(traces, questions, ks=args.ks)
    if not scored["positives"]:
        raise SystemExit(
            f"日志中的 qid 与 {args.questions} 完全不匹配（示例：{scored['unmatched'][:3]}）"
        )
    metrics = aggregate(scored, ks=args.ks, n_boot=args.boot, seed=args.seed)
    modes = list(metrics["by_mode"])
    main_mode = args.main_mode or ("hybrid" if "hybrid" in modes else modes[0])

    run_meta = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:1]:
        try:
            run_meta = json.loads(line)
        except json.JSONDecodeError:
            pass

    return {
        "schema": "l1-metrics/1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_meta.get("run_id") or path.stem,
        "log_file": raglog.relative_path(path),
        "content_hash": run_meta.get("content_hash"),
        "index": run_meta.get("index"),
        "models": run_meta.get("models"),
        "dataset_sha256": dataset_hash(args.questions),
        "ks": list(args.ks),
        "modes": modes,
        "main_mode": main_mode,
        "boot": {"n_boot": args.boot, "seed": args.seed},
        "metrics": metrics,
        # 键必须是字符串才能写进 baseline.json
        "paired": {
            f"{a}~{b}": row for (a, b), row in compare_modes(scored, k=args.paired_k).items()
        },
        "paired_k": args.paired_k,
        "deltas_vs_prev": _prev_pair(scored, path, questions, args),
        "negatives": _negative_summary(scored["negatives"], scored["positives"]),
        "unmatched": scored["unmatched"],
    }


def _prev_pair(scored, path, questions, args):
    """和「同目录上一份日志」做逐题配对，方便看这次改动有没有实际效果。"""
    siblings = sorted(path.parent.glob("*.jsonl"))
    if len(siblings) < 2 or siblings[-1] != path:
        return {}
    prev_path = siblings[-2]
    prev_scored = score_traces(load_traces(prev_path), questions, ks=args.ks)
    matrix_now = {(r["qid"], r["mode"]): r for r in scored["positives"]}
    matrix_prev = {(r["qid"], r["mode"]): r for r in prev_scored["positives"]}
    shared = sorted(set(matrix_now) & set(matrix_prev))
    if not shared:
        return {}
    result = {}
    for mode in sorted({mode for _, mode in shared}):
        pairs = [(key, matrix_now[key], matrix_prev[key]) for key in shared if key[1] == mode]
        now = [float(now_row["hit"][args.paired_k]) for _, now_row, _ in pairs]
        before = [float(prev_row["hit"][args.paired_k]) for _, _, prev_row in pairs]
        result[mode] = paired_bootstrap(now, before, n_boot=args.boot, seed=args.seed)
    return {"previous_run": prev_path.stem, "by_mode": result}


def cmd_measure(args):
    payload = build_payload(args)
    report = render_markdown(payload)
    print(report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else REPORT_DIR / f"{payload['run_id']}.md"
    out.write_text(report, encoding="utf-8")
    print(f"报告：{out}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.write_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"基线：{BASELINE_PATH}")
    return 0


# ---------------------------------------------------------------- compare

def cmd_compare(args):
    baseline_path = Path(args.baseline) if getattr(args, "baseline", None) else BASELINE_PATH
    if not baseline_path.exists():
        raise SystemExit(f"还没有基线：{baseline_path}（先跑 measure --write-baseline）")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload = build_payload(args)
    current = {"metrics": payload["metrics"], **{k: payload[k] for k in ("content_hash", "dataset_sha256")}}
    result = diff_baseline(baseline, current)

    print(f"基线文件：{baseline_path}")
    print(f"基线 run：{baseline.get('run_id')}（{baseline.get('created_at')}）")
    print(f"当前 run：{payload['run_id']}（{payload['created_at']}）")
    for note in result["notes"]:
        print(f"  [注意] {note}")
    print(f"\n{'切片':34} {'指标':6} {'k':>3} {'基线':>8} {'当前':>8} {'差值':>8} {'容差':>8}  判定")
    for row in result["deltas"]:
        if row["metric"] not in ("recall", "hit", "full", "mrr"):
            continue
        if result["verdict"] == "not_comparable":
            # 指纹/题集不同时 diff_baseline 不判回归，这里也不能把大幅差值写成「噪声内」
            verdict = "不可比"
        elif row["regressed"]:
            verdict = "回归"
        elif row["improved"]:
            verdict = "改善"
        else:
            verdict = "噪声内"
        print(
            f"{row['slice'][:34]:34} {row['metric']:6} {str(row['k']):>3} "
            f"{row['base']:>8.3f} {row['now']:>8.3f} {row['delta']:>+8.3f} {row['tolerance']:>8.3f}  {verdict}"
        )
    print(f"\n结论：{result['verdict']}")
    if result["verdict"] == "regression":
        print("存在超出 CI 半宽的跌幅。")
        return 1
    return 0


def cmd_list(args):
    log_dir = raglog.log_dir()
    manifest = log_dir / "runs.jsonl"
    if not manifest.exists():
        print(f"{log_dir} 下还没有 run")
        return 0
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "run":
            rows.append(payload)
    for payload in rows[-args.limit:]:
        print(
            f"{payload['run_id']}  {payload.get('started_at', '')[:19]}  "
            f"hash={payload.get('content_hash')}  {payload.get('file')}"
        )
    print(f"\n共 {len(rows)} 个 run；logs/runs.jsonl")
    return 0


# ---------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="L1 检索评估：run（落日志）/ measure（从日志算）/ compare（门禁）/ list",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="跑检索并把结果写进 raglog")
    run_parser.add_argument("--modes", default=",".join(MODES))
    run_parser.add_argument("--k", type=int, default=max(TOP_K, 10))
    run_parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    run_parser.add_argument("--repeat", type=int, default=1, help="每题重复次数（测延迟抖动用）")
    run_parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    run_parser.set_defaults(func=cmd_run)

    measure_parser = sub.add_parser("measure", help="从日志算指标并出报告")
    measure_parser.add_argument("run", nargs="?", default="latest", help="run_id / latest / 日志路径")
    measure_parser.add_argument("--ks", default="4,10")
    measure_parser.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    measure_parser.add_argument("--seed", type=int, default=0)
    measure_parser.add_argument("--main-mode", default=None, help="报告里做题型切片的模式")
    measure_parser.add_argument("--paired-k", type=int, default=4)
    measure_parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    measure_parser.add_argument("--out", default=None)
    measure_parser.add_argument("--json", action="store_true")
    measure_parser.add_argument("--write-baseline", action="store_true")
    measure_parser.set_defaults(func=cmd_measure)

    compare_parser = sub.add_parser("compare", help="与基线对比（跌出 CI 半宽 → exit 1）")
    compare_parser.add_argument("run", nargs="?", default="latest")
    compare_parser.add_argument(
        "--baseline", default=None,
        help="指定基线 JSON（默认 experiments/baseline.json）；换 embedding 模型后用它对比旧基线",
    )
    compare_parser.add_argument("--ks", default="4,10")
    compare_parser.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    compare_parser.add_argument("--seed", type=int, default=0)
    compare_parser.add_argument("--main-mode", default=None)
    compare_parser.add_argument("--paired-k", type=int, default=4)
    compare_parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    compare_parser.set_defaults(func=cmd_compare)

    list_parser = sub.add_parser("list", help="列出历史 run")
    list_parser.add_argument("--limit", type=int, default=10)
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        print("\n提示：旧的「关键词覆盖率」评估已废弃，见 docs/eval-and-logging-design.md 4.1")
        return 0
    if args.command in ("measure", "compare") and not isinstance(args.ks, tuple):
        args.ks = parse_ks(args.ks)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
