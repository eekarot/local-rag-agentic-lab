"""RAG 日志 review CLI：把 JSONL 日志渲染成人能读的卡片 / 失败清单 / 统计 / 两次 run 的 diff。

用法：
    python -m src.raglog_report latest --md          # 每题一张卡片
    python -m src.raglog_report latest --html        # 单文件 HTML，可点击 [证据 n] 跳转
    python -m src.raglog_report latest --failures    # 只列被标记的题
    python -m src.raglog_report latest --stats       # 延迟与分布
    python -m src.raglog_report A.jsonl --diff B.jsonl
    python -m src.raglog_report --prune 14           # 清理 14 天前的日志

设计规范见 docs/eval-and-logging-design.md 7.5 节。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .config import ROOT_DIR
from .raglog_schema import (
    CITATION_PATTERN,
    EV_ERROR,
    EV_GENERATE,
    EV_RETRIEVE,
    EV_RUN,
    EV_TRACE,
    FLAT_SCORE_RATIO,
    SOURCE_COLLAPSE_RATIO,
)

LOG_ROOT = ROOT_DIR / "logs"

FLAG_LABELS = {
    "empty": "空检索",
    "flat_scores": "分数平坦",
    "source_collapse": "来源坍缩",
    "no_citation": "零引用",
    "citation_out_of_range": "引用越界",
    "abstained": "弃答",
    "error": "有错误",
}


# ---------------------------------------------------------------- 读日志

def _read_jsonl(path):
    events = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] {path}:{line_no} 不是合法 JSON，已跳过", file=sys.stderr)
    return events


def _newest_run():
    manifest = LOG_ROOT / "runs.jsonl"
    if manifest.exists():
        for line in reversed(manifest.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate = ROOT_DIR / entry["file"] if not Path(entry["file"]).is_absolute() else Path(entry["file"])
            if candidate.exists():
                return candidate
    files = [p for p in LOG_ROOT.rglob("*.jsonl")
             if p.name != "runs.jsonl" and not p.name.endswith(".labels.jsonl")]
    if not files:
        raise SystemExit(f"没有找到任何日志：{LOG_ROOT}（先跑一次 python -m src.rag）")
    return max(files, key=lambda p: p.stat().st_mtime)


def _resolve(spec):
    if spec in (None, "", "latest"):
        return _newest_run()
    if spec == "list":
        return None
    path = Path(spec)
    if path.exists():
        return path
    manifest = LOG_ROOT / "runs.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if spec in line:
                entry = json.loads(line)
                candidate = ROOT_DIR / entry["file"] if not Path(entry["file"]).is_absolute() else Path(entry["file"])
                if candidate.exists():
                    return candidate
    raise SystemExit(f"找不到 run：{spec}")


def load_run(spec):
    path = _resolve(spec)
    events = _read_jsonl(path)
    run_event = next((e for e in events if e.get("event") == EV_RUN), {})
    groups = defaultdict(list)
    for event in events:
        if event.get("event") == EV_RUN:
            continue
        trace_id = event.get("trace_id")
        if trace_id:
            groups[trace_id].append(event)
    traces = []
    for trace_id, items in groups.items():
        traces.append(_build_trace(trace_id, items))
    return {
        "path": path,
        "run": run_event,
        "traces": traces,
        "events": events,
    }


def _build_trace(trace_id, events):
    retrieve = next((e for e in events if e.get("event") == EV_RETRIEVE), {})
    generate = next((e for e in events if e.get("event") == EV_GENERATE), {})
    closing = next((e for e in events if e.get("event") == EV_TRACE), {})
    trace = {
        "trace_id": trace_id,
        "qid": next((e.get("qid") for e in events if e.get("qid")), None),
        "mode": next((e.get("mode") for e in events if e.get("mode")), None),
        "k": next((e.get("k") for e in events if e.get("k")), None),
        "retrieve": retrieve,
        "generate": generate,
        "close": closing,
        "errors": [e for e in events if e.get("event") == EV_ERROR],
        "decisions": [e for e in events if e.get("name") == "agent.decision"],
        "spans": [e for e in events if e.get("name", "").startswith("span:")],
        "events": events,
    }
    trace["query"] = retrieve.get("query") or closing.get("query")
    trace["latency_ms"] = closing.get("latency_ms")
    return trace


def trace_flags(trace, ratio=FLAT_SCORE_RATIO, collapse=SOURCE_COLLAPSE_RATIO):
    out = []
    retrieve = trace["retrieve"] or {}
    generate = trace["generate"] or {}
    hits = retrieve.get("hits") or []
    if retrieve:
        if retrieve.get("empty"):
            out.append("empty")
        scores = retrieve.get("scores") or {}
        rule = (retrieve.get("fuse") or {}).get("rule")
        gap = scores.get("gap")
        gap_ratio = scores.get("gap_ratio")
        if gap_ratio is None and gap is not None and scores.get("top1"):
            gap_ratio = gap / abs(scores["top1"])
        # RRF 的分数就是 1/(k+rank) 累加，相邻名次天然只差 ~2.7e-4：
        # 用相对阈值会把每条 RRF 都判成平坦，所以 RRF 只在完全同分时才标记。
        if rule == "rrf":
            if gap is not None and abs(gap) < 1e-9:
                out.append("flat_scores")
        elif gap_ratio is not None and gap_ratio < ratio:
            out.append("flat_scores")
        unique = retrieve.get("n_unique_sources")
        k = trace.get("k") or len(hits)
        if unique is not None and k and not retrieve.get("empty") and unique / k < collapse:
            out.append("source_collapse")
        if trace["mode"] in ("hybrid", "hybrid_score") and hits:
            if all(h.get("rank_vector") is None and h.get("rank_bm25") is None for h in hits):
                out.append("no_branch_origin")
    if generate:
        if generate.get("citations_missing"):
            out.append("no_citation")
        if generate.get("cited_valid") is False:
            out.append("citation_out_of_range")
        if generate.get("abstained"):
            out.append("abstained")
    if trace["errors"]:
        out.append("error")
    return out


# ---------------------------------------------------------------- 工具

def _pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _text_of(event, name):
    if not event:
        return ""
    if event.get(name):
        return event[name]
    ref = event.get(f"{name}_ref")
    if ref:
        path = ROOT_DIR / ref if not Path(ref).is_absolute() else Path(ref)
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def _first_line(text, limit=120):
    line = re.sub(r"\s+", " ", text or "").strip()
    return line[:limit] + ("…" if len(line) > limit else "")


def _fmt_ms(value):
    return "—" if value is None else f"{value:.0f} ms"


def _trace_title(trace):
    label = trace["qid"] or trace["trace_id"]
    return f"{label}"


# ---------------------------------------------------------------- Markdown

def render_md(run, traces, limit=None, show_events=False):
    out = []
    meta = run["run"]
    index = meta.get("index") or {}
    out.append(f"# RAG 运行 review：`{meta.get('run_id', run['path'].stem)}`\n")
    out.append("| 项 | 值 |\n|---|---|")
    out.append(f"| 日志文件 | `{run['path'].relative_to(ROOT_DIR) if run['path'].is_relative_to(ROOT_DIR) else run['path']}` |")
    out.append(f"| content_hash | `{meta.get('content_hash')}` |")
    out.append(f"| 索引 | {index.get('n_records')} 块 / {index.get('dim')} 维 / tokenizer={index.get('tokenizer')} / chunk={index.get('chunk_size')}+{index.get('chunk_overlap')} |")
    out.append(f"| 模型 | chat=`{(meta.get('models') or {}).get('chat')}` embed=`{(meta.get('models') or {}).get('embed')}` |")
    out.append(f"| 问题数 | {len(traces)} |")
    flagged = [t for t in traces if trace_flags(t)]
    out.append(f"| 被标记 | {len(flagged)} |")
    out.append("")

    shown = traces if limit is None else traces[:limit]
    for trace in shown:
        hits = (trace["retrieve"] or {}).get("hits") or []
        generate = trace["generate"] or {}
        branch = (trace["retrieve"] or {}).get("branch") or {}
        vector = branch.get("vector") or {}
        bm25 = branch.get("bm25") or {}
        flags = trace_flags(trace)
        out.append(f"## {_trace_title(trace)} — `{trace.get('mode')}` k={trace.get('k')}\n")
        out.append(f"**问题**：{trace['query'] or '（未记录）'}\n")
        chips = " ".join(f"`{FLAG_LABELS.get(f, f)}`" for f in flags) or "`正常`"
        out.append(f"标记：{chips}\n")
        out.append("| 阶段 | 耗时 | 说明 |\n|---|---|---|")
        out.append(f"| query embedding | {_fmt_ms(vector.get('embed_ms'))} | candidates={vector.get('candidates')} |")
        out.append(f"| 向量全表打分 | {_fmt_ms(vector.get('search_ms'))} | top1_cos={vector.get('top1_cos')} |")
        out.append(f"| 检索合计 | {_fmt_ms((trace['retrieve'] or {}).get('latency_ms'))} | bm25 命中词项 {bm25.get('terms_hit')}，postings {bm25.get('postings_touched')} |")
        out.append(f"| 生成 | {_fmt_ms(generate.get('latency_ms'))} | tokens {generate.get('prompt_eval_count')}+{generate.get('eval_count')} |")
        out.append("")
        if hits:
            out.append("| # | 来源 / chunk | score | 向量排名 | BM25 排名 | cos | 首行 |")
            out.append("|---|---|---|---|---|---|---|")
            for hit in hits:
                source = f"{Path(str(hit.get('source'))).name} #{hit.get('chunk_id')}"
                out.append(
                    f"| {hit.get('rank')} | `{source}` | {hit.get('score')} | "
                    f"{hit.get('rank_vector') if hit.get('rank_vector') is not None else '—'} | "
                    f"{hit.get('rank_bm25') if hit.get('rank_bm25') is not None else '—'} | "
                    f"{hit.get('cos')} | {_first_line(hit.get('text'), 60)} |"
                )
            out.append("")
            for hit in hits:
                out.append(f"<details><summary>证据 {hit.get('rank')} 全文</summary>\n\n```text\n{hit.get('text')}\n```\n\n</details>\n")
        answer = _text_of(generate, "answer")
        if answer:
            out.append("**答案**：\n")
            out.append("```text\n" + answer + "\n```\n")
        if generate:
            cited = generate.get("cited") or []
            out.append(
                f"引用：{cited or '无'}（有效={generate.get('cited_valid')}，零引用={generate.get('citations_missing')}，"
                f"弃答={generate.get('abstained')} 启发式）\n"
            )
        if trace["errors"]:
            for error in trace["errors"]:
                out.append(f"> ❌ `{error.get('where')}`：{error.get('message')}\n")
        if show_events:
            out.append("<details><summary>原始事件</summary>\n\n```json\n"
                       + "\n".join(json.dumps(e, ensure_ascii=False) for e in trace["events"])
                       + "\n```\n\n</details>\n")
        out.append("---\n")
    if limit is not None and len(traces) > limit:
        out.append(f"（只显示前 {limit} 条，共 {len(traces)} 条；用 `--limit 0` 看全部）\n")
    return "\n".join(out)


# ---------------------------------------------------------------- HTML

HTML_STYLE = """
:root{--fg:#1d1d1f;--muted:#6b7280;--line:#e5e7eb;--bg:#fff;--accent:#2563eb;--warn:#b45309;--bad:#b91c1c}
*{box-sizing:border-box}
body{margin:0;padding:24px 32px;font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:var(--fg);background:var(--bg);max-width:1200px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:16px;margin:32px 0 8px;padding-top:16px;border-top:1px solid var(--line)}
h2 a{color:inherit;text-decoration:none}
.meta{color:var(--muted);font-size:12.5px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f9fafb;font-weight:600}
code{font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
.q{font-size:15px;font-weight:600;margin:4px 0 8px}
.chip{display:inline-block;padding:1px 8px;margin:0 4px 4px 0;border-radius:999px;font-size:12px;background:#eef2ff;color:#3730a3}
.chip.warn{background:#fef3c7;color:var(--warn)}
.chip.bad{background:#fee2e2;color:var(--bad)}
.answer{white-space:pre-wrap;background:#f9fafb;border:1px solid var(--line);border-radius:8px;padding:12px;margin:8px 0}
.answer a{color:var(--accent);text-decoration:none;font-weight:600}
details{margin:6px 0;border:1px solid var(--line);border-radius:8px;padding:6px 10px;background:#fcfcfd}
summary{cursor:pointer;color:var(--muted);font-size:13px}
details pre{white-space:pre-wrap;word-break:break-word;margin:8px 0 4px;font-size:12.5px}
.hit{scroll-margin-top:16px}
.hit:target{outline:2px solid var(--accent);border-radius:6px}
.bar{display:inline-block;height:8px;background:#c7d2fe;border-radius:4px;vertical-align:middle}
.summary{display:flex;gap:24px;flex-wrap:wrap;margin:8px 0 16px}
.summary div{font-size:12.5px;color:var(--muted)}
.summary b{display:block;font-size:18px;color:var(--fg)}
.filter{margin:12px 0;font-size:13px}
"""

HTML_SCRIPT = """
function toggle(flag){
  document.querySelectorAll('[data-flags]').forEach(function(el){
    if(!flag){el.style.display='';return;}
    el.style.display = el.dataset.flags.indexOf(flag)>=0 ? '' : 'none';
  });
}
"""


def _chip_class(flag):
    if flag in ("empty", "error", "citation_out_of_range"):
        return "chip bad"
    if flag in ("flat_scores", "source_collapse", "no_citation", "abstained"):
        return "chip warn"
    return "chip"


def _answer_html(text, trace_id, k):
    escaped = html.escape(text or "")
    pattern = re.compile(re.escape("[证据 ") + r"(\d+)\]")

    def link(match):
        number = int(match.group(1))
        if k and not 1 <= number <= k:
            return f'<span style="color:#b91c1c">[证据 {number}]</span>'
        return f'<a href="#ev-{trace_id}-{number}">[证据 {number}]</a>'

    return pattern.sub(link, escaped)


def render_html(run, traces, limit=None, title=None):
    meta = run["run"]
    index = meta.get("index") or {}
    flagged = [t for t in traces if trace_flags(t)]
    modes = Counter(t.get("mode") for t in traces)

    parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        f"<title>{html.escape(title or ('RAG review ' + str(meta.get('run_id'))))}</title>",
        f"<style>{HTML_STYLE}</style></head><body>",
        f"<h1>RAG 运行 review · {html.escape(str(meta.get('run_id', run['path'].stem)))}</h1>",
        f"<div class='meta'>日志：<code>{html.escape(str(run['path']))}</code> · "
        f"content_hash <code>{html.escape(str(meta.get('content_hash')))}</code></div>",
        f"<div class='meta'>索引：{index.get('n_records')} 块 / {index.get('dim')} 维 / "
        f"tokenizer={html.escape(str(index.get('tokenizer')))} / chunk={index.get('chunk_size')}+{index.get('chunk_overlap')} · "
        f"模型：chat=<code>{html.escape(str((meta.get('models') or {}).get('chat')))}</code> "
        f"embed=<code>{html.escape(str((meta.get('models') or {}).get('embed')))}</code></div>",
        "<div class='summary'>",
        f"<div><b>{len(traces)}</b>traces</div>",
        f"<div><b>{len(flagged)}</b>被标记</div>",
    ]
    for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
        parts.append(f"<div><b>{count}</b>{html.escape(str(mode))}</div>")
    parts.append("</div>")

    if flagged:
        parts.append("<div class='filter'><b>被标记的题：</b> ")
        for trace in flagged:
            flags = trace_flags(trace)
            parts.append(
                f"<a href='#t-{trace['trace_id']}'>{html.escape(_trace_title(trace))}</a>"
                f"({'/'.join(FLAG_LABELS.get(f, f) for f in flags)}) "
            )
        parts.append("</div>")

    shown = traces if limit is None else traces[:limit]
    for trace in shown:
        trace_id = trace["trace_id"]
        retrieve = trace["retrieve"] or {}
        generate = trace["generate"] or {}
        hits = retrieve.get("hits") or []
        branch = retrieve.get("branch") or {}
        vector = branch.get("vector") or {}
        bm25 = branch.get("bm25") or {}
        flags = trace_flags(trace)
        scores = retrieve.get("scores") or {}
        max_score = max(((h.get("score") or 0) for h in hits), default=0) or 1.0

        parts.append(f"<h2 id='t-{trace_id}'><a href='#t-{trace_id}'>{html.escape(_trace_title(trace))}</a> "
                     f"<span class='meta'>· {html.escape(str(trace.get('mode')))} · k={trace.get('k')}</span></h2>")
        parts.append(f"<div class='q'>{html.escape(str(trace['query'] or '（未记录）'))}</div>")
        parts.append("<div>" + "".join(
            f"<span class='{_chip_class(f)}'>{html.escape(FLAG_LABELS.get(f, f))}</span>" for f in flags
        ) + "</div>")

        parts.append("<table><tr><th>阶段</th><th>耗时</th><th>说明</th></tr>")
        parts.append(f"<tr><td>query embedding</td><td>{_fmt_ms(vector.get('embed_ms'))}</td><td>candidates={vector.get('candidates')}</td></tr>")
        parts.append(f"<tr><td>向量全表打分</td><td>{_fmt_ms(vector.get('search_ms'))}</td><td>top1_cos={vector.get('top1_cos')}</td></tr>")
        parts.append(f"<tr><td>检索合计</td><td>{_fmt_ms(retrieve.get('latency_ms'))}</td>"
                     f"<td>bm25 命中词项 {bm25.get('terms_hit')} / posting {bm25.get('postings_touched')}；"
                     f"gap={scores.get('gap')}（相对 {scores.get('gap_ratio')}）；来源数={retrieve.get('n_unique_sources')}</td></tr>")
        parts.append(f"<tr><td>生成</td><td>{_fmt_ms(generate.get('latency_ms'))}</td>"
                     f"<td>tokens {generate.get('prompt_eval_count')}+{generate.get('eval_count')}</td></tr></table>")

        if hits:
            parts.append("<table><tr><th>#</th><th>来源 / chunk</th><th>score</th><th>向量</th><th>BM25</th><th>cos</th><th>首行</th></tr>")
            for hit in hits:
                width = int(60 * (hit.get("score") or 0) / max_score)
                source = f"{Path(str(hit.get('source'))).name} #{hit.get('chunk_id')}"
                parts.append(
                    f"<tr class='hit' id='ev-{trace_id}-{hit.get('rank')}'>"
                    f"<td>{hit.get('rank')}</td><td><code>{html.escape(source)}</code></td>"
                    f"<td>{hit.get('score')} <span class='bar' style='width:{width}px'></span></td>"
                    f"<td>{hit.get('rank_vector') if hit.get('rank_vector') is not None else '—'}</td>"
                    f"<td>{hit.get('rank_bm25') if hit.get('rank_bm25') is not None else '—'}</td>"
                    f"<td>{hit.get('cos')}</td>"
                    f"<td>{html.escape(_first_line(hit.get('text'), 70))}</td></tr>"
                )
            parts.append("</table>")
            for hit in hits:
                parts.append(f"<details><summary>证据 {hit.get('rank')} 全文</summary>"
                             f"<pre>{html.escape(hit.get('text') or '')}</pre></details>")

        answer = _text_of(generate, "answer")
        if answer:
            parts.append(f"<div class='answer'>{_answer_html(answer, trace_id, trace.get('k') or 0)}</div>")
        if generate:
            parts.append(
                f"<div class='meta'>引用 {generate.get('cited') or '无'} · "
                f"有效={generate.get('cited_valid')} · 零引用={generate.get('citations_missing')} · "
                f"弃答={generate.get('abstained')}（启发式）</div>"
            )
        for error in trace["errors"]:
            parts.append(f"<div class='meta' style='color:#b91c1c'>❌ {html.escape(str(error.get('where')))}："
                         f"{html.escape(str(error.get('message')))}</div>")

    parts.append(f"<script>{HTML_SCRIPT}</script></body></html>")
    return "".join(parts)


# ---------------------------------------------------------------- Failures / stats

def render_failures(run, traces):
    rows = []
    for trace in traces:
        flags = trace_flags(trace)
        if not flags:
            continue
        retrieve = trace["retrieve"] or {}
        generate = trace["generate"] or {}
        rows.append(
            f"- **{_trace_title(trace)}** `{trace.get('mode')}` "
            f"{' / '.join(FLAG_LABELS.get(f, f) for f in flags)}\n"
            f"  - 问题：{trace['query']}\n"
            f"  - 证据数 {len(retrieve.get('hits') or [])}，来源数 {retrieve.get('n_unique_sources')}，"
            f"gap={(retrieve.get('scores') or {}).get('gap')}，引用 {generate.get('cited')}\n"
            f"  - 答案开头：{_first_line(_text_of(generate, 'answer'), 90)}\n"
        )
    header = f"# 失败清单（{len(rows)}/{len(traces)}）\n\n"
    return header + ("\n".join(rows) if rows else "没有触发任何告警。\n")


def render_stats(run, traces):
    lines = ["# 运行统计\n"]
    by_mode = defaultdict(list)
    for trace in traces:
        by_mode[trace.get("mode") or "?"].append(trace)

    def collect(attr, getter):
        return [v for v in (getter(t) for t in traces) if isinstance(v, (int, float))]

    retrieve_ms = collect("retrieve_ms", lambda t: (t["retrieve"] or {}).get("latency_ms"))
    embed_ms = collect("embed_ms", lambda t: (((t["retrieve"] or {}).get("branch") or {}).get("vector") or {}).get("embed_ms"))
    search_ms = collect("search_ms", lambda t: (((t["retrieve"] or {}).get("branch") or {}).get("vector") or {}).get("search_ms"))
    gen_ms = collect("gen_ms", lambda t: (t["generate"] or {}).get("latency_ms"))

    lines.append("| 阶段 | n | p50 | p95 | max |")
    lines.append("|---|---|---|---|---|")
    for name, values in (("检索合计", retrieve_ms), ("query embedding", embed_ms),
                         ("向量打分", search_ms), ("生成", gen_ms)):
        if values:
            lines.append(f"| {name} | {len(values)} | {_pct(values, .5):.0f} ms | {_pct(values, .95):.0f} ms | {max(values):.0f} ms |")
        else:
            lines.append(f"| {name} | 0 | — | — | — |")

    lines.append("\n| 模式 | traces | 空检索 | 来源坍缩 | 零引用 | 引用越界 | 弃答 |")
    lines.append("|---|---|---|---|---|---|---|")
    for mode, items in sorted(by_mode.items()):
        counts = Counter(f for t in items for f in trace_flags(t))
        lines.append(
            f"| {mode} | {len(items)} | {counts['empty']} | {counts['source_collapse']} | "
            f"{counts['no_citation']} | {counts['citation_out_of_range']} | {counts['abstained']} |"
        )

    sources = Counter()
    for trace in traces:
        for hit in (trace["retrieve"] or {}).get("hits") or []:
            sources[hit.get("source")] += 1
    if sources:
        lines.append("\n| 来源（top 10） | 命中次数 |")
        lines.append("|---|---|")
        for source, count in sources.most_common(10):
            lines.append(f"| `{source}` | {count} |")
    return "\n".join(lines)


def render_diff(run_a, run_b):
    meta_a, meta_b = run_a["run"], run_b["run"]
    lines = ["# 两个 run 的 diff\n"]
    if meta_a.get("content_hash") != meta_b.get("content_hash"):
        lines.append(
            "> ⚠️ `content_hash` 不同（索引/模型/参数有变化），这不是干净对比，"
            "差异不能全部归因于代码或题目变化。\n"
        )
    lines.append("| 项 | A | B |\n|---|---|---|")
    lines.append(f"| run_id | `{meta_a.get('run_id')}` | `{meta_b.get('run_id')}` |")
    lines.append(f"| content_hash | `{meta_a.get('content_hash')}` | `{meta_b.get('content_hash')}` |")
    lines.append(f"| traces | {len(run_a['traces'])} | {len(run_b['traces'])} |")
    flags_a = sum(1 for t in run_a["traces"] if trace_flags(t))
    flags_b = sum(1 for t in run_b["traces"] if trace_flags(t))
    lines.append(f"| 被标记 | {flags_a} | {flags_b} |")
    lines.append("")

    def key(trace):
        return trace["qid"] or trace["query"] or trace["trace_id"]

    map_a = {key(t): t for t in run_a["traces"]}
    map_b = {key(t): t for t in run_b["traces"]}
    rows = []
    for name in sorted(set(map_a) | set(map_b)):
        a, b = map_a.get(name), map_b.get(name)
        if a is None or b is None:
            rows.append(f"| {name} | {'缺失' if a is None else '有'} | {'缺失' if b is None else '有'} | — | — |")
            continue
        ids_a = {h.get("id") for h in ((a["retrieve"] or {}).get("hits") or [])}
        ids_b = {h.get("id") for h in ((b["retrieve"] or {}).get("hits") or [])}
        added, removed = ids_b - ids_a, ids_a - ids_b
        answer_a = _text_of(a["generate"] or {}, "answer")
        answer_b = _text_of(b["generate"] or {}, "answer")
        r_ms = (a["retrieve"] or {}).get("latency_ms") or 0
        r_ms_b = (b["retrieve"] or {}).get("latency_ms") or 0
        rows.append(
            f"| {name} | +{len(added)} | -{len(removed)} | "
            f"{'变了' if answer_a != answer_b else '相同'} | {r_ms_b - r_ms:+.0f} ms |"
        )
        if added or removed:
            for chunk in sorted(added):
                rows.append(f"| ↳ 新增 | `{chunk}` | | | |")
            for chunk in sorted(removed):
                rows.append(f"| ↳ 消失 | `{chunk}` | | | |")
    shared = set(map_a) & set(map_b)
    if map_a and map_b and not shared:
        lines.append(
            "> ⚠️ 两边的 trace 标识完全不重叠：A 侧 key 形如 `" + next(iter(map_a)) +
            "`，B 侧形如 `" + next(iter(map_b)) + "`。"
            "确认两次运行是否用了同一套 `qid`（评估集必须传 `qid`），否则 diff 无意义。\n"
        )
    lines.append("| trace | 新增证据 | 消失证据 | 答案 | 检索耗时变化 |")
    lines.append("|---|---|---|---|---|")
    lines.extend(rows)
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def prune(days):
    cutoff = time.time() - days * 86400
    removed = 0
    for path in sorted(LOG_ROOT.glob("*/*")):
        if path.name == "runs.jsonl":
            continue
        if path.stat().st_mtime < cutoff:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
    print(f"已清理 {removed} 项（早于 {days} 天）。")


def _write_or_print(text, out_path, default_name=None, run=None):
    if out_path:
        target = Path(out_path)
    elif default_name and run is not None:
        target = run["path"].with_suffix("")
        target = target.parent / f"{run['path'].stem}{default_name}"
    else:
        print(text)
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"已写入 {target}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.raglog_report", description="RAG 日志 review 工具")
    parser.add_argument("run", nargs="?", default="latest", help="run_id / 日志路径 / latest / list")
    parser.add_argument("--md", action="store_true", help="输出 Markdown review 卡片（默认）")
    parser.add_argument("--html", action="store_true", help="输出单文件 HTML（可点击证据跳转）")
    parser.add_argument("--failures", action="store_true", help="只输出被标记的题")
    parser.add_argument("--stats", action="store_true", help="输出延迟与分布统计")
    parser.add_argument("--diff", metavar="OTHER", help="与另一个 run 做逐题 diff")
    parser.add_argument("--out", metavar="PATH", help="写入文件而不是打印")
    parser.add_argument("--limit", type=int, default=None, help="最多渲染多少条 trace（0 = 全部）")
    parser.add_argument("--events", action="store_true", help="Markdown 中附原始事件")
    parser.add_argument("--prune", type=int, metavar="DAYS", help="清理超过 DAYS 天的日志后退出")
    args = parser.parse_args(argv)

    if args.prune is not None:
        prune(args.prune)
        return 0

    if args.run == "list":
        manifest = LOG_ROOT / "runs.jsonl"
        if not manifest.exists():
            print("还没有任何 run。")
            return 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            print(f"{entry.get('run_id')}  {entry.get('started_at')}  {entry.get('file')}  "
                  f"{len(entry.get('params') or {})} 组参数")
        return 0

    run = load_run(args.run)
    limit = None if not args.limit else (None if args.limit <= 0 else args.limit)

    if args.diff:
        other = load_run(args.diff)
        text = render_diff(run, other)
        _write_or_print(text, args.out)
        return 0
    if args.failures:
        _write_or_print(render_failures(run, run["traces"]), args.out)
        return 0
    if args.stats:
        _write_or_print(render_stats(run, run["traces"]), args.out)
        return 0
    if args.html:
        text = render_html(run, run["traces"], limit=limit)
        _write_or_print(text, args.out, default_name=".review.html", run=run)
        return 0
    text = render_md(run, run["traces"], limit=limit, show_events=args.events)
    if args.out:
        _write_or_print(text, args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
