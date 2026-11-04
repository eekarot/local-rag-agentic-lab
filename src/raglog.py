"""RAG 运行日志：trace / span / 事件，落 JSONL，供 raglog_report 渲染 review 报告。

设计规范见 docs/eval-and-logging-design.md 第 7 章。三条硬约束：

1. **不改上层签名**：上下文用 contextvars 传递，`retrieve()` / `answer()` 不接收 logger 参数；
2. **零成本关闭**：`RAG_LOG_LEVEL=off` 时 `enabled()` 立即返回 False，埋点处短路；
3. **写日志不能拖慢请求**：事件在内存缓冲，trace 结束时一次写入。

用法：

    with raglog.trace(qid="eval-007", mode="hybrid", k=4) as t:
        hits = retrieve(q, mode="hybrid", k=4, stats=stats)
        t.retrieve(hits, query=q, stats=stats, latency_ms=4.2)
        t.answer(text, model=OLLAMA_CHAT_MODEL, latency_ms=6120)

环境变量：RAG_LOG_DIR / RAG_LOG_LEVEL / RAG_LOG_TEXT(full|off) / RAG_LOG_SAMPLE。
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import random
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import ROOT_DIR
from .raglog_schema import (
    ABSTAIN_STRONG,
    ABSTAIN_WEAK,
    CITATION_PATTERN,
    DEFAULT_LEVEL,
    EV_ERROR,
    EV_EVENT,
    EV_GENERATE,
    EV_PROMPT,
    EV_RETRIEVE,
    EV_RUN,
    EV_SPAN,
    EV_TRACE,
    LEVEL_ORDER,
    SCHEMA_VERSION,
    TEXT_LIMIT,
)

_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("raglog_trace", default=None)
_LOCK = threading.Lock()

_STATE = {
    "log_dir": Path(os.getenv("RAG_LOG_DIR") or (ROOT_DIR / "logs")),
    "level": (os.getenv("RAG_LOG_LEVEL") or DEFAULT_LEVEL).strip().lower(),
    "text_mode": (os.getenv("RAG_LOG_TEXT") or "full").strip().lower(),
    "sample": float(os.getenv("RAG_LOG_SAMPLE") or 1.0),
    "run_id": None,
    "run_written": False,
    "run_event": None,
    # 索引元信息覆盖：新库链路（Qdrant parent/child）不是 storage/index.json
    "index_meta": None,
    "models": None,
}
if _STATE["level"] not in LEVEL_ORDER:
    _STATE["level"] = DEFAULT_LEVEL


# ---------------------------------------------------------------- 基础工具

def configure(**kwargs):
    """运行时覆盖配置（测试与脚本用）。支持 log_dir / level / text_mode / sample。"""
    if "log_dir" in kwargs and kwargs["log_dir"] is not None:
        _STATE["log_dir"] = Path(kwargs["log_dir"])
    if "level" in kwargs and kwargs["level"] is not None:
        level = str(kwargs["level"]).lower()
        if level not in LEVEL_ORDER:
            raise ValueError(f"level 必须是 {'、'.join(LEVEL_ORDER)}")
        _STATE["level"] = level
    if "text_mode" in kwargs and kwargs["text_mode"] is not None:
        _STATE["text_mode"] = str(kwargs["text_mode"]).lower()
    if "sample" in kwargs and kwargs["sample"] is not None:
        _STATE["sample"] = float(kwargs["sample"])
    return _STATE["level"]


def reset_run():
    """强制下一个 trace 开新 run（换索引/换题目集后调用）。"""
    _STATE["run_id"] = None
    _STATE["run_written"] = False
    _STATE["run_event"] = None
    # 覆盖也要一起清：否则换回旧链路时 run 事件会继续报新库的元信息
    _STATE["index_meta"] = None
    _STATE["models"] = None


def set_index_meta(meta, models=None):
    """替换 run 事件里的索引元信息。

    必须在 `reset_run()` **之后**调用（reset 会清掉覆盖），
    典型用法：`reset_run()` → `set_index_meta(...)` → 开始写 trace。
    """
    _STATE["index_meta"] = dict(meta) if meta else None
    _STATE["models"] = dict(models) if models else None


def enabled(level="detail"):
    """埋点快速路径：关闭时返回 False，调用方直接短路，不构造任何 dict。"""
    return LEVEL_ORDER.get(_STATE["level"], 0) >= LEVEL_ORDER.get(level, 2)


def current():
    return _ACTIVE.get()


def _now():
    return datetime.now().astimezone()


def _iso(moment=None):
    return (moment or _now()).isoformat(timespec="milliseconds")


def _sha32(text):
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32]


def _rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


# 对外别名：渲染路径时用（内部实现细节仍叫 _rel）
relative_path = _rel


def _short_id():
    return uuid.uuid4().hex[:12]


def parse_citations(text):
    """解析答案里的 [证据 n]，返回去重且升序的编号列表。"""
    return sorted({int(n) for n in re.findall(CITATION_PATTERN, text or "")})


def looks_abstained(text, cited=None):
    """弃答启发式：只用于打标记，调用方必须知道这不是判定结果。

    规则（实测调过）：
    - 出现「无法回答」这类强信号 → 判定为弃答（不管有没有引用）；
    - 只有弱信号（"证据不足/未提供/找不到"）时，要求答案是**零引用**才判弃答。
      否则会把「正文里讨论证据不足这个概念」的正常答案误判。
    """
    lowered = (text or "").lower()
    if any(pattern in lowered for pattern in ABSTAIN_STRONG):
        return True
    if cited:
        return False
    return any(pattern in lowered for pattern in ABSTAIN_WEAK)


# ---------------------------------------------------------------- run 元信息

def _build_run_event(params=None):
    """构造 run 事件：把「这次结果是用什么索引、什么模型、什么参数跑出来的」钉死。

    索引读取失败不影响日志功能（例如单元测试没有索引），此时降级为 index=None。
    """
    index_meta, models = None, {}
    override = _STATE.get("index_meta")
    if override:
        return {
            "schema": SCHEMA_VERSION,
            "python": ".".join(str(v) for v in __import__("sys").version_info[:3]),
            "platform": __import__("platform").platform(),
            "index": dict(override),
            "models": dict(_STATE.get("models") or {}),
            "params": params or {},
            "content_hash": _sha32(json.dumps(
                {"index": override, "models": _STATE.get("models") or {}, "params": params or {}},
                sort_keys=True, ensure_ascii=False,
            ))[:16],
        }
    try:  # 延迟导入，避免 retrieve -> raglog 的循环依赖
        from .config import OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL
        from .retrieve import index_stats, load_index

        raw = load_index()
        stats = index_stats(raw["records"])
        index_meta = {
            "n_records": stats["chunks"],
            "bm25_terms": stats["terms"],
            "embedding_model": raw.get("embedding_model"),
            "dim": raw.get("embedding_dim"),
            "tokenizer": stats["tokenizer"],
            "chunk_size": raw.get("chunk_size"),
            "chunk_overlap": raw.get("chunk_overlap"),
        }
        models = {"chat": OLLAMA_CHAT_MODEL, "embed": OLLAMA_EMBED_MODEL}
    except Exception as exc:  # noqa: BLE001  日志不能因索引异常而失败
        index_meta = {"error": f"{type(exc).__name__}: {exc}"}

    payload = {
        "schema": SCHEMA_VERSION,
        "python": ".".join(str(v) for v in __import__("sys").version_info[:3]),
        "platform": __import__("platform").platform(),
        "index": index_meta,
        "models": models,
        "params": params or {},
    }
    identity = json.dumps(
        {"index": index_meta, "models": models, "params": payload["params"]},
        sort_keys=True,
        ensure_ascii=False,
    )
    payload["content_hash"] = _sha32(identity)[:16]
    return payload


def _ensure_run():
    if _STATE["run_id"] is not None:
        return _STATE["run_id"]
    started = _now()
    run_id = f"{started:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:4]}"
    run_event = _build_run_event()
    run_event.update({"event": EV_RUN, "level": "info", "run_id": run_id, "started_at": _iso(started)})
    _STATE["run_id"] = run_id
    _STATE["run_event"] = run_event
    _STATE["run_started"] = started
    return run_id


def log_dir():
    return _STATE["log_dir"]


def level():
    return _STATE["level"]


def run_path(run_id=None):
    run_id = run_id or _STATE["run_id"]
    day = run_id.split("T")[0] if run_id else f"{_now():%Y%m%d}"
    day = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
    return _STATE["log_dir"] / day / f"{run_id}.jsonl"


def _append_lines(path, lines):
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(blob)


def _write_run_manifest(run_event):
    manifest = _STATE["log_dir"] / "runs.jsonl"
    line = {
        "event": EV_RUN,
        "run_id": run_event["run_id"],
        "started_at": run_event["started_at"],
        "file": _rel(run_path(run_event["run_id"])),
        "content_hash": run_event.get("content_hash"),
        "index": run_event.get("index"),
        "models": run_event.get("models"),
        "params": run_event.get("params"),
    }
    _append_lines(manifest, [line])


# ---------------------------------------------------------------- Trace

class Trace:
    """一次提问（或一次评估用例）的上下文。事件先缓冲，结束时一次落盘。"""

    def __init__(self, query=None, qid=None, mode=None, k=None, meta=None, record=True):
        self.query = query
        self.qid = qid
        self.mode = mode
        self.k = k
        self.meta = meta or {}
        self.record = record
        self.run_id = _STATE["run_id"]
        self.trace_id = _short_id()
        self.started = _now()
        self._t0 = time.perf_counter()
        self._events = []
        self._closed = False
        self.path = run_path(self.run_id)
        self._artifacts = self.path.parent / "artifacts"
        self.status = "ok"

    # ---- 通用 ----

    def _base(self, event, level="info", **fields):
        payload = {
            "ts": _iso(),
            "schema": SCHEMA_VERSION,
            "event": event,
            "level": level,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
        }
        if self.qid is not None:
            payload["qid"] = self.qid
        if self.mode is not None:
            payload["mode"] = self.mode
        if self.k is not None:
            payload["k"] = self.k
        payload.update(fields)
        return payload

    def event(self, name, level="info", **fields):
        if not self.record:
            return None
        payload = self._base(EV_EVENT, level=level, name=name, **fields)
        self._events.append(payload)
        return payload

    def error(self, where, message, **fields):
        self.status = "error"
        if not self.record:
            return None
        payload = self._base(
            EV_ERROR, level="error", where=where, message=str(message)[:2000], **fields
        )
        self._events.append(payload)
        _append_lines(self.path, [payload])  # 错误立刻落盘，崩溃也能留下痕迹
        return payload

    # ---- 文本字段（超长外置） ----

    def _text_fields(self, name, value):
        value = value or ""
        chars = len(value)
        if _STATE["text_mode"] == "off":
            return {f"{name}_chars": chars, f"{name}_sha32": _sha32(value)}
        if chars > TEXT_LIMIT:
            self._artifacts.mkdir(parents=True, exist_ok=True)
            target = self._artifacts / f"{self.trace_id}-{name}.txt"
            target.write_text(value, encoding="utf-8")
            return {
                f"{name}_ref": _rel(target),
                f"{name}_chars": chars,
                f"{name}_sha32": _sha32(value),
                f"{name}_truncated": True,
            }
        return {name: value, f"{name}_chars": chars}

    # ---- 检索 ----

    def retrieve(self, hits, query=None, stats=None, latency_ms=None, **extra):
        hit_list = list(hits or [])
        if not self.record:
            return None
        stats = stats or {}
        vector = stats.get("vector") or {}
        bm25 = stats.get("bm25") or {}
        fuse = stats.get("fuse") or {}
        v_ranks = vector.get("ranks") or {}
        b_ranks = bm25.get("ranks") or {}
        cos = vector.get("cos") or {}
        bm25_scores = bm25.get("scores") or {}

        scores = [float(item.get("score", 0.0)) for item in hit_list]
        gap = (scores[0] - scores[-1]) if len(scores) > 1 else None
        gap_ratio = (gap / abs(scores[0]) if gap is not None and scores[0] else None)
        detailed = enabled("detail")
        payload = self._base(
            EV_RETRIEVE,
            query=query if query is not None else self.query,
            query_chars=len(query if query is not None else (self.query or "")),
            latency_ms=round(latency_ms, 2) if latency_ms is not None else None,
            empty=not hit_list,
            n_unique_sources=len({item.get("source") for item in hit_list}),
            scores={
                "top1": round(scores[0], 6) if scores else None,
                "topk": round(scores[-1], 6) if scores else None,
                "gap": round(gap, 6) if gap is not None else None,
                "gap_ratio": round(gap_ratio, 6) if gap_ratio is not None else None,
            },
        )
        if bm25 or vector:
            # 两个分支各自独立记录：纯 vector 模式没有 bm25 统计，
            # 条件写成 `if bm25:` 会让 vector 模式的日志彻底没有 branch（实测漏了 12 条/轮）。
            if bm25:
                payload["tokens"] = bm25.get("tokens")
            payload["branch"] = {
                "bm25": {
                    "terms_hit": bm25.get("terms_hit"),
                    "postings_touched": bm25.get("postings_touched"),
                    "candidates": bm25.get("candidates"),
                } if bm25 else None,
                "vector": {
                    "candidates": vector.get("candidates"),
                    "cos_mean": vector.get("cos_mean"),
                    "cos_std": vector.get("cos_std"),
                    "top1_cos": vector.get("top1_cos"),
                    "topk_cos_floor": vector.get("topk_cos_floor"),
                    "embed_ms": vector.get("embed_ms"),
                    "search_ms": vector.get("search_ms"),
                } if vector else None,
            }
        if fuse:
            payload["fuse"] = fuse
        if detailed:
            payload["hits"] = [
                {
                    "rank": rank,
                    "id": item.get("id"),
                    "chunk_id": item.get("chunk_id"),
                    "source": item.get("source"),
                    "len": len(item.get("text") or ""),
                    "score": round(float(item.get("score", 0.0)), 6),
                    "rank_vector": v_ranks.get(item.get("id")),
                    "rank_bm25": b_ranks.get(item.get("id")),
                    "cos": round(cos[item.get("id")], 6) if item.get("id") in cos else None,
                    "bm25": round(bm25_scores[item.get("id")], 4) if item.get("id") in bm25_scores else None,
                    "text": (item.get("text") or "")[:TEXT_LIMIT],
                }
                for rank, item in enumerate(hit_list, start=1)
            ]
        else:
            payload["hit_ids"] = [item.get("id") for item in hit_list]
        payload.update(extra)
        self._events.append(payload)
        return payload

    # ---- Prompt / 生成 ----

    def prompt(self, messages, context_chars=None, evidence_count=None, evidence_truncated=False):
        if not self.record:
            return None
        serialized = json.dumps(messages, ensure_ascii=False)
        payload = self._base(
            EV_PROMPT,
            level="debug",
            prompt_sha32=_sha32(serialized),
            prompt_chars=len(serialized),
            messages=messages if enabled("debug") else None,
            context_chars=context_chars,
            evidence_count=evidence_count,
            evidence_truncated=evidence_truncated,
        )
        self._events.append(payload)
        return payload

    def answer(
        self,
        text,
        model=None,
        latency_ms=None,
        tokens_prompt=None,
        tokens_completion=None,
        finish_reason=None,
        extra=None,
    ):
        if not self.record:
            return None
        cited = parse_citations(text)
        k = self.k or 0
        payload = self._base(
            EV_GENERATE,
            level="info",
            model=model,
            latency_ms=round(latency_ms, 2) if latency_ms is not None else None,
            prompt_eval_count=tokens_prompt,
            eval_count=tokens_completion,
            finish_reason=finish_reason,
            cited=cited,
            cited_valid=all(1 <= n <= k for n in cited) if cited else None,
            citations_missing=not cited,
            abstained=looks_abstained(text, cited),
            abstained_heuristic=True,
        )
        payload.update(self._text_fields("answer", text))
        if extra:
            payload.update(extra)
        self._events.append(payload)
        return payload

    # ---- 收尾 ----

    def close(self, status=None):
        if self._closed:
            return
        self._closed = True
        if status:
            self.status = status
        if not self.record:
            return
        closing = self._base(
            EV_TRACE,
            level="info",
            status=self.status,
            query=self.query,
            latency_ms=round((time.perf_counter() - self._t0) * 1000, 2),
            events=len(self._events) + 1,
            **self.meta,
        )
        self._events.append(closing)
        try:
            _append_lines(self.path, self._events)
        except OSError as exc:  # 磁盘写失败不能影响主流程
            print(f"[raglog] 写日志失败：{exc}")


@contextmanager
def trace(query=None, qid=None, mode=None, k=None, meta=None):
    """开一个 trace。若当前已有 trace（评估脚本外层开了），则复用外层，不嵌套文件。"""
    outer = _ACTIVE.get()
    if outer is not None:
        if qid is not None and outer.qid is None:
            outer.qid = qid
        yield outer
        return

    _ensure_run()
    # level=off 必须彻底不落盘：只靠 enabled() 判断采样会漏掉这条路径（实测踩过）
    record = _STATE["level"] != "off" and (
        _STATE["sample"] >= 1.0 or random.random() < _STATE["sample"]
    )
    tracer = Trace(query=query, qid=qid, mode=mode, k=k, meta=meta, record=record)
    token = _ACTIVE.set(tracer)
    try:
        if record:
            run_event = dict(_STATE["run_event"])
            if not _STATE["run_written"]:
                _STATE["run_written"] = True
                _append_lines(tracer.path, [run_event])
                try:
                    _write_run_manifest(run_event)
                except OSError:
                    pass
            tracer._events.insert(0, {**run_event, "trace_id": tracer.trace_id})
        yield tracer
    except BaseException as exc:
        tracer.status = "exception"
        tracer.error("trace", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _ACTIVE.reset(token)
        tracer.close()


@contextmanager
def span(name, **fields):
    """记录一个阶段耗时。level < detail 时不做任何事。"""
    active = _ACTIVE.get()
    if active is None or not active.record or not enabled("detail"):
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        active.event(
            f"span:{name}",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            **fields,
        )


def emit(name, **fields):
    """往当前 trace 打一个通用事件（无 trace 时静默丢弃）。"""
    active = _ACTIVE.get()
    if active is None:
        return None
    return active.event(name, **fields)
