"""D31 Reranker：cross-encoder 精排（ONNX Runtime，不依赖 torch）。

**为什么不是文档里写的 `bge-reranker-v2-m3`（B08 结论）**

1. 它只发布 PyTorch 权重；本机没有 torch/sentence-transformers，装 CPU 版 torch 要多
   1–2 GB RSS，与 D39 的 16 GB 预算正面冲突。
2. 已经在的 `onnxruntime` + `tokenizers` 够用，且换成 ONNX 之后模型体积能压到 266–449 MB。
3. B08 探针（`experiments/probe_reranker.py`）实测两个 ONNX 候选：

   | 候选 | 体积 | 128 tok | 256 tok | 512 tok |
   |---|---|---|---|---|
   | `mmarco-mMiniLMv2-L12-H384` fp32 | 449 MB | 23 ms/对 | 28–53 ms/对 | 105–113 ms/对 |
   | `jina-reranker-v2-base-multilingual` int8 | 266 MB | 39 ms/对 | 88 ms/对 | 194 ms/对 |

   int8 在这台机器（无 VNNI 收益）反而慢一倍，所以默认走 fp32 小模型 + 256 token 截断。

**重排的单位是 child，不是 parent。** 父块 p50 1,564 字符，截到 256 token 会砍掉大部分正文；
而检索本来就命中过某个 child，那个 child 的正文才是「和 query 真正相关的一段」。
所以打分对象是「命中的 child 正文」，父块分数取组内 max（`collapse_to_parents` 之后
每个父块只留一个代表 child，天然就是 max）。

设计要求（D31 验收）：

- **可开关**：`RerankConfig(enabled=False)` 时输出与 D26 完全一致，一个字节都不变；
- **批量**：`batch_size` 控制单次 ONNX 推理的批大小；
- **并发上限**：CPU 上同时跑多个 cross-encoder 只会互相拖慢，默认 `max_concurrency=1`；
- **失败不炸**：编码器异常时回退原序并在 `stats()` 里留 `errors` / `fallbacks` 两个计数。
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

#: B08 探针选出来的默认模型（fp32 小模型 + 256 token 截断）
DEFAULT_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_ONNX_FILE = "onnx/model.onnx"
DEFAULT_CACHE_DIR = "storage/models"


_SENTENCE_END = "。！？；" + chr(10)
_CJK = re.compile("[\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-z0-9_]+")


class RerankError(RuntimeError):
    """重排链路不可用（模型缺失、ONNX 会话创建失败等）。"""


@dataclass
class RerankConfig:
    """重排参数。默认值是 B08 探针实测出来的，不是拍的。"""

    enabled: bool = True
    max_length: int = 256          # 128/256/512 三档实测后的拐点
    batch_size: int = 8
    max_concurrency: int = 1       # CPU 上并发重排只会互相拖慢
    top_k_pool: int = 20           # 进重排的候选数（E 的 k=10 列表不够重排用）
    fallback_on_error: bool = True
    model_id: str = DEFAULT_MODEL_ID
    onnx_file: str = DEFAULT_ONNX_FILE
    cache_dir: str = DEFAULT_CACHE_DIR

    def __post_init__(self) -> None:
        for name in ("max_length", "batch_size", "max_concurrency", "top_k_pool"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数，实际 {value!r}")


@dataclass
class RerankHit:
    """重排的最小输入单元。故意不带「路由」概念，方便 D26/D34 两边复用。"""

    chunk_id: str
    parent_id: str = ""
    text: str = ""
    score: float = 0.0
    rank: int = 0
    rerank_score: float | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "rerank_score": self.rerank_score,
            "fields": dict(self.fields),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RerankHit":
        return cls(
            chunk_id=data["chunk_id"],
            parent_id=data.get("parent_id", ""),
            text=data.get("text", ""),
            score=float(data.get("score", 0.0)),
            rank=int(data.get("rank", 0)),
            rerank_score=data.get("rerank_score"),
            fields=dict(data.get("fields") or {}),
        )


class OnnxPairEncoder:
    """`(query, passage)` → 一个标量分数。ONNX 会话**懒加载**（首次调用时才建）。"""

    def __init__(self, config: RerankConfig):
        self.config = config
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()
        self.load_ms: float | None = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可读信息
            raise RerankError(f"重排依赖缺失：{exc}") from exc

        started = time.perf_counter()
        self._tokenizer = Tokenizer.from_file(
            hf_hub_download(self.config.model_id, "tokenizer.json", cache_dir=self.config.cache_dir)
        )
        self._tokenizer.enable_truncation(max_length=self.config.max_length)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            hf_hub_download(self.config.model_id, self.config.onnx_file, cache_dir=self.config.cache_dir),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {item.name for item in self._session.get_inputs()}
        self.load_ms = (time.perf_counter() - started) * 1000

    def encode_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        scores: list[float] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = list(texts[start:start + self.config.batch_size])
            encoded = self._tokenizer.encode_batch([(query, text) for text in batch])
            feed = {
                "input_ids": np.array([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": np.array([item.attention_mask for item in encoded], dtype=np.int64),
            }
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.array([item.type_ids for item in encoded], dtype=np.int64)
            logits = np.asarray(self._session.run(None, feed)[0], dtype=np.float32)
            scores.extend(np.asarray(logits).reshape(len(batch), -1)[:, 0].tolist())
        return np.asarray(scores, dtype=np.float32)


class _GuardedEncoder:
    """把并发上限和计数套在真正的编码器外面（上层只认 `encode_pairs`）。"""

    def __init__(self, inner, limiter: threading.Semaphore, stats: dict, lock: threading.Lock):
        self.inner = inner
        self._limiter = limiter
        self._stats = stats
        self._lock = lock

    def encode_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
        with self._limiter:                      # 并发上限在这里生效
            with self._lock:
                self._stats["calls"] += 1
                self._stats["pairs"] += len(texts)
            started = time.perf_counter()
            try:
                return self.inner.encode_pairs(query, texts)
            except Exception:
                with self._lock:
                    self._stats["errors"] += 1
                raise
            finally:
                with self._lock:
                    self._stats["encode_ms"] += (time.perf_counter() - started) * 1000


class CrossEncoderReranker:
    """容器：config + 受控编码器 + 计数。`reranker.encoder` 可直接喂给 `rerank_hits`。"""

    def __init__(self, encoder=None, config: RerankConfig | None = None):
        self.config = config or RerankConfig()
        self._stats = {"calls": 0, "pairs": 0, "errors": 0, "fallbacks": 0, "encode_ms": 0.0}
        self._lock = threading.Lock()
        inner = encoder if encoder is not None else OnnxPairEncoder(self.config)
        self.encoder = _GuardedEncoder(inner, threading.Semaphore(self.config.max_concurrency),
                                       self._stats, self._lock)

    def stats(self) -> dict[str, Any]:
        """**实时字典**（不是拷贝）：`rerank_hits` 会往里累加 `fallbacks`。"""
        inner = getattr(self.encoder, "inner", None)
        if getattr(inner, "load_ms", None) is not None:
            self._stats["load_ms"] = round(inner.load_ms, 1)
        self._stats["encode_ms"] = round(self._stats["encode_ms"], 1)
        return self._stats


def rerank_hits(
    query: str,
    hits: Sequence[RerankHit],
    *,
    encoder,
    config: RerankConfig | None = None,
    top_n: int | None = None,
    stats: Callable[[], dict[str, Any]] | dict | None = None,
    text_getter: Callable[[RerankHit], str] | None = None,
) -> list[RerankHit]:
    """按 cross-encoder 分数重排。

    - 关闭开关：原样返回（只做 `top_n` 截断），不碰 encoder；
    - 并列：保持原顺序（`rank` 兜底），保证两次运行结果一致；
    - 异常：`fallback_on_error=True` 时回退原序并记 `fallbacks`。
    """
    config = config or RerankConfig()
    candidates = list(hits)
    if not config.enabled or not candidates:
        return candidates[:top_n] if top_n else candidates

    def _bump(name: str) -> None:
        live = stats() if callable(stats) else stats
        if isinstance(live, dict):
            live[name] = live.get(name, 0) + 1

    texts = [(text_getter(hit) if text_getter else hit.text) or "" for hit in candidates]
    try:
        scores = np.asarray(encoder.encode_pairs(query, texts), dtype=np.float32)
    except Exception:
        if not config.fallback_on_error:
            raise
        _bump("fallbacks")
        return candidates[:top_n] if top_n else candidates

    if len(scores) != len(candidates):
        if not config.fallback_on_error:
            raise RerankError(f"编码器返回 {len(scores)} 个分数，候选 {len(candidates)} 个")
        _bump("fallbacks")
        return candidates[:top_n] if top_n else candidates

    for hit, score in zip(candidates, scores):
        hit.rerank_score = float(score)
    ordered = sorted(candidates, key=lambda hit: (-hit.rerank_score, hit.rank, hit.chunk_id))
    return ordered[:top_n] if top_n else ordered


def _query_terms(query: str) -> set[str]:
    """查询词集合：英文按词、中文按字 bigram（不引 jieba，避免为了挑窗口拉依赖）。"""
    lowered = (query or "").lower()
    terms = set(_LATIN.findall(lowered))
    han = "".join(_CJK.findall(lowered))
    terms.update(han[i:i + 2] for i in range(max(0, len(han) - 1)))
    if len(han) == 1:
        terms.add(han)
    return {term for term in terms if len(term) >= 2}


def _split_units(text: str) -> list[str]:
    """按句末标点切「窗口积木」，尽量不把一个句子切开。"""
    units: list[str] = []
    buffer = ""
    for char in text:
        buffer += char
        if char in _SENTENCE_END:
            if buffer.strip():
                units.append(buffer)
            buffer = ""
    if buffer.strip():
        units.append(buffer)
    return units


def _hard_split(unit: str, max_tokens: int, counter) -> list[str]:
    """把一个超长句子按 token 比例硬切开（技术文档里整段没句号的段落很常见）。"""
    total = max(1, counter.count(unit))
    if total <= max_tokens:
        return [unit]
    ratio = len(unit) / total                      # 字符 / token
    step = max(1, int(max_tokens * ratio * 0.9))   # 留 10% 余量，避免估算误差
    return [unit[i:i + step] for i in range(0, len(unit), step)] or [unit]


def best_window(text: str, query: str, *, max_tokens: int, counter) -> str:
    """在父块正文里挑一段「最可能回答 query」的窗口，长度 ≤ max_tokens。

    为什么需要它（实测数字，见 `docs/evidence/window-strategy-k8s-v1.json`）：
    候选父块 p50 = **322 token**，其中 **63% 超过 256**；直接截开头会拿到标题/目录，
    真正相关的那句常在中间。窗口挑错不是小事：整批换成「块首 256 token」，
    整体 hit@10 从 0.8345 掉到 0.6978（中→中 −18.5 点）。做法是：

    1. 按句末标点切积木；
    2. 逐个积木累计 token，滑出一个一个 ≤ max_tokens 的候选窗口；
    3. 选与 query 词重叠最多的窗口（并列取靠前的）。

    没有任何重叠时退回开头窗口——至少不比 `text[:N]` 更差。
    """
    if not text:
        return ""
    if counter.count(text) <= max_tokens:
        return text

    # 超长「句子」先硬切开：否则它整块超预算会被跳过，里面那句话就永远选不到
    units: list[str] = []
    for unit in _split_units(text) or [text]:
        units.extend(_hard_split(unit, max_tokens, counter))
    terms = _query_terms(query)

    windows: list[tuple[int, int, str]] = []      # (start, end, text)
    start = 0
    tokens = 0
    for end, unit in enumerate(units):
        unit_tokens = max(1, counter.count(unit))
        tokens += unit_tokens
        while tokens > max_tokens and start < end:
            tokens -= max(1, counter.count(units[start]))
            start += 1
        if tokens <= max_tokens:
            windows.append((start, end, "".join(units[start:end + 1])))
    if not windows:
        return units[0][: max(4, max_tokens * 2)]

    best = windows[0]
    best_score = -1.0
    for window in windows:
        lowered = window[2].lower()
        hit = sum(1 for term in terms if term in lowered)
        # 同样重叠数时优先长窗口（信息量大），仍然并列则取靠前的
        score = hit + len(window[2]) / 1e6
        if score > best_score:
            best, best_score = window, score
    return best[2]
