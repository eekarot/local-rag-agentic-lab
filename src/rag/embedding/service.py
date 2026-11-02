"""D13：Embedding 服务（§7.2）。

对外只有两个方法，语义分开是为了将来能换「非对称」模型（query 要加前缀）：

    encode_documents(texts) -> np.ndarray (n, 1024) float32
    encode_queries(texts)   -> np.ndarray (n, 1024) float32

与方案 §7.2 的一处偏差：签名写的是 `list[list[float]]`，实现返回
`np.ndarray(float32)`。理由：8 万 chunk × 1024 维 = 8.3e7 个 Python float，
光对象头就要 6 亿字节量级；而且下游（Qdrant upsert / numpy 基准）全都要数组。
**已记入 §17.12 执行记录**，需要 list 时 `.tolist()` 即可。

关键行为（每条都有用例守着）：

- 维度 ≠ 1024 → `EmbeddingDimensionError`
- NaN / Inf / 全零 → `EmbeddingValueError`（§7.2「不得静默产生空向量」）
- 空串 / 非字符串 / 超长 → `EmbeddingInputError`
- 同一文本第二次调用直接读缓存（doc 与 query 分开命名空间）
- 批量失败：先原地重试，仍是 400 才二分（Ollama 偶发 400，实测约 30% 请求，
  与批大小无关的那部分靠重试解决，真正的「批太大」才靠二分）
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

import numpy as np

from src.config import EMBED_BATCH_SIZE, OLLAMA_EMBED_MODEL, STORAGE_DIR

from .cache import EmbeddingCache
from .errors import (
    EmbeddingBackendError,
    EmbeddingDimensionError,
    EmbeddingInputError,
    EmbeddingValueError,
)

VECTOR_SIZE = 1024                                  # BGE-M3 Dense 维度
DEFAULT_MODEL = OLLAMA_EMBED_MODEL                  # 默认 bge-m3
DEFAULT_VERSION = f"{DEFAULT_MODEL}-v1"             # 换预处理口径时才递增
DEFAULT_CACHE_PATH = STORAGE_DIR / "cache" / "embeddings.sqlite3"
MAX_CHARS = int(os.getenv("RAG_EMBED_MAX_CHARS", "8192"))

# Ollama 的 400 有两种：偶发（runner 重启）与批太大。前者重试能过，后者必须二分。
# 旧实现只匹配 "HTTP 400"，漏掉了 requests 的 "400 Client Error"，导致二分分支永不触发。
BATCH_400_MARKERS = ("HTTP 400", "400 Client Error")


def _looks_like_400(exc: BaseException) -> bool:
    text = str(exc)
    return any(marker in text for marker in BATCH_400_MARKERS)


class EmbeddingBackend(Protocol):
    """后端协议：把一批文本变成一批向量（顺序一致）。"""

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - 协议
        ...


class OllamaBackend:
    """走 Ollama `/api/embed`（决策 B01：不引 FlagEmbedding 本地推理）。"""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        from src import ollama_client

        previous = ollama_client.OLLAMA_EMBED_MODEL
        if previous != self.model:  # 允许通过构造参数临时换模型
            ollama_client.OLLAMA_EMBED_MODEL = self.model
        try:
            vectors = ollama_client.embed_texts(texts)
        except Exception as exc:  # 网络/模型不存在/接口不支持，统一成明确报错
            raise EmbeddingBackendError(
                f"调用 Ollama embedding 失败（model={self.model}）：{exc}"
            ) from exc
        finally:
            ollama_client.OLLAMA_EMBED_MODEL = previous
        if not vectors:
            raise EmbeddingBackendError(
                f"Ollama 没有返回向量（model={self.model}）；请确认模型已 pull 且支持 /api/embed"
            )
        return vectors


@dataclass
class _PassStats:
    hits: int = 0
    misses: int = 0
    memo_hits: int = 0
    backend_batches: int = 0
    backend_texts: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class EmbeddingService:
    def __init__(
        self,
        *,
        backend: EmbeddingBackend | None = None,
        cache_path: str | os.PathLike[str] | None = None,
        cache_enabled: bool = True,
        model: str = DEFAULT_MODEL,
        version: str = DEFAULT_VERSION,
        dim: int = VECTOR_SIZE,
        batch_size: int = EMBED_BATCH_SIZE,
        normalize: bool = False,
        max_chars: int = MAX_CHARS,
        retry_attempts: int = 2,
        memo_size: int = 1024,
    ):
        self.model = model
        self.version = version
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.normalize = normalize
        self.max_chars = max_chars
        self.retry_attempts = max(1, retry_attempts)
        self.backend = backend or OllamaBackend(model)
        self.cache_enabled = cache_enabled
        self.cache = EmbeddingCache(cache_path or DEFAULT_CACHE_PATH) if cache_enabled else None
        # 进程内 LRU：SQLite 命中实测 6.8 ms（4 KB BLOB + 建连接），离 ADR 的 10 ms
        # 只剩 30% 余量，换块慢盘就破。查询场景的 key 数量很少（几十到几千），
        # 缓一层内存即可把命中打到 0.1 ms 量级。1024 条 ≈ 4 MB。
        self._memo: OrderedDict[str, np.ndarray] = OrderedDict()
        self.memo_size = max(0, memo_size)
        self._total = _PassStats()
        self._last = _PassStats()

    # -- 对外 API ------------------------------------------------------

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, kind="doc")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, kind="query")

    def stats(self) -> dict:
        """累计统计 + 最近一次调用的命中率（D13 验收看的是后者）。"""
        return {
            "hits": self._total.hits,
            "misses": self._total.misses,
            "hit_rate": round(self._total.hit_rate, 4),
            "memo_hits": self._total.memo_hits,
            "backend_batches": self._total.backend_batches,
            "backend_texts": self._total.backend_texts,
            "last_pass": {
                "hits": self._last.hits,
                "misses": self._last.misses,
                "hit_rate": round(self._last.hit_rate, 4),
            },
            "cached_rows": self.cache.stats()["rows"] if self.cache else 0,
            "cached_bytes": self.cache.stats()["bytes"] if self.cache else 0,
        }

    # -- 内部 ----------------------------------------------------------

    def _encode(self, texts: Sequence[str], *, kind: str) -> np.ndarray:
        items = self._validate(texts)
        self._last = _PassStats()
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        matrix = np.zeros((len(items), self.dim), dtype=np.float32)
        keys = [
            EmbeddingCache.key_for(kind=kind, model=self.model, version=self.version, text=text)
            for text in items
        ]

        pending: list[int] = []
        if self.cache is not None:
            missing: list[int] = []
            for index, key in enumerate(keys):
                vector = self._memo.get(key)
                if vector is None:
                    missing.append(index)
                else:
                    self._memo.move_to_end(key)
                    matrix[index] = vector
                    self._total.hits += 1
                    self._last.hits += 1
                    self._total.memo_hits += 1
            cached = self.cache.get_many([keys[i] for i in missing], expected_dim=self.dim) if missing else {}
            for index in missing:
                key = keys[index]
                vector = cached.get(key)
                if vector is None:
                    pending.append(index)
                else:
                    matrix[index] = vector
                    self._remember(key, vector)
                    self._total.hits += 1
                    self._last.hits += 1
        else:
            pending = list(range(len(items)))

        for start in range(0, len(pending), self.batch_size):
            batch_indexes = pending[start:start + self.batch_size]
            batch_texts = [items[i] for i in batch_indexes]
            vectors = self._embed_with_retry(batch_texts)
            for index, vector in zip(batch_indexes, vectors):
                matrix[index] = self._check_vector(vector, text=items[index])
            self._total.misses += len(batch_indexes)
            self._last.misses += len(batch_indexes)

        if self.cache is not None and pending:
            self.cache.put_many(
                (keys[i], matrix[i], self.model, self.version, kind, len(items[i]))
                for i in pending
            )
            for i in pending:
                self._remember(keys[i], matrix[i])

        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, norms, out=matrix, where=norms > 0)
        return matrix

    def _remember(self, key: str, vector: np.ndarray) -> None:
        if self.memo_size <= 0:
            return
        self._memo[key] = np.array(vector, dtype=np.float32, copy=True)
        self._memo.move_to_end(key)
        while len(self._memo) > self.memo_size:
            self._memo.popitem(last=False)

    def _validate(self, texts: Sequence[str]) -> list[str]:
        if texts is None:
            raise EmbeddingInputError("texts 不能为 None")
        items = list(texts)
        for index, text in enumerate(items):
            if not isinstance(text, str):
                raise EmbeddingInputError(f"第 {index} 条不是字符串：{type(text).__name__}")
            if not text.strip():
                raise EmbeddingInputError(f"第 {index} 条是空文本，空文本没有语义，不能编码")
            if len(text) > self.max_chars:
                raise EmbeddingInputError(
                    f"第 {index} 条长度 {len(text)} 超过上限 {self.max_chars} 字符；"
                    f"应在切分阶段就把块压到 max_chars 以内"
                )
        return items

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for _ in range(self.retry_attempts):
            try:
                vectors = self.backend.embed(texts)
                self._total.backend_batches += 1
                self._last.backend_batches += 1
                self._total.backend_texts += len(texts)
                self._last.backend_texts += len(texts)
                if len(vectors) != len(texts):
                    raise EmbeddingBackendError(
                        f"后端返回 {len(vectors)} 条向量，输入 {len(texts)} 条，无法对齐"
                    )
                return vectors
            except EmbeddingBackendError as exc:
                last_error = exc
                if not _looks_like_400(exc) or len(texts) == 1:
                    raise
        # 重试仍 400：判定为「批太大」，二分重来（单条也失败就直接抛）
        assert last_error is not None
        if len(texts) == 1:
            raise last_error
        mid = len(texts) // 2
        return self._embed_with_retry(texts[:mid]) + self._embed_with_retry(texts[mid:])

    def _check_vector(self, vector: Iterable[float], *, text: str) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.size != self.dim:
            raise EmbeddingDimensionError(
                f"向量维度 {array.size} 与约定 {self.dim} 不一致（文本前 30 字：{text[:30]!r}）。"
                f"换 embedding 模型必须改 VECTOR_SIZE 并重建 collection（D21）"
            )
        if not np.all(np.isfinite(array)):
            kind = "NaN" if np.isnan(array).any() else "Inf"
            raise EmbeddingValueError(f"向量含 {kind}（文本前 30 字：{text[:30]!r}）")
        if not np.any(array):
            raise EmbeddingValueError(
                f"向量全零，检索会静默返回 0 分（文本前 30 字：{text[:30]!r}）"
            )
        return array


# --------------------------------------------------------------------- CLI


def cli_main(argv: Sequence[str] | None = None) -> int:
    """缓存运维：`stats` / `purge` / `encode`。"""
    import argparse
    import json as jsonlib
    import time

    parser = argparse.ArgumentParser(prog="embedding", description="Embedding 服务缓存运维（D13）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats", help="看缓存行数/体积/命中统计")
    purge = sub.add_parser("purge", help="清缓存（可只清某个模型）")
    purge.add_argument("--model")
    purge.add_argument("--yes", action="store_true")
    encode = sub.add_parser("encode", help="编码一段文本，看维度/耗时/是否命中缓存")
    encode.add_argument("text")
    encode.add_argument("--kind", choices=("doc", "query"), default="query")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cache_path = os.getenv("RAG_EMBED_CACHE") or DEFAULT_CACHE_PATH
    if args.cmd == "stats":
        cache = EmbeddingCache(cache_path)
        print(jsonlib.dumps(cache.stats(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "purge":
        if not args.yes:
            print("拒绝执行：清缓存会让下次编码重新调用模型，确认请加 --yes")
            return 2
        cache = EmbeddingCache(cache_path)
        print(f"已删除 {cache.purge(model=args.model)} 行")
        return 0
    service = EmbeddingService(cache_path=cache_path)
    started = time.perf_counter()
    matrix = service.encode_queries([args.text]) if args.kind == "query" else service.encode_documents([args.text])
    elapsed = (time.perf_counter() - started) * 1000
    print(f"[{args.kind}] shape={matrix.shape} dtype={matrix.dtype} "
          f"耗时 {elapsed:.1f} ms 命中={service.stats()['last_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
