"""检索路由（D24）。

三路召回的统一抽象：**路由只负责「拿一段文本检索出父块排名」，不关心这段文本是
中文还是英文**。哪一路吃哪段文本由编排层（`orchestrator.MultiRouteRetriever`）决定。

这样拆的好处是可测性：把 `embedder` / `searcher` / `index` 作为可调用对象注入，
不起 Qdrant、不加载真实模型就能验证「关掉一路不影响其它路」这条验收。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

# 三路的名字固定下来，评估与 debug 字段都按它对齐，不许各写各的
ROUTE_DENSE_ZH = "dense_zh"
ROUTE_DENSE_EN = "dense_en"
ROUTE_LEXICAL_EN = "lexical_en"
ROUTE_LEXICAL_ZH = "lexical_zh"      # 不在三路里，只用于 B05 消融（中文 BM25 去留）
DEFAULT_ROUTES = (ROUTE_DENSE_ZH, ROUTE_DENSE_EN, ROUTE_LEXICAL_EN)


@dataclass(frozen=True)
class RouteHit:
    """一路检索结果里的一条父块。"""

    parent_id: str
    score: float
    rank: int = 0
    route: str = ""
    chunk_id: str | None = None


@dataclass
class RouteResult:
    """一路的完整结果。被关闭时 `enabled=False`、`hits=[]`，但仍然出现在结果里。"""

    route: str
    hits: list[RouteHit] = field(default_factory=list)
    latency_ms: float = 0.0
    enabled: bool = True
    query_used: str = ""
    note: str = ""
    candidates: int = 0
    #: 这一路失败时的错误文本（`TypeError: ...` / `CircuitOpenError: ...`）。
    #: **留在这里而不是吞掉**：降级判定要靠它区分「缺模型」与「普通 bug」。
    error: str = ""

    @property
    def ids(self) -> list[str]:
        return [hit.parent_id for hit in self.hits]


class Retriever(Protocol):
    """路由协议。"""

    name: str
    input_lang: str      # "zh" | "en"

    def search(self, text: str, k: int) -> list[RouteHit]:  # pragma: no cover
        ...


def collapse_to_parents(
    rows: Iterable[Mapping[str, Any] | Sequence[Any]],
    k: int,
    *,
    route: str = "",
) -> list[RouteHit]:
    """子块命中 → 父块排名：**去重保序、分取最高**。

    - 保序：Qdrant / BM25 已经按分数降序返回，父块按「首次出现」的位置排
    - 取最高：同一父块多个 child 命中时只留最高分（和评测侧 `parent_ranked` 同口径）
    - 支持 tuple `(chunk_id, parent_id, score)` 与 dict 两种输入
    """
    best: dict[str, RouteHit] = {}
    order: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            parent_id = row.get("parent_id") or row.get("id")
            chunk_id = row.get("chunk_id")
            score = float(row.get("score") or 0.0)
        else:
            chunk_id, parent_id, score = row[0], row[1], float(row[2])
        if not parent_id:
            continue
        existing = best.get(parent_id)
        if existing is None:
            best[parent_id] = RouteHit(
                parent_id=parent_id, score=score, rank=0, route=route, chunk_id=chunk_id
            )
            order.append(parent_id)
        elif score > existing.score:
            best[parent_id] = RouteHit(
                parent_id=parent_id, score=score, rank=0, route=route, chunk_id=chunk_id
            )
    trimmed = order[: max(k, 0)]
    return [
        RouteHit(parent_id=parent_id, score=best[parent_id].score, rank=rank,
                 route=route, chunk_id=best[parent_id].chunk_id)
        for rank, parent_id in enumerate(trimmed, start=1)
    ]


class DenseRoute:
    """向量路：embed → Qdrant ANN（多要 `k*pool_factor` 条）→ 收敛成父块排名。"""

    input_lang = "zh"

    def __init__(
        self,
        *,
        name: str = ROUTE_DENSE_ZH,
        embedder: Callable[[Sequence[str]], Any],
        searcher: Callable[[Any, int], Sequence[Sequence[Any]]],
        pool_factor: int = 4,
        input_lang: str = "zh",
    ):
        self.name = name
        self.input_lang = input_lang
        self._embedder = embedder
        self._searcher = searcher
        self.pool_factor = pool_factor

    def search(self, text: str, k: int) -> list[RouteHit]:
        vector = np.asarray(self._embedder([text])[0], dtype=np.float32)
        rows = self._searcher(vector, max(int(k) * int(self.pool_factor), int(k)))
        return collapse_to_parents(rows, k, route=self.name)


class LexicalRoute:
    """词面路：BM25 over 父块正文。

    `index` 只需要实现 `search(text, k) -> [(chunk_id, parent_id, score), ...]`，
    所以既可以接 D25 的英文正分词器（`src/rag/store/lexical.py`），
    也可以接 B05 待定的中文 jieba 索引。
    """

    input_lang = "en"

    def __init__(
        self,
        *,
        index: Any,
        name: str = ROUTE_LEXICAL_EN,
        input_lang: str = "en",
        pool_factor: int = 4,
    ):
        self.name = name
        self.input_lang = input_lang
        self._index = index
        self.pool_factor = pool_factor

    def search(self, text: str, k: int) -> list[RouteHit]:
        rows = self._index.search(text, max(int(k) * int(self.pool_factor), int(k)))
        return collapse_to_parents(rows, k, route=self.name)


class JiebaLexicalRoute:
    """中文 BM25（`src/bm25.py` 的 jieba 索引）。**不在三路默认配置里**。

    存在的唯一理由：B05「中文 BM25 去留」需要一路可开关的对照。
    """

    input_lang = "zh"

    def __init__(
        self,
        *,
        index: Any,
        records: Sequence[Mapping[str, Any]],
        name: str = ROUTE_LEXICAL_ZH,
        input_lang: str = "zh",
        pool_factor: int = 4,
    ):
        self.name = name
        self.input_lang = input_lang
        self._index = index
        self._records = list(records)
        self.pool_factor = pool_factor

    def search(self, text: str, k: int) -> list[RouteHit]:
        rows = self._index.search(
            text, max(int(k) * int(self.pool_factor), int(k)), self._records
        )
        # jieba 索引返回 (record, score)，record 里有 parent_id / id 与 chunk_id
        normalized = [
            (record.get("chunk_id"), record.get("parent_id") or record.get("id"), score)
            for record, score in rows
        ]
        return collapse_to_parents(normalized, k, route=self.name)


def timed(route: Retriever, text: str, k: int) -> tuple[list[RouteHit], float]:
    started = time.perf_counter()
    hits = route.search(text, k)
    return hits, (time.perf_counter() - started) * 1000.0
