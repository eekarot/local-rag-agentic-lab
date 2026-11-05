"""三路召回编排（D24）+ 融合出口（D26）。

编排层只做四件事：
1. **输入映射**：`dense_zh` 吃中文原句，`dense_en` / `lexical_en` 吃英文 `query_en`
2. **开关**：`disable=("dense_en",)` → 该路 `enabled=False`、`hits=[]`，其余路逐条不变
3. **计时**：每路独立计时，进 `debug["latency_ms"]`
4. **融合**：三路 RRF → `fused`（去重 + 累加）
5. **容错**：单路抛异常只记在 `RouteResult.error`，其余路照常融合；
   出问题的路汇总成 `degraded`（缺向量模型 vs 普通 bug，判定只看异常类型）

**开关一致性为什么天然成立**：路由之间不共享可变状态，也不互相喂文本；
每次 `retrieve()` 都从同一份 `query_zh/query_en` 出发。用例 `test_disabling_a_route_does_not_change_other_routes`
把这条锁死。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .degrade import CapabilityError, CircuitOpenError, DegradedInfo, degraded_from_errors
from .fusion import RRF_K, FusedHit, rrf_fuse_routes
from .routes import (
    DEFAULT_ROUTES,
    Retriever,
    RouteHit,
    RouteResult,
    timed,
)

FALLBACK_NOTE = "query_en 缺失，回退中文原句（该路结果应视为噪声源）"


@dataclass
class RetrievalResult:
    query_zh: str
    query_en: str | None
    k: int
    routes: dict[str, RouteResult] = field(default_factory=dict)
    fused: list[FusedHit] = field(default_factory=list)
    debug: dict = field(default_factory=dict)
    #: 非 None 表示这次检索**缺了东西**（向量模型不可用 / 某一路抛异常）。
    #: CLI、`/retrieve`、`/ask`、SSE 的 `sources` 都把它原样带出去。
    degraded: DegradedInfo | None = None

    @property
    def enabled_routes(self) -> list[str]:
        return [name for name, result in self.routes.items() if result.enabled]

    def hits_of(self, route: str) -> list[RouteHit]:
        return self.routes[route].hits if route in self.routes else []


class MultiRouteRetriever:
    """把若干路由组合成「一次请求 → 三路 Top-k + 融合列表」。"""

    def __init__(self, routes: Iterable[Retriever], *, rrf_k: int = RRF_K):
        self.routes_list = list(routes)
        if not self.routes_list:
            raise ValueError("至少要给一路路由")
        self._by_name: dict[str, Retriever] = {}
        for route in self.routes_list:
            if route.name in self._by_name:
                raise ValueError(f"路由名重复：{route.name}")
            self._by_name[route.name] = route
        if rrf_k <= 0:
            raise ValueError(f"rrf_k 必须为正数（收到 {rrf_k}）")
        self.rrf_k = rrf_k

    @property
    def route_names(self) -> list[str]:
        return [route.name for route in self.routes_list]

    def retrieve(
        self,
        query_zh: str,
        *,
        query_en: str | None = None,
        k: int = 10,
        disable: Sequence[str] = (),
        fuse: bool = True,
    ) -> RetrievalResult:
        disabled = {name for name in disable}
        unknown = disabled - set(self._by_name)
        if unknown:
            raise ValueError(f"未知路由：{sorted(unknown)}；可选 {sorted(self._by_name)}")

        started = time.perf_counter()
        results: dict[str, RouteResult] = {}
        errors: dict[str, str] = {}
        broken_capability: str | None = None
        for route in self.routes_list:
            needs_en = getattr(route, "input_lang", "zh") == "en"
            text, note = query_zh, ""
            if needs_en:
                if query_en:
                    text = query_en
                else:
                    note = FALLBACK_NOTE

            if route.name in disabled:
                results[route.name] = RouteResult(
                    route=route.name, hits=[], latency_ms=0.0, enabled=False,
                    query_used=text, note=note or "已按请求关闭", candidates=0,
                )
                continue

            try:
                hits, latency_ms = timed(route, text, k)
            except Exception as exc:              # noqa: BLE001 - 单路故障不许带走其它路
                latency_ms = 0.0
                message = f"{type(exc).__name__}: {exc}"
                errors[route.name] = message
                if isinstance(exc, (CapabilityError, CircuitOpenError)):
                    broken_capability = getattr(exc, "capability", "embed_model")
                results[route.name] = RouteResult(
                    route=route.name, hits=[], latency_ms=latency_ms, enabled=False,
                    query_used=text, note=f"检索失败：{type(exc).__name__}", candidates=0,
                    error=message,
                )
                continue
            results[route.name] = RouteResult(
                route=route.name, hits=hits, latency_ms=latency_ms, enabled=True,
                query_used=text, note=note, candidates=len(hits),
            )

        fused: list[FusedHit] = []
        if fuse:
            fused = rrf_fuse_routes(
                {name: result.ids for name, result in results.items() if result.enabled},
                rrf_k=self.rrf_k,
                top_k=k,
            )

        degraded = degraded_from_errors(errors, capability=broken_capability)
        debug = {
            "k": k,
            "rrf_k": self.rrf_k,
            "routes": list(results),
            "disabled": sorted(disabled),
            "candidates": {name: result.candidates for name, result in results.items()},
            "latency_ms": {name: round(result.latency_ms, 3) for name, result in results.items()},
            "query_used": {name: result.query_used for name, result in results.items()},
            "notes": {name: result.note for name, result in results.items() if result.note},
            "errors": dict(errors),
            "degraded": degraded.as_dict() if degraded else None,
            "n_fused": len(fused),
            "total_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        return RetrievalResult(
            query_zh=query_zh, query_en=query_en, k=k, routes=results, fused=fused,
            debug=debug, degraded=degraded,
        )


def default_route_names() -> tuple[str, ...]:
    return DEFAULT_ROUTES
