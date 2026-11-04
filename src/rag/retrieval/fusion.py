"""应用层 RRF 多路融合（D26）。

`score(d) = Σ_route 1 / (rrf_k + rank_route(d))`，rank 从 1 开始，`rrf_k=60`
（Cormack et al. 2009 的常用取值）。

三条硬约束（都写成用例）：
1. **跨列表按 id 去重**：同一父块多路命中只输出一次
2. **同 id 多路命中分数累加**：不是取最大，是逐项相加
3. **并列名次可复现**：同分时按 `best_rank` 升序、再按 id 升序兜底

注意与 `union_pool` 的区别：并集是消融矩阵 C/D 的口径（各路 Top-k 的上限），
RRF 是 E 的口径（融合成一个 `k` 条的列表）。两者不能混用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    parent_id: str
    score: float
    routes: tuple[str, ...] = ()
    best_rank: int = 0
    per_route: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @property
    def n_routes(self) -> int:
        return len(self.routes)


def rrf_fuse_routes(
    lists: Mapping[str, Sequence[str]],
    *,
    rrf_k: int = RRF_K,
    top_k: int | None = None,
) -> list[FusedHit]:
    """多路 RRF 融合。`lists` = {路由名: 父块 id 的有序列表}。"""
    if rrf_k <= 0:
        raise ValueError(f"rrf_k 必须为正数（收到 {rrf_k}）")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    order: list[str] = []
    for route, ids in lists.items():
        seen_in_route: set[str] = set()
        for index, item_id in enumerate(ids or (), start=1):
            if not item_id or item_id in seen_in_route:
                continue      # 同一列表里重复出现只算一次（取最好名次）
            seen_in_route.add(item_id)
            if item_id not in scores:
                scores[item_id] = 0.0
                ranks[item_id] = {}
                order.append(item_id)
            scores[item_id] += 1.0 / (rrf_k + index)
            ranks[item_id][route] = index

    fused = [
        FusedHit(
            parent_id=item_id,
            score=float(scores[item_id]),
            routes=tuple(ranks[item_id].keys()),
            best_rank=min(ranks[item_id].values()),
            per_route=tuple(ranks[item_id].items()),
        )
        for item_id in order
    ]
    # 分数降序 → 最好名次升序 → id 升序（保证同分可复现）
    fused.sort(key=lambda hit: (-hit.score, hit.best_rank, hit.parent_id))
    if top_k is not None:
        fused = fused[: max(int(top_k), 0)]
    return fused


def union_pool(
    lists: Mapping[str, Sequence[str]],
    *,
    per_route_k: int | None = None,
) -> list[str]:
    """多路 Top-k 的**并集**（保序去重）。消融矩阵 C/D 的候选池口径。"""
    out: list[str] = []
    seen: set[str] = set()
    for ids in lists.values():
        limit = len(ids or ()) if per_route_k is None else max(int(per_route_k), 0)
        for item_id in list(ids or ())[:limit]:
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            out.append(item_id)
    return out
