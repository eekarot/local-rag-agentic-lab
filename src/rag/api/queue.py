"""有界队列 + 单飞（B14 定案）：**一次只生成一个，其余排队；排不下就明确拒绝**。

为什么不是「放开并发」（实测数字）：
- 生成 p50 **53.5 s**、reranker 常驻 **945 MB**、`max_concurrency=1`、可用内存 **5.9 GB**；
- CPU 上跑 9B，并发不会让 10 个人各快 10 倍，只会让**10 个人一起等更久**，最后一起超时；
- 所以宁可让第 4 个人在 0 ms 看到 `429 + Retry-After`，也不要让他在 60 s 后看到超时。

语义（`tests/test_api_queue.py` 盯着）：

    inflight <= max_inflight            正在跑的上限
    inflight + waiting <= max_inflight + max_queue   总准入上限（超出→QueueFullError→429）
    等待超过 timeout_s                  →QueueTimeoutError→429（带 Retry-After）

两个实现上的坑：
- `Condition.wait()` 可能被唤醒后发现名额又被别人抢走 → 必须**循环重查**条件；
- 等待用的是**超时等待**而不是 `sleep` 轮询 → 放名额时能立刻唤醒，少一个 50 ms 的抖动。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class QueueFullError(RuntimeError):
    """准入就失败：正在跑 + 排队已满。→ HTTP 429 + Retry-After。"""

    def __init__(self, message: str, *, retry_after_s: int = 1) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class QueueTimeoutError(RuntimeError):
    """排队超时：位子是有的，但太久没轮到。→ HTTP 429 + Retry-After。"""

    def __init__(self, message: str, *, retry_after_s: int = 1) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


@dataclass
class _Stats:
    served: int = 0
    rejected_full: int = 0
    rejected_timeout: int = 0
    waited_ms_total: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "served": self.served,
            "rejected_full": self.rejected_full,
            "rejected_timeout": self.rejected_timeout,
            "waited_ms_total": round(self.waited_ms_total, 1),
        }


@dataclass
class Ticket:
    """占住的名额。`release()` 幂等（重复调用不会把计数减成负数）。"""

    queue: "SingleFlightQueue"
    waited_ms: float = 0.0
    _released: bool = field(default=False, repr=False)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.queue.release(self)

    def __enter__(self) -> "Ticket":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.release()
        return False


class SingleFlightQueue:
    """有界信号量。名字里的 single-flight 指「默认只放 1 个生成进去」。"""

    def __init__(
        self,
        *,
        max_inflight: int = 1,
        max_queue: int = 2,
        timeout_s: float = 120.0,
        retry_after_s: float = 60.0,
        name: str = "生成",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_inflight < 1:
            raise ValueError(f"max_inflight 必须 ≥1（收到 {max_inflight}）")
        if max_queue < 0:
            raise ValueError(f"max_queue 不许为负（收到 {max_queue}）")
        self.max_inflight = int(max_inflight)
        self.max_queue = int(max_queue)
        self.timeout_s = float(timeout_s)
        self.retry_after_s = float(retry_after_s)
        self.name = name
        self._clock = clock
        self._cond = threading.Condition()
        self._inflight = 0
        self._waiting = 0
        self._stats = _Stats()

    # -- 准入 --------------------------------------------------------------

    def acquire(self, *, blocking: bool = True) -> Ticket:
        """拿一个名额。拿不到就是**明确的两类拒绝**，不返回 None、不静默排队到天荒地老。"""
        with self._cond:
            # 空位优先：先看能不能直接跑，再看排队容量（顺序反了会把空队列判满）
            if self._inflight < self.max_inflight:
                self._inflight += 1
                return Ticket(queue=self)

            if self._waiting >= self.max_queue:
                self._stats.rejected_full += 1
                raise QueueFullError(
                    f"{self.name}队列已满（{self.max_inflight} 个在跑 + {self.max_queue} 个排队），"
                    f"请 {self._retry_after()} s 后重试（Retry-After: {self._retry_after()}）。",
                    retry_after_s=self._retry_after(),
                )

            # 需要排队
            self._waiting += 1
            started = self._clock()
            deadline = started + self.timeout_s
            try:
                while self._inflight >= self.max_inflight:
                    remaining = deadline - self._clock()
                    if not blocking or remaining <= 0:
                        self._stats.rejected_timeout += 1
                        raise QueueTimeoutError(
                            f"{self.name}队列排队超时（{self.timeout_s:.0f} s 没轮到），"
                            f"请 {self._retry_after()} s 后重试（Retry-After: {self._retry_after()}）。",
                            retry_after_s=self._retry_after(),
                        )
                    self._cond.wait(remaining)
                self._inflight += 1
                waited = (self._clock() - started) * 1000.0
                return Ticket(queue=self, waited_ms=waited)
            finally:
                self._waiting -= 1

    def release(self, ticket: Ticket | None = None) -> None:
        with self._cond:
            self._inflight = max(0, self._inflight - 1)
            self._stats.served += 1
            if ticket is not None:
                self._stats.waited_ms_total += ticket.waited_ms
            self._cond.notify()          # 只叫醒一个：其他人在循环里会重新检查

    def __call__(self):
        """便于 `with queue:` 使用。"""
        return _TicketContext(self)

    # -- 观测 --------------------------------------------------------------

    def _retry_after(self) -> int:
        """`Retry-After` 用**预期服务时长**（≈生成 p50），而不是排队超时值。

        实测：生成 24–54 s，排队超时若设 20 s，排队者 100% 会 429 ——
        `max_queue` 就形同虚设。所以超时设 120 s（>2×p50，排到才算数），
        而 `Retry-After` 给 60 s（≈p50），让客户端回来时大概率能直接进。
        """
        return max(1, int(round(self.retry_after_s)))

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            return {
                "name": self.name,
            "max_inflight": self.max_inflight,
                "max_queue": self.max_queue,
                "timeout_s": self.timeout_s,
                "retry_after_s": self._retry_after(),
                "inflight": self._inflight,
                "waiting": self._waiting,
                **_stats_copy(self._stats),
            }


class _TicketContext:
    def __init__(self, queue: SingleFlightQueue) -> None:
        self._queue = queue
        self._ticket: Ticket | None = None

    def __enter__(self) -> Ticket:
        self._ticket = self._queue.acquire()
        return self._ticket

    def __exit__(self, *exc_info: Any) -> bool:
        if self._ticket is not None:
            self._ticket.release()
        return False


def _stats_copy(stats: _Stats) -> dict[str, Any]:
    return stats.as_dict()


def make_queue_from_env(*, prefix: str = "RAG", concurrency: int = 1, queue: int = 2,
                        timeout_s: float = 120.0, retry_after_s: float = 60.0,
                        name: str = "生成") -> SingleFlightQueue:
    """按环境变量建队列：`RAG_MAX_INFLIGHT` / `RAG_MAX_QUEUE` / `RAG_QUEUE_TIMEOUT` / `RAG_RETRY_AFTER`。"""
    import os

    def _int(name: str, default: int) -> int:
        raw = os.getenv(f"{prefix}_{name}")
        if raw is None or not str(raw).strip():
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"环境变量 {prefix}_{name} 必须是整数（收到 {raw!r}）") from exc

    def _float(name: str, default: float) -> float:
        raw = os.getenv(f"{prefix}_{name}")
        if raw is None or not str(raw).strip():
            return default
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"环境变量 {prefix}_{name} 必须是数字（收到 {raw!r}）") from exc

    return SingleFlightQueue(
        max_inflight=_int("MAX_INFLIGHT", concurrency),
        max_queue=_int("MAX_QUEUE", queue),
        timeout_s=_float("QUEUE_TIMEOUT", timeout_s),
        retry_after_s=_float("RETRY_AFTER", retry_after_s),
        name=name,
    )
