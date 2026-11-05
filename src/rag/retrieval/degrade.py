"""检索降级：**没有 Ollama 也要能给出「能给的答案」**，并且要说清楚自己降级了。

三件互相独立的东西，缺一件降级就是假的：

| 机制 | 解决什么 | 不做会怎样 |
|---|---|---|
| `guarded()` + `Breaker` | Ollama 没起时 `embed_texts` 要 **2.9 s** 才失败（requests 重试） | 每个请求都白烧 2.9 s，降级比不降级还慢 |
| 编排层单路容错 | `dense_zh` 抛异常会带走 `lexical_en` | 一路挂全挂（改造前 `src/rag/retrieval/` 里一个 `except` 都没有） |
| `DegradedInfo` | 静默降级 = 用户以为「知识库里没有」 | 答案变差但没人知道为什么 |

两条反面约束（`tests/test_degradation.py` 盯着）：

- **普通 bug 不许冒充「缺模型」**：路由里抛 `TypeError` 也记成 embed 不可用，会让人去重装 Ollama；
- **不许永久降级**：TTL 到期后必须允许半开重试，否则用户起好 Ollama 还得重启服务。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


class CircuitOpenError(RuntimeError):
    """熔断打开期间的短路异常：**没有真的发请求**，所以也不会再等 2.9 s。"""

    capability = "embed_model"


class CapabilityError(RuntimeError):
    """某个**外部能力**（向量模型等）调用失败。

    为什么要专门一个类型：光看异常文本判断「是不是 Ollama 挂了」会误判
    （`TypeError` 里恰好带 "ollama" 就会去重装模型）。`guarded()` 只包住那一个
    能力调用，所以从它逃出来的异常**按构造就**是该能力的故障。
    """

    def __init__(self, capability: str, message: str) -> None:
        super().__init__(message)
        self.capability = capability


@dataclass(frozen=True)
class DegradedInfo:
    """「哪些能力缺了 + 怎么修」。字段刻意与 `src/rag/ops/probes.py` 的 `Check` 对齐。"""

    reason: str                     # embed_unavailable | route_failed
    missing: list[str] = field(default_factory=list)     # ["embed_model"] / 路由名
    fix: str = ""
    detail: str = ""
    routes: list[str] = field(default_factory=list)      # 受影响的检索路

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "missing": list(self.missing),
            "fix": self.fix,
            "detail": self.detail,
            "routes": list(self.routes),
        }

    def render(self) -> str:
        rows = [f"[降级] {self.reason}：{self.detail or '部分检索路不可用'}"]
        if self.routes:
            rows.append(f"  停用的路：{', '.join(self.routes)}")
        if self.fix:
            rows.extend(f"  修复：{line}" for line in self.fix.splitlines())
        return "".join(chr(10) + row for row in rows)


class Breaker:
    """失败即打开的简易熔断（单飞服务够用，不引入状态机库）。

    - 失败 → 打开，`ttl_s` 内 `is_open()` 为真，调用方短路；
    - TTL 到期 → `is_open()` 转假（半开）：允许一次真实尝试，成功则闭合、失败则重新计时。
    """

    def __init__(self, *, ttl_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.ttl_s = float(ttl_s)
        self._clock = clock
        self._opened_at: float | None = None
        self.detail = ""

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        return (self._clock() - self._opened_at) < self.ttl_s

    def record_failure(self, detail: str = "") -> None:
        self._opened_at = self._clock()
        self.detail = detail or "未知错误"

    def record_success(self) -> None:
        self._opened_at = None
        self.detail = ""

    @property
    def age_s(self) -> float | None:
        return None if self._opened_at is None else self._clock() - self._opened_at


def guarded(fn: Callable[[Sequence[str]], Any], breaker: Breaker,
            *, capability: str = "embed_model") -> Callable[[Sequence[str]], Any]:
    """把熔断套在某个外部调用上（这里是 embedder）。

    熔断打开时抛 `CircuitOpenError`（**不再发起网络请求**），由编排层记进
    `RouteResult.error` 并汇总成 `DegradedInfo(reason="embed_unavailable")`。
    """

    def call(*args: Any, **kwargs: Any) -> Any:
        if breaker.is_open():
            raise CircuitOpenError(
                f"熔断打开（{breaker.age_s:.0f}s 前失败：{breaker.detail}），{breaker.ttl_s:.0f}s 内不再重试"
            )
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            breaker.record_failure(f"{type(exc).__name__}: {exc}")
            # 包成 CapabilityError：上层据此判定"缺的是哪项能力"，不靠猜文本
            raise CapabilityError(capability, f"{type(exc).__name__}: {exc}") from exc
        breaker.record_success()
        return result

    call.__wrapped__ = fn          # type: ignore[attr-defined] - 便于测试与排查取回原函数
    return call


#: embed 能力不可用时给用户看的修复命令（与 doctor 的 `fix` 口径一致）
EMBED_FIX = (
    "ollama serve  # 先确认 Ollama 在跑（桌面版/服务均可）" + chr(10)
    + "ollama pull bge-m3  # 向量模型必须与建库时一致：bge-m3（换模型需整库重建）"
)

#: 降级后**还能做什么**——只说缺什么会让人以为服务挂了
EMBED_NOTE = "降级期间只剩 lexical_en（英文词面）：中文问题基本搜不到，英文关键词仍可用"

_MAX_DETAIL = 240


def _short(text: str, limit: int = _MAX_DETAIL) -> str:
    """错误文本压成一行：`detail` 是给人看的一句话，不是堆栈。"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def degraded_from_errors(errors: dict[str, str],
                         *, capability: str | None = None) -> DegradedInfo | None:
    """`{路由名: 错误文本}` + 出问题的能力 → `DegradedInfo`。

    判定**只看异常类型**（`CapabilityError` / `CircuitOpenError` 的 `.capability`），
    不看错误文本里有没有 "ollama" 三个字母（N12 的教训）。
    """
    if not errors:
        return None
    if capability:
        detail = next(message for message in errors.values() if "Capability" in message
                      or "CircuitOpen" in message) if any(
            "Capability" in m or "CircuitOpen" in m for m in errors.values()) else next(iter(errors.values()))
        return DegradedInfo(
            reason="embed_unavailable" if capability == "embed_model" else f"{capability}_unavailable",
            missing=[capability],
            fix=(EMBED_FIX + chr(10) + "# " + EMBED_NOTE) if capability == "embed_model" else "",
            detail=_short(detail),
            routes=sorted(errors),
        )
    name, message = next(iter(errors.items()))
    return DegradedInfo(
        reason="route_failed",
        missing=sorted(errors),
        fix="看 detail 里的异常类型定位；单路故障不影响其它路，但融合结果会变差",
        detail=_short(f"{name}: {message}"),
        routes=sorted(errors),
    )
