"""SSE 桥：**同步生成器（线程）↔ async 响应**之间的翻译层。

为什么需要这一层（不是"多此一举"）：

- 生成是同步阻塞的（`requests` + Ollama），放进 `async def` 会卡死事件循环
  —— 一个请求就能让 `/health` 无响应；
- 但 SSE 必须是 async 生成器（要 `await` 判断客户端断开、要按时发心跳）；
- 于是：**生成跑在独立线程里，事件通过有界队列交给事件循环**。

三条只有长连接才会暴露的事，都在这层解决：

1. **断开即取消**：`request.is_disconnected()` 或响应被关（`GeneratorExit`）→ `cancel()`
   → 线程里的生成器 `close()` → `chat_stream` 的 `finally` 关掉 Ollama 连接。
   不取消的话，生成会继续烧完 50 s，而「1 跑 + 2 排」的名额很快被僵尸占满；
2. **心跳**：`sse_comment("ping")` 发 `: ping`（SSE 注释行，不产生事件），
   防反代/浏览器掐断长时间无数据的连接；
3. **超时保留已发内容**：超时发 `error(status=504, partial=已生成部分)`，
   不能因为最后一步失败就把前面 40 s 的 `sources` 与正文一起吞掉。

有界队列（`maxsize`）是故意的：线程不许无限堆积事件——慢客户端会反压到生成端，
而不是把内存吃光。
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, Callable, Iterable

#: 单条事件最多积压多少：慢客户端反压到生成端，而不是撑爆内存
EVENT_BUFFER = 64

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # 反代（nginx）默认会缓冲响应体 → 流式退化成"最后一次性吐出"
    "X-Accel-Buffering": "no",
}


def sse_event(name: str, data: Any) -> str:
    """一条 SSE 事件。`data` 单行（`json.dumps` 不换行），中文不转义方便人读。"""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}" + chr(10) + f"data: {payload}" + chr(10) + chr(10)


def sse_comment(text: str) -> str:
    """SSE 注释行：客户端忽略，但连接上**有字节流动**，反代不会掐断。"""
    return f": {text}" + chr(10) + chr(10)


class StreamBridge:
    """把一个同步事件生成器跑到线程里，主线程按需取事件。"""

    def __init__(self, factory: Callable[[], Iterable[dict]], *,
                 maxsize: int = EVENT_BUFFER) -> None:
        self._factory = factory
        self._events: queue.Queue = queue.Queue(maxsize=maxsize)
        self._cancelled = threading.Event()
        self._thread: threading.Thread | None = None

    # -- 生命周期 ----------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="rag-stream", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        generator = None
        try:
            generator = self._factory()
            for event in generator:
                # 取消后不再往队列里塞：让线程尽快退出，把 close 传到上游
                while not self._cancelled.is_set():
                    try:
                        self._events.put(event, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self._cancelled.is_set():
                    break
        except Exception as exc:                  # noqa: BLE001 - 兜底：绝不让线程静默死掉
            self._put_nowait({
                "event": "error",
                "data": {"status": 500, "detail": f"{type(exc).__name__}: {exc}",
                         "retryable": True, "partial": ""},
            })
        finally:
            if generator is not None:
                close = getattr(generator, "close", None)
                if callable(close):
                    close()          # ← 这里把 close 一路传到 chat_stream 的 finally
            self._put_nowait(None)   # 结束哨兵

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- 取事件 ------------------------------------------------------------

    def get(self, timeout: float) -> dict | None:
        """取一条事件；`None` 表示"暂时没有"（调用方据此发心跳/查断开）。"""
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def _put_nowait(self, event: dict | None) -> None:
        while True:
            try:
                self._events.put(event, timeout=0.2)
                return
            except queue.Full:
                if event is None:
                    return       # 队列满且已取消：哨兵丢了也无所谓，线程要退出了
                if self._cancelled.is_set():
                    return
