"""O3：把 `answer_query()` 变成 HTTP 服务（FastAPI）。

三条设计约束，都是被前面的坑逼出来的：

1. **依赖注入从第一天就做**：`create_app(deps=...)` 接收一个 `ApiDeps`，
   所以测试可以塞假 retriever / 假 reranker / 假模型，一个用例 0.1 s 跑完；
   而生产用一个"首次请求时才装配"的懒加载函数（`production_deps()`）——
   模块被 import 时**不加载 ONNX、不连 Qdrant、不读 25k 父块**，否则
   `uvicorn --reload` 和任何 `import src.rag.api.app` 都会莫名等十几秒。
2. **端点是 `def` 而不是 `async def`**：`answer_query()` 是同步阻塞的
   （ONNX 推理 0.7 s、Ollama 生成 50 s+）。用 `async def` + 阻塞调用会卡死事件循环，
   一个请求就能让 `/health` 无响应；写成 `def`，Starlette 自动丢进线程池。
3. **错误要可读**：模型/向量库炸了返回 500，`detail` 里带异常类型与消息
   （`RuntimeError: Ollama 挂了：connection refused`），不甩栈、不吞掉。

并发与流式（SSE、有界队列、超时、断开取消）**不在这一层**，见 O4a/O4b/O4c——
本文件只保证"一个请求进来，能拿到结构化的答案与来源"。
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .queue import QueueFullError, QueueTimeoutError, make_queue_from_env, SingleFlightQueue
from .streaming import SSE_HEADERS, StreamBridge, sse_comment, sse_event
from ..ask.pipeline import (
    DEFAULT_BUDGET,
    DEFAULT_K,
    DEFAULT_POOL,
    answer_query,
    answer_query_stream,
    retrieve_with_meta,
)
from .schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RetrieveHit,
    RetrieveResponse,
)

#: 版本号（开源发布时对齐 git tag；先写死，O11 打包时改为读包元数据）
API_VERSION = "0.1.0"

#: 正文预览长度：够判断"是不是这块"，又不至于让 /retrieve 返回一屏
PREVIEW_CHARS = 200

#: `translate` 的返回：(语言, 英文 query 或 None)
TranslateFn = Callable[[str], tuple[str, str | None]]


@dataclass
class ApiDeps:
    """HTTP 层需要的全部协作者。字段刻意少，且**都是协议级的**（有就用，不关心类型）。

    与 `src.ask.build_components()` 的区别：那边返回给 CLI 用的 dict（还带日志、argparse 语义），
    这边只描述"服务要什么"，这样测试不用伪造 argparse。
    """

    retriever: Any
    parents: Mapping[str, Mapping[str, Any]]
    counter: Any
    reranker: Any
    ask: Callable[[list[dict[str, str]]], str] | None = None
    translate: TranslateFn = lambda question: ("zh", None)   # noqa: E731 - 默认不做翻译
    routes: tuple[str, ...] = ("dense_zh", "dense_en", "lexical_en")
    version: str = API_VERSION
    #: 两条队列**分开**：生成慢且吃内存（单飞），检索快且纯 CPU（可并发）。
    #: 合成一条会让批量检索把问答堵死，也会让一次生成把检索饿死。
    ask_queue: Any = None
    retrieve_queue: Any = None
    #: SSE 心跳间隔（秒）：反代会掐断长时间无数据的连接
    heartbeat_s: float = 15.0
    #: 单次流式请求的总时长上限（秒）：超时发 `error(status=504, partial=...)`
    request_timeout_s: float = 300.0
    extra: dict[str, Any] = field(default_factory=dict)

    def queue_for(self, kind: str) -> Any:
        """没注入就按默认建一个（测试注入假的，生产按环境变量）。"""
        explicit = self.ask_queue if kind == "ask" else self.retrieve_queue
        if explicit is not None:
            return explicit
        if kind == "ask":
            self.ask_queue = make_queue_from_env()
        else:
            self.retrieve_queue = make_queue_from_env(
                prefix="RAG_RETRIEVE", concurrency=4, queue=2, timeout_s=5.0, name="检索")
        return self.ask_queue if kind == "ask" else self.retrieve_queue


# -- 生产装配（懒加载） ----------------------------------------------------

_LAZY_LOCK = threading.Lock()
_LAZY_DEPS: ApiDeps | None = None


def production_deps() -> ApiDeps:
    """真件装配：Qdrant 三路检索 + 父块 + 重排 + Ollama 生成 + 翻译。

    只在**第一次请求**时执行一次；`storage/` 里没有索引时这里会抛错，
    由 `/health` 的调用方看到具体原因（O5 会把这里拆成逐项探针）。
    """
    global _LAZY_DEPS
    if _LAZY_DEPS is not None:
        return _LAZY_DEPS
    with _LAZY_LOCK:
        if _LAZY_DEPS is not None:      # 双检：两个请求同时进来只装一次
            return _LAZY_DEPS

        from src import ollama_client
        from src.config import OLLAMA_CHAT_MODEL, STORAGE_DIR
        from src.rag.ingestion.tokenizer import TokenCounter
        from src.rag.lang.detect import detect
        from src.rag.retrieval.rerank import CrossEncoderReranker, RerankConfig
        from src.rag.retrieval.store_wiring import build_default_retriever, load_parent_records
        from src.rag.translate.cache import TranslationCache
        from src.rag.translate.service import TranslateService

        translator = TranslateService(
            provider=lambda prompt: ollama_client.chat(
                [{"role": "user", "content": prompt}], temperature=0.0, think=False
            ),
            cache=TranslationCache(STORAGE_DIR / "cache" / "translations.sqlite3",
                                   model=OLLAMA_CHAT_MODEL, version="v1"),
            max_retries=1,
        )

        def translate(question: str) -> tuple[str, str | None]:
            """语言识别 + 中文提问翻一路英文。翻译层五态出口，永不抛异常。"""
            language = detect(question)
            if language != "zh":
                return language, question
            result = translator.translate([question])[0]
            return language, (result.text or None)

        retriever = build_default_retriever(warmup_texts=[])
        parents = {row["id"]: row for row in load_parent_records()}
        _LAZY_DEPS = ApiDeps(
            retriever=retriever,
            parents=parents,
            counter=TokenCounter(),
            reranker=CrossEncoderReranker(config=RerankConfig()),
            ask=lambda messages: ollama_client.chat(messages, temperature=0.0, think=False),
            translate=translate,
            heartbeat_s=float(os.getenv("RAG_HEARTBEAT", "15")),
            request_timeout_s=float(os.getenv("RAG_REQUEST_TIMEOUT", "300")),
            ask_queue=make_queue_from_env(),
            retrieve_queue=make_queue_from_env(
                prefix="RAG_RETRIEVE", concurrency=4, queue=2, timeout_s=5.0, name="检索"),
        )
        return _LAZY_DEPS


# -- 组装 ------------------------------------------------------------------


def _paths(raw: Any) -> list[str]:
    """`section_path` 可能是 list、JSON 字符串或 None。HTTP 出口必须是 list（N11 的教训）。"""
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str) and raw.strip():
        import json

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        return [str(item) for item in parsed] if isinstance(parsed, list) else [str(parsed)]
    return []


def _busy(prefix: str, exc: QueueFullError | QueueTimeoutError) -> HTTPException:
    """429：**明确拒绝并告诉对方多久后回来**，不把人塞进注定超时的长队。"""
    retry_after = int(getattr(exc, "retry_after_s", 1) or 1)
    return HTTPException(
        status_code=429,
        detail=f"{prefix}{exc}".strip(),
        headers={"Retry-After": str(retry_after)},
    )


def _failed(prefix: str, exc: Exception) -> HTTPException:
    """统一 500：保留异常类型与消息，方便调用方与日志对账。"""
    return HTTPException(status_code=500, detail=f"{prefix}{type(exc).__name__}: {exc}")


def create_app(*, deps: ApiDeps | None = None) -> FastAPI:
    """建 app。`deps=None` 时走 `production_deps()`（首次请求才真正装配）。"""
    app = FastAPI(
        title="Local RAG Lab API",
        version=API_VERSION,
        description="本地 RAG 知识库问答：三路召回 + RRF + 重排 + 引用体检。",
    )

    def get_deps() -> ApiDeps:
        return deps if deps is not None else production_deps()

    @app.post("/api/v1/ask", response_model=AskResponse, tags=["rag"],
              summary="完整问答：检索 → 重排 → 组装 → 生成 → 引用体检")
    def ask_endpoint(request: AskRequest) -> AskResponse:
        current = get_deps()
        # `retrieve_only` 不调模型 → 不占生成名额（否则一次批量检索能把问答堵死）
        ticket = None
        if not request.retrieve_only:
            try:
                ticket = current.queue_for("ask").acquire()
            except (QueueFullError, QueueTimeoutError) as exc:
                raise _busy("生成队列已满：", exc) from exc
        try:
            language, query_en = current.translate(request.question)
            result = answer_query(
                request.question,
                retriever=current.retriever,
                parents=current.parents,
                counter=current.counter,
                reranker=current.reranker,
                ask=None if request.retrieve_only else current.ask,
                query_en=query_en,
                k=request.k,
                pool=request.pool,
                budget=request.budget,
                rerank=request.rerank,
            )
        except Exception as exc:                       # noqa: BLE001 - 统一转 500 + 可读 detail
            raise _failed("问答失败：", exc) from exc
        finally:
            if ticket is not None:
                ticket.release()

        payload = result.as_dict()
        # `as_dict()` 里 blocks 是**数量**（CLI 只需数量），HTTP 客户端要块本体才能自己渲染引用
        payload["blocks"] = result.blocks
        payload["block_count"] = len(result.blocks)
        payload["language"] = language or payload.get("language") or "zh"
        return AskResponse(**payload)

    @app.get("/api/v1/retrieve", response_model=RetrieveResponse, tags=["rag"],
             summary="只检索：返回融合+重排后的候选（不调用生成模型）")
    def retrieve_endpoint(
        q: str = Query(min_length=1, max_length=2000, description="查询文本"),
        k: int = Query(default=DEFAULT_K, ge=1, le=200, description="返回条数"),
        pool: int = Query(default=DEFAULT_POOL, ge=1, le=200, description="每路检索深度"),
        rerank: bool = Query(default=True, description="是否重排"),
        debug: bool = Query(default=False, description="附带 candidates 等诊断字段"),
    ) -> RetrieveResponse:
        current = get_deps()
        try:
            ticket = current.queue_for("retrieve").acquire()
        except (QueueFullError, QueueTimeoutError) as exc:
            raise _busy("检索队列已满：", exc) from exc
        try:
            language, query_en = current.translate(q)
            outcome = retrieve_with_meta(
                q, retriever=current.retriever, parents=current.parents,
                counter=current.counter, reranker=current.reranker,
                pool=pool, rerank=rerank,
            )
            ranked = outcome.hits
        except Exception as exc:                       # noqa: BLE001
            raise _failed("检索失败：", exc) from exc
        finally:
            ticket.release()

        hits: list[RetrieveHit] = []
        for position, hit in enumerate(ranked[:k], start=1):
            parent = current.parents.get(hit.parent_id) or {}
            hits.append(RetrieveHit(
                parent_id=hit.parent_id,
                rank=position,
                rrf_rank=hit.rank,
                score=round(float(hit.score), 6),
                rerank_score=None if hit.rerank_score is None else round(float(hit.rerank_score), 4),
                section_path=_paths(parent.get("section_path")),
                source_uri=parent.get("source_uri"),
                title=parent.get("title"),
                preview=(parent.get("text") or "")[:PREVIEW_CHARS],
            ))
        return RetrieveResponse(
            query=q,
            question_en=query_en,
            language=language,
            hits=hits,
            route_hits=outcome.route_hits,
            candidates=outcome.candidates,
            degraded=outcome.degraded.as_dict() if outcome.degraded else None,
        )

    @app.get("/api/v1/ask/stream", tags=["rag"],
             summary="流式问答（SSE）：queued → sources → delta* → done / error",
             response_class=StreamingResponse,
             responses={200: {"content": {"text/event-stream": {}}}})
    async def ask_stream_endpoint(
        request: Request,
        q: str = Query(min_length=1, max_length=2000, description="问题"),
        k: int = Query(default=DEFAULT_K, ge=1, le=50),
        pool: int = Query(default=DEFAULT_POOL, ge=1, le=200),
        budget: int = Query(default=DEFAULT_BUDGET, ge=500, le=32000),
        rerank: bool = Query(default=True),
    ) -> StreamingResponse:
        """用 GET 是为了浏览器 `EventSource` 能直接用（它只能发 GET、不能带 body）。

        HTTP 状态一旦开始流就是 200，**之后的失败用 `error` 事件表达**（带 `status`）。
        """
        current = get_deps()
        snapshot = current.queue_for("ask").snapshot()

        def factory():
            ask_queue = current.queue_for("ask")
            try:
                ticket = ask_queue.acquire()
            except (QueueFullError, QueueTimeoutError) as exc:
                yield {
                    "event": "error",
                    "data": {
                        "status": 429,
                        "detail": str(exc),
                        "retry_after_s": int(getattr(exc, "retry_after_s", 1) or 1),
                        "retryable": True,
                        "partial": "",
                    },
                }
                return
            try:
                language, query_en = current.translate(q)
                yield from answer_query_stream(
                    q,
                    retriever=current.retriever,
                    parents=current.parents,
                    counter=current.counter,
                    reranker=current.reranker,
                    ask=current.ask,
                    query_en=query_en,
                    k=k, pool=pool, budget=budget, rerank=rerank,
                )
            finally:
                ticket.release()

        bridge = StreamBridge(factory)

        async def event_stream():
            partial: list[str] = []
            yield sse_event("queued", {
                "inflight": snapshot["inflight"],
                "waiting": snapshot["waiting"],
                "max_inflight": snapshot["max_inflight"],
                "max_queue": snapshot["max_queue"],
                "retry_after_s": snapshot["retry_after_s"],
                # 名额已满 → 大概率要排一会儿，先说清楚，别让人对着空白等
                "will_wait": snapshot["inflight"] >= snapshot["max_inflight"],
            })
            bridge.start()
            started = time.monotonic()
            last_beat = started
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    event = bridge.get(0.05)
                    if event is None:
                        now = time.monotonic()
                        if now - started > current.request_timeout_s:
                            yield sse_event("error", {
                                "status": 504,
                                "detail": f"生成超时（>{current.request_timeout_s:.0f} s）",
                                "retryable": True,
                                "partial": "".join(partial),
                            })
                            break
                        if now - last_beat >= current.heartbeat_s:
                            last_beat = now
                            yield sse_comment("ping")
                        continue
                    if event["event"] == "delta":
                        partial.append(event["data"].get("text", ""))
                    yield sse_event(event["event"], event["data"])
                    if event["event"] in ("done", "error"):
                        break
            finally:
                # 断开（或被取消）→ 关掉线程里的生成器 → 关闭 Ollama 连接
                bridge.cancel()
                bridge.join(5.0)

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers=dict(SSE_HEADERS))

    @app.get("/health", response_model=HealthResponse, tags=["ops"],
             summary="存活探针（O5 会扩成逐项就绪检查）")
    def health_endpoint() -> HealthResponse:
        current = get_deps()
        return HealthResponse(
            status="ok",
            version=current.version,
            parents=len(current.parents),
            routes=list(current.routes),
            rerank_enabled=bool(getattr(getattr(current.reranker, "config", None), "enabled", True)),
            generation_enabled=current.ask is not None,
            queue=current.queue_for("ask").snapshot(),
        )

    @app.exception_handler(Exception)
    async def unhandled(request, exc: Exception) -> JSONResponse:   # noqa: ANN001
        """兜底：没被路由捕获的异常也回结构化 JSON，而不是 HTML 500 页。"""
        return JSONResponse(status_code=500,
                            content={"detail": f"{type(exc).__name__}: {exc}"})

    return app


#: `uvicorn src.rag.api.app:app` 用的实例（此处不装配，首个请求才加载真件）
app = create_app()

__all__ = ["ApiDeps", "app", "create_app", "production_deps", "API_VERSION"]
