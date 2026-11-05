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

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from ..ask.pipeline import (
    DEFAULT_BUDGET,
    DEFAULT_K,
    DEFAULT_POOL,
    answer_query,
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
    extra: dict[str, Any] = field(default_factory=dict)


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
            language, query_en = current.translate(q)
            outcome = retrieve_with_meta(
                q, retriever=current.retriever, parents=current.parents,
                counter=current.counter, reranker=current.reranker,
                pool=pool, rerank=rerank,
            )
            ranked = outcome.hits
        except Exception as exc:                       # noqa: BLE001
            raise _failed("检索失败：", exc) from exc

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
