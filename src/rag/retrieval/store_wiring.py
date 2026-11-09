"""把三路召回接到真实存储上（Qdrant + DocumentStore + EmbeddingService）。

只做装配，不做逻辑——逻辑都在 `routes.py` / `fusion.py` / `orchestrator.py`，
所以那三个模块可以完全用假协作者测。

缓存：英文 BM25 索引（D25 的正分词器）落地在 `storage/cache/parent_bm25_en.json`，
stamp = `sha256(parent_count:max(source_end))`。语料没变就不重建（约 15–20 s）。
"""

from __future__ import annotations

import os

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .degrade import Breaker, guarded
from .orchestrator import MultiRouteRetriever
from .routes import (
    ROUTE_DENSE_EN,
    ROUTE_DENSE_ZH,
    ROUTE_LEXICAL_EN,
    ROUTE_LEXICAL_ZH,
    DenseRoute,
    JiebaLexicalRoute,
    LexicalRoute,
)

STORAGE_DIR = Path(__file__).resolve().parents[3] / "storage"


def load_parent_records(store_path: str | Path | None = None) -> list[dict[str, Any]]:
    """从 Document Store 拉父块（id / 正文 / 标题 / 章节 / 来源 / 文档 / 序号）。

    `document_id` 与 `parent_index` 是 D32 组装器判「同文档相邻」的唯一依据：
    少了它们，`assemble()` 会把所有块当成不同文档、`adjacent_overlap_ratio` 也不可信。
    """
    from ..store.document_store import DocumentStore

    store = DocumentStore(store_path or (STORAGE_DIR / "documents.sqlite3"))
    out: list[dict[str, Any]] = []
    with store._connect() as conn:      # noqa: SLF001  同包内部
        rows = conn.execute(
            "SELECT p.parent_id, p.document_id, p.parent_index, p.content, p.title,"
            " p.section_path, COALESCE(p.language, d.language) AS language, d.source_uri"
            " FROM parents p JOIN documents d ON d.document_id = p.document_id"
        ).fetchall()
    for row in rows:
        out.append({
            "id": row["parent_id"],
            "chunk_id": row["parent_id"],
            "document_id": row["document_id"],
            "parent_index": row["parent_index"],
            "text": row["content"] or "",
            "title": row["title"],
            # parents.section_path 落库是 JSON 文本；原样传下去会让 assemble() 把字符串
            # 按字符拆开（N11：prompt 里每块头部变成 " > 深 > 入 > …"）
            "section_path": _path_list(row["section_path"]),
            # 键名必须是 `source_uri`：组装器读的是 record["source_uri"]。
            # 原先写成 "source"，于是每个块的来源路径都是空串 —— CLI 来源表只有标题、
            # 网页端「来源」列全空、SSE 的 sources 事件里 source_uri: ""（A5 以陌生人
            # 视角看来源表才发现）。`source` 保留给老调用方。
            "source_uri": row["source_uri"],
            "source": row["source_uri"],
            # 语言是 C0d「按候选块语言选重排 query」的依据：没有它就只能在重排时靠正则猜，
            # 而库里本来就有这一列（`parents.language`），没理由现场猜。
            "language": row["language"],
        })
    return out


def _path_list(raw: Any) -> list[str]:
    """把库里的 JSON 文本转成 list[str]；None/坏数据一律退化成空列表。"""
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _parents_stamp(store_path: str | Path | None = None) -> str:
    from ..store.document_store import DocumentStore

    store = DocumentStore(store_path or (STORAGE_DIR / "documents.sqlite3"))
    with store._connect() as conn:      # noqa: SLF001
        row = conn.execute("SELECT count(*), max(source_end) FROM parents").fetchone()
    return hashlib.sha256(f"{row[0]}:{row[1]}".encode()).hexdigest()[:16]


def build_english_bm25(
    parents: list[dict[str, Any]] | None = None,
    *,
    store_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
):
    """英文正分词器 BM25（D25 的 `src/rag/store/lexical.py`），带磁盘缓存。"""
    from ..store.lexical import BM25Index, build_bm25

    cache_dir = Path(cache_dir or (STORAGE_DIR / "cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "parent_bm25_en.json"
    stamp = _parents_stamp(store_path)

    if cached.exists():
        try:
            payload = json.loads(cached.read_text(encoding="utf-8"))
            if payload.get("stamp") == stamp:
                index = BM25Index()
                for item in payload["docs"]:
                    index.add(item["chunk_id"], item["parent_id"], item["text"])
                index.build()
                log(f"[lexical_en] 复用缓存（{index.size} 父块）")
                return index
        except (ValueError, KeyError):
            pass

    started = time.perf_counter()
    parents = parents if parents is not None else load_parent_records(store_path)
    index = build_bm25(parents)
    payload = {
        "stamp": stamp,
        "docs": [
            {"chunk_id": row["id"], "parent_id": row["id"], "text": row["text"]}
            for row in parents
        ],
    }
    cached.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log(f"[lexical_en] 建索引：{index.size} 父块 / {time.perf_counter() - started:.1f} s")
    return index


def build_default_retriever(
    *,
    collection: str | None = None,
    store_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    pool_factor: int = 4,
    rrf_k: int = 60,
    include_zh_lexical: bool = False,
    warmup_texts: Sequence[str] = (),
    log: Callable[[str], None] = print,
    breaker: Breaker | None = None,
) -> MultiRouteRetriever:
    """三路（可选四路）真实装配：`dense_zh` + `dense_en` + `lexical_en`（+ `lexical_zh`）。

    `warmup_texts`：一次性批量预编码（走 EmbeddingService 的批量 + SQLite 缓存）。
    不热身的话 `dense_en` 的每条新 query 都是冷调用 ~2 s，会把 P50 完全带偏
    ——D28 报延迟前必须热身，这是硬要求。
    """
    from ..embedding.service import EmbeddingService
    from ..store.qdrant_store import default_collection, make_client
    from ..store.search import qdrant_search

    client = make_client(timeout=180)
    collection = collection or default_collection()
    embedder = EmbeddingService()
    parents = load_parent_records(store_path)

    def searcher(vector, k):
        results = qdrant_search(client, collection, vector, k=k)
        return [
            (result.payload.get("chunk_id"), result.payload.get("parent_id"), result.score)
            for result in results
        ]

    # 两路 dense **共用一个 embed 能力 + 一个熔断**：Ollama 挂掉时一次失败就够，
    # 不该每个请求、每路各等 2.9 s（requests 的重试），也不该让两个 breaker 各自半开
    embed_breaker = breaker or Breaker(ttl_s=float(os.getenv("RAG_BREAKER_TTL", "30")))

    def embed(texts):
        return embedder.encode_queries(list(texts))

    embed = guarded(embed, embed_breaker, capability="embed_model")

    if warmup_texts:
        # 热身是**优化**不是前提：Ollama 没起时它必然失败，此时应带着已打开的熔断
        # 继续启动（后续 dense 路自动跳过），而不是让整个 CLI / 服务起不来
        started = time.perf_counter()
        try:
            embed(list(dict.fromkeys(warmup_texts)))
            log(f"[warmup] 预编码 {len(set(warmup_texts))} 条 query / "
                f"{time.perf_counter() - started:.1f} s")
        except Exception as exc:                  # noqa: BLE001
            log(f"[warmup] 跳过（向量模型不可用：{type(exc).__name__}）"
                f"｜dense 两路将降级，lexical 仍可用")

    routes = [
        DenseRoute(name=ROUTE_DENSE_ZH, embedder=embed, searcher=searcher,
                   pool_factor=pool_factor, input_lang="zh"),
        DenseRoute(name=ROUTE_DENSE_EN, embedder=embed, searcher=searcher,
                   pool_factor=pool_factor, input_lang="en"),
        LexicalRoute(index=build_english_bm25(parents, store_path=store_path,
                                              cache_dir=cache_dir, log=log),
                     name=ROUTE_LEXICAL_EN, input_lang="en"),
    ]

    if include_zh_lexical:
        from ...bm25 import BM25Index, build_bm25

        zh_records = [dict(row, chunk_id=row["id"], parent_id=row["id"]) for row in parents]
        zh_index = BM25Index(build_bm25(zh_records))
        routes.append(JiebaLexicalRoute(index=zh_index, records=zh_records,
                                        name=ROUTE_LEXICAL_ZH, input_lang="zh"))

    retriever = MultiRouteRetriever(routes, rrf_k=rrf_k)
    retriever.breaker = embed_breaker          # 供 /health、doctor 读当前状态
    return retriever
