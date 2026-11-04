"""把三路召回接到真实存储上（Qdrant + DocumentStore + EmbeddingService）。

只做装配，不做逻辑——逻辑都在 `routes.py` / `fusion.py` / `orchestrator.py`，
所以那三个模块可以完全用假协作者测。

缓存：英文 BM25 索引（D25 的正分词器）落地在 `storage/cache/parent_bm25_en.json`，
stamp = `sha256(parent_count:max(source_end))`。语料没变就不重建（约 15–20 s）。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

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
    """从 Document Store 拉父块（id / 正文 / 标题 / 章节 / 来源）。"""
    from ..store.document_store import DocumentStore

    store = DocumentStore(store_path or (STORAGE_DIR / "documents.sqlite3"))
    out: list[dict[str, Any]] = []
    with store._connect() as conn:      # noqa: SLF001  同包内部
        rows = conn.execute(
            "SELECT p.parent_id, p.content, p.title, p.section_path, d.source_uri"
            " FROM parents p JOIN documents d ON d.document_id = p.document_id"
        ).fetchall()
    for row in rows:
        out.append({
            "id": row["parent_id"],
            "chunk_id": row["parent_id"],
            "text": row["content"] or "",
            "title": row["title"],
            "section_path": row["section_path"],
            "source": row["source_uri"],
        })
    return out


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
) -> MultiRouteRetriever:
    """三路（可选四路）真实装配：`dense_zh` + `dense_en` + `lexical_en`（+ `lexical_zh`）。

    `warmup_texts`：一次性批量预编码（走 EmbeddingService 的批量 + SQLite 缓存）。
    不热身的话 `dense_en` 的每条新 query 都是冷调用 ~2 s，会把 P50 完全带偏
    ——D28 报延迟前必须热身，这是硬要求。
    """
    from ..embedding.service import EmbeddingService
    from ..store.qdrant_store import COLLECTION_NAME, make_client
    from ..store.search import qdrant_search

    client = make_client(timeout=180)
    collection = collection or COLLECTION_NAME
    embedder = EmbeddingService()
    parents = load_parent_records(store_path)

    def searcher(vector, k):
        results = qdrant_search(client, collection, vector, k=k)
        return [
            (result.payload.get("chunk_id"), result.payload.get("parent_id"), result.score)
            for result in results
        ]

    def embed(texts):
        return embedder.encode_queries(list(texts))

    if warmup_texts:
        started = time.perf_counter()
        embedder.encode_queries(list(dict.fromkeys(warmup_texts)))
        log(f"[warmup] 预编码 {len(set(warmup_texts))} 条 query / "
            f"{time.perf_counter() - started:.1f} s")

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

    return MultiRouteRetriever(routes, rrf_k=rrf_k)
