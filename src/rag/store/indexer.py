"""D16：幂等 Upsert + 文档版本管理 + 按文档删除/重建。

幂等的支点只有一个：**同一个 chunk 永远映射到同一个 point id**
（`point_id_for` 的 UUID5）。因此重复导入是「覆盖」而不是「新增」，
不需要「先删后插」那种会把数据删出空窗的做法。

版本变化的处理：
    现有该文档的点集（scroll 出来） − 本次要写的点集 = **陈旧点集** → 精确删除。
不用「按 document_id 全删再写」，因为那样中间会有一段时间库里没有这篇文档
（查询会短暂丢结果），而且失败后留下半篇数据比失败本身更难查。

写入顺序：**先 Document Store（事务），再 Qdrant（幂等）**。
反过来的话，Qdrant 写成功而 Store 写失败，就会出现「命中却回溯不到父块」的孤儿向量；
而按这个顺序失败，重跑一次即可自愈（Store 覆盖、Qdrant 覆盖 + 清陈旧点）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..embedding.service import EmbeddingService
from .document_store import DocumentStore, load_chunks_dir
from .errors import VectorStoreError
from .qdrant_store import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    QdrantClient,
    ensure_collection,
    make_client,
    payload_from_child,
    point_id_for,
)

#: scroll 拉点时的页大小（Qdrant 单页上限 10000）
SCROLL_PAGE = 1000


class IndexerError(VectorStoreError):
    """索引层错误。"""


class Indexer:
    """把「文档 + Parent + Child」写进 Qdrant 与 Document Store。"""

    def __init__(
        self,
        *,
        client: QdrantClient | None = None,
        collection: str = COLLECTION_NAME,
        store: DocumentStore | None = None,
        embedder: EmbeddingService | None = None,
        batch_size: int = 256,
        vector_size: int = VECTOR_SIZE,
    ):
        self.client = client or make_client()
        self.collection = collection
        self.store = store
        self.embedder = embedder
        self.batch_size = max(1, batch_size)
        self.vector_size = vector_size

    # -- 索引 -----------------------------------------------------------

    def index_document(
        self,
        document: Mapping[str, Any],
        parents: Sequence[Mapping[str, Any]],
        children: Sequence[Mapping[str, Any]],
        *,
        vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> dict[str, int]:
        """写入一篇文档。可重复调用（幂等）。"""
        document_id = document["document_id"]
        if not children:
            raise IndexerError(f"{document_id} 没有 child，拒绝写入空文档")
        stray = [c["chunk_id"] for c in children if c["document_id"] != document_id]
        if stray:
            raise ValueError(
                f"{len(stray)} 个 child 的 document_id 与参数不一致（例如 {stray[0]}）"
            )
        unmatched = sorted({c["parent_id"] for c in children} - {p["parent_id"] for p in parents})
        if unmatched:
            raise ValueError(
                f"{len(unmatched)} 个 child 的 parent_id 不在本次 parents 里（例如 {unmatched[0]}）"
            )

        ensure_collection(self.client, self.collection, vector_size=self.vector_size)

        # 1) 先落 Document Store（事务；失败则 Qdrant 完全没动）
        if self.store is not None:
            self.store.replace_document(document, parents, children)

        # 2) 算向量
        matrix = self._vectors_for(children, vectors)

        # 3) 记下这篇文档现有的点，用于事后清陈旧点
        existing = self._point_ids(document_id)

        # 4) 分批 upsert（point id 由 chunk_id 决定 → 覆盖而不是新增）
        upserted = 0
        for start in range(0, len(children), self.batch_size):
            batch = children[start:start + self.batch_size]
            points = [
                QdrantPoint(id=point_id_for(child["chunk_id"]),
                            vector=matrix[start + offset].tolist(),
                            payload=payload_from_child(child))
                for offset, child in enumerate(batch)
            ]
            self.client.upsert(self.collection, points=points, wait=True)
            upserted += len(points)

        # 5) 删除本次不再存在的点（版本变化 / 文档变短）
        fresh = {point_id_for(child["chunk_id"]) for child in children}
        stale = sorted(existing - fresh)
        if stale:
            self.client.delete(self.collection, points_selector=stale, wait=True)

        return {"documents": 1, "parents": len(parents), "children": len(children),
                "upserted": upserted, "deleted_stale": len(stale)}

    def index_chunks_dir(self, chunks_dir: str | os.PathLike[str], *, limit: int | None = None) -> dict[str, int]:
        """把一个 `chunks/` 目录逐篇写入（D17 的产物 → Qdrant）。"""
        grouped = load_chunks_dir(chunks_dir)
        totals = {k: 0 for k in ("documents", "parents", "children", "upserted", "deleted_stale")}
        items = list(grouped["by_document"].items())
        if limit is not None:
            items = items[:limit]
        for _document_id, bundle in items:
            counts = self.index_document(bundle["document"], bundle["parents"], bundle["children"])
            for key in totals:
                totals[key] += counts[key]
        return totals

    # -- 删除 -----------------------------------------------------------

    def delete_document(self, document_id: str) -> dict[str, int]:
        """按 `document_id` 删点 + 删 Store 行。"""
        ensured = False
        if self.collection in [c.name for c in self.client.get_collections().collections]:
            ensured = True
        deleted = 0
        if ensured:
            before = self.count(document_id)
            from qdrant_client import models

            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id))])
                ),
                wait=True,
            )
            deleted = before
        store_deleted = self.store.delete_document(document_id) if self.store is not None else 0
        return {"points": deleted, "documents": store_deleted}

    # -- 查询 -----------------------------------------------------------

    def count(self, document_id: str | None = None) -> int:
        if self.collection not in [c.name for c in self.client.get_collections().collections]:
            return 0
        from qdrant_client import models

        count_filter = None
        if document_id is not None:
            count_filter = models.Filter(must=[models.FieldCondition(
                key="document_id", match=models.MatchValue(value=document_id))])
        return int(self.client.count(self.collection, count_filter=count_filter, exact=True).count)

    def chunk_ids(self, document_id: str) -> set[str]:
        """该文档在 Qdrant 里的 chunk_id 集合（供 D18/D19 核对）。"""
        from qdrant_client import models

        found: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id))]),
                limit=SCROLL_PAGE,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            found.update(p.payload["chunk_id"] for p in points if p.payload and "chunk_id" in p.payload)
            if offset is None:
                return found

    def stats(self) -> dict[str, Any]:
        cfg = None
        if self.collection in [c.name for c in self.client.get_collections().collections]:
            from .qdrant_store import collection_config

            cfg = collection_config(self.client, self.collection)
        return {
            "collection": self.collection,
            "points": self.count(),
            "documents": self.store.count("documents") if self.store is not None else None,
            "parents": self.store.count("parents") if self.store is not None else None,
            "vector_size": cfg["vector_size"] if cfg else None,
        }

    # -- 内部 -----------------------------------------------------------

    def _vectors_for(
        self,
        children: Sequence[Mapping[str, Any]],
        vectors: Mapping[str, Sequence[float]] | None,
    ) -> np.ndarray:
        if vectors is not None:
            missing = [c["chunk_id"] for c in children if c["chunk_id"] not in vectors]
            if missing:
                raise IndexerError(f"{len(missing)} 个 chunk 没有提供向量（例如 {missing[0]}）")
            matrix = np.asarray([vectors[c["chunk_id"]] for c in children], dtype=np.float32)
            if matrix.shape != (len(children), self.vector_size):
                raise IndexerError(
                    f"向量矩阵形状 {matrix.shape} 与期望 ({len(children)}, {self.vector_size}) 不符"
                )
            return matrix
        if self.embedder is None:
            raise IndexerError("没有提供向量也没有 embedder；索引前必须二选一")
        # 编码用 embedding_text（Document/Section 前缀），存活文本用 content
        texts = [c.get("embedding_text") or c["content"] for c in children]
        return np.asarray(self.embedder.encode_documents(texts), dtype=np.float32)

    def _point_ids(self, document_id: str) -> set[str]:
        from qdrant_client import models

        if self.collection not in [c.name for c in self.client.get_collections().collections]:
            return set()
        found: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id))]),
                limit=SCROLL_PAGE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            found.update(str(p.id) for p in points)
            if offset is None:
                return found


#: 延迟导入包装，避免模块顶部和 qdrant 模型混在一起
from qdrant_client import models as _models  # noqa: E402


class QdrantPoint(_models.PointStruct):
    """`PointStruct` 的别名，让 index_document 里的构造读起来短一点。"""
