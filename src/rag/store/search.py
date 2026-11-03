"""D19：Dense 检索（Qdrant）+ 与精确检索基准对照。

对外三件事：

1. `qdrant_search()` —— 带 payload 过滤的 dense 检索（打分降序）；
2. `exact_topk()` —— 同一批向量的 numpy 暴力 Top-K，**标准答案**；
3. `ann_recall()` —— ANN 结果与标准答案的交集比例。

为什么暴力实现必须留着：HNSW 是近似的，`ef` 调小会变快也会丢结果，而"丢了"在返回
结果里看不出来。没有精确 Top-K 做分母，任何「ANN 召回 100%」的说法都是自欺。

两个实现细节（踩过）：
- 精确检索必须用**从 Qdrant scroll 出来的同一批向量**，不能另取数据源；
- Qdrant 的 Cosine 返回的是**相似度**（越大越像），不是距离，别当距离排序。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .qdrant_store import COLLECTION_NAME, make_client

SCROLL_PAGE = 1000


@dataclass
class SearchResult:
    point_id: str
    chunk_id: str
    score: float
    payload: dict[str, Any]


@dataclass
class ExactResult:
    ids: list[int]           # 指向传入矩阵的行号
    scores: np.ndarray


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """逐行 L2 归一化。全零行保持全零（不产生 NaN）。"""
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def exact_topk(matrix: np.ndarray, query: np.ndarray, k: int) -> ExactResult:
    """精确 Top-K：余弦相似度 = 归一化后的点积。"""
    normalized = normalize_rows(matrix)
    q = normalize_rows(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
    scores = normalized @ q
    order = np.argsort(-scores)[: min(k, len(scores))]
    return ExactResult(ids=[int(i) for i in order], scores=scores[order])


def ann_recall(ann_ids: Iterable[Any], exact_ids: Sequence[Any]) -> float:
    """召回 = |ANN ∩ 精确| / |精确|。

    分母用**精确结果的条数**，不是 ANN 返回的条数：ANN 少返回几条时不能被算成满分。
    """
    exact_list = list(exact_ids)
    if not exact_list:
        return 0.0
    return len(set(ann_ids) & set(exact_list)) / len(exact_list)


def qdrant_search(
    client,
    collection: str = COLLECTION_NAME,
    query: Sequence[float] | np.ndarray | None = None,
    *,
    k: int = 50,
    query_filter=None,
    with_payload: bool = True,
    ef: int | None = None,
) -> list[SearchResult]:
    """dense 检索。`ef` 用来做 D37 的 HNSW 参数实验（越大越准越慢）。"""
    from qdrant_client import models

    vector = np.asarray(query, dtype=np.float32).reshape(-1).tolist()
    search_params = None
    if ef is not None:
        search_params = models.SearchParams(hnsw_ef=int(ef))
    points = client.query_points(
        collection,
        query=vector,
        limit=k,
        query_filter=query_filter,
        with_payload=with_payload,
        search_params=search_params,
    ).points
    results = []
    for point in points:
        payload = point.payload or {}
        results.append(SearchResult(
            point_id=str(point.id),
            chunk_id=payload.get("chunk_id") or str(point.id),
            score=float(point.score),
            payload=payload,
        ))
    return results


def scroll_vectors(client, collection: str = COLLECTION_NAME, *, limit: int | None = None
                   ) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """把 collection 里的向量与 payload 全量拉出来（D19 做标准答案用）。"""
    vectors: list[list[float]] = []
    point_ids: list[str] = []
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=SCROLL_PAGE, offset=offset, with_payload=True, with_vectors=True
        )
        for point in points:
            vectors.append(list(point.vector))
            point_ids.append(str(point.id))
            payloads.append(point.payload or {})
        if offset is None or (limit is not None and len(vectors) >= limit):
            break
    matrix = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)
    return matrix, point_ids, payloads


def measure_ann_recall(
    client,
    collection: str = COLLECTION_NAME,
    *,
    queries: Sequence[Sequence[float]],
    k: int = 50,
    matrix: np.ndarray | None = None,
    point_ids: Sequence[str] | None = None,
    ef: int | None = None,
) -> dict[str, Any]:
    """对给定 query 批量算 ANN Recall（D19/D36 的度量入口）。"""
    if matrix is None or point_ids is None:
        matrix, point_ids, _payloads = scroll_vectors(client, collection)
    id_to_row = {pid: index for index, pid in enumerate(point_ids)}
    recalls, per_query = [], []
    for query in queries:
        hits = qdrant_search(client, collection, query, k=k, ef=ef)
        ann_rows = [id_to_row[h.point_id] for h in hits if h.point_id in id_to_row]
        exact = exact_topk(matrix, np.asarray(query, dtype=np.float32), k)
        recall = ann_recall(ann_rows, exact.ids)
        recalls.append(recall)
        per_query.append({"k": k, "returned": len(hits), "recall": round(recall, 4)})
    return {
        "collection": collection,
        "points": int(matrix.shape[0]) if matrix.size else 0,
        "k": k,
        "ef": ef,
        "queries": len(queries),
        "recall_mean": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "recall_min": round(float(np.min(recalls)), 4) if recalls else 0.0,
        "per_query": per_query,
    }


# --------------------------------------------------------------------- CLI


def cli_main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="search", description="Dense 检索与 ANN Recall 基准（D19）")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--ef", type=int, default=None, help="HNSW ef（越大越准越慢）")
    parser.add_argument("--queries", type=int, default=10, help="用集合里随机取多少个向量当 query")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query", default=None, help="用文本 query 演示检索（走 Ollama）")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    client = make_client()
    try:
        if args.query:
            from ..embedding.service import EmbeddingService

            vector = EmbeddingService().encode_queries([args.query])[0]
            hits = qdrant_search(client, args.collection, vector, k=args.k, ef=args.ef)
            for rank, hit in enumerate(hits, start=1):
                source = hit.payload.get("document_id", "?")[:14]
                text = (hit.payload.get("text") or "")[:60].replace("\n", " ")
                print(f"  {rank:2d}. {hit.score:.4f}  {source}  {text}")
            return 0
        matrix, point_ids, _ = scroll_vectors(client, args.collection)
        if matrix.size == 0:
            print("collection 是空的，先用 scripts/ingest.py 灌数据")
            return 1
        rng = np.random.default_rng(args.seed)
        picks = rng.choice(matrix.shape[0], size=min(args.queries, matrix.shape[0]), replace=False)
        report = measure_ann_recall(
            client, args.collection, queries=[matrix[i] for i in picks],
            k=args.k, matrix=matrix, point_ids=point_ids, ef=args.ef,
        )
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"  points={report['points']} k={report['k']} ef={report['ef']} "
                  f"queries={report['queries']}")
            print(f"  ANN Recall@{report['k']}: mean={report['recall_mean']:.4f} "
                  f"min={report['recall_min']:.4f}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(cli_main())
