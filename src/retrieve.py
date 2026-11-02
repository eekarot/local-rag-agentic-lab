import json
import time

import numpy as np

from .bm25 import BM25Index, build_bm25
from .config import EMBEDDINGS_PATH, HYBRID_ALPHA, INDEX_PATH, RRF_K, TOP_K
from .ollama_client import embed_texts
from .tokenizer import tokenize, tokenizer_name  # noqa: F401  (对外的 tokenize 兼容入口)

# 检索模式：vector / bm25（keyword 为旧别名）/ hybrid（RRF）/ hybrid_score（分数加权）
MODES = ("vector", "bm25", "hybrid", "hybrid_score")
_MODE_ALIASES = {"keyword": "bm25", "rrf": "hybrid", "weighted": "hybrid_score"}

_INDEX_CACHE = None
_BM25_CACHE = {"records": None, "index": None}
_MATRIX_CACHE = {"records": None, "matrix": None}


def load_index(force=False):
    """读取索引并缓存。BM25 倒排表与向量矩阵都依赖同一份 records 对象。"""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None and not force:
        return _INDEX_CACHE
    if not INDEX_PATH.exists():
        raise RuntimeError("找不到索引。请先运行：python -m src.build_index")
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if not data.get("records"):
        raise RuntimeError("索引为空，请检查 data/ 中的文档。")
    _INDEX_CACHE = data
    return data


def _resolve_mode(mode):
    mode = (mode or "hybrid").strip().lower()
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {'、'.join(MODES)}（keyword 为 bm25 的旧别名）")
    return mode


def get_bm25_index(records):
    """优先用索引文件里预建的倒排表；老索引没有就现场建一次并缓存。"""
    if _BM25_CACHE["records"] is records:
        return _BM25_CACHE["index"]
    payload = None
    if _INDEX_CACHE is not None and _INDEX_CACHE.get("records") is records:
        payload = _INDEX_CACHE.get("bm25")
    if not payload:
        payload = build_bm25(records)
    index = BM25Index(payload)
    _BM25_CACHE["records"] = records
    _BM25_CACHE["index"] = index
    return index


def _load_embeddings(records):
    """加载向量矩阵：优先读 embeddings.npy，老索引回退到 records[].embedding。"""
    n_records = len(records)
    if EMBEDDINGS_PATH.exists():
        vectors = np.load(EMBEDDINGS_PATH)
        if vectors.ndim != 2 or vectors.shape[0] != n_records:
            raise RuntimeError(
                f"{EMBEDDINGS_PATH.name} 与索引不匹配：向量矩阵 {vectors.shape}，"
                f"记录 {n_records} 条。请重建索引：python -m src.build_index"
            )
    elif records and "embedding" in records[0]:
        # 向后兼容：旧版索引把向量存在 JSON 里
        vectors = np.asarray([r["embedding"] for r in records], dtype=np.float32)
    else:
        raise RuntimeError(
            "索引缺少向量：既没有 embeddings.npy，records 里也没有 embedding 字段。"
            "请重建索引：python -m src.build_index"
        )
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _embedding_matrix(records):
    """把 embedding 预堆成归一化矩阵，每个进程只加载一次（按 records 对象身份缓存）。"""
    if _MATRIX_CACHE["records"] is records:
        return _MATRIX_CACHE["matrix"]
    matrix = _load_embeddings(records)
    _MATRIX_CACHE["records"] = records
    _MATRIX_CACHE["matrix"] = matrix
    return matrix


def warmup(load_vectors=True):
    """预热：解析索引 / 包 BM25 / 读一次向量矩阵，避免第一次查询出现尖峰。"""
    index = load_index()
    records = index["records"]
    get_bm25_index(records)
    if load_vectors:
        _embedding_matrix(records)
    return index


def _pack(record, score):
    """返回给上层的结果去掉 embedding，避免每条结果携带上千维浮点数组。"""
    return {
        "id": record["id"],
        "source": record["source"],
        "chunk_id": record["chunk_id"],
        "text": record["text"],
        "score": float(score),
    }


def vector_search(query, records, k=TOP_K, stats=None):
    """向量检索。stats 不为 None 时顺带记录候选规模与分数分布（供日志 review 用）。"""
    matrix = _embedding_matrix(records)
    started = time.perf_counter()
    qvec = np.asarray(embed_texts([query])[0], dtype=np.float32)
    embed_ms = (time.perf_counter() - started) * 1000
    qnorm = float(np.linalg.norm(qvec)) or 1.0
    started = time.perf_counter()
    scores = matrix @ (qvec / qnorm)
    top = np.argsort(-scores)[:k]
    if stats is not None:
        stats.update({
            "embed_ms": round(embed_ms, 2),
            "search_ms": round((time.perf_counter() - started) * 1000, 2),
            "candidates": int(scores.shape[0]),
            # 全库分布：负样本要用 (top1 - mean) / std 判断「离群程度」，
            # 绝对余弦不可跨模型比较。这两个数只花 0.2 ms，但让日志自带标尺。
            "cos_mean": float(scores.mean()),
            "cos_std": float(scores.std()),
            "top1_cos": float(scores[top[0]]) if len(top) else None,
            "topk_cos_floor": float(scores[top[-1]]) if len(top) else None,
            "ranks": {records[i]["id"]: rank for rank, i in enumerate(top, start=1)},
            "cos": {records[i]["id"]: float(scores[i]) for i in top},
        })
        if stats.get("debug") and len(top):
            # 只在 debug 级别算：99.9 分位反映「同分/平坦」程度，全量排序约 3 ms
            stats["cos_p999"] = float(np.percentile(scores, 99.9))
    return [_pack(records[i], scores[i]) for i in top]


def bm25_search(query, records, k=TOP_K, bm25=None, stats=None):
    index = bm25 or get_bm25_index(records)
    ranked = index.search(query, k, records, stats=stats)
    return [_pack(record, score) for record, score in ranked]


def keyword_search(query, records, k=TOP_K, bm25=None, stats=None):
    """旧函数名保留，内部已换成 BM25。"""
    return bm25_search(query, records, k=k, bm25=bm25, stats=stats)


def hybrid_search(query, records, k=TOP_K, bm25=None, rrf_constant=RRF_K, depth=3, stats=None):
    """RRF：只用排名融合，不比较不同量纲的分数。"""
    pool = max(k * depth, k)
    bm25 = bm25 or get_bm25_index(records)
    vec_stats = ({"debug": bool(stats.get("debug"))} if stats is not None else None)
    bm_stats = {} if stats is not None else None
    rankings = (
        vector_search(query, records, pool, stats=vec_stats),
        bm25_search(query, records, pool, bm25=bm25, stats=bm_stats),
    )
    fused = {}
    by_id = {r["id"]: r for r in records}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item["id"]] = fused.get(item["id"], 0.0) + 1.0 / (rrf_constant + rank)
    ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:k]
    if stats is not None:
        stats["vector"] = vec_stats
        stats["bm25"] = bm_stats
        stats["fuse"] = {
            "rule": "rrf",
            "rrf_k": rrf_constant,
            "pool": pool,
            "candidates_in": len(fused),
            "candidates_out": len(ids),
        }
    return [_pack(by_id[item_id], fused[item_id]) for item_id in ids if item_id in by_id]


def _minmax(items):
    if not items:
        return {}
    values = [item["score"] for item in items]
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return {item["id"]: 1.0 for item in items}
    return {item["id"]: (item["score"] - low) / (high - low) for item in items}


def hybrid_score_search(query, records, k=TOP_K, bm25=None, alpha=HYBRID_ALPHA, depth=3, stats=None):
    """分数加权融合：两路 min-max 归一化后按 alpha 相加（更看重高分项）。"""
    pool = max(k * depth, k)
    bm25 = bm25 or get_bm25_index(records)
    vec_stats = ({"debug": bool(stats.get("debug"))} if stats is not None else None)
    bm_stats = {} if stats is not None else None
    vec_hits = vector_search(query, records, pool, stats=vec_stats)
    bm25_hits = bm25_search(query, records, pool, bm25=bm25, stats=bm_stats)
    vec_norm = _minmax(vec_hits)
    bm25_norm = _minmax(bm25_hits)
    by_id = {r["id"]: r for r in records}
    candidates = {item["id"] for item in vec_hits} | {item["id"] for item in bm25_hits}
    scored = []
    # sorted() 不是为了好看：集合的遍历顺序受 PYTHONHASHSEED 影响，
    # 直接遍历 set 会让同分结果的顺序在不同进程间变化，评测无法复现。
    for item_id in sorted(candidates):
        score = alpha * bm25_norm.get(item_id, 0.0) + (1 - alpha) * vec_norm.get(item_id, 0.0)
        scored.append((item_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    if stats is not None:
        stats["vector"] = vec_stats
        stats["bm25"] = bm_stats
        stats["fuse"] = {
            "rule": "minmax",
            "alpha": alpha,
            "pool": pool,
            "candidates_in": len(candidates),
            "candidates_out": len(scored[:k]),
        }
    return [_pack(by_id[item_id], score) for item_id, score in scored[:k]]


def retrieve(query, mode="hybrid", k=TOP_K, records=None, stats=None):
    """检索入口。stats 仅在需要日志时传入，为 None 时行为与原来完全一致。"""
    mode = _resolve_mode(mode)
    if records is None:
        records = load_index()["records"]
    if mode == "vector":
        # 单路模式也要把统计塞回 stats["vector"]：
        # vector_search 是就地 update 传进去的那个 dict，不包一层的话
        # Trace.retrieve 读到 stats.get("vector") 是空的，日志就只剩 hit 没分数分布。
        vec_stats = {} if stats is not None else None
        results = vector_search(query, records, k, stats=vec_stats)
        if stats is not None:
            stats["vector"] = vec_stats
        return results
    if mode == "bm25":
        bm_stats = {} if stats is not None else None
        results = bm25_search(query, records, k, stats=bm_stats)
        if stats is not None:
            stats["bm25"] = bm_stats
        return results
    bm25 = get_bm25_index(records)
    if mode == "hybrid":
        return hybrid_search(query, records, k, bm25=bm25, stats=stats)
    return hybrid_score_search(query, records, k, bm25=bm25, stats=stats)


def index_stats(records=None):
    """给命令行 / 评测用的索引规模与分词方案信息。"""
    if records is None:
        records = load_index()["records"]
    bm25 = get_bm25_index(records)
    return {
        "chunks": len(records),
        "terms": len(bm25.postings),
        "avgdl": round(bm25.avgdl, 1),
        "tokenizer": tokenizer_name(),
        "k1": bm25.k1,
        "b": bm25.b,
    }
