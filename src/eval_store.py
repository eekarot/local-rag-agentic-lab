"""新库（Qdrant parent/child 链路）的 L1 检索评估。

旧链路 `src.retrieve` 读 `storage/index.json` 的 81,027 个 chunk；新链路是
Qdrant `rag_chunks_v1`（38,865 child）+ Document Store（25,411 parent）。
两套 id 空间不相交，所以本模块**不复用** `src.retrieve`，只复用指标层
（`src.eval_logs` / `src.eval_metrics`）与日志层（`src.raglog`）——**指标仍然只从日志算**。

三个模式（全部在**父块**层排名，与 `gold_points.parent_ids` 的口径一致）：

| 模式 | 实现 | 为什么需要它 |
|---|---|---|
| `dense` | bge-m3 query 向量 → Qdrant ANN → 子块命中收敛成父块 | 语义召回，跨语言主力 |
| `lexical` | 父块正文 jieba BM25（`src.bm25`） | 型号/数字这类词面命中，中文分词 |
| `hybrid` | RRF 融合前两路（k=60） | 两路互不重叠的部分合起来 |

`rrf` 而不是权重和：两路分数量纲不同（余弦 vs BM25），加权和要调权重、且随库变化；
RRF 只用名次，无参数（除 k=60）且对单路退化安全。

用法：

    python -m src.eval_store run --k 10
    python -m src.evaluate measure latest --main-mode hybrid --write-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import raglog
from .config import QUESTIONS_PATH, STORAGE_DIR

MODES = ("dense", "lexical", "hybrid")

#: dense 先捞这么多子块，再收敛成父块（父子比 ~1.53，多捞 4 倍足够）
DEFAULT_POOL_FACTOR = 4
RRF_K = 60


# ---------------------------------------------------------------- 排名工具


def parent_ranked(hits: Iterable[Mapping[str, Any]], k: int) -> list[dict[str, Any]]:
    """子块命中 → 父块排名（去重保序，分取最高）。"""
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for hit in hits:
        parent_id = hit.get("parent_id") or hit.get("id")
        if not parent_id:
            continue
        score = float(hit.get("score") or 0.0)
        existing = best.get(parent_id)
        if existing is None:
            best[parent_id] = {
                "parent_id": parent_id,
                "chunk_id": hit.get("chunk_id"),
                "score": score,
                "title": hit.get("title"),
                "section_path": hit.get("section_path"),
                "source": hit.get("source"),
            }
            order.append(parent_id)
        elif score > existing["score"]:
            existing["score"] = score
            existing["chunk_id"] = hit.get("chunk_id") or existing["chunk_id"]
    ranked = [best[pid] for pid in order]
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked[:k]


def rrf_fuse(*rankings: Iterable[Mapping[str, Any]], k: int, rrf_k: int = RRF_K) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion：score = Σ 1/(rrf_k + rank)，rank 从 1 开始。"""
    scores: dict[str, float] = {}
    order: list[str] = []
    for ranking in rankings:
        seen: set[str] = set()
        for rank, row in enumerate(ranking, start=1):
            parent_id = row.get("parent_id") or row.get("id")
            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)
            if parent_id not in scores:
                scores[parent_id] = 0.0
                order.append(parent_id)
            scores[parent_id] += 1.0 / (rrf_k + rank)
    ranked = [{"parent_id": pid, "score": round(scores[pid], 6)} for pid in order]
    # 稳定排序：分数相同按首次出现顺序（`order` 已保证）
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked[:k]


def gold_parents(item: Mapping[str, Any]) -> set[str]:
    """题的父块级金标：优先 `gold_points`（新库），退回 `gold_chunks`（旧库）。"""
    points = (item.get("gold_points") or {}).get("parent_ids") or []
    if points:
        return set(points)
    return {chunk["id"] for chunk in item.get("gold_chunks") or [] if chunk.get("id")}


class CorpusCosStats:
    """query 对**全库**余弦的均值与标准差（负样本 z 分数的分母）。

    为什么不用「Top-k 的余弦」估：那是有偏的（全是高分）。为什么能不算全量：
    向量都是 L2 归一化的，所以 `cos(q, v) = q·v`，

        E[cos]   = q·μ,              μ = mean(v_i)
        E[cos²]  = qᵀ M q,           M = (1/N) Σ v_i v_iᵀ

    只要一次扫全库算出 μ（1024 维）和 M（1024×1024 ≈ 4 MB），之后每题都是 O(d²)，
    而且**精确**，不是抽样近似。
    """

    def __init__(self, mu, second_moment):
        self.mu = mu
        self.second_moment = second_moment

    @classmethod
    def build(cls, matrix):
        import numpy as np

        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.size == 0:
            raise ValueError("空矩阵")
        mu = matrix.mean(axis=0)
        moment = (matrix.T @ matrix) / matrix.shape[0]
        return cls(mu=mu, second_moment=moment.astype(np.float32))

    def mean(self, query) -> float:
        import numpy as np

        return float(np.dot(np.asarray(query, dtype=np.float32), self.mu))

    def second(self, query) -> float:
        import numpy as np

        q = np.asarray(query, dtype=np.float32)
        return float(q @ self.second_moment @ q)

    def std(self, query) -> float:
        import math

        variance = self.second(query) - self.mean(query) ** 2
        return math.sqrt(max(variance, 0.0))

    def cos_stats(self, query) -> dict[str, float]:
        return {"cos_mean": self.mean(query), "cos_std": self.std(query)}

    def z(self, query, cos: float) -> float:
        std = self.std(query)
        return (cos - self.mean(query)) / std if std else 0.0

    def save(self, path) -> None:
        import numpy as np

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mu=self.mu, second_moment=self.second_moment)

    @classmethod
    def load(cls, path):
        import numpy as np

        payload = np.load(path)
        return cls(mu=payload["mu"], second_moment=payload["second_moment"])


# ---------------------------------------------------------------- 真实检索器


class StoreRetriever:
    """把 Qdrant dense 与父块 BM25 包成统一接口。构造较贵（建 BM25），复用一个实例。"""

    def __init__(
        self,
        *,
        collection: str | None = None,
        store_path: str | None = None,
        pool_factor: int = DEFAULT_POOL_FACTOR,
        cache_dir: Path | None = None,
        log=print,
    ):
        from .rag.store.document_store import DocumentStore
        from .rag.store.qdrant_store import COLLECTION_NAME, make_client

        self.collection = collection or COLLECTION_NAME
        self.client = make_client(timeout=180)
        self._store = DocumentStore(store_path or (STORAGE_DIR / "documents.sqlite3"))
        self.pool_factor = pool_factor
        self._embedder = None
        self._bm25 = None
        self._parents: list[dict[str, Any]] = []
        self._cos_stats = None
        self._cache_dir = cache_dir or (STORAGE_DIR / "cache")
        self._log = log
        self._build_lexical()

    # -- 词面路 ----------------------------------------------------------
    def _load_parents(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._store._connect() as conn:      # noqa: SLF001  同包内部
            rows = conn.execute(
                "SELECT p.parent_id, p.content, p.title, p.section_path, d.source_uri"
                " FROM parents p JOIN documents d ON d.document_id = p.document_id"
            ).fetchall()
        for row in rows:
            out.append({
                "id": row["parent_id"],
                "text": row["content"] or "",
                "title": row["title"],
                "section_path": row["section_path"],
                "source": row["source_uri"],
            })
        return out

    def _build_lexical(self) -> None:
        from .bm25 import BM25Index, build_bm25

        cached = self._cache_dir / "parent_bm25.json"
        stamp = self._lexical_stamp()
        if cached.exists():
            try:
                payload = json.loads(cached.read_text(encoding="utf-8"))
                if payload.get("stamp") == stamp:
                    self._parents = payload["parents"]
                    self._bm25 = BM25Index(payload["bm25"])
                    self._log(f"[lexical] 复用父块 BM25 缓存（{self._bm25.n_docs} 父块）")
                    return
            except (ValueError, KeyError):
                pass

        started = time.perf_counter()
        self._parents = self._load_parents()
        payload = build_bm25(self._parents)
        self._bm25 = BM25Index(payload)
        elapsed = time.perf_counter() - started
        self._log(f"[lexical] 建父块 BM25：{len(self._parents)} 父块 / {elapsed:.1f} s")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(
            json.dumps({"stamp": stamp, "parents": self._parents, "bm25": payload}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _lexical_stamp(self) -> str:
        import hashlib

        with self._store._connect() as conn:      # noqa: SLF001
            row = conn.execute("SELECT count(*), max(source_end) FROM parents").fetchone()
        return hashlib.sha256(f"{row[0]}:{row[1]}".encode()).hexdigest()[:16]

    # -- 语义路 ----------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from .rag.embedding.service import EmbeddingService

            self._embedder = EmbeddingService()
        return self._embedder

    def cos_stats(self) -> "CorpusCosStats":
        """懒加载全库余弦统计（第一次约 20 s 扫全库向量，之后走 4 MB 缓存）。"""
        if self._cos_stats is None:
            cached = self._cache_dir / "corpus_cos_stats.npz"
            if cached.exists():
                self._cos_stats = CorpusCosStats.load(cached)
            else:
                from .rag.store.search import scroll_vectors

                matrix, _ids, _payloads = scroll_vectors(self.client, self.collection)
                self._cos_stats = CorpusCosStats.build(matrix)
                self._cos_stats.save(cached)
                self._log(f"[dense] 全库余弦统计已缓存（{matrix.shape[0]} 向量 → {self._cache_dir / 'corpus_cos_stats.npz'}）")
        return self._cos_stats

    def dense(self, query: str, k: int) -> list[dict[str, Any]]:
        from .rag.store.search import qdrant_search

        vector = self.embedder.encode_queries([query])[0]
        results = qdrant_search(
            self.client, self.collection, vector, k=max(k * self.pool_factor, k)
        )
        hits = []
        for result in results:
            payload = result.payload or {}
            hits.append({
                "parent_id": payload.get("parent_id"),
                "chunk_id": payload.get("chunk_id"),
                "score": result.score,
                "title": payload.get("title"),
                "section_path": payload.get("section_path"),
                "cos": result.score,
            })
        return parent_ranked(hits, k)

    def lexical(self, query: str, k: int) -> list[dict[str, Any]]:
        results = self._bm25.search(query, k, self._parents)
        return [
            {
                "parent_id": record["id"],
                "chunk_id": None,
                "score": float(score),
                "title": record.get("title"),
                "section_path": record.get("section_path"),
                "source": record.get("source"),
                "bm25": float(score),
            }
            for record, score in results
        ]

    def hybrid(self, query: str, k: int) -> list[dict[str, Any]]:
        return rrf_fuse(self.dense(query, k), self.lexical(query, k), k=k)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:      # noqa: BLE001  关不掉不影响结果
            pass


# ---------------------------------------------------------------- 跑批


def run_questions(
    questions: Sequence[Mapping[str, Any]],
    retriever,
    *,
    modes: Sequence[str],
    k: int,
    repeat: int = 1,
    index_meta: Mapping[str, Any] | None = None,
    models: Mapping[str, Any] | None = None,
    log=print,
) -> int:
    """逐题逐模式检索并写 raglog；返回写入的 trace 数。"""
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise ValueError(f"未知模式：{unknown}（可用：{list(MODES)}）")

    raglog.reset_run()
    if index_meta:
        raglog.set_index_meta(index_meta, models=models)
    written = 0
    started = time.perf_counter()
    for index, item in enumerate(questions, start=1):
        for mode in modes:
            for _ in range(repeat):
                with raglog.trace(
                    query=item["question"],
                    qid=item["qid"],
                    mode=mode,
                    k=k,
                    meta={"stage": "l1-store", "qtype": item.get("type")},
                ) as tracer:
                    t0 = time.perf_counter()
                    ranked = getattr(retriever, mode)(item["question"], k)
                    latency = (time.perf_counter() - t0) * 1000
                    hits = [
                        {
                            "id": row["parent_id"],
                            "chunk_id": row.get("chunk_id"),
                            "source": row.get("source") or item.get("source"),
                            "text": row.get("text") or "",
                            "score": row.get("score") or 0.0,
                        }
                        for row in ranked
                    ]
                    stats = _vector_stats(retriever, mode, item["question"], hits)
                    tracer.retrieve(hits, query=item["question"], stats=stats, latency_ms=latency)
                    written += 1
        if index % 20 == 0 or index == len(questions):
            log(f"  {index}/{len(questions)} 题（{time.perf_counter() - started:.1f} s）")
    return written


def _vector_stats(retriever, mode: str, query: str, hits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """给日志补上 z 分数要用的统计量（只有语义路才有余弦）。"""
    if mode not in ("dense", "hybrid") or not hits:
        return {}
    try:
        vector = retriever.embedder.encode_queries([query])[0]
        stats = retriever.cos_stats().cos_stats(vector)
    except Exception:      # noqa: BLE001  统计量缺失不该打断跑批
        return {}
    top1 = max((row.get("score") or 0.0) for row in hits) if mode == "dense" else None
    return {"vector": {**stats, "top1_cos": top1}}


def cmd_run(args) -> int:
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    missing = [item["qid"] for item in questions
               if item.get("type") != "negative" and not (item.get("gold_points") or {}).get("parent_ids")]
    if missing:
        print(f"⚠️ {len(missing)} 题没有 gold_points（会从正样本里掉出去）：{', '.join(missing[:5])}…")

    retriever = StoreRetriever(collection=args.collection, store_path=args.store)
    index_meta = {
        "runner": "store",
        "store": "qdrant+sqlite",
        "collection": retriever.collection,
        "parents": len(retriever._parents),          # noqa: SLF001
        "embedding_model": "bge-m3",
        "dim": 1024,
        "tokenizer": "jieba-cut_for_search",
        "dataset_sha256": __import__("hashlib").sha256(Path(args.questions).read_bytes()).hexdigest(),
    }
    if {"dense", "hybrid"} & set(modes):
        # 一次性批量预计算 query 向量（和 evaluate.py 同一动机：别让 embedding 固定开销
        # 落进每题延迟，303 次单条调用 = 10 分钟，一次批量 = 几秒）
        unique = list(dict.fromkeys(item["question"] for item in questions))
        started = time.perf_counter()
        retriever.embedder.encode_queries(unique)
        print(f"  预热 query 向量：{len(unique)} 条 / {time.perf_counter() - started:.1f} s")

    print(f"评估集 {len(questions)} 题 × {len(modes)} 模式，k={args.k}；日志 {raglog.log_dir()}")
    try:
        written = run_questions(
            questions, retriever, modes=modes, k=args.k, repeat=args.repeat,
            index_meta=index_meta, models={"chat": None, "embed": "bge-m3"},
        )
    finally:
        retriever.close()
    print(f"完成：{written} 条 trace")
    print(f"日志：{raglog.relative_path(raglog.run_path())}")
    print(f"下一步：python -m src.evaluate measure latest --main-mode {modes[-1] if modes else 'dense'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval_store",
        description="新库（Qdrant parent/child）L1 检索评估：run",
    )
    sub = parser.add_subparsers(dest="command")
    run_parser = sub.add_parser("run", help="跑检索并写 raglog")
    run_parser.add_argument("--modes", default=",".join(MODES))
    run_parser.add_argument("--k", type=int, default=10)
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--repeat", type=int, default=1)
    run_parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    run_parser.add_argument("--collection", default=None)
    run_parser.add_argument("--store", default=None, help="Document Store 路径")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
