"""D25：英文 lexical 检索 —— Qdrant full-text 召回 + 应用层 BM25 排序。

为什么不能「直接用 Qdrant 的 full-text 当 BM25」：Qdrant 的 `TextIndexParams` 提供的
是**布尔匹配**（`MatchText`），返回的是「含这些词的点」，没有 BM25 的那套
idf/长度归一化分数。真按 id 顺序取前 10 条，等于随机挑 10 条含关键词的块。

所以这一路拆成两段：

1. **召回**：Qdrant `text` payload 索引 + `MatchText` 过滤出候选（`ensure_text_index`）；
2. **排序**：候选的 `text` 取回来，在应用层用 BM25 打分排序（`BM25Index`）。

分词这里**刻意不用 jieba**：英文路径就该用英文分词。中文问法送进来只会剩下
型号/缩写这些拉丁词元（`HNSW 是什么` → `["hnsw"]`），这正是「中文 query 走英文
lexical 路会很差」的原因，D25 要量的就是这件事。

    python -m src.rag.store.lexical --build-index                 # 建 Qdrant text 索引
    python -m src.rag.store.lexical --questions experiments/questions.json --k 10
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: 英文词元：字母/数字开头，允许数字与下划线（`ef_construct`、`b300`、`9b`）
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_]*")

#: 极常见词对 BM25 没有区分度，直接丢掉（不追求完整停用词表）
STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been do does did "
    "what which how why does do with by from as at it its this that these those".split()
)


def tokenize_en(text: str) -> list[str]:
    """英文分词：小写 → 词元 → 去停用词/单字符。中文汉字会被直接丢掉。"""
    tokens = _TOKEN.findall((text or "").lower())
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


class BM25Index:
    """最小可用的 BM25（k1=1.2, b=0.75），只为英文 lexical 路服务。"""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[tuple[str, str]] = []   # (chunk_id, parent_id)
        self._lengths: list[int] = []
        self._freqs: list[Counter] = []
        self._df: Counter = Counter()
        self._avg_length = 0.0
        self._built = False

    def add(self, chunk_id: str, parent_id: str, text: str) -> None:
        tokens = tokenize_en(text)
        self._chunks.append((chunk_id, parent_id))
        self._lengths.append(len(tokens))
        self._freqs.append(Counter(tokens))
        self._built = False

    def build(self) -> "BM25Index":
        total = len(self._chunks)
        self._avg_length = (sum(self._lengths) / total) if total else 0.0
        self._df = Counter()
        for freq in self._freqs:
            self._df.update(freq.keys())
        self._built = True
        return self

    @property
    def size(self) -> int:
        return len(self._chunks)

    def _idf(self, term: str) -> float:
        # 加 0.5 的平滑：idf 永远为正，避免出现「词越常见分越高」的负分反转
        n = len(self._chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[str, str, float]]:
        """返回 [(chunk_id, parent_id, score), ...]，按分数降序。"""
        if not self._built:
            self.build()
        terms = tokenize_en(query)
        if not terms or not self._chunks:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for index, freq in enumerate(self._freqs):
                tf = freq.get(term)
                if not tf:
                    continue
                length = self._lengths[index] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self._avg_length or 1))
                scores[index] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self._chunks[i][0], self._chunks[i][1], round(score, 4)) for i, score in ranked]

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": len(self._chunks),
            "parents": len({parent for _chunk, parent in self._chunks}),
            "terms": len(self._df),
            "avg_tokens": round(self._avg_length, 2),
        }


def parent_hits(results: Iterable[tuple[str, str, float]], k: int | None = None) -> list[str]:
    """把 chunk 级结果收敛成父块序列（去重保序）——指标口径与 D14/D20 一致。"""
    seen: set[str] = set()
    out: list[str] = []
    for index, (_chunk_id, parent_id, _score) in enumerate(results):
        if k is not None and index >= k:
            break
        if parent_id in seen:
            continue
        seen.add(parent_id)
        out.append(parent_id)
    return out


# ---------------------------------------------------------------------------
# Qdrant 侧：text payload 索引 + 候选过滤
# ---------------------------------------------------------------------------


def ensure_text_index(client, collection: str, field: str = "text") -> dict[str, Any]:
    """给 payload 字段建 full-text 索引（幂等）。"""
    from qdrant_client import models

    client.create_payload_index(
        collection_name=collection,
        field_name=field,
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            min_token_len=2,
        ),
    )
    return {"collection": collection, "field": field, "index": "text/word/lowercase"}


def fulltext_filter(client, collection: str, text: str, *, limit: int = 512) -> list[dict[str, Any]]:
    """用 Qdrant full-text 过滤出候选 payload（不排序，排序交给 BM25）。

    语义是 **OR**（任一词命中即入选）而不是 AND：AND 会把正常问句卡死——
    实测 `What is the peak speed of NDR InfiniBand per direction in a DGX SuperPOD?`
    在 AND 下候选 **0 条**（要求 per、direction 这些词在同一条 payload 里全都出现），
    而 OR 能正常召回。过滤只负责「别把无关的点捞进来」，排序交给 BM25。

    **caveat**：`scroll` 按点 id 顺序返回，不是按相关度，所以 `limit` 给小了会漏掉
    真正该排前的候选。生产路径要么把 limit 放大到能覆盖全部命中（几千条量级），
    要么干脆本地 BM25（本模块 `load_payloads` + `build_bm25` 就是完整索引）。
    """
    from qdrant_client import models

    terms = [token for token in tokenize_en(text)][:12]
    if not terms:
        return []
    conditions = [
        models.FieldCondition(key="text", match=models.MatchText(text=term)) for term in terms
    ]
    # only-should 过滤器默认就是「至少命中 1 个」（min_should 缺省 = 1）
    flt = models.Filter(should=conditions)
    points, _next = client.scroll(
        collection_name=collection,
        scroll_filter=flt,
        limit=limit,
        with_payload=["chunk_id", "parent_id", "text"],
        with_vectors=False,
    )
    return [dict(point.payload or {}) for point in points]


def load_payloads(client, collection: str, *, batch: int = 1024, log=None) -> list[dict[str, Any]]:
    """把全库 payload 拉回来建本地 BM25（38k 点约 200 MB 文本）。"""
    out: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=batch,
            with_payload=["chunk_id", "parent_id", "text"],
            with_vectors=False,
        )
        out.extend(dict(point.payload or {}) for point in points)
        if log and len(out) % (batch * 10) == 0:
            log(f"[lexical] 已拉取 {len(out)} 点")
        if offset is None:
            break
    return out


def build_bm25(payloads: Iterable[Mapping[str, Any]]) -> BM25Index:
    index = BM25Index()
    for payload in payloads:
        chunk_id = payload.get("chunk_id")
        if not chunk_id:
            continue
        index.add(chunk_id, payload.get("parent_id") or "", payload.get("text") or "")
    return index.build()


# ---------------------------------------------------------------------------
# CLI：D25 验收（英文 query vs 中文 query，同一条英文 lexical 路）
# ---------------------------------------------------------------------------


def _gold_parents(store, questions: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """每题的金标父块（按 gold_spans 在父块正文里做子串匹配，口径同 D20）。"""
    return {item["qid"]: store.parents_containing(item.get("gold_spans") or [])
            for item in questions if item.get("type") != "negative"}


def _evaluate(index: BM25Index, questions, gold_map, *, k: int, field: str) -> dict[str, Any]:
    per_question = []
    for item in questions:
        gold = gold_map.get(item["qid"]) or set()
        if not gold:
            continue
        query = item.get(field) or ""
        ranked = parent_hits(index.search(query, k=k), k=k)
        matched = [pid for pid in ranked if pid in gold]
        per_question.append({
            "qid": item["qid"],
            "topic": item.get("topic"),
            "gold": len(gold),
            "matched": len(matched),
            "recall": round(len(matched) / len(gold), 4),
            "hit": 1 if matched else 0,
        })
    hit = sum(row["hit"] for row in per_question)
    return {
        "sample": len(per_question),
        "hit": hit,
        "hit_rate": round(hit / len(per_question), 4) if per_question else None,
        "recall": round(sum(row["recall"] for row in per_question) / len(per_question), 4)
        if per_question else None,
        "rows": per_question,
    }


def main(argv: Sequence[str] | None = None) -> int:
    from ...config import QUESTIONS_PATH, STORAGE_DIR

    from .document_store import DocumentStore
    from .qdrant_store import COLLECTION_NAME, make_client

    parser = argparse.ArgumentParser(prog="python -m src.rag.store.lexical", description="D25 英文 lexical 检索")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--build-index", action="store_true", help="给 payload.text 建 full-text 索引")
    parser.add_argument("--questions", default=None, help="评估集路径（做中英 query 对比）")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--keyword-only", action="store_true",
                        help="只统计关键词类题（number_spec / acronym_model）")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    client = make_client(timeout=180)
    if args.build_index:
        print(ensure_text_index(client, args.collection))
        client.close()
        return 0

    if not args.questions:
        parser.print_help()
        client.close()
        return 2

    print(f"[lexical] 拉取 {args.collection} 的 payload…")
    index = build_bm25(load_payloads(client, args.collection, log=print))
    client.close()
    print(f"[lexical] BM25 索引：{index.stats()}")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    store = DocumentStore(STORAGE_DIR / "documents.sqlite3")
    try:
        gold_map = _gold_parents(store, questions)
    finally:
        store.close()

    judgeable = [item for item in questions
                 if (item.get("type") != "negative") and gold_map.get(item["qid"])]
    if args.keyword_only:
        judgeable = [item for item in judgeable
                     if item.get("topic") in ("number_spec", "acronym_model")]
    print(f"[lexical] 可判定题：{len(judgeable)}（金标父块在库内）")

    zh = _evaluate(index, judgeable, gold_map, k=args.k, field="question")
    en = _evaluate(index, judgeable, gold_map, k=args.k, field="query_en")
    print(f"\n同一批题、同一条英文 lexical 路（hit@{args.k} / Recall@{args.k}）：")
    print(f"  中文 query：{zh['hit']}/{zh['sample']} = {zh['hit_rate']:.2%}，Recall {zh['recall']:.2%}")
    print(f"  英文 query：{en['hit']}/{en['sample']} = {en['hit_rate']:.2%}，Recall {en['recall']:.2%}")
    print(f"  差值：hit {en['hit'] - zh['hit']:+d} 条，Recall {(en['recall'] - zh['recall']) * 100:+.2f} pt")

    payload = {"k": args.k, "sample": zh["sample"], "zh": zh, "en": en}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
