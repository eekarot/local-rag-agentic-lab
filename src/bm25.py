"""Okapi BM25 倒排索引。

公式（标准 Okapi BM25，k1 控制词频饱和，b 控制文档长度惩罚）：

    idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))        # 恒为正，避免 df>N/2 时负分
    score(q) = Σ_{t∈q} qtf(t) · idf(t) · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))

索引结构（可 JSON 序列化，直接放进 storage/index.json 的 "bm25" 字段）：

    {"k1":1.5, "b":0.75, "doc_len":[...], "postings": {"词项": {"文档下标": tf}}}

只在 build_index 阶段分词一次，查询时只访问命中词项的 posting list，
查询成本从「每次全量重算」降到 O(命中 posting 长度)。

**旧链专用**：新链的词法检索走 `src/rag/store/lexical.py`，本模块只为旧入口保留
（见 `src/rag.py` 的 deprecated 说明）。
"""


import math
from collections import Counter

from .config import BM25_K1, BM25_B
from .tokenizer import tokenize


def build_bm25(records, k1=BM25_K1, b=BM25_B):
    """为所有文本块建立倒排索引。records 顺序即文档下标。"""
    postings = {}
    doc_len = []
    for idx, record in enumerate(records):
        tokens = tokenize(record.get("text", ""))
        doc_len.append(len(tokens))
        for term, tf in Counter(tokens).items():
            postings.setdefault(term, {})[str(idx)] = tf
    return {
        "k1": float(k1),
        "b": float(b),
        "doc_len": doc_len,
        "postings": postings,
    }


class BM25Index:
    """查询侧：把 JSON payload 包一层，缓存 idf。"""

    def __init__(self, payload):
        self.k1 = float(payload.get("k1", BM25_K1))
        self.b = float(payload.get("b", BM25_B))
        self.doc_len = payload.get("doc_len") or []
        self.n_docs = len(self.doc_len)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.postings = payload.get("postings") or {}
        self._idf_cache = {}
        self._term_docs = {}

    def _postings(self, term):
        """JSON 里文档下标是字符串，这里转成 int 并缓存。"""
        cached = self._term_docs.get(term)
        if cached is None:
            raw = self.postings.get(term)
            cached = {int(doc_id): tf for doc_id, tf in raw.items()} if raw else {}
            self._term_docs[term] = cached
        return cached

    def idf(self, term):
        value = self._idf_cache.get(term)
        if value is None:
            df = len(self._postings(term))
            value = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)) if df else 0.0
            self._idf_cache[term] = value
        return value

    def term_scores(self, term):
        """返回 [(文档下标, 该词项贡献的 BM25 分), ...]。"""
        postings = self._postings(term)
        if not postings:
            return []
        idf = self.idf(term)
        k1, b, avgdl = self.k1, self.b, self.avgdl or 1.0
        out = []
        for doc_idx, tf in postings.items():
            dl = self.doc_len[doc_idx] if doc_idx < self.n_docs else avgdl
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            out.append((doc_idx, idf * (tf * (k1 + 1) / denom)))
        return out

    def scores(self, query_tokens):
        """查询打分：query 内重复词项按 qtf 线性加权（标准 BM25 做法）。"""
        acc = {}
        for term, qtf in Counter(query_tokens).items():
            for doc_idx, score in self.term_scores(term):
                acc[doc_idx] = acc.get(doc_idx, 0.0) + score * qtf
        return acc

    def search(self, query, k, records, stats=None):
        q_tokens = tokenize(query)
        if not q_tokens:
            if stats is not None:
                stats.update({"tokens": [], "terms_hit": 0, "postings_touched": 0, "candidates": 0,
                              "ranks": {}, "scores": {}})
            return []
        acc = self.scores(q_tokens)
        ranked = sorted(
            ((idx, score) for idx, score in acc.items() if score > 0),
            key=lambda x: x[1],
            reverse=True,
        )[:k]
        if stats is not None:
            # 只做统计，不参与排序：分辨「关键词题没命中」是分词问题还是倒排问题
            unique_terms = list(dict.fromkeys(q_tokens))
            stats.update({
                "tokens": list(q_tokens),
                "terms_hit": sum(1 for term in unique_terms if self._postings(term)),
                "postings_touched": sum(len(self._postings(term)) for term in unique_terms),
                "candidates": sum(1 for score in acc.values() if score > 0),
                "ranks": {records[idx]["id"]: rank for rank, (idx, _) in enumerate(ranked, start=1)
                          if idx < len(records)},
                "scores": {records[idx]["id"]: float(score) for idx, score in ranked
                           if idx < len(records)},
            })
        return [
            (records[idx], score)
            for idx, score in ranked
            if idx < len(records)
        ]
