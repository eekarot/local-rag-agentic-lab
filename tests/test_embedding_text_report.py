"""D14：`embedding_text` 前缀统计 + 有/无前缀 A/B（Recall@k 差异报告）。"""

from __future__ import annotations

import unittest
import uuid

import numpy as np

from src.rag.ingestion.embedding_text_report import (
    PrefixStats,
    ab_summary,
    prefix_stats,
    query_metrics,
    reconstruction_stats,
)
from src.rag.ingestion.parent_builder import build_embedding_text
from src.rag.store.qdrant_store import VECTOR_SIZE, make_client
from tests._slow import run_slow, server_skip_reason, slow


def _child(text, content="正文", section=("RAG", "HNSW"), title="RAG 指南"):
    return {
        "chunk_id": f"chk_{abs(hash((text, content))) % 10**12:012x}",
        "title": title,
        "section_path": list(section),
        "content": content,
        "embedding_text": text,
    }


class PrefixStatsTest(unittest.TestCase):
    def test_all_prefixed(self):
        records = [
            _child(build_embedding_text("指南", ["RAG", "HNSW"], "正文")),
            _child(build_embedding_text("指南", ["RAG"], "正文 2")),
        ]
        stats = prefix_stats(records)
        self.assertEqual(stats.total, 2)
        self.assertEqual(stats.with_prefix, 2)
        self.assertEqual(stats.without_prefix, 0)
        self.assertEqual(stats.reasons, {})

    def test_document_only_prefix_counts_as_prefixed(self):
        # 只有 Document: 行、没有 Section: 行（顶层正文没有章节）也算有前缀
        records = [_child("Document: 指南\nContent: 正文")]
        stats = prefix_stats(records)
        self.assertEqual(stats.with_prefix, 1)

    def test_content_only_is_missing(self):
        records = [_child("Content: 正文")]
        stats = prefix_stats(records)
        self.assertEqual(stats.without_prefix, 1)
        self.assertEqual(stats.reasons, {"content_only": 1})

    def test_identical_to_content_is_missing(self):
        records = [_child("正文", content="正文")]
        stats = prefix_stats(records)
        self.assertEqual(stats.without_prefix, 1)
        self.assertEqual(stats.reasons, {"identical_to_content": 1})

    def test_missing_field_is_missing(self):
        record = _child("正文")
        record.pop("embedding_text")
        stats = prefix_stats([record])
        self.assertEqual(stats.without_prefix, 1)
        self.assertEqual(stats.reasons, {"missing_field": 1})

    def test_empty(self):
        stats = prefix_stats([])
        self.assertEqual(stats, PrefixStats(total=0, with_prefix=0, without_prefix=0, reasons={}))

    def test_as_dict_reports_ratio(self):
        stats = prefix_stats([_child("Document: A\nContent: x"), _child("正文", content="正文")])
        payload = stats.as_dict()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["with_prefix"], 1)
        self.assertEqual(payload["prefix_ratio"], 0.5)
        self.assertEqual(payload["reasons"], {"identical_to_content": 1})


class ReconstructionTest(unittest.TestCase):
    def test_reconstructable_from_stored_fields(self):
        content = "正文内容"
        text = build_embedding_text("RAG 指南", ["RAG", "HNSW"], content)
        stats = reconstruction_stats([_child(text, content=content)])
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["mismatched"], 0)

    def test_mismatch_detected(self):
        stats = reconstruction_stats([_child("Document: 别的标题\nContent: 正文", content="正文")])
        self.assertEqual(stats["mismatched"], 1)
        self.assertEqual(stats["examples"][0]["expected"], "Document: RAG 指南\nSection: RAG / HNSW\nContent: 正文")


class QueryMetricsTest(unittest.TestCase):
    def test_full_match(self):
        metrics = query_metrics(["p1", "p2"], {"p1", "p2"}, k=2)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["hit"], 1)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_partial_match(self):
        metrics = query_metrics(["p1", "x"], {"p1", "p2"}, k=2)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["hit"], 1)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_miss(self):
        metrics = query_metrics(["x", "y"], {"p1"}, k=2)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["hit"], 0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_dedupes_same_parent(self):
        # 同一个父块的两个子块都命中，只算一次（Recall 的分母是父块）
        metrics = query_metrics(["p1", "p1"], {"p1", "p2"}, k=2)
        self.assertEqual(metrics["recall"], 0.5)

    def test_mrr_uses_first_matched_rank(self):
        # 先按父块去重再排名：同一父块的兄弟子块只占一个位次，
        # 所以 ["x","x","p1"] 去重后 p1 是第 2 名
        self.assertEqual(query_metrics(["x", "x", "p1"], {"p1"}, k=3)["mrr"], 0.5)
        # 三个不同父块时才真的是第 3 名
        self.assertEqual(query_metrics(["x", "y", "p1"], {"p1"}, k=3)["mrr"], round(1 / 3, 4))

    def test_no_gold_yields_none_recall(self):
        # 该题在本次索引范围内没有可判定的金标 → 不参与均值，而不是算 0
        metrics = query_metrics(["p1"], set(), k=2)
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["hit"])


class AbSummaryTest(unittest.TestCase):
    def _row(self, gold, prefixed, plain, k=2):
        return {"qid": "q", "k": k, "gold": set(gold), "prefixed": prefixed, "plain": plain}

    def test_means_and_delta(self):
        rows = [
            self._row({"p1"}, ["p1", "x"], ["x", "p1"]),
            self._row({"p1"}, ["p1", "y"], ["y", "z"]),
        ]
        summary = ab_summary(rows)
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["prefixed"]["recall"], 1.0)
        self.assertEqual(summary["plain"]["recall"], 0.5)
        self.assertEqual(summary["delta"]["recall"], 0.5)
        self.assertEqual(summary["prefixed"]["hit"], 1.0)
        self.assertEqual(summary["plain"]["hit"], 0.5)

    def test_skips_unjudgeable_queries(self):
        rows = [self._row(set(), ["p1"], ["p1"]), self._row({"p1"}, ["p1"], ["x"])]
        summary = ab_summary(rows)
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["judged"], 1)
        self.assertEqual(summary["prefixed"]["recall"], 1.0)
        self.assertEqual(summary["plain"]["recall"], 0.0)


def _server_available():
    try:
        make_client(timeout=3).get_collections()
        return True
    except Exception:
        return False


SERVER = run_slow() and _server_available()   # 默认不探：省 3 s 且不联网


class _DeterministicEmbedder:
    """内容决定向量：相同文本恒等，含前缀的文本与纯正文必然不同。"""

    def encode_documents(self, texts, **kwargs):
        return np.vstack([self._one(t) for t in texts])

    def encode_queries(self, texts, **kwargs):
        return np.vstack([self._one(t) for t in texts])

    @staticmethod
    def _one(text):
        # 剥掉 Document/Section 前缀再哈希：让「前缀变体」和「纯正文变体」得到同一向量，
        # 这样 A/B 的两个分支都能命中，测的是管线而不是随机性
        body = text.split("Content:", 1)[-1].strip()
        rng = np.random.default_rng(abs(hash(body)) % (2**32))
        vec = rng.normal(size=VECTOR_SIZE).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)


@unittest.skipUnless(SERVER, server_skip_reason("D14 服务端用例"))
@slow
class RunAbServerTest(unittest.TestCase):
    """两种口径真的各建一个 collection，报告要能算出差异。"""

    def test_builds_two_collections_and_reports(self):
        import asyncio

        from src.rag.ingestion.embedding_text_report import run_ab
        from src.rag.store.qdrant_store import ensure_collection

        client = make_client(timeout=60)
        suffix = uuid.uuid4().hex[:8]
        chunks = []
        for i in range(20):
            content = f"第 {i} 条正文内容"
            chunks.append(
                {
                    "chunk_id": f"chk_{i:04x}{suffix[:12]}",
                    "parent_id": f"par_{i:04x}{suffix[:12]}",
                    "title": "指南",
                    "section_path": ["RAG", f"S{i}"],
                    "content": content,
                    "embedding_text": build_embedding_text("指南", ["RAG", f"S{i}"], content),
                }
            )
        names = []
        try:
            report = run_ab(
                chunks,
                [{"qid": "q1", "question": "第 3 条正文内容", "gold_parent_ids": {"par_0003" + suffix[:12]}}],
                embedder=_DeterministicEmbedder(),
                client=client,
                collection_prefix=f"ab_d14_{suffix}",
                collection_names=names,
            )
        finally:
            for name in client.get_collections().collections:
                if name.name.startswith(f"ab_d14_{suffix}"):
                    client.delete_collection(name.name)
            client.close()
        self.assertEqual(report["chunks"], 20)
        self.assertEqual(report["queries"], 1)
        # 假 embedder 剥掉前缀后哈希，两个变体应给出同样的排名 → delta 为 0
        self.assertEqual(report["plain"]["recall"], 1.0)
        self.assertEqual(report["prefixed"]["recall"], 1.0)
        self.assertEqual(report["delta"]["recall"], 0.0)
        self.assertEqual(len(report["rows"]), 1)
