"""S1（gold 落地）：gold_spans → 新库 parent/child 映射。

为什么必须做这一步：`questions.json` 里的 `gold_chunks` 是**旧 81,027-chunk 索引**的
`rag_basics.md::chunk-0000`，而现在的库是 Qdrant `rag_chunks_v1`（38,865 child /
25,411 parent）+ Document Store 的 parent/child 结构。两套 id 空间完全不相交，
不做映射，新库的 Recall 恒为 0。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval_gold import (
    STATUS_AMBIGUOUS,
    STATUS_NEGATIVE,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_UNMATCHED,
    apply_gold_points,
    build_gold_points,
    map_question,
    normalize,
)
from src.rag.store.document_store import DocumentStore


def _parent(parent_id, content, *, doc_id="doc_1", title="指南", section="§1", index=0):
    return {
        "parent_id": parent_id,
        "document_id": doc_id,
        "parent_index": index,
        "content": content,
        "title": title,
        "section_path": section,
        "token_count": len(content.split()),
        "source_start": 0,
        "source_end": len(content),
        "chunker_version": "v1",
        "document_version": "1",
        "language": "zh",
    }


def _child(chunk_id, parent_id, *, doc_id="doc_1"):
    return {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        "document_id": doc_id,
        "child_index": 0,
        "content": "x",
        "content_hash": "h" * 16,
        "token_count": 10,
        "source_start": 0,
        "source_end": 10,
        "chunker_version": "v1",
        "document_version": "1",
        "embedding_version": "bge-m3-v1",
        "status": "active",
    }


def _document(doc_id="doc_1", source="md/a.md"):
    return {
        "document_id": doc_id,
        "source_uri": source,
        "file_name": "a.md",
        "file_type": "md",
        "title": "指南",
        "language": "zh",
        "document_version": "1",
        "parser_version": "v1",
        "status": "active",
    }


class StoreFixture(unittest.TestCase):
    """一个临时 DocumentStore：两个文档、三个父块。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = DocumentStore(Path(tmp.name) / "store.sqlite3")
        self.store.replace_document(
            _document(),
            [
                _parent("par_a", "RAG 是把检索和生成拆开的做法。\n\n它先检索再生成。"),
                # 故意写成一行长文本 + 换行，用于测「空白归一化」匹配
                _parent("par_b", "HNSW  是在图上做\n近似最近邻搜索的算法，用于向量检索。", index=1),
            ],
            [_child("chk_a", "par_a"), _child("chk_b", "par_b")],
        )
        self.store.replace_document(
            _document("doc_2", "md/b.md"),
            [_parent("par_c", "Top-K 太大会引入噪声。", doc_id="doc_2", index=0)],
            [_child("chk_c", "par_c", doc_id="doc_2")],
        )


class NormalizeTest(unittest.TestCase):
    def test_collapses_all_whitespace(self):
        self.assertEqual(normalize("a  b\n\nc\td"), "a b c d")

    def test_keeps_chinese(self):
        self.assertEqual(normalize("你好  世界"), "你好 世界")


class MapQuestionTest(StoreFixture):
    def test_exact_span_hits_one_parent(self):
        item = {"qid": "q1", "gold_spans": ["HNSW 是在图上做\n近似最近邻搜索的算法，用于向量检索。"]}
        result = map_question(self.store, item)
        self.assertEqual(result["parent_ids"], ["par_b"])
        self.assertEqual(result["child_ids"], ["chk_b"])
        self.assertEqual(result["status"], STATUS_OK)

    def test_whitespace_insensitive_match(self):
        # 语料里是「HNSW  是在图上做\n」双空格 + 换行，span 写单空格
        item = {"qid": "q2", "gold_spans": ["HNSW 是在图上做 近似最近邻搜索的算法，用于向量检索。"]}
        result = map_question(self.store, item)
        self.assertEqual(result["parent_ids"], ["par_b"])
        self.assertEqual(result["match_mode"], "normalized")

    def test_partial_when_one_span_misses(self):
        item = {"qid": "q3", "gold_spans": ["Top-K 太大会引入噪声。", "这句语料里没有"]}
        result = map_question(self.store, item)
        self.assertEqual(result["status"], STATUS_PARTIAL)
        self.assertEqual(result["parent_ids"], ["par_c"])
        self.assertEqual(result["unmatched_spans"], ["这句语料里没有"])

    def test_unmatched_when_no_span_hits(self):
        item = {"qid": "q4", "gold_spans": ["完全不存在的一句话"]}
        result = map_question(self.store, item)
        self.assertEqual(result["status"], STATUS_UNMATCHED)
        self.assertEqual(result["parent_ids"], [])

    def test_ambiguous_span_is_flagged(self):
        # 「检索」同时出现在 par_a 和 par_b → 超过 max_parents=1 → gold 太泛，必须留痕
        item = {"qid": "q5", "gold_spans": ["检索"]}
        result = map_question(self.store, item, max_parents=1)
        self.assertEqual(result["status"], STATUS_AMBIGUOUS)
        self.assertEqual(len(result["parent_ids"]), 2)

    def test_negative_question_is_skipped(self):
        item = {"qid": "q6", "type": "negative", "gold_spans": []}
        result = map_question(self.store, item)
        self.assertEqual(result["status"], STATUS_NEGATIVE)

    def test_explicit_gold_points_are_kept(self):
        item = {
            "qid": "q7",
            "gold_spans": ["Top-K 太大会引入噪声。"],
            "gold_points": {"parent_ids": ["par_manual"]},
        }
        result = map_question(self.store, item)
        self.assertEqual(result["parent_ids"], ["par_manual"])
        self.assertEqual(result["status"], STATUS_OK)


class BuildTest(StoreFixture):
    def test_build_and_apply(self):
        questions = [
            {"qid": "q1", "gold_spans": ["RAG 是把检索和生成拆开的做法。"]},
            {"qid": "q2", "type": "negative", "gold_spans": []},
        ]
        mapping = build_gold_points(questions, self.store)
        self.assertEqual(sorted(mapping), ["q1", "q2"])
        updated = apply_gold_points(questions, mapping)
        self.assertEqual(updated[0]["gold_points"]["parent_ids"], ["par_a"])
        self.assertNotIn("gold_points", updated[1])       # 负样本不写
        self.assertEqual(questions[0].get("gold_points"), None)   # 不改原对象

    def test_stats(self):
        questions = [
            {"qid": "q1", "gold_spans": ["RAG 是把检索和生成拆开的做法。"]},
            {"qid": "q2", "gold_spans": ["不存在"]},
            {"qid": "q3", "type": "negative", "gold_spans": []},
        ]
        mapping = build_gold_points(questions, self.store)
        self.assertEqual(mapping["q1"]["status"], STATUS_OK)
        self.assertEqual(mapping["q2"]["status"], STATUS_UNMATCHED)
        self.assertEqual(mapping["q3"]["status"], STATUS_NEGATIVE)

    def test_store_path_and_roundtrip(self):
        questions = [{"qid": "q1", "gold_spans": ["Top-K 太大会引入噪声。"]}]
        mapping = build_gold_points(questions, self.store)
        payload = json.dumps(mapping, ensure_ascii=False)
        self.assertIn("par_c", payload)


if __name__ == "__main__":
    unittest.main()
