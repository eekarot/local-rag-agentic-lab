"""SciFact 评测的口径粘合层（不联网、不碰 Qdrant）。

这里守的是**三个口径问题**，每一个都会让数字看起来更好或更差而没人发现：
1. qrels 是**文档级**，而检索返回**父块级** —— 同一文档命中多个父块必须只算一次，
   否则 nDCG 会被同文档的多个块虚高（同一篇摘要被切成 2–3 个父块是常态）；
2. BEIR 的 qrels 是 TSV（含表头、score≥1 才算相关），不是 JSON；
3. nDCG@10 必须是 BEIR 论文的二元相关 + log2 折扣口径，才和公开数字可比（BM25 ≈0.665）。
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_beir import doc_ranking, parent_to_doc, read_qrels, score_query


class ReadQrelsTest(unittest.TestCase):
    def _write(self, text):
        path = Path(tempfile.mkdtemp(prefix="scifact-qrels-")) / "test.tsv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_skips_header_and_groups_by_query(self):
        path = self._write("query-id\tcorpus-id\tscore\n1\t10\t1\n1\t11\t1\n2\t20\t1\n")
        self.assertEqual(read_qrels(path), {"1": {"10", "11"}, "2": {"20"}})

    def test_score_zero_is_not_relevant(self):
        path = self._write("query-id\tcorpus-id\tscore\n1\t10\t0\n1\t11\t1\n")
        self.assertEqual(read_qrels(path), {"1": {"11"}})

    def test_missing_score_column_defaults_to_relevant(self):
        path = self._write("1\t10\n")
        self.assertEqual(read_qrels(path), {"1": {"10"}})


class ParentToDocTest(unittest.TestCase):
    def test_doc_id_from_file_name(self):
        parents = {"p1": {"file_name": "12345.md"}, "p2": {"source_uri": "docs/678.md"}}
        self.assertEqual(parent_to_doc(parents), {"p1": "12345", "p2": "678"})

    def test_windows_paths_are_normalized(self):
        parents = {"p1": {"source_uri": "datasets\\scifact-v1\\docs\\42.md"}}
        self.assertEqual(parent_to_doc(parents), {"p1": "42"})

    def test_record_without_name_is_skipped(self):
        self.assertEqual(parent_to_doc({"p1": {}}), {})


class LoadDocMapTest(unittest.TestCase):
    """父块→文档 id 必须从 documents 表拿（file_name），不能靠 parents 表。"""

    def test_reads_from_join(self):
        import sqlite3

        from scripts.eval_beir import load_doc_map

        path = Path(tempfile.mkdtemp(prefix="scifact-store-")) / "documents.sqlite3"
        conn = sqlite3.connect(str(path))
        conn.executescript(
            "CREATE TABLE documents (document_id TEXT PRIMARY KEY, file_name TEXT, source_uri TEXT);"
            "CREATE TABLE parents (parent_id TEXT PRIMARY KEY, document_id TEXT, content TEXT);"
            "INSERT INTO documents VALUES ('d1', '111.md', 'docs/111.md');"
            "INSERT INTO documents VALUES ('d2', '222.md', 'docs/222.md');"
            "INSERT INTO parents VALUES ('p1', 'd1', 'x');"
            "INSERT INTO parents VALUES ('p2', 'd1', 'y');"
            "INSERT INTO parents VALUES ('p3', 'd2', 'z');"
        )
        conn.commit()
        conn.close()
        self.assertEqual(load_doc_map(path), {"p1": "111", "p2": "111", "p3": "222"})

    def test_missing_store_says_how_to_fix(self):
        from scripts.eval_beir import load_doc_map

        with self.assertRaises(SystemExit) as ctx:
            load_doc_map(Path(tempfile.mkdtemp(prefix="scifact-none-")) / "nope.sqlite3")
        self.assertIn("Document Store", str(ctx.exception))


class DisableForwardingTest(unittest.TestCase):
    """消融开关必须真的传到 `retriever.retrieve()`，否则「只剩 BM25」还是三路。"""

    def test_pipeline_forwards_disable_only_when_set(self):
        from src.rag.ask.pipeline import retrieve_with_meta

        seen = {}

        class _Route:
            ids = ["p1"]
            enabled = True

        class _Result:
            routes = {"dense_zh": _Route()}
            degraded = None

        class _Retriever:
            def retrieve(self, question, k=None, query_en=None, **kwargs):
                seen.update(kwargs)
                return _Result()

        parents = {"p1": {"text": "x", "content": "x", "language": "en"}}
        retrieve_with_meta("q", retriever=_Retriever(), parents=parents, counter=None,
                           reranker=None, rerank=False, k=1)
        self.assertNotIn("disable", seen)          # 没传就一个参数都不多给

        retrieve_with_meta("q", retriever=_Retriever(), parents=parents, counter=None,
                           reranker=None, rerank=False, k=1, disable=("lexical_en",))
        self.assertEqual(seen["disable"], ("lexical_en",))

    def test_eval_beir_accepts_disable(self):
        import inspect

        from scripts import eval_beir

        self.assertIn("disable", inspect.signature(eval_beir.evaluate).parameters)


class DocRankingTest(unittest.TestCase):
    def test_same_document_counts_once_and_keeps_best_rank(self):
        mapping = {"a1": "A", "a2": "A", "b1": "B"}
        self.assertEqual(doc_ranking(["a1", "a2", "b1"], mapping), ["A", "B"])

    def test_limit_applies_after_dedup(self):
        mapping = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}
        self.assertEqual(doc_ranking(["a1", "a2", "b1", "c1"], mapping, limit=2), ["A", "B"])

    def test_unmapped_parents_are_dropped(self):
        self.assertEqual(doc_ranking(["x", "a1"], {"a1": "A"}), ["A"])


class ScoreQueryTest(unittest.TestCase):
    def test_perfect_ranking(self):
        score = score_query(["A", "B"], {"A"}, k=10)
        self.assertEqual(score["ndcg@10"], 1.0)
        self.assertEqual(score["mrr@10"], 1.0)
        self.assertEqual(score["recall@10"], 1.0)

    def test_second_position_gets_log_discount(self):
        score = score_query(["X", "A"], {"A"}, k=10)
        # DCG = 1/log2(2+1) = 0.6309，IDCG = 1 → nDCG = 0.6309
        self.assertAlmostEqual(score["ndcg@10"], round(1 / 1.5849625007211562, 4), places=4)
        self.assertEqual(score["mrr@10"], 0.5)

    def test_miss_is_zero_not_error(self):
        score = score_query(["X", "Y"], {"A"}, k=10)
        self.assertEqual((score["ndcg@10"], score["hit@10"], score["mrr@10"]), (0.0, 0, 0.0))

    def test_two_gold_docs_recall_is_fraction(self):
        score = score_query(["A", "X"], {"A", "B"}, k=10)
        self.assertEqual(score["recall@10"], 0.5)
        self.assertEqual(score["full@10"], 0)

    def test_k_truncates_before_scoring(self):
        score = score_query(["X", "A"], {"A"}, k=1)
        self.assertEqual(score["ndcg@1"], 0.0)


class BeirReferenceTest(unittest.TestCase):
    """公开对账用的参考值必须写在代码里（含出处），免得以后拿错数字比。"""

    def test_bm25_reference_is_documented(self):
        from scripts.eval_beir import BM25_NDCG10_SCIFACT

        self.assertAlmostEqual(BM25_NDCG10_SCIFACT, 0.665, places=3)


if __name__ == "__main__":
    unittest.main()
