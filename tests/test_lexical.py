"""D25：英文 lexical 检索（Qdrant full-text 召回 + 应用层 BM25 排序）。"""

from __future__ import annotations

import unittest

from src.rag.store.lexical import BM25Index, parent_hits, tokenize_en


class TokenizeTest(unittest.TestCase):
    def test_lowercases_and_splits_on_punctuation(self):
        self.assertEqual(tokenize_en("DGX B300 has 288 GB"), ["dgx", "b300", "has", "288", "gb"])

    def test_keeps_identifiers_with_underscore_and_dot(self):
        self.assertEqual(tokenize_en("ef_construct=200, qwen3.5:9b"), ["ef_construct", "200", "qwen3", "9b"])

    def test_drops_chinese_characters(self):
        # 英文 lexical 路不该拿中文词去匹配英文语料
        self.assertEqual(tokenize_en("HNSW 是什么"), ["hnsw"])

    def test_drops_single_characters_and_stopwords(self):
        self.assertEqual(tokenize_en("a b of the 200"), ["200"])


class BM25Test(unittest.TestCase):
    def _index(self):
        index = BM25Index()
        index.add("c1", "p1", "DGX B300 has 288 GB of GPU memory")
        index.add("c2", "p1", "DGX B300 default system memory is 2 TB")
        index.add("c3", "p2", "InfiniBand NDR reaches 400 Gbps per direction")
        index.add("c4", "p3", "chunk size should not exceed the embedding context length")
        index.build()
        return index

    def test_rare_term_ranks_first(self):
        results = self._index().search("infiniband gbps", k=3)
        self.assertEqual(results[0][0], "c3")

    def test_common_term_matches_multiple_chunks(self):
        results = self._index().search("dgs", k=5)   # 打错字 → 没有命中
        self.assertEqual(results, [])
        results = self._index().search("dgx b300", k=5)
        self.assertEqual({chunk for chunk, _parent, _score in results}, {"c1", "c2"})

    def test_returns_parent_id_alongside_chunk(self):
        results = self._index().search("memory", k=5)
        self.assertEqual({parent for _chunk, parent, _score in results}, {"p1"})

    def test_k_limits_results_and_is_sorted(self):
        results = self._index().search("dgx b300 memory gpu system", k=1)
        self.assertEqual(len(results), 1)
        full = self._index().search("dgx b300 memory gpu system", k=10)
        scores = [score for _chunk, _parent, score in full]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self._index().search("") , [])
        self.assertEqual(self._index().search("的是什么") , [])

    def test_stats(self):
        stats = self._index().stats()
        self.assertEqual(stats["chunks"], 4)
        self.assertEqual(stats["parents"], 3)
        self.assertGreater(stats["terms"], 10)


class ParentHitsTest(unittest.TestCase):
    def test_dedupes_parents_and_counts(self):
        results = [("c1", "p1", 9.0), ("c2", "p1", 8.0), ("c3", "p2", 7.0)]
        self.assertEqual(parent_hits(results, k=10), ["p1", "p2"])
        self.assertEqual(parent_hits(results, k=1), ["p1"])


if __name__ == "__main__":
    unittest.main()
