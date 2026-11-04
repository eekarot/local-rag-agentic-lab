"""新库（Qdrant parent/child）的 L1 评估：run 逻辑的单测。"""

from __future__ import annotations

import unittest

from src.eval_store import (
    MODES,
    gold_parents,
    parent_ranked,
    rrf_fuse,
    run_questions,
)


def _hit(parent_id, chunk_id, score):
    return {"id": parent_id, "parent_id": parent_id, "chunk_id": chunk_id, "score": score}


class ParentRankedTest(unittest.TestCase):
    def test_dedupes_siblings_and_keeps_best(self):
        # 同一父块的两个子块都命中 → 父块只占一个位次，分取最高那个
        hits = [
            _hit("par_a", "chk_a2", 0.91),
            _hit("par_a", "chk_a1", 0.88),
            _hit("par_b", "chk_b1", 0.80),
        ]
        ranked = parent_ranked(hits, k=10)
        self.assertEqual([row["parent_id"] for row in ranked], ["par_a", "par_b"])
        self.assertEqual(ranked[0]["score"], 0.91)

    def test_respects_k_after_dedupe(self):
        hits = [_hit(f"par_{i}", f"chk_{i}", 1.0 - i / 100) for i in range(10)]
        self.assertEqual(len(parent_ranked(hits, k=3)), 3)

    def test_ignores_hits_without_parent_id(self):
        ranked = parent_ranked([{"id": None, "score": 0.5}, _hit("par_a", "chk_a", 0.4)], k=5)
        self.assertEqual([row["parent_id"] for row in ranked], ["par_a"])

    def test_sorted_by_score_desc(self):
        hits = [_hit("par_b", "chk_b", 0.5), _hit("par_a", "chk_a", 0.7)]
        self.assertEqual([row["parent_id"] for row in parent_ranked(hits, k=5)], ["par_a", "par_b"])


class RrfTest(unittest.TestCase):
    def test_fuses_two_rankings(self):
        dense = [{"parent_id": "par_a"}, {"parent_id": "par_b"}]
        lexical = [{"parent_id": "par_b"}, {"parent_id": "par_c"}]
        fused = rrf_fuse(dense, lexical, k=10, rrf_k=60)
        ids = [row["parent_id"] for row in fused]
        # par_b 在两路都靠前 → 第一
        self.assertEqual(ids[0], "par_b")
        self.assertEqual(set(ids), {"par_a", "par_b", "par_c"})

    def test_single_route_preserves_order(self):
        dense = [{"parent_id": "par_a"}, {"parent_id": "par_b"}]
        self.assertEqual([row["parent_id"] for row in rrf_fuse(dense, k=10)], ["par_a", "par_b"])

    def test_dedupe_within_route(self):
        dense = [{"parent_id": "par_a"}, {"parent_id": "par_a"}]
        fused = rrf_fuse(dense, k=10)
        self.assertEqual([row["parent_id"] for row in fused], ["par_a"])

    def test_empty_routes(self):
        self.assertEqual(rrf_fuse([], [], k=10), [])

    def test_tie_broken_by_first_seen(self):
        dense = [{"parent_id": "par_a"}]
        lexical = [{"parent_id": "par_b"}]
        self.assertEqual([row["parent_id"] for row in rrf_fuse(dense, lexical, k=10)],
                         ["par_a", "par_b"])


class GoldTest(unittest.TestCase):
    def test_prefers_gold_points(self):
        item = {
            "gold_points": {"parent_ids": ["par_a", "par_b"]},
            "gold_chunks": [{"id": "old::chunk-0000", "grade": 2}],
        }
        self.assertEqual(gold_parents(item), {"par_a", "par_b"})

    def test_falls_back_to_gold_chunks(self):
        item = {"gold_chunks": [{"id": "chk_a", "grade": 2}]}
        self.assertEqual(gold_parents(item), {"chk_a"})

    def test_negative_has_no_gold(self):
        self.assertEqual(gold_parents({"type": "negative"}), set())


class _FakeRetriever:
    """假检索器：按 query 返回固定父块序列，用于测 run_questions 的编排。"""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def dense(self, query, k):
        self.calls.append(("dense", query, k))
        return [{"parent_id": pid, "chunk_id": f"chk_{pid}", "score": 1.0}
                for pid in self.table.get("dense", {}).get(query, [])][:k]

    def lexical(self, query, k):
        self.calls.append(("lexical", query, k))
        return [{"parent_id": pid, "chunk_id": f"chk_{pid}", "score": 0.5}
                for pid in self.table.get("lexical", {}).get(query, [])][:k]


class RunQuestionsTest(unittest.TestCase):
    """run_questions 把每次检索变成一条 trace —— 指标只能从日志算，所以这里只验写入内容。"""

    def setUp(self):
        import tempfile
        from pathlib import Path

        from src import raglog

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_dir = Path(tmp.name)
        self.previous = raglog._STATE["log_dir"]
        raglog._STATE["log_dir"] = self.log_dir
        raglog.reset_run()
        self.addCleanup(self._restore)

    def _restore(self):
        from src import raglog

        raglog._STATE["log_dir"] = self.previous
        raglog.reset_run()

    def test_writes_one_trace_per_question_per_mode(self):
        from src import raglog

        questions = [
            {"qid": "q1", "question": "RAG 是什么？", "type": "concept",
             "gold_points": {"parent_ids": ["par_a"]}},
            {"qid": "q2", "question": "不存在的答案？", "type": "negative"},
        ]
        retriever = _FakeRetriever({"dense": {"RAG 是什么？": ["par_a", "par_b"]}})
        written = run_questions(questions, retriever, modes=["dense", "lexical"], k=5)
        self.assertEqual(written, 4)

        traces = raglog.read_run() if hasattr(raglog, "read_run") else None
        path = raglog.run_path()
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        events = [__import__("json").loads(line) for line in lines]
        retrieves = [e for e in events if e.get("event") == "retrieve"]
        self.assertEqual(len(retrieves), 4)
        first = retrieves[0]
        self.assertEqual(first["qid"], "q1")
        self.assertEqual(first["hits"][0]["id"], "par_a")
        self.assertEqual(first["hits"][0]["rank"], 1)
        self.assertIsNone(traces)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            run_questions([], _FakeRetriever({}), modes=["nope"], k=5)

    def test_modes_constant(self):
        self.assertIn("dense", MODES)
        self.assertIn("lexical", MODES)
        self.assertIn("hybrid", MODES)


if __name__ == "__main__":
    unittest.main()


class CorpusCosStatsTest(unittest.TestCase):
    """z 分数要用「query 对全库的余弦」的均值/标准差，这里验它和暴力计算一致。"""

    def _matrix(self):
        import numpy as np

        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(200, 8)).astype("float32")
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix

    def test_mean_and_std_match_bruteforce(self):
        import numpy as np

        from src.eval_store import CorpusCosStats

        matrix = self._matrix()
        stats = CorpusCosStats.build(matrix)
        query = matrix[3]
        cos = matrix @ query
        self.assertAlmostEqual(stats.mean(query), float(cos.mean()), places=5)
        self.assertAlmostEqual(stats.std(query), float(cos.std()), places=5)

    def test_z_score_sign(self):
        import numpy as np

        from src.eval_store import CorpusCosStats

        matrix = self._matrix()
        stats = CorpusCosStats.build(matrix)
        query = matrix[0]
        cos = float(matrix[0] @ query)
        z = (cos - stats.mean(query)) / stats.std(query)
        self.assertGreater(z, 2.0)      # 与自己最像，必须在均值右侧 2σ 以上

    def test_save_and_load(self):
        import tempfile
        from pathlib import Path

        from src.eval_store import CorpusCosStats

        matrix = self._matrix()
        stats = CorpusCosStats.build(matrix)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.npz"
            stats.save(path)
            loaded = CorpusCosStats.load(path)
            self.assertAlmostEqual(stats.mean(matrix[5]), loaded.mean(matrix[5]), places=6)
            self.assertAlmostEqual(stats.std(matrix[5]), loaded.std(matrix[5]), places=6)
