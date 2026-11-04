"""日志 → 指标 的解析与聚合测试：用合成日志，不依赖真实 run。"""

import json
import tempfile
import unittest
from pathlib import Path

from src.eval_logs import aggregate, load_traces, latest_run, score_traces


def _write_log(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n",
        encoding="utf-8",
    )


def _retrieve(trace_id, qid, mode, k, hit_ids, latency=10.0):
    return {
        "event": "retrieve",
        "trace_id": trace_id,
        "qid": qid,
        "mode": mode,
        "k": k,
        "latency_ms": latency,
        "hits": [{"rank": i, "id": doc_id, "source": f"doc{i}"} for i, doc_id in enumerate(hit_ids, 1)],
    }


QUESTIONS = [
    {
        "qid": "q1",
        "question": "Q1",
        "type": "fact",
        "lang": "zh",
        "gold_chunks": [{"id": "a", "grade": 2}],
    },
    {
        "qid": "q2",
        "question": "Q2",
        "type": "concept",
        "lang": "zh",
        "gold_chunks": [{"id": "z", "grade": 2}],
    },
    {
        "qid": "q3",
        "question": "Q3（负样本）",
        "type": "negative",
        "lang": "zh",
        "gold_chunks": [],
    },
]


class TestLoadTraces(unittest.TestCase):
    def test_extracts_ranked_hits_and_skips_non_retrieve_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2026-01-01" / "run1.jsonl"
            _write_log(path, [
                {"event": "run", "run_id": "run1", "content_hash": "abc"},
                _retrieve("t1", "q1", "hybrid", 10, ["x", "a", "y"]),
                {"event": "generate", "trace_id": "t1", "qid": "q1"},
                {"event": "trace", "trace_id": "t1", "qid": "q1", "status": "ok"},
                {"event": "trace", "trace_id": "t2", "qid": "q2", "status": "error"},
            ])
            traces = load_traces(path)

        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace["qid"], "q1")
        self.assertEqual(trace["mode"], "hybrid")
        self.assertEqual(trace["k"], 10)
        self.assertEqual(trace["hits"], ["x", "a", "y"])
        self.assertEqual(trace["status"], "ok")

    def test_latest_run_reads_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_log(root / "runs.jsonl", [
                {"event": "run", "run_id": "old", "file": "logs/2026-01-01/old.jsonl"},
                {"event": "run", "run_id": "new", "file": "logs/2026-01-02/new.jsonl"},
            ])
            self.assertEqual(latest_run(root), "new")

    def test_latest_run_missing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(latest_run(Path(tmp)))


class TestScoreTraces(unittest.TestCase):
    def _traces(self):
        return [
            {"qid": "q1", "mode": "hybrid", "k": 10, "hits": ["x", "a", "y"], "status": "ok", "latency_ms": 5.0},
            {"qid": "q2", "mode": "hybrid", "k": 10, "hits": ["z", "x", "y"], "status": "ok", "latency_ms": 7.0},
            {"qid": "q3", "mode": "hybrid", "k": 10, "hits": ["x", "y", "w"], "status": "ok", "latency_ms": 6.0},
        ]

    def test_scores_positive_questions_and_keeps_negatives_separate(self):
        scored = score_traces(self._traces(), QUESTIONS, ks=(4, 10))
        self.assertEqual(len(scored["positives"]), 2)
        self.assertEqual(len(scored["negatives"]), 1)
        q1 = next(row for row in scored["positives"] if row["qid"] == "q1")
        self.assertEqual(q1["first_rank"], 2)
        self.assertTrue(q1["hit"][4])
        self.assertAlmostEqual(q1["mrr"][10], 0.5)
        self.assertEqual(q1["type"], "fact")
        self.assertEqual(scored["negatives"][0]["first_rank"], None)

    def test_unmatched_qids_are_reported_not_silently_dropped(self):
        traces = self._traces() + [
            {"qid": "s0-01", "mode": "hybrid", "k": 10, "hits": ["a"], "status": "ok", "latency_ms": 1.0}
        ]
        scored = score_traces(traces, QUESTIONS, ks=(4,))
        self.assertEqual(scored["unmatched"], ["s0-01"])

    def test_rows_carry_top1_cos_z_when_branch_stats_present(self):
        """负样本要比的是「相对全库多离群」，不是绝对余弦（余弦尺度随模型变）。"""
        traces = [
            {
                "qid": "q3", "mode": "hybrid", "k": 10, "hits": ["x", "y"], "status": "ok",
                "branch": {"vector": {"top1_cos": 0.70, "cos_mean": 0.50, "cos_std": 0.10}},
            },
            {
                "qid": "q1", "mode": "hybrid", "k": 10, "hits": ["a", "x"], "status": "ok",
                "branch": {"vector": {"top1_cos": 0.62, "cos_mean": 0.50, "cos_std": 0.05}},
            },
        ]
        scored = score_traces(traces, QUESTIONS, ks=(4,))
        neg = scored["negatives"][0]
        pos = next(row for row in scored["positives"] if row["qid"] == "q1")
        self.assertAlmostEqual(neg["top1_cos"], 0.70)
        self.assertAlmostEqual(neg["top1_cos_z"], 2.0)
        self.assertAlmostEqual(pos["top1_cos_z"], 2.4)

    def test_top1_cos_z_is_none_for_old_logs_without_branch_stats(self):
        scored = score_traces(self._traces(), QUESTIONS, ks=(4,))
        self.assertIsNone(scored["negatives"][0]["top1_cos_z"])
        self.assertIsNone(scored["positives"][0]["top1_cos_z"])

    def test_branch_with_null_vector_is_tolerated(self):
        """纯 bm25 模式的日志里 branch.vector 是 null，不能因此崩掉整个 measure。"""
        traces = [{
            "qid": "q1", "mode": "bm25", "k": 10, "hits": ["a"], "status": "ok",
            "branch": {"bm25": {"terms_hit": 2, "postings_touched": 30, "candidates": 9}, "vector": None},
        }]
        scored = score_traces(traces, QUESTIONS, ks=(4,))
        row = scored["positives"][0]
        self.assertIsNone(row["top1_cos"])
        self.assertIsNone(row["top1_cos_z"])

    def test_top1_cos_z_survives_zero_std(self):
        traces = [{
            "qid": "q3", "mode": "hybrid", "k": 10, "hits": ["x"], "status": "ok",
            "branch": {"vector": {"top1_cos": 0.7, "cos_mean": 0.5, "cos_std": 0.0}},
        }]
        scored = score_traces(traces, QUESTIONS, ks=(4,))
        self.assertIsNone(scored["negatives"][0]["top1_cos_z"])


class TestAggregate(unittest.TestCase):
    def _scored(self):
        return score_traces([
            {"qid": "q1", "mode": "hybrid", "k": 10, "hits": ["a", "x"], "status": "ok", "latency_ms": 5.0},
            {"qid": "q2", "mode": "hybrid", "k": 10, "hits": ["x", "z"], "status": "ok", "latency_ms": 7.0},
            {"qid": "q1", "mode": "bm25", "k": 10, "hits": ["x", "y"], "status": "ok", "latency_ms": 3.0},
            {"qid": "q2", "mode": "bm25", "k": 10, "hits": ["z", "x"], "status": "ok", "latency_ms": 4.0},
        ], QUESTIONS, ks=(4, 10))

    def test_overall_slice_reports_both_recall_definitions_with_ci(self):
        result = aggregate(self._scored(), ks=(4, 10), n_boot=200, seed=0)
        hybrid = result["by_mode"]["hybrid"]
        self.assertEqual(hybrid["n"], 2)
        self.assertAlmostEqual(hybrid["hit"][4]["mean"], 1.0)
        self.assertAlmostEqual(hybrid["full"][4]["mean"], 1.0)
        self.assertAlmostEqual(hybrid["mrr"][10]["mean"], (1.0 + 0.5) / 2)
        self.assertIn("half_width", hybrid["hit"][4])

    def test_slices_by_type_and_language(self):
        result = aggregate(self._scored(), ks=(4,), n_boot=100, seed=0)
        self.assertIn("fact", result["by_type"])
        self.assertIn("concept", result["by_type"])
        self.assertIn("zh", result["by_lang"])
        self.assertEqual(result["by_type"]["fact"]["n"], 2)  # 每题一个模式各一条

    def test_latency_percentiles_per_mode(self):
        result = aggregate(self._scored(), ks=(4,), n_boot=50, seed=0)
        self.assertAlmostEqual(result["by_mode"]["bm25"]["latency_ms"]["p50"], 3.5)
        self.assertEqual(result["by_mode"]["bm25"]["latency_ms"]["n"], 2)


if __name__ == "__main__":
    unittest.main()


class GoldSpaceTest(unittest.TestCase):
    """命中 id 属于哪个金标空间，必须自动判对——判错就是静默变 0。"""

    def _trace(self, hits, qid="q1", mode="vector"):
        return {"trace_id": f"t-{mode}", "qid": qid, "mode": mode, "k": 10,
                "hits": hits, "latency_ms": 1.0}

    def _question(self):
        return {
            "qid": "q1", "type": "fact",
            "gold_chunks": [{"id": "md/a.md::chunk-0000", "grade": 2}],
            "gold_points": {"parent_ids": ["par_" + "a" * 32]},
        }

    def test_detects_legacy_chunk_ids(self):
        from src.eval_logs import detect_gold_space

        self.assertEqual(detect_gold_space([self._trace(["md/a.md::chunk-0000"])]), "legacy_chunks")

    def test_detects_parent_ids(self):
        from src.eval_logs import detect_gold_space

        self.assertEqual(detect_gold_space([self._trace(["par_" + "b" * 32])]), "parent_points")

    def test_legacy_log_scores_against_gold_chunks_not_points(self):
        from src.eval_logs import score_traces

        scored = score_traces([self._trace(["md/a.md::chunk-0000"])], [self._question()], ks=(10,))
        self.assertEqual(len(scored["positives"]), 1)
        self.assertTrue(scored["positives"][0]["hit"][10])

    def test_parent_log_scores_against_gold_points(self):
        from src.eval_logs import score_traces

        scored = score_traces([self._trace(["par_" + "a" * 32])], [self._question()], ks=(10,))
        self.assertTrue(scored["positives"][0]["hit"][10])

    def test_wrong_space_would_zero_out(self):
        # 反向确认：显式指定错的空间，分数必须掉到 0（说明这个坑是真实的）
        from src.eval_logs import score_traces

        scored = score_traces(
            [self._trace(["md/a.md::chunk-0000"])], [self._question()],
            ks=(10,), gold_space="parent_points",
        )
        self.assertFalse(scored["positives"][0]["hit"][10])
