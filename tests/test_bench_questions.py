"""基准题集与评估指标口径的守卫。

两件事：

1. **题集本身要自洽**：qid 唯一、正样本有 gold、负样本没有 gold、`query_en` 人工写过、
   两个 id 空间（`gold_chunks`=parent / `gold_points`=child）不混；
2. **指标口径不许退化成「一个综合分」**：`hit@k` / `full@k` / `recall@k` 三者分母不同，
   必须同时报；负样本不进 recall 分母。
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "datasets" / "k8s-zh-en-v1" / "questions.jsonl"

import importlib.util


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_bench = _load("eval_bench", ROOT / "scripts" / "eval_bench.py")


class QuestionsIntegrityTest(unittest.TestCase):
    def setUp(self):
        if not QUESTIONS.is_file():
            self.skipTest("题集还没生成（先跑 scripts/build_bench_questions.py append）")
        self.rows = [json.loads(line) for line in
                     QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_qids_unique_and_count(self):
        qids = [row["qid"] for row in self.rows]
        self.assertEqual(len(qids), len(set(qids)), "qid 必须唯一")
        self.assertGreaterEqual(len(qids), 100, "第一版基准至少 100 题")

    def test_every_question_is_rewritten_text(self):
        for row in self.rows:
            self.assertTrue(row["question"].strip(), row["qid"])
            self.assertTrue(row["query_en"].strip(),
                            f"{row['qid']}：query_en 必须人工写，不许留空（机翻回填不允许）")

    def test_positives_have_gold_and_negatives_do_not(self):
        for row in self.rows:
            if row["type"] == "negative":
                self.assertEqual(row["gold_chunks"], [], f"{row['qid']}：负样本不该有 gold")
                self.assertEqual(row["gold_spans"], [], f"{row['qid']}：负样本不该有 span")
            else:
                self.assertTrue(row["gold_chunks"], f"{row['qid']}：正样本必须有 gold_chunks")
                self.assertTrue(row["gold_spans"], f"{row['qid']}：正样本必须有 gold_spans")

    def test_two_id_spaces_are_not_mixed(self):
        for row in self.rows:
            for cid in row["gold_chunks"]:
                self.assertTrue(cid.startswith("par_"), f"{row['qid']}：gold_chunks 应是 parent id")
            for pid in row.get("gold_points", []):
                self.assertTrue(pid.startswith("chk_"), f"{row['qid']}：gold_points 应是 child id")

    def test_evidence_language_is_declared_for_positives(self):
        for row in self.rows:
            if row["type"] == "negative":
                continue
            self.assertIn(row["expected_evidence_lang"], ("zh-cn", "en"), row["qid"])

    def test_type_vocabulary(self):
        allowed = {"single_hop", "multi_hop", "numeric", "cross_lingual", "negative"}
        for row in self.rows:
            self.assertIn(row["type"], allowed, f"{row['qid']}：未知题型 {row['type']}")

    def test_cross_lingual_group_exists_both_directions(self):
        x = [row for row in self.rows if row["type"] == "cross_lingual"]
        self.assertTrue(any(row["expected_evidence_lang"] == "en" for row in x), "缺「中文问→英文证据」")
        self.assertTrue(any(row["expected_evidence_lang"] == "zh-cn" for row in x), "缺「英文问→中文证据」")

    def test_negative_questions_stay_out_of_corpus(self):
        """负样本必须是「语料里确实没有」的问题，不能是「我们没写 gold」。"""
        negatives = [row for row in self.rows if row["type"] == "negative"]
        self.assertGreaterEqual(len(negatives), 3)
        for row in negatives:
            self.assertTrue(row["note"].startswith("负样本"), f"{row['qid']}：负样本要留痕说明")


class SummarizeMetricTest(unittest.TestCase):
    """指标口径：三个分母都要报，负样本不进分母。"""

    def _row(self, *, gold, hit, full, recall, rank):
        return {"gold": gold, "hit": hit, "full": full, "recall": recall, "first_hit_rank": rank}

    def test_reports_all_three_denominators(self):
        rows = [
            self._row(gold=["a"], hit=True, full=True, recall=1.0, rank=1),
            self._row(gold=["a", "b"], hit=True, full=False, recall=0.5, rank=3),
            self._row(gold=["a"], hit=False, full=False, recall=0.0, rank=None),
            self._row(gold=[], hit=False, full=False, recall=None, rank=None),   # 负样本
        ]
        stats = eval_bench.summarize(rows, 10)
        self.assertEqual(stats["positives"], 3, "负样本不进分母")
        self.assertAlmostEqual(stats["hit@10"], 0.6667, places=4)
        self.assertAlmostEqual(stats["full@10"], 0.3333, places=4)
        self.assertAlmostEqual(stats["recall@10"], 0.5, places=4)
        self.assertAlmostEqual(stats["mrr"], (1 / 1 + 1 / 3) / 3, places=4)

    def test_all_miss_gives_zero_mrr(self):
        rows = [self._row(gold=["a"], hit=False, full=False, recall=0.0, rank=None)]
        stats = eval_bench.summarize(rows, 10)
        self.assertEqual(stats["mrr"], 0.0, "全未命中时 MRR 必须是 0，不能留空")

    def test_no_positives_is_not_a_zero(self):
        stats = eval_bench.summarize([self._row(gold=[], hit=False, full=False, recall=None, rank=None)], 10)
        self.assertEqual(stats, {"n": 1, "positives": 0},
                         "全是负样本时不该报 0% 命中，那样会误导成「系统全错」")


if __name__ == "__main__":
    unittest.main()
