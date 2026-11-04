"""D34 / D35 的纯函数用例：消融报告口径 + L2 汇总口径。

这两块最容易出的错不是算错，而是**口径混淆**：

- F 是「列表指标」，G 是「上下文指标」，混着报会把「token 变少」说成「检索变准」；
- L2 的正样本和负样本必须**分开算**，合成一个平均会让「弃答率高」看着像「答得准」。
"""

import unittest

from src.eval_answer import HUMAN_COLUMNS, pick_questions, summarize, write_sheet
from src.eval_rerank_ablation import legacy_context, render_report


def _question(qid, gold=("p1",), negative=False):
    item = {"qid": qid, "question": f"问题 {qid}",
            "type": "negative" if negative else "fact",
            "gold_points": {"parent_ids": list(gold) if not negative else []}}
    return item


class LegacyContextTest(unittest.TestCase):
    def test_one_block_per_hit_without_dedupe(self):
        parents = {"p1": {"document_id": "d1", "parent_index": 0, "text": "甲"},
                   "p2": {"document_id": "d1", "parent_index": 1, "text": "乙"}}
        legacy = legacy_context([{"parent_id": "p1"}, {"parent_id": "p1"}], parents)
        self.assertEqual(len(legacy["blocks"]), 2)      # 改造前不去重
        self.assertEqual(legacy["blocks"][0]["parent_indexes"], [0])

    def test_fragmentation_reported(self):
        parents = {"p1": {"document_id": "d1", "parent_index": 0, "text": "甲"},
                   "p2": {"document_id": "d1", "parent_index": 1, "text": "乙"}}
        legacy = legacy_context([{"parent_id": "p1"}, {"parent_id": "p2"}], parents)
        self.assertAlmostEqual(legacy["fragmentation"], 0.5)


class RenderTest(unittest.TestCase):
    def _payload(self):
        return {
            "n": 2, "pool": 20, "ks": [5, 10],
            "cells": {
                "E": {"metrics": {k: {"recall": 0.5, "hit": 0.5, "mrr": 0.4, "ndcg": 0.4} for k in ("5", "10")}},
                "F": {"metrics": {k: {"recall": 0.6, "hit": 0.5, "mrr": 0.5, "ndcg": 0.5} for k in ("5", "10")},
                      "latency_ms": {"p50": 700.0, "p95": 900.0}},
            },
            "pairs": [{"k": 10, "metric": "ndcg", "a_mean": 0.4, "b_mean": 0.5, "delta": 0.1,
                       "lo": 0.01, "hi": 0.2}],
            "context": {"legacy_blocks": 10.0, "legacy_tokens": 6000.0, "legacy_overlap": 0.2,
                        "legacy_fragmentation": 0.4, "assembled_blocks": 7.0, "assembled_tokens": 4500.0,
                        "assembled_overlap": 0.0, "assembled_fragmentation": 0.3,
                        "merged": 0.7, "duplicate_parent": 0.0, "duplicate_text": 0.1,
                        "near_duplicate_text": 0.0, "improved_overlap": 0.2, "token_saving": 0.25},
        }

    def test_report_contains_both_sections_and_labels(self):
        text = render_report(self._payload())
        self.assertIn("F = E + Reranker", text)
        self.assertIn("G = F + Context Assembler", text)
        self.assertIn("P50 700.0 ms", text)
        self.assertIn("相邻重叠块占比", text)
        self.assertIn("省 25.0%", text)          # token 节省必须出现在报告里

    def test_missing_p_value_renders_dash(self):
        payload = self._payload()
        payload["pairs"][0].pop("p_value", None)
        self.assertIn("—", render_report(payload))


class L2PickTest(unittest.TestCase):
    def test_picks_positives_then_negatives_in_fixed_order(self):
        questions = [_question("q1"), _question("q2"), _question("n1", negative=True),
                     _question("q3"), _question("n2", negative=True)]
        picked = pick_questions(questions, positives=2, negatives=1)
        self.assertEqual([item["qid"] for item in picked], ["q1", "q2", "n1"])

    def test_negative_without_gold_not_counted_as_positive(self):
        questions = [_question("n1", negative=True)]
        self.assertEqual(pick_questions(questions, positives=5, negatives=5)[0]["type"], "negative")


def _record(qid, negative=False, abstained=False, cited_blocks=2, hit_gold=1, gold_size=1,
            cited=(1, 2), out_of_range=(), malformed=(), called=True, tokens=4000, latency=1000.0):
    return {
        "qid": qid, "type": "negative" if negative else "fact", "question": "q", "answer": "a",
        "abstained": abstained, "blocks": 3, "context_tokens": tokens, "cited": list(cited),
        "out_of_range": list(out_of_range), "malformed": list(malformed),
        "cited_blocks": cited_blocks, "cited_hit_gold": hit_gold, "gold_size": gold_size,
        "is_negative": negative, "llm_called": called, "latency_ms": latency, "total_ms": latency,
        "parse_rate": 1.0,
    }


class L2SummaryTest(unittest.TestCase):
    def test_positive_and_negative_rates_are_separate(self):
        records = [
            _record("q1", cited_blocks=2, hit_gold=1),
            _record("q2", abstained=True, cited_blocks=0, hit_gold=0),
            _record("n1", negative=True, abstained=False, cited_blocks=0, hit_gold=0, gold_size=0),
            _record("n2", negative=True, abstained=True, cited_blocks=0, hit_gold=0, gold_size=0),
        ]
        summary = summarize(records)
        self.assertEqual(summary["positives"], 2)
        self.assertEqual(summary["negatives"], 2)
        self.assertAlmostEqual(summary["citation_precision"], 0.5)   # 1/2 引用块命中
        self.assertAlmostEqual(summary["over_abstain"], 0.5)
        self.assertAlmostEqual(summary["wrong_confidence"], 0.5)
        self.assertAlmostEqual(summary["negatives_abstained"], 0.5)

    def test_invalid_citation_rate_counts_out_of_range_and_malformed(self):
        records = [_record("q1", cited=(1, 2), out_of_range=(9,), malformed=("[证据 二]",))]
        summary = summarize(records)
        self.assertAlmostEqual(summary["invalid_citation_rate"], 2 / 4)

    def test_no_evidence_shortcut_rate(self):
        records = [_record("q1", called=False, abstained=True), _record("q2")]
        self.assertAlmostEqual(summarize(records)["no_evidence_shortcut"], 0.5)

    def test_empty_input_is_safe(self):
        summary = summarize([])
        self.assertEqual(summary["n"], 0)
        self.assertEqual(summary["citation_precision"], 0.0)

    def test_sheet_has_human_columns_blank(self):
        path = write_sheet([_record("q1")], "experiments/reports/_test_sheet.csv")
        text = path.read_text(encoding="utf-8-sig")
        header = text.splitlines()[0]
        for column in HUMAN_COLUMNS:
            self.assertIn(column, header)
        row = text.splitlines()[1]
        self.assertTrue(row.endswith(","))       # 最后四个人工列留空
        path.unlink()


if __name__ == "__main__":
    unittest.main()
