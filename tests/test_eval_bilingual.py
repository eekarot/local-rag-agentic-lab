"""D27：评估集扩到 100–200 条中英成对题（骨架 + 校验）。

结构见方案 §11.2，落地口径见 §17.11.3（B06）：`question` 就是 `query_zh`，
新增 `query_en`（**只用于英文 Dense / 英文 BM25 两路**，不进主指标）。

「成对」不是自动机翻就算——B06 明确要求 `query_en` 人工撰写，避免把翻译器的
缺陷写进评估集。所以这里的分工是：

* `draft`：机器生成**草稿**（`query_en_status="draft"`），人只需改；
* `review`：人工改完标 `reviewed`（或 `edited`）；
* `check`：只要还有 `draft`，评估集就不算可用——**拦在 CI 里**。
"""

from __future__ import annotations

import unittest

from src.eval_bilingual import (
    DIFFICULTIES,
    TOPICS,
    check,
    coverage,
    draft_items,
    infer_topic,
    pair_rate,
    protect_tokens,
    token_retention,
)


def _item(qid="eval-001", question="DGX B300 的显存是多少？", topic="number_spec",
          difficulty="normal", query_en="How much GPU memory does the DGX B300 have?",
          status="reviewed", **extra):
    item = {
        "qid": qid,
        "question": question,
        "type": "fact",
        "lang": "zh",
        "topic": topic,
        "difficulty": difficulty,
        "query_en": query_en,
        "query_en_status": status,
        "gold_spans": ["8 x 288 GB = 2.3 TB total"],
        "gold_chunks": [{"id": "x::chunk-0000", "grade": 2}],
        "reference_answer": "2.3 TB",
    }
    item.update(extra)
    return item


class TopicTaxonomyTest(unittest.TestCase):
    def test_eight_topics_match_spec(self):
        # 方案 §11.2 列的 8 类问题
        self.assertEqual(len(TOPICS), 8)
        self.assertEqual(set(TOPICS), {
            "concept_zh", "zh_with_en_term", "acronym_model", "number_spec",
            "paraphrase", "multi_condition", "cross_chunk", "no_answer",
        })

    def test_difficulties(self):
        self.assertEqual(set(DIFFICULTIES), {"easy", "normal", "hard"})

    def test_infer_negative_is_no_answer(self):
        self.assertEqual(infer_topic({"type": "negative", "question": "DGX B300 有几种颜色？"}), "no_answer")

    def test_infer_multi_hop_is_cross_chunk(self):
        self.assertEqual(infer_topic({"type": "multi_hop", "question": "显存和内存各是多少？"}), "cross_chunk")

    def test_infer_number_first(self):
        # 有数字就是「数字/参数/版本号」类，即使里面也有型号
        self.assertEqual(infer_topic({"type": "fact", "question": "DGX B300 的系统内存最多能扩到多少？"}), "number_spec")

    def test_infer_acronym_model(self):
        self.assertEqual(infer_topic({"type": "fact", "question": "MIG 关闭后要怎么恢复 NVLink？"}), "acronym_model")

    def test_infer_zh_with_en_term(self):
        self.assertEqual(infer_topic({"type": "fact", "question": "Reranker 能否找回没召回的证据？"}), "zh_with_en_term")

    def test_infer_plain_concept(self):
        self.assertEqual(infer_topic({"type": "concept", "question": "为什么文档要切分成小段？"}), "concept_zh")


class ProtectTokensTest(unittest.TestCase):
    def test_extracts_numbers_acronyms_and_model_names(self):
        tokens = protect_tokens("DGX B300 的 HNSW ef_construct 设成 200，显存 288 GB，走 PCIe Gen6")
        for want in ("DGX B300", "HNSW", "ef_construct", "288", "200", "PCIe Gen6"):
            self.assertIn(want, tokens)

    def test_ignores_chinese_only(self):
        self.assertEqual(protect_tokens("为什么文档要切分成小段"), set())

    def test_retention_full(self):
        item = _item()
        self.assertEqual(token_retention(item["question"], item["query_en"]), 1.0)

    def test_retention_drops_when_missing(self):
        item = _item(query_en="How much GPU memory does the machine have?")   # 丢了 DGX B300
        self.assertLess(token_retention(item["question"], item["query_en"]), 1.0)


class PairRateTest(unittest.TestCase):
    def test_full_pair(self):
        self.assertEqual(pair_rate([_item(), _item(qid="eval-002")]), 1.0)

    def test_missing_or_copied_query_en_not_paired(self):
        items = [
            _item(),
            _item(qid="eval-002", query_en=""),
            _item(qid="eval-003", query_en="DGX B300 的显存是多少？"),   # 直接抄中文
        ]
        self.assertEqual(pair_rate(items), round(1 / 3, 4))


class CoverageTest(unittest.TestCase):
    def test_counts_topics_and_difficulties(self):
        items = [_item(topic=t) for t in TOPICS]
        result = coverage(items)
        self.assertEqual(result["topics_covered"], 8)
        self.assertEqual(result["topics_missing"], [])
        self.assertEqual(result["counts"]["number_spec"], 1)

    def test_reports_missing(self):
        result = coverage([_item(topic="number_spec")])
        self.assertEqual(result["topics_covered"], 1)
        self.assertIn("no_answer", result["topics_missing"])


class CheckTest(unittest.TestCase):
    def test_clean_batch_has_no_errors(self):
        items = [_item(qid=f"eval-{i:03d}", topic=TOPICS[i % 8]) for i in range(8)]
        for i, item in enumerate(items):
            if item["topic"] == "no_answer":
                item["type"] = "negative"
            elif item["topic"] == "cross_chunk":
                item["type"] = "multi_hop"
            if item["type"] == "negative":
                item["gold_spans"] = []
                item["gold_chunks"] = []
                item["note"] = "价格类：全库无报价"
        errors = check(items)
        self.assertEqual(errors, [])

    def test_missing_query_en_is_error(self):
        errors = check([_item(query_en="")])
        self.assertTrue(any("query_en" in e for e in errors), errors)

    def test_copied_query_en_is_error(self):
        errors = check([_item(query_en="DGX B300 的显存是多少？")])
        self.assertTrue(any("query_en" in e for e in errors), errors)

    def test_unreviewed_draft_is_error(self):
        errors = check([_item(status="draft")])
        self.assertTrue(any("draft" in e for e in errors), errors)

    def test_unknown_topic_is_error(self):
        errors = check([_item(topic="随便写的")])
        self.assertTrue(any("topic" in e for e in errors), errors)

    def test_missing_difficulty_is_error(self):
        errors = check([_item(difficulty="")])
        self.assertTrue(any("difficulty" in e for e in errors), errors)

    def test_allow_drafts_flag(self):
        # 起草阶段允许 draft，CI 阶段不允许（默认 allow_drafts=False）
        self.assertEqual(check([_item(status="draft")], allow_drafts=True,
                               require_topic_coverage=False), [])

    def test_dropped_protected_token_is_error(self):
        errors = check([_item(query_en="How much GPU memory does the machine have?")])
        self.assertTrue(any("术语" in e or "原样" in e for e in errors), errors)


class DraftTest(unittest.TestCase):
    def test_draft_fills_only_empty_and_marks_status(self):
        items = [_item(qid="eval-001", query_en="", status=""), _item(qid="eval-002")]
        calls = []

        def translator(texts):
            calls.append(list(texts))
            return ["Drafted question?" for _ in texts]

        changed = draft_items(items, translator, force=False)
        self.assertEqual(changed, 1)
        self.assertEqual(calls[0], ["DGX B300 的显存是多少？"])  # 已有 query_en 的不重复翻
        self.assertEqual(items[0]["query_en"], "Drafted question?")
        self.assertEqual(items[0]["query_en_status"], "draft")
        self.assertEqual(items[1]["query_en_status"], "reviewed")

    def test_draft_force_redrafts(self):
        items = [_item(status="draft", query_en="old")]
        draft_items(items, lambda texts: ["new draft?" for _ in texts], force=True)
        self.assertEqual(items[0]["query_en"], "new draft?")

    def test_draft_skips_negative_with_empty_question(self):
        items = [_item(qid="eval-009", type="negative", question="", query_en="", status="")]
        draft_items(items, lambda texts: ["x" for _ in texts])
        self.assertEqual(items[0]["query_en"], "")


if __name__ == "__main__":
    unittest.main()
