"""D33 Prompt 与引用规范用例。

三条验收线（§17.5 D33）：引用越界计数要能被算出来、证据不足要能弃答、
引用解析成功率 ≥99%。这里把口径钉死，避免以后「解析失败」被静默算成「没引用」。
"""

import unittest

from src.rag.answer.citations import (
    CitationReport,
    citation_summary,
    parse_citations,
    resolve_sources,
)
from src.rag.answer.prompt import (
    ABSTAIN_TEXT_EN,
    ABSTAIN_TEXT_ZH,
    SYSTEM_PROMPT,
    SYSTEM_RULES,
    build_messages,
    detect_abstain,
    looks_chinese,
    should_abstain,
)
from src.rag.context.assembler import ContextBlock, AssembledContext


def _block(index, text="#", tokens=10, doc="d1"):
    return ContextBlock(index=index, document_id=doc, parent_ids=[f"p{index}"], chunk_ids=[f"c{index}"],
                        section_path=["第1章"], source_uri="file.md", title=doc, text=text, tokens=tokens)


class ParseCitationTest(unittest.TestCase):
    def test_plain_and_tight_forms(self):
        report = parse_citations("先看 [证据 1]，再看 [证据2]。")
        self.assertEqual(report.cited, [1, 2])
        self.assertEqual(report.occurrences, 2)
        self.assertEqual(report.parse_rate, 1.0)

    def test_full_width_brackets(self):
        self.assertEqual(parse_citations("【证据 3】这么写也要认。").cited, [3])

    def test_multiple_numbers_in_one_bracket(self):
        report = parse_citations("参见 [证据 2,3]。")
        self.assertEqual(report.cited, [2, 3])
        self.assertEqual(report.occurrences, 2)

    def test_duplicates_deduped_but_counted(self):
        report = parse_citations("[证据 1] 和 [证据 1] 是同一处证据。")
        self.assertEqual(report.cited, [1])
        self.assertEqual(report.occurrences, 2)

    def test_out_of_range_detected(self):
        report = parse_citations("见 [证据 7]。", k=5)
        self.assertEqual(report.out_of_range, [7])
        self.assertEqual(report.cited, [])
        self.assertTrue(report.has_citation)

    def test_malformed_counted_separately(self):
        report = parse_citations("见 [证据 二]。")
        self.assertEqual(report.malformed, ["[证据 二]"])
        self.assertEqual(report.occurrences, 0)
        self.assertEqual(report.parse_rate, 0.0)

    def test_parse_rate_mixed(self):
        report = parse_citations("[证据 1] 和 [证据 二]。")
        self.assertAlmostEqual(report.parse_rate, 0.5)

    def test_no_citation_has_full_parse_rate_but_no_citation(self):
        report = parse_citations("完全没有引用的一句话。")
        self.assertEqual(report.parse_rate, 1.0)
        self.assertFalse(report.has_citation)

    def test_precision(self):
        report = parse_citations("[证据 1] [证据 2] [证据 9]", k=5)
        self.assertAlmostEqual(report.precision, 2 / 3)

    def test_abstain_detected(self):
        self.assertTrue(parse_citations(ABSTAIN_TEXT_ZH).abstained)
        self.assertTrue(parse_citations(ABSTAIN_TEXT_EN).abstained)
        self.assertFalse(parse_citations("答案是 42。").abstained)


class ResolveSourceTest(unittest.TestCase):
    def test_resolves_to_blocks(self):
        blocks = [_block(1, doc="d1"), _block(2, doc="d2")]
        sources = resolve_sources(parse_citations("[证据 2]"), [b.as_dict() for b in blocks])
        self.assertEqual([item["document_id"] for item in sources], ["d2"])

    def test_out_of_range_resolves_to_nothing(self):
        self.assertEqual(resolve_sources(parse_citations("[证据 9]", k=2), [_block(1).as_dict()]), [])

    def test_summary_across_answers(self):
        summary = citation_summary([parse_citations("[证据 1]", k=3), parse_citations(ABSTAIN_TEXT_ZH)])
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["abstain_rate"], 0.5)
        self.assertEqual(summary["out_of_range"], 0)
        self.assertEqual(summary["no_citation"], 1)


class PromptTest(unittest.TestCase):
    def setUp(self):
        self.ctx = AssembledContext(blocks=[_block(1, "第一段正文。"), _block(2, "第二段正文。", doc="d2")],
                                    tokens=20, dropped=[], stats={})

    def test_all_seven_rules_present(self):
        self.assertEqual(len(SYSTEM_RULES), 7)
        for number, rule in enumerate(SYSTEM_RULES, start=1):
            self.assertIn(f"{number}. {rule}", SYSTEM_PROMPT)

    def test_messages_numbered_evidence_and_sources(self):
        bundle = build_messages("HNSW 是什么？", self.ctx)
        self.assertEqual(bundle.messages[0]["role"], "system")
        user = bundle.messages[1]["content"]
        self.assertIn("[证据 1]", user)
        self.assertIn("[证据 2]", user)
        self.assertIn("document=d1", user)
        self.assertIn("section=第1章", user)
        self.assertIn("HNSW 是什么？", user)

    def test_language_hint_follows_question(self):
        self.assertIn("用中文回答", build_messages("中文提问？", self.ctx).messages[1]["content"])
        self.assertIn("Answer in English", build_messages("English question?", self.ctx).messages[1]["content"])
        self.assertEqual(build_messages("English question?", self.ctx).language, "en")

    def test_empty_context_abstains_without_model_call(self):
        bundle = build_messages("随便问问？", AssembledContext(blocks=[], tokens=0, dropped=[], stats={}))
        self.assertTrue(bundle.abstained)
        self.assertEqual(bundle.reason, "no_evidence")
        self.assertEqual(bundle.messages, [{"role": "assistant", "content": ABSTAIN_TEXT_ZH}])

    def test_empty_context_abstains_in_english(self):
        bundle = build_messages("Anything?", AssembledContext(blocks=[], tokens=0, dropped=[], stats={}))
        self.assertEqual(bundle.messages[0]["content"], ABSTAIN_TEXT_EN)

    def test_should_abstain_counts_tokens(self):
        self.assertTrue(should_abstain([]))
        self.assertFalse(should_abstain([_block(1, tokens=5)]))
        self.assertTrue(should_abstain([_block(1, tokens=0)]))

    def test_bundle_dict_has_what_logger_needs(self):
        payload = build_messages("问题？", self.ctx).as_dict()
        self.assertEqual(payload["tokens"], 20)
        self.assertEqual(len(payload["blocks"]), 2)
        self.assertTrue(payload["context"].startswith("[证据 1]"))

    def test_looks_chinese(self):
        self.assertTrue(looks_chinese("中文问题"))
        self.assertFalse(looks_chinese("what is hnsw"))
        self.assertFalse(looks_chinese("HNSW 是什么？".replace("是什么", "")))

    def test_detect_abstain(self):
        self.assertTrue(detect_abstain(ABSTAIN_TEXT_ZH))
        self.assertFalse(detect_abstain("HNSW 是一种近似最近邻索引。"))


if __name__ == "__main__":
    unittest.main()
