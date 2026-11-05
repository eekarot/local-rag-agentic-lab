"""弃答判定（N12）：**「证据不足」出现在答案里 ≠ 模型弃答**。

D35 首版的 40 条答案里，14 条被标成弃答但其实在正常回答——因为模型在描述系统行为时
会引用这个词：

    如果模型反复认为证据不足并不断搜索，循环可能永远无法结束 [证据 1]      ← 答得很完整

同时，负样本的正确弃答也常常**同时引用**它检查过的块（"未提及电话，仅提供邮箱 [证据 1][证据 2]"），
所以「弃答 ∧ 有引用 ⇒ 误判」这条捷径是错的，会把正确行为算成错误自信。

正确判据是**位置 + 强度**：
- 结论型短语（"无法找到"/"知识库中没有"/"未提供足够"）出现在哪里都算弃答；
- 弱信号（"证据不足"）只在**答案开头**（前 120 字符）才算——它出现在中间多半是在转述内容。
"""

import unittest

from src.rag.answer.citations import detect_abstain, parse_citations


class StrongPatternTest(unittest.TestCase):
    def test_leading_conclusion_is_abstain(self):
        self.assertTrue(detect_abstain("根据检索结果，无法找到 NVIDIA 客服电话。[证据 1]"))

    def test_conclusion_anywhere_counts(self):
        self.assertTrue(detect_abstain("证据都看过了。结论：知识库中没有提供该型号的显存容量。[证据 3]"))

    def test_english_conclusion(self):
        self.assertTrue(detect_abstain("The provided context cannot answer this question. [证据 1]"))


class WeakPatternTest(unittest.TestCase):
    def test_weak_pattern_at_head_counts(self):
        self.assertTrue(detect_abstain("证据不足，无法确认该参数。"))

    def test_topic_mention_in_body_is_not_abstain(self):
        """真实误判样本（eval-009）：完整回答了「为什么要设最大步数」。"""
        answer = ("根据提供的知识库内容，Agent Loop 需要设置最大步数的主要原因如下："
                  "如果模型反复认为证据不足并不断搜索，循环可能永远无法结束。"
                  "设置最大步数可以让系统在可控范围内终止 [证据 1]。")
        self.assertFalse(detect_abstain(answer))

    def test_abstain_with_citations_still_counts(self):
        """负样本的正确弃答会同时引用它检查过的块——不能因为有引用就否决弃答。"""
        answer = ("根据提供的知识库内容，无法找到该客服电话。检索到的文档只提供了邮箱 [证据 1][证据 2]，"
                  "并未提及电话号码。")
        self.assertTrue(detect_abstain(answer))


class CitationReportTest(unittest.TestCase):
    def test_parse_citations_uses_same_rule(self):
        good = "它会在证据不足时改写查询 [证据 1]。"
        self.assertFalse(parse_citations(good, k=3).abstained)
        self.assertTrue(parse_citations("证据不足，无法回答。", k=3).abstained)


class AbsenceScopeTest(unittest.TestCase):
    """N15：弃答判定必须**分辨"谁缺"**——是"证据里没有 X"，不是"这个块装不下 X"。

    真实事故（2026-11-05，O3 的 HTTP 端到端）：`/api/v1/ask` 返回 `abstained=true`，
    但答案完整、引用了 [证据 2] 和 [证据 1]。触发句是——

        「当切分片段（chunk）过小时，单个块可能无法包含完整的上下文信息 [证据 2]。」

    `无法` + `包含` 满足了「否定 + 缺失」的句法形状，但这句在**描述 chunk 的容量**，
    跟"知识库答不了"毫无关系。所以判据再加一条：**缺失主张必须落在证据/资料这个主语上**。
    """

    def test_content_capacity_sentence_is_not_abstain(self):
        from src.rag.answer.citations import detect_abstain

        answer = (
            "切分过短主要会带来以下问题："
            "当切分片段（chunk）过小时，单个块可能无法包含完整的上下文信息 [证据 2]。"
        )
        self.assertFalse(detect_abstain(answer))

    def test_capability_sentence_about_code_is_not_abstain(self):
        from src.rag.answer.citations import detect_abstain

        self.assertFalse(detect_abstain("这种写法的单线程实现无法提供更高的吞吐量 [证据 1]。"))

    def test_absence_about_knowledge_base_is_abstain(self):
        from src.rag.answer.citations import detect_abstain

        self.assertTrue(detect_abstain("根据提供的知识库内容，未包含 NVIDIA 客服电话信息。"))
        self.assertTrue(detect_abstain("检索到的证据中没有找到该产品的零售价。"))
        self.assertTrue(detect_abstain("现有文档未提供该型号的保修年限。"))

    def test_absence_about_documents_in_english_is_abstain(self):
        from src.rag.answer.citations import detect_abstain

        self.assertTrue(detect_abstain("The provided documents do not contain the pricing information."))
        self.assertFalse(detect_abstain("A smaller chunk cannot contain the full context [evidence 1]."))


if __name__ == "__main__":
    unittest.main()
