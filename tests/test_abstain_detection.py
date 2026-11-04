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


if __name__ == "__main__":
    unittest.main()
