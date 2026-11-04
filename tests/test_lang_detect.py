"""D22：语言识别（zh / en，含「中文 + 英文术语」混排）。"""

from __future__ import annotations

import unittest

from src.rag.lang.detect import (
    detect,
    detect_batch,
    latin_word_count,
    cjk_char_count,
    language_ratio,
    summarize,
)


class CountTest(unittest.TestCase):
    def test_cjk_count_ignores_punctuation(self):
        self.assertEqual(cjk_char_count("HNSW 是什么？"), 3)

    def test_latin_word_count_counts_tokens_not_letters(self):
        # 这是关键：按「词」算而不是按「字母」算，否则缩写会把比例拉偏
        self.assertEqual(latin_word_count("DGX B300 的系统内存"), 2)
        self.assertEqual(latin_word_count("What is RAG?"), 3)

    def test_ratio_of_mixed_is_dominated_by_chinese(self):
        self.assertGreater(language_ratio("DGX B300 的系统内存最多能扩到多少？"), 0.8)


class DetectTest(unittest.TestCase):
    def test_pure_chinese(self):
        self.assertEqual(detect("为什么文档要切分成小段？"), "zh")

    def test_pure_english(self):
        self.assertEqual(detect("Why should documents be split into chunks?"), "en")

    def test_acronym_only_question_is_zh(self):
        # 验收条件里点名的一条：英文缩写 + 中文问句 → 仍然算中文
        self.assertEqual(detect("HNSW 是什么？"), "zh")

    def test_acronym_plus_number_question_is_zh(self):
        self.assertEqual(detect("DGX B300 装了几张 GPU？"), "zh")

    def test_english_query_with_product_names(self):
        self.assertEqual(detect("How many GPUs does a DGX B300 have?"), "en")

    def test_code_identifier_does_not_flip_to_en(self):
        self.assertEqual(detect("ef_construct 设成多少合适？"), "zh")

    def test_no_letters_is_unknown(self):
        self.assertEqual(detect("8192"), "unknown")
        self.assertEqual(detect(""), "unknown")
        self.assertEqual(detect("？？？，。"), "unknown")

    def test_long_english_quote_inside_chinese_still_zh(self):
        question = "NVIDIA 文档里那句“the chunk size should not exceed the maximum context length”是什么意思？"
        self.assertEqual(detect(question), "zh")

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(detect("   \n  What is RRF?\t "), "en")


class BatchTest(unittest.TestCase):
    def test_batch_keeps_order(self):
        self.assertEqual(detect_batch(["HNSW 是什么？", "What is RRF?", "8192"]),
                         ["zh", "en", "unknown"])

    def test_summarize_counts(self):
        stats = summarize(["zh", "zh", "en", "unknown"])
        self.assertEqual(stats["zh"], 2)
        self.assertEqual(stats["en"], 1)
        self.assertEqual(stats["unknown"], 1)
        self.assertEqual(stats["total"], 4)


class AccuracyGateTest(unittest.TestCase):
    """验收线：≥95%。这里用回归集（不是完整题集）守住「改规则不许变差」。"""

    ZH = [
        "RAG 是什么？",
        "为什么文档要切分成 chunk？",
        "Embedding 模型负责生成最终答案吗？",
        "Top-K 太大有什么风险？",
        "RRF 是用来做什么的？",
        "RNG 之外还有哪些检索方式？",
        "DGX B300 的系统内存最多能扩到多少？",
        "HNSW 是什么？",
        "NVLink 和 PCIe 有什么区别？",
        "ef_construct 设成多少合适？",
        "chunk 切得过短会有什么问题？",
        "MIG 关闭后要恢复点对点，需要什么在运行？",
        "BMC 的默认用户名该怎么处理？",
        "vGPU 属于哪个授权套餐？",
        "TopK 调太大会怎样？",
    ]
    EN = [
        "What is RAG?",
        "Why should documents be split into chunks?",
        "Does the Embedding model generate the final answer?",
        "What are the risks of Top-K being too large?",
        "What is RRF used for?",
        "How many GPUs does a DGX B300 have?",
        "How do the GPUs in a DGX B300 connect to the network?",
        "Which parameters significantly affect RAG accuracy?",
        "What is the maximum chunk size for the embedding model?",
        "Can a Reranker recover evidence never retrieved in stage one?",
        "What is the peak speed of NDR InfiniBand per direction?",
        "How many accelerators does the free BCM license support?",
        "Where do I configure the default BMC username?",
        "What does the Agent Loop do on each iteration?",
        "When is Agentic RAG worth the extra latency?",
    ]

    def test_accuracy_at_least_95_percent(self):
        pairs = [(q, "zh") for q in self.ZH] + [(q, "en") for q in self.EN]
        correct = sum(1 for text, want in pairs if detect(text) == want)
        accuracy = correct / len(pairs)
        self.assertGreaterEqual(accuracy, 0.95, f"准确率 {accuracy:.2%}，未达 95%")


if __name__ == "__main__":
    unittest.main()
