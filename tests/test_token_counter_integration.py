"""D05 集成测试：真实 BGE-M3 词表 + Ollama 计数对齐。

默认跳过（需要 storage/tokenizer/bge-m3/tokenizer.json 与运行中的 Ollama）：
    RAG_RUN_SLOW=1 python -m unittest tests.test_token_counter_integration -v

第一个用例是需求本身（误差 ≤2%）；
第二个用例是**负向对照**——拿另一个模型的计数来比，必须判为不一致，
否则说明这个校验根本分辨不出错，等于没测。
"""

import json
import os
import unittest
from pathlib import Path

from src.rag.ingestion.tokenizer import DEFAULT_TOKENIZER_PATH, TokenCounter
from tests._slow import run_slow, slow

ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT_DIR / "storage" / "index.json"
RUN_OLLAMA = os.environ.get("RAG_OLLAMA_CHECK") == "1"

SAMPLE_SIZE = 200


def sample_texts(n=SAMPLE_SIZE):
    records = json.loads(INDEX_PATH.read_text(encoding="utf-8"))["records"]
    step = max(1, len(records) // n)
    return [records[i]["text"] for i in range(0, len(records), step)][:n]


@unittest.skipUnless(DEFAULT_TOKENIZER_PATH.is_file(), "缺少 bge-m3 tokenizer.json")
@slow
class RealVocabTest(unittest.TestCase):
    def test_counts_real_chunks_in_expected_range(self):
        counter = TokenCounter()
        counts = counter.count_batch(sample_texts(50))
        self.assertTrue(all(c > 0 for c in counts))
        # 现行切分是 500 字符，对应 100–200 token（实测均值 134）
        self.assertLess(max(counts), 400)

    def test_truncate_on_real_chinese_text(self):
        counter = TokenCounter()
        text = "HNSW 是用多层图做近似最近邻搜索的算法。" * 40
        out = counter.truncate(text, 50)
        self.assertLessEqual(counter.count(out), 50)
        self.assertGreater(counter.count(out), 40)

    def test_truncate_returns_exact_prefix_of_original(self):
        """真实词表下必须逐字符前缀：D07 要靠它算 source_start/source_end。

        decode(ids[:n]) 会把空格/标点规范化，得到的不是原文子串，
        用它算偏移会错位，所以截断要按 token 的字符 offset 回溯原文。
        """
        counter = TokenCounter()
        text = sample_texts(1)[0]
        self.assertGreater(counter.count(text), 150)
        for budget in (60, 120, 200):
            with self.subTest(budget=budget):
                out = counter.truncate(text, budget)
                self.assertLessEqual(counter.count(out), budget)
                self.assertTrue(text.startswith(out), f"截断结果不是原文前缀（budget={budget}）")

    def test_truncate_keeps_punctuation_and_spacing(self):
        counter = TokenCounter()
        text = "First sentence. Second sentence, with commas; and spaces. " * 30
        out = counter.truncate(text, 40)
        self.assertTrue(text.startswith(out))
        self.assertFalse(out.endswith(" ") and not text[len(out)].isspace())


@unittest.skipUnless(RUN_OLLAMA, "设置 RAG_RUN_SLOW=1 才跑（依赖 Ollama 服务；`RAG_OLLAMA_CHECK=1` 仍作旧名兼容）")
@unittest.skipUnless(DEFAULT_TOKENIZER_PATH.is_file(), "缺少 bge-m3 tokenizer.json")
@slow
class OllamaAlignmentTest(unittest.TestCase):
    def test_bge_m3_alignment_on_real_chunks(self):
        counter = TokenCounter()
        report = counter.verify_against_ollama(sample_texts(8))
        self.assertTrue(report["aligned"], report)
        self.assertEqual(report["median_rel_diff"], 0.0)

    def test_negative_control_other_model_is_detected(self):
        """拿 nomic 的计数冒充 bge-m3，必须判为不一致（实测中位 5.97%、最大差 107）。"""
        counter = TokenCounter()

        def embed_with_other_model(batch):
            import requests

            from src.config import OLLAMA_BASE_URL

            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": "nomic-embed-text", "input": list(batch)},
                timeout=300,
            )
            return response.json()

        report = counter.verify_against_ollama(sample_texts(8), embed_fn=embed_with_other_model)
        self.assertFalse(report["aligned"], report)
        self.assertGreater(report["median_rel_diff"], 0.02)


if __name__ == "__main__":
    unittest.main()
