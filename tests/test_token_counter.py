"""D05：Token 计数契约（BGE-M3 口径，B01 决策下的本地实现）。

单元测试用临时构造的极简 tokenizer.json，不依赖 17 MB 的真实词表；
真实词表与 Ollama 的对齐见 test_token_counter_integration.py。
"""

import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, models, pre_tokenizers

from src.rag.ingestion.errors import TokenizerUnavailable
from src.rag.ingestion.tokenizer import TokenCounter, cli_main


def build_tiny_tokenizer(path):
    vocab = {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "hello": 4, "world": 5}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(path)
    return path


class TokenCounterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.path = build_tiny_tokenizer(os.path.join(cls.tmpdir.name, "tokenizer.json"))
        cls.counter = TokenCounter(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_empty_text_counts_as_zero(self):
        self.assertEqual(self.counter.count(""), 0)
        self.assertEqual(self.counter.count("   \n\t "), 0)

    def test_counts_tokens_not_characters(self):
        self.assertEqual(self.counter.count("hello world"), 2)
        self.assertEqual(self.counter.count("a b c"), 3)

    def test_unknown_words_are_counted_not_dropped(self):
        # 词表里不存在的中文/型号也必须占 token，不能被当 0 处理
        # （fixture 是空白切分的极简词表，所以这里只验证「未登录也计 1」这一条契约）
        self.assertEqual(self.counter.count("知识库 zzz"), 2)
        self.assertEqual(self.counter.count("zzz qqq"), 2)

    def test_non_empty_text_always_counts_at_least_one(self):
        for text in ("x", "z", "\u4e2d", "...", "0"):
            with self.subTest(text=text):
                self.assertGreaterEqual(self.counter.count(text), 1)

    def test_count_batch_matches_individual_counts(self):
        texts = ["hello world", "", "a b c", "知识库 zzz"]
        self.assertEqual(self.counter.count_batch(texts), [self.counter.count(t) for t in texts])

    def test_count_batch_on_empty_input(self):
        self.assertEqual(self.counter.count_batch([]), [])


class TruncateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.counter = TokenCounter(build_tiny_tokenizer(os.path.join(cls.tmpdir.name, "t.json")))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_text_within_budget_is_returned_unchanged(self):
        self.assertEqual(self.counter.truncate("hello world", 5), "hello world")

    def test_truncate_respects_budget_and_keeps_prefix(self):
        text = "hello world a b c"
        out = self.counter.truncate(text, 2)
        self.assertLessEqual(self.counter.count(out), 2)
        self.assertTrue(text.startswith(out))

    def test_truncate_to_zero_tokens_returns_empty(self):
        self.assertEqual(self.counter.truncate("hello world", 0), "")

    def test_truncate_empty_text(self):
        self.assertEqual(self.counter.truncate("", 10), "")

    def test_truncate_never_returns_text_over_budget(self):
        text = " ".join(["hello", "world", "a", "b", "c"] * 20)
        for budget in (1, 3, 7, 30):
            with self.subTest(budget=budget):
                self.assertLessEqual(self.counter.count(self.counter.truncate(text, budget)), budget)


class OllamaCalibrationTest(unittest.TestCase):
    """校验口径：逐条对比，Ollama 单条计数 == 内容 token + 2（BOS/EOS）。

    不用「聚合总量」对比的原因：实测 200 条 chunk 上，错模型的聚合差只有 0.35%，
    误差互相抵消后就分不出词表对不对；逐条对比才有分辨力。
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.counter = TokenCounter(build_tiny_tokenizer(os.path.join(self.tmpdir.name, "t.json")))

    def fake_api(self, scale=1.0, offset=0):
        """模拟 Ollama：单条返回内容 token×scale + 2 个特殊 token + offset。"""
        counter = self.counter

        def embed(batch):
            total = sum(int(counter.count(t) * scale) + 2 + offset for t in batch)
            return {"prompt_eval_count": total}

        return embed

    def test_aligned_when_api_matches_content_plus_specials(self):
        texts = ["hello world", "a b c"]
        report = self.counter.verify_against_ollama(texts, embed_fn=self.fake_api())
        self.assertTrue(report["aligned"])
        self.assertEqual(report["median_rel_diff"], 0.0)
        self.assertEqual(report["max_abs_diff"], 0)
        self.assertEqual(report["local_total"], report["ollama_total"])

    def test_absolute_offset_beyond_tolerance_is_unaligned(self):
        report = self.counter.verify_against_ollama(["a b c"], embed_fn=self.fake_api(offset=5))
        self.assertFalse(report["aligned"])
        self.assertEqual(report["max_abs_diff"], 5)

    def test_relative_drift_beyond_tolerance_is_unaligned(self):
        report = self.counter.verify_against_ollama(
            ["hello world a b c"], embed_fn=self.fake_api(scale=1.2)
        )
        self.assertFalse(report["aligned"])
        self.assertGreater(report["median_rel_diff"], 0.02)

    def test_report_carries_sample_size_and_both_tolerances(self):
        report = self.counter.verify_against_ollama(["hello world"], embed_fn=self.fake_api())
        self.assertEqual(report["n_texts"], 1)
        self.assertEqual(report["tolerance"], 0.02)
        self.assertEqual(report["max_abs_tolerance"], 2)
        self.assertEqual(len(report["per_text"]), 1)

    def test_empty_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            self.counter.verify_against_ollama([], embed_fn=self.fake_api())


class UnavailableTokenizerTest(unittest.TestCase):
    def test_missing_file_raises_with_download_hint(self):
        with self.assertRaises(TokenizerUnavailable) as ctx:
            TokenCounter(os.path.join(tempfile.gettempdir(), "no-such-dir", "tokenizer.json"))
        message = str(ctx.exception)
        self.assertIn("tokenizer.json", message)
        self.assertIn("ModelScope", message)

    def test_error_message_includes_the_path(self):
        missing = os.path.join(tempfile.gettempdir(), "nope-tokenizer.json")
        with self.assertRaises(TokenizerUnavailable) as ctx:
            TokenCounter(missing)
        self.assertIn("nope-tokenizer.json", str(ctx.exception))


class CliTest(unittest.TestCase):
    """D05 验收命令：python -m src.rag.ingestion.tokenizer count "文本"。"""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.path = build_tiny_tokenizer(os.path.join(cls.tmpdir.name, "tokenizer.json"))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        self.env = patch.dict(os.environ, {"RAG_TOKENIZER_PATH": self.path})
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_cli(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli_main(argv)
        return code, out.getvalue()

    def test_count_prints_token_number(self):
        code, out = self.run_cli(["count", "hello world"])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "2")

    def test_count_reads_stdin_lines_when_text_is_dash(self):
        stdin = io.StringIO("hello world\na b c\n")
        out = io.StringIO()
        # 3.12 没有 contextlib.redirect_stdin（3.13 才加）
        with patch("sys.stdin", stdin), contextlib.redirect_stdout(out):
            code = cli_main(["count", "-"])
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue().split(), ["2", "3"])

    def test_missing_command_exits_with_argparse_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli_main([])


if __name__ == "__main__":
    unittest.main()
