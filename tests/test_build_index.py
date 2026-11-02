"""build_index 的 embedding 重试逻辑：Ollama 偶发 400 时必须重试，而不是直接二分或中止。

实测背景：bge-m3 在 batch=256 时约 30% 的请求返回 400（runner 重启，tokenize 代理连不上），
同一批再发一次就成功。原实现「400 就二分」会在两个方向上都吃亏：
既会白白把 batch 拆小（吞吐掉一半），又可能在 batch=1 时仍然失败而中断整个 317 批的构建。
"""
import unittest

import numpy as np

import src.build_index as build_index


class _FakeEmbed:
    """可编排的假 embedding：按调用次数决定失败还是成功。"""

    def __init__(self, failures_before_success=None, fail_when=None):
        self.calls = []
        self.failures = failures_before_success or 0
        self.fail_when = fail_when  # callable(len(texts)) -> bool，为 True 时抛 400

    def __call__(self, texts):
        self.calls.append(len(texts))
        if self.fail_when is not None and self.fail_when(len(texts)):
            raise RuntimeError("Ollama API 返回 HTTP 400：tokenize 连接失败")
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("Ollama API 返回 HTTP 400：tokenize 连接失败")
        return [[float(i), 0.0] for i in range(len(texts))]


class EmbedRetryTest(unittest.TestCase):
    def setUp(self):
        self._orig_embed = build_index.embed_texts
        self._orig_sleep = build_index.time.sleep
        build_index.time.sleep = lambda seconds: None  # 测试里不要真的睡

    def tearDown(self):
        build_index.embed_texts = self._orig_embed
        build_index.time.sleep = self._orig_sleep

    def test_retries_400_then_succeeds_without_bisecting(self):
        fake = _FakeEmbed(failures_before_success=2)
        build_index.embed_texts = fake
        vectors = build_index._embed_batch(["a", "b"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(fake.calls, [2, 2, 2], "同一批应原地重试，不应拆小")

    def test_non_400_error_is_not_retried(self):
        calls = []

        def fake(texts):
            calls.append(len(texts))
            raise RuntimeError("Ollama 服务不可用：Connection refused")

        build_index.embed_texts = fake
        with self.assertRaises(RuntimeError):
            build_index._embed_batch(["a"])
        self.assertEqual(len(calls), 1, "服务不可用要立刻抛出，把进度留给进度文件")

    def test_persistent_400_on_batches_bisects_down_to_single(self):
        """batch>1 永远 400、单条成功：必须二分到 1 条才能拿到结果。"""
        fake = _FakeEmbed(fail_when=lambda n: n > 1)
        build_index.embed_texts = fake
        vectors = build_index._embed_batch(["a", "b"])
        self.assertEqual(len(vectors), 2)
        self.assertIn(1, fake.calls, "应该尝试过单条请求")

    def test_gives_up_after_retries_when_even_single_batch_fails(self):
        fake = _FakeEmbed(fail_when=lambda n: True)
        build_index.embed_texts = fake
        with self.assertRaises(RuntimeError):
            build_index._embed_batch(["a"])
        self.assertEqual(len(fake.calls), build_index.EMBED_400_RETRIES)

    def test_rejects_mismatched_vector_count(self):
        build_index.embed_texts = lambda texts: [[0.0, 0.0]]  # 少返回一条
        with self.assertRaises(RuntimeError):
            build_index._embed_batch(["a", "b"])


if __name__ == "__main__":
    unittest.main()
