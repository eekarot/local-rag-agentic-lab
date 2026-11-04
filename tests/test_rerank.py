"""D31 Reranker 用例。

用**注入的假编码器**锁接口和语义（真模型只在集成用例里跑，见 `tests/test_rerank_integration.py`）：

- 送进编码器的是 `(query, 候选文本)` 对，候选文本由调用方给（默认取 `text`）；
- 输出顺序按重排分降序，**并列时保持原序**（否则同一批题两次跑结果不一致）；
- 关闭开关时**一个字节都不变**（D31 验收：与 D26 逐字节一致）；
- 编码器抛异常时回退原序并留痕，不把异常抛给上层；
- 并发上限必须真的生效（CPU 上同时跑 4 个 278M cross-encoder 只会互相拖慢）。
"""

import threading
import time
import unittest

import numpy as np

from src.rag.retrieval.rerank import (
    CrossEncoderReranker,
    RerankConfig,
    RerankHit,
    best_window,
    rerank_hits,
)


class FakeEncoder:
    """把候选文本里出现的数字当分数，便于断言顺序；同时记录调用。"""

    def __init__(self, scores=None, delay=0.0, raises=None):
        self.scores = scores
        self.delay = delay
        self.raises = raises
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def encode_pairs(self, query, texts):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append((query, list(texts)))
            if self.delay:
                time.sleep(self.delay)
            if self.raises:
                raise self.raises
            if self.scores is not None:
                return np.asarray(self.scores[: len(texts)], dtype=np.float32)
            return np.asarray([float(t.split("#")[-1]) for t in texts], dtype=np.float32)
        finally:
            with self._lock:
                self.active -= 1


def _hits(*specs):
    return [RerankHit(chunk_id=f"c{i}", parent_id=f"p{i}", text=t, score=s, rank=i + 1)
            for i, (t, s) in enumerate(specs)]


class RerankHitTest(unittest.TestCase):
    def test_fields_and_dict_round_trip(self):
        hit = RerankHit(chunk_id="c1", parent_id="p1", text="t", score=0.5, rank=3)
        self.assertEqual(hit.as_dict()["rank"], 3)
        self.assertIsNone(hit.rerank_score)
        self.assertIsNone(RerankHit.from_dict(hit.as_dict()).rerank_score)


class RerankOrderTest(unittest.TestCase):
    def test_reorders_by_rerank_score(self):
        hits = _hits(("a#1", 0.9), ("b#5", 0.8), ("c#3", 0.7))
        out = rerank_hits("q", hits, encoder=FakeEncoder())
        self.assertEqual([h.chunk_id for h in out], ["c1", "c2", "c0"])
        self.assertEqual([h.rerank_score for h in out], [5.0, 3.0, 1.0])

    def test_ties_keep_original_order(self):
        hits = _hits(("a#2", 0.9), ("b#2", 0.8), ("c#2", 0.7))
        out = rerank_hits("q", hits, encoder=FakeEncoder())
        self.assertEqual([h.chunk_id for h in out], ["c0", "c1", "c2"])

    def test_top_n_truncates_after_reordering(self):
        hits = _hits(("a#1", 0.9), ("b#5", 0.8), ("c#3", 0.7))
        out = rerank_hits("q", hits, encoder=FakeEncoder(), top_n=2)
        self.assertEqual([h.chunk_id for h in out], ["c1", "c2"])

    def test_query_and_texts_are_passed_through(self):
        encoder = FakeEncoder()
        rerank_hits("问题", _hits(("x#1", 1.0)), encoder=encoder)
        self.assertEqual(encoder.calls[0], ("问题", ["x#1"]))

    def test_custom_text_getter_is_used(self):
        encoder = FakeEncoder()
        hits = _hits(("body#1", 1.0))
        hits[0].fields["rerank_text"] = "alt#7"
        rerank_hits("q", hits, encoder=encoder, text_getter=lambda h: h.fields["rerank_text"])
        self.assertEqual(encoder.calls[0][1], ["alt#7"])

    def test_empty_input_does_not_call_encoder(self):
        encoder = FakeEncoder()
        self.assertEqual(rerank_hits("q", [], encoder=encoder), [])
        self.assertEqual(encoder.calls, [])


class RerankSwitchTest(unittest.TestCase):
    def test_disabled_returns_input_unchanged(self):
        hits = _hits(("a#1", 0.9), ("b#5", 0.8))
        before = [h.as_dict() for h in hits]
        out = rerank_hits("q", hits, encoder=FakeEncoder(), config=RerankConfig(enabled=False))
        self.assertEqual([h.as_dict() for h in out], before)
        self.assertEqual([h.rerank_score for h in out], [None, None])

    def test_disabled_still_respects_top_n(self):
        hits = _hits(("a#1", 0.9), ("b#5", 0.8), ("c#3", 0.7))
        out = rerank_hits("q", hits, encoder=FakeEncoder(), config=RerankConfig(enabled=False), top_n=2)
        self.assertEqual([h.chunk_id for h in out], ["c0", "c1"])

    def test_error_falls_back_to_original_order_and_leaves_note(self):
        hits = _hits(("a#1", 0.9), ("b#5", 0.8))
        reranker = CrossEncoderReranker(encoder=FakeEncoder(raises=RuntimeError("boom")))
        out = rerank_hits("q", hits, encoder=reranker.encoder, config=RerankConfig(fallback_on_error=True),
                          stats=reranker.stats)
        self.assertEqual([h.chunk_id for h in out], ["c0", "c1"])
        self.assertEqual(reranker.stats()["errors"], 1)
        self.assertEqual(reranker.stats()["fallbacks"], 1)


class ConcurrencyTest(unittest.TestCase):
    def test_max_concurrency_one_serializes_calls(self):
        encoder = FakeEncoder(delay=0.05)
        reranker = CrossEncoderReranker(encoder=encoder, config=RerankConfig(max_concurrency=1))

        def work():
            rerank_hits("q", _hits(("a#1", 1.0), ("b#2", 2.0)), encoder=reranker.encoder,
                        config=reranker.config, stats=reranker.stats)

        threads = [threading.Thread(target=work) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(encoder.max_active, 1)

    def test_calls_counter_accumulates(self):
        reranker = CrossEncoderReranker(encoder=FakeEncoder())
        rerank_hits("q", _hits(("a#1", 1.0), ("b#2", 2.0)), encoder=reranker.encoder,
                    config=reranker.config, stats=reranker.stats)
        rerank_hits("q", _hits(("a#1", 1.0)), encoder=reranker.encoder,
                    config=reranker.config, stats=reranker.stats)
        stats = reranker.stats()
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["pairs"], 3)


class ConfigTest(unittest.TestCase):
    def test_defaults_are_the_measured_ones(self):
        config = RerankConfig()
        self.assertEqual(config.enabled, True)
        self.assertEqual(config.max_length, 256)     # B08 探针：128/256/512 的延迟拐点
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.max_concurrency, 1)  # CPU 上并发只会互相拖慢
        self.assertEqual(config.top_k_pool, 20)

    def test_negative_values_rejected(self):
        with self.assertRaises(ValueError):
            RerankConfig(max_length=0)
        with self.assertRaises(ValueError):
            RerankConfig(batch_size=0)
        with self.assertRaises(ValueError):
            RerankConfig(max_concurrency=0)


class BestWindowTest(unittest.TestCase):
    """父块 ~1,500 字符而重排只截 256 token，所以要先挑一段「提到 query 词」的窗口。

    朴素做法是直接截开头（`text[:N]`），但技术文档的父块开头往往是标题或目录，
    真正相关的那句在中间。这里锁的是「窗口要跟着 query 走」。
    """

    def _counter(self):
        class FakeCounter:
            max_length = 0

            def count(self, text):
                return max(1, len(text) // 4)   # 4 字符 ≈ 1 token，够验证切窗逻辑

        return FakeCounter()

    def test_picks_window_containing_query_terms(self):
        head = "目录 " + "第一章 概述 " * 40
        middle = "重启 CMDaemon 之前需要确认所有节点已经进入维护模式。"
        tail = "附录 " + "参考资料 " * 40
        window = best_window(head + middle + tail, "怎么重启 CMDaemon？",
                             max_tokens=60, counter=self._counter())
        self.assertIn("CMDaemon", window)
        self.assertNotIn("参考资料", window)

    def test_falls_back_to_head_when_no_overlap(self):
        text = "第一段完全没有相关词。" * 5 + "第二段也没有。" * 5
        window = best_window(text, "zzz qqq", max_tokens=40, counter=self._counter())
        self.assertTrue(text.startswith(window[:8]))

    def test_short_text_returned_unchanged(self):
        text = "短文本。"
        self.assertEqual(best_window(text, "短文本", max_tokens=256, counter=self._counter()), text)

    def test_respects_token_budget(self):
        counter = self._counter()
        window = best_window("句子。" * 400, "句子", max_tokens=50, counter=counter)
        self.assertLessEqual(counter.count(window), 60)


if __name__ == "__main__":
    unittest.main()
