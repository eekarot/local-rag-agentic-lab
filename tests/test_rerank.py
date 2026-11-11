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
    rerank_tiles,
    tile_windows,
    window_candidates,
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


class WindowCandidatesTest(unittest.TestCase):
    """把「有哪些可选窗口」从 `best_window()` 里拆出来（C0f）。

    为什么要拆：C0f 的核心发现是**跨语言时词面重叠这个打分信号可能是空的**，
    那时无论怎么加权都是在挑「第一个窗口」。要换挑法，就必须先能拿到**全部**候选窗口，
    而且枚举逻辑只能有一份实现——两份就会变成「测的窗口 ≠ 跑的窗口」。

    这个类只锁枚举本身的三条性质，不锁「怎么挑」（那是 `best_window` 的事）：
    窗口覆盖全文、每个窗口各自守预算、短文本不切。
    """

    def _counter(self):
        class FakeCounter:
            max_length = 0

            def count(self, text):
                return max(1, len(text) // 4)   # 4 字符 ≈ 1 token

        return FakeCounter()

    def test_long_text_yields_several_windows_covering_the_whole_text(self):
        text = "".join(f"第{i}段说了一件事。" for i in range(1, 21))
        # 这个用例用「1 字符 = 1 token」的计数器：逐句相加没有取整误差，预算可以严格断言。
        # （真词表下拼接后最多也就 1.000×，但它**故意**按逐句估算而不是拼接后卡死 ——
        #  精确卡死会让整体 hit@10 掉 2.85 点，见 C0f / N59。）
        exact = type("ExactCounter", (), {"count": staticmethod(len)})()
        windows = window_candidates(text, max_tokens=20, counter=exact)
        self.assertGreater(len(windows), 1, "长文本必须切出多个候选窗口，否则没得挑")
        for window in windows:
            self.assertLessEqual(exact.count(window), 20, "每个候选窗口都得守预算")
        # 覆盖性：首窗口从块头起、末窗口到块尾止；且右端严格向右推进
        self.assertTrue(text.startswith(windows[0]))
        self.assertTrue(text.endswith(windows[-1]))
        ends = [text.index(window) + len(window) for window in windows]
        self.assertEqual(ends, sorted(set(ends)),
                         "候选窗口必须按右端向右推进（左端只在超预算时才抬，所以前几个窗口是前缀）")

    def test_best_window_is_always_one_of_the_candidates(self):
        head = "目录 " + "第一章 概述 " * 40
        middle = "重启 CMDaemon 之前需要确认所有节点已经进入维护模式。"
        tail = "附录 " + "参考资料 " * 40
        text = head + middle + tail
        picked = best_window(text, "怎么重启 CMDaemon？", max_tokens=60, counter=self._counter())
        self.assertIn(picked, window_candidates(text, max_tokens=60, counter=self._counter()),
                      "`best_window()` 只能从候选里挑，不能另造一个窗口")

    def test_short_text_is_a_single_window(self):
        self.assertEqual(window_candidates("短文本。", max_tokens=256, counter=self._counter()),
                         ["短文本。"], "不超预算的文本不该被切，也不该返回空")

    def test_empty_text_has_no_window(self):
        self.assertEqual(window_candidates("", max_tokens=256, counter=self._counter()), [])


if __name__ == "__main__":
    unittest.main()


class TileWindowsTest(unittest.TestCase):
    """C0f：粗粒度分段（给「让交叉编码器挑窗口」用）。

    与 `window_candidates()` 的分工必须写清，否则下一个人会以为两个函数是重复实现：
    - `window_candidates()`：**全部可滑位置**（p50 每块 ~24 个），给「按词面重叠挑」用，
      便宜且细，但让 cross-encoder 逐段打分就是 40×24 次编码（实测一条 query 31 s）；
    - `tile_windows()`：**按预算切完不重叠**（p50 每块 1–2 段），给交叉编码器打分用。

    因此这里锁的是「切完、不留缝、不重叠、守预算」，不是「切得最优」。
    """

    def _counter(self):
        class FakeCounter:
            max_length = 0

            def count(self, text):
                return max(1, len(text) // 4)   # 4 字符 ≈ 1 token

        return FakeCounter()

    def test_covers_whole_text_without_gaps(self):
        text = "".join(f"第{i}段说了一件事。" for i in range(1, 21))
        tiles = tile_windows(text, max_tokens=20, counter=self._counter())
        self.assertGreater(len(tiles), 1, "超预算的文本必须被切成多段")
        self.assertEqual("".join(tiles), text, "分段必须首尾相接、不丢字、不重复")

    def test_each_tile_respects_budget(self):
        text = "".join(f"第{i}段说了一件事。" for i in range(1, 21))
        for tile in tile_windows(text, max_tokens=20, counter=self._counter()):
            self.assertLessEqual(self._counter().count(tile), 20)

    def test_short_text_is_a_single_tile(self):
        self.assertEqual(tile_windows("短文本。", max_tokens=256, counter=self._counter()),
                         ["短文本。"])

    def test_empty_text_is_one_empty_tile(self):
        """空文本也给一段（不是空列表）：调用方要按下标取「第一段」，不该被迫先判长度。"""
        self.assertEqual(tile_windows("", max_tokens=256, counter=self._counter()), [""])


class RerankTilesTest(unittest.TestCase):
    """C0f：让交叉编码器在**粗分段**里挑窗口再排序（`--window tiles`）。

    为什么必须换挑法：`best_window()` 靠「与 query 的词面重叠」，实测 gold 块里
    「随便哪一段都含答案句」= 1.0，而词面挑中的只有 0.79 —— 病在挑错，不在挑不到。
    端到端探针（`docs/evidence/c0f-window-strategy-k8s-v1.json`）四格全赢：
    整体 hit@10 0.8286→0.8571、MRR 0.5116→0.5433，英→中 hit 0.7083→0.7917。

    锁三件事：送进编码器的是**每段**（一次批量，不是逐候选调一次）、
    候选分取**段内最高分**、并列时保持原序（两次运行结果必须一致）。
    """

    def _counter(self):
        class FakeCounter:
            max_length = 0

            def count(self, text):
                # 每 5 个字符算 1 token，好让「一段 = 一句」这件事在测试里可预测
                return max(1, len(text) // 5)

        return FakeCounter()

    def test_scores_every_tile_and_keeps_the_best(self):
        text = "第一段内容。" * 400                      # 远超 256 token，必然拆成多段
        tiles = tile_windows(text, max_tokens=RerankConfig().max_length, counter=self._counter())
        self.assertGreater(len(tiles), 1, "前提：这段文本本来就该拆成多段")
        # 让**最后一段**拿最高分：证明「取段内最高分」不是只看第一段
        encoder = FakeEncoder(scores=[1.0] * (len(tiles) - 1) + [9.0])
        ranked = rerank_tiles("q", _hits((text, 0.5)), encoder=encoder,
                              config=RerankConfig(enabled=True), counter=self._counter())
        self.assertEqual(ranked[0].rerank_score, 9.0, "候选分应取段内最高分")
        self.assertEqual(ranked[0].fields["window"], tiles[-1], "还要记录**是哪一段**赢的（诊断用）")

    def test_encodes_all_tiles_in_one_call(self):
        long_text = "句子内容。" * 400
        hits = _hits((long_text, 0.5), ("短句内容。", 0.5))
        encoder = FakeEncoder(scores=[0.1, 0.2, 0.3, 0.4, 0.5])
        rerank_tiles("q", hits, encoder=encoder, config=RerankConfig(enabled=True),
                     counter=self._counter())
        self.assertEqual(len(encoder.calls), 1, "必须一次批量编码，不能逐候选各调一次")
        texts = encoder.calls[0][1]
        self.assertGreater(len(texts), len(hits), "长候选要拆成多段，所以送进去的段数多于候选数")

    def test_ties_keep_input_order(self):
        hits = _hits(("甲 #5", 0.5), ("乙 #5", 0.4), ("丙 #5", 0.3))
        ranked = rerank_tiles("q", hits, encoder=FakeEncoder(), config=RerankConfig(enabled=True),
                              counter=self._counter())
        self.assertEqual([hit.chunk_id for hit in ranked], ["c0", "c1", "c2"],
                         "同分必须保持原序，否则同一批题两次跑结果不一致")

    def test_encoder_failure_falls_back_to_input_order(self):
        hits = _hits(("甲 #1", 0.5), ("乙 #9", 0.4))
        stats = {"fallbacks": 0}
        ranked = rerank_tiles("q", hits, encoder=FakeEncoder(raises=RuntimeError("boom")),
                              config=RerankConfig(enabled=True), counter=self._counter(),
                              stats=stats)
        self.assertEqual([hit.chunk_id for hit in ranked], ["c0", "c1"])
        self.assertEqual(stats["fallbacks"], 1, "回退必须留痕，不许静默")

    def test_disabled_returns_input_order_without_calling_encoder(self):
        hits = _hits(("甲 #1", 0.5), ("乙 #9", 0.4))
        encoder = FakeEncoder()
        ranked = rerank_tiles("q", hits, encoder=encoder, config=RerankConfig(enabled=False),
                              counter=self._counter())
        self.assertEqual([hit.chunk_id for hit in ranked], ["c0", "c1"])
        self.assertEqual(encoder.calls, [], "关掉重排时一个编码调用都不该发")

    def test_respects_top_n(self):
        hits = _hits(("甲 #1", 0.5), ("乙 #9", 0.4), ("丙 #5", 0.3))
        ranked = rerank_tiles("q", hits, encoder=FakeEncoder(), config=RerankConfig(enabled=True),
                              counter=self._counter(), top_n=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].chunk_id, "c1")
