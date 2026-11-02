"""D13：Embedding 服务（批量 / 维度与 NaN 校验 / 内容 hash 缓存）。

规格见方案 §7.2 与 D13 验收行，另有 ADR §17.11.1 追加的两条硬指标：
- query 缓存命中率 ≥60%，**命中延迟 <10 ms**（单条 query embedding 实测 2.1 s，
  全是 Ollama 固定开销，所以缓存是硬需求不是优化）；
- 同一文本二次调用命中率 ≥95%、P95 延迟降 ≥10×。

全部用例都用 `FakeBackend`，不碰网络；真模型校验在 `test_embedding_integration.py`。
"""

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.rag.embedding.cache import EmbeddingCache
from src.rag.embedding.errors import (
    EmbeddingBackendError,
    EmbeddingDimensionError,
    EmbeddingInputError,
    EmbeddingValueError,
)
from src.rag.embedding.service import DEFAULT_MODEL, DEFAULT_VERSION, VECTOR_SIZE, EmbeddingService


class FakeBackend:
    """可控的假后端：记录每批大小，可注入延迟/维度错误/NaN/批过大 400。"""

    def __init__(self, dim=VECTOR_SIZE, delay=0.0, zero_at=None, nan_at=None,
                 inf_at=None, fail_batch_gt=None, fail_times=0):
        self.dim = dim
        self.delay = delay
        self.zero_at = zero_at
        self.nan_at = nan_at
        self.inf_at = inf_at
        self.fail_batch_gt = fail_batch_gt   # 批大于该值时抛 400（模拟「批太大」）
        self.fail_times = fail_times         # 前 N 次调用直接 400（模拟偶发失败）
        self.calls: list[int] = []
        self.seen: list[str] = []

    def embed(self, texts):
        self.calls.append(len(texts))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise EmbeddingBackendError("HTTP 400: 偶发失败")
        if self.fail_batch_gt is not None and len(texts) > self.fail_batch_gt:
            raise EmbeddingBackendError("HTTP 400: batch too large")
        if self.delay:
            time.sleep(self.delay * len(texts))
        out = []
        for idx, text in enumerate(texts):
            self.seen.append(text)
            if self.zero_at is not None and idx == self.zero_at:
                out.append([0.0] * self.dim)
                continue
            vec = np.full(self.dim, 0.001, dtype=np.float64)
            # 让向量随文本变化，便于断言「顺序没被打乱」
            vec[0] = (sum(map(ord, text)) % 97) / 97.0
            if self.nan_at is not None and idx == self.nan_at:
                vec[1] = float("nan")
            if self.inf_at is not None and idx == self.inf_at:
                vec[1] = float("inf")
            out.append(vec.tolist())
        return out


class ServiceTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_path = Path(self._tmp.name) / "embed.sqlite3"

    def make(self, backend=None, **kwargs):
        kwargs.setdefault("cache_path", self.cache_path)
        kwargs.setdefault("backend", backend or FakeBackend())
        return EmbeddingService(**kwargs)


class ShapeAndOrderTest(ServiceTestBase):
    def test_returns_float32_matrix_in_input_order(self):
        backend = FakeBackend()
        service = self.make(backend)
        texts = ["alpha", "beta", "gamma"]
        matrix = service.encode_documents(texts)
        self.assertEqual(matrix.shape, (3, VECTOR_SIZE))
        self.assertEqual(matrix.dtype, np.float32)
        # 期望值：按 FakeBackend 的规则独立算出第 0 维
        expected = [(sum(map(ord, t)) % 97) / 97.0 for t in texts]
        np.testing.assert_allclose(matrix[:, 0], expected, rtol=1e-6)

    def test_batches_by_batch_size(self):
        backend = FakeBackend()
        service = self.make(backend, batch_size=2)
        service.encode_documents([f"t{i}" for i in range(5)])
        self.assertEqual(backend.calls, [2, 2, 1])

    def test_queries_and_documents_stay_separate(self):
        backend = FakeBackend()
        service = self.make(backend)
        service.encode_documents(["same text"])
        service.encode_queries(["same text"])
        self.assertEqual(len(backend.calls), 2, "doc/query 分属不同缓存命名空间（为将来非对称模型留位）")

    def test_normalize_option_gives_unit_vectors(self):
        service = self.make(normalize=True)
        matrix = service.encode_queries(["a", "bb"])
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), np.ones(2), rtol=1e-5)


class ValidationTest(ServiceTestBase):
    def test_dimension_mismatch_raises(self):
        service = self.make(FakeBackend(dim=512))
        with self.assertRaises(EmbeddingDimensionError) as ctx:
            service.encode_documents(["x"])
        self.assertIn("512", str(ctx.exception))
        self.assertIn(str(VECTOR_SIZE), str(ctx.exception))

    def test_nan_raises(self):
        with self.assertRaises(EmbeddingValueError) as ctx:
            self.make(FakeBackend(nan_at=0)).encode_documents(["x"])
        self.assertIn("NaN", str(ctx.exception))

    def test_inf_raises(self):
        with self.assertRaises(EmbeddingValueError) as ctx:
            self.make(FakeBackend(inf_at=0)).encode_documents(["x"])
        self.assertIn("Inf", str(ctx.exception))

    def test_zero_vector_raises(self):
        """§7.2「不得静默产生空向量」：全零向量会让余弦相似度全 0，检索静默失效。"""
        with self.assertRaises(EmbeddingValueError) as ctx:
            self.make(FakeBackend(zero_at=0)).encode_documents(["x"])
        self.assertIn("全零", str(ctx.exception))

    def test_empty_text_rejected(self):
        with self.assertRaises(EmbeddingInputError):
            self.make().encode_documents(["ok", "   "])

    def test_non_string_rejected(self):
        with self.assertRaises(EmbeddingInputError):
            self.make().encode_documents(["ok", None])

    def test_too_long_text_rejected(self):
        service = self.make(max_chars=10)
        with self.assertRaises(EmbeddingInputError) as ctx:
            service.encode_documents(["x" * 11])
        self.assertIn("10", str(ctx.exception))

    def test_empty_batch_returns_empty_matrix(self):
        matrix = self.make().encode_documents([])
        self.assertEqual(matrix.shape, (0, VECTOR_SIZE))


class CacheTest(ServiceTestBase):
    def test_second_call_is_all_cache_hits(self):
        backend = FakeBackend()
        service = self.make(backend)
        texts = ["one", "two", "three"]
        first = service.encode_documents(texts)
        calls_after_first = list(backend.calls)
        second = service.encode_documents(texts)
        self.assertEqual(backend.calls, calls_after_first, "第二批不该再调后端")
        np.testing.assert_array_equal(first, second)
        self.assertEqual(service.stats()["hits"], 3)
        self.assertEqual(service.stats()["misses"], 3)

    def test_cache_persists_across_service_instances(self):
        first = self.make(FakeBackend()).encode_documents(["persisted"])
        backend2 = FakeBackend()
        second = self.make(backend2).encode_documents(["persisted"])
        self.assertEqual(backend2.calls, [], "新实例应直接读缓存")
        np.testing.assert_array_equal(first, second)

    def test_model_or_version_change_invalidates_cache(self):
        self.make(FakeBackend()).encode_documents(["t"])
        backend = FakeBackend()
        self.make(backend, model="nomic-embed-text").encode_documents(["t"])
        self.assertEqual(backend.calls, [1], "换模型必须重新编码")
        backend2 = FakeBackend()
        self.make(backend2, version="v2").encode_documents(["t"])
        self.assertEqual(backend2.calls, [1], "换版本必须重新编码")

    def test_cache_disabled_always_calls_backend(self):
        backend = FakeBackend()
        service = self.make(backend, cache_enabled=False)
        service.encode_documents(["t"])
        service.encode_documents(["t"])
        self.assertEqual(backend.calls, [1, 1])

    def test_partial_hit_only_encodes_missing(self):
        backend = FakeBackend()
        service = self.make(backend)
        service.encode_documents(["a", "b"])
        backend.calls.clear()
        service.encode_documents(["a", "b", "c"])
        self.assertEqual(backend.calls, [1], "只该编码新的那一条")

    def test_corrupt_cache_row_is_treated_as_miss(self):
        service = self.make(FakeBackend())
        service.encode_documents(["x"])
        # 手工破坏：把向量 blob 截断
        with service.cache._connect() as conn:  # noqa: SLF001 - 测试故意碰内部
            conn.execute("UPDATE embeddings SET vector = x'00'")
        backend = FakeBackend()
        service2 = self.make(backend)
        service2.encode_documents(["x"])
        self.assertEqual(backend.calls, [1], "坏行按 miss 处理并重算")

    def test_cache_stats_bytes(self):
        service = self.make(FakeBackend())
        service.encode_documents(["a", "b"])
        stats = service.stats()
        self.assertEqual(stats["cached_rows"], 2)
        self.assertGreater(stats["cached_bytes"], VECTOR_SIZE * 4)


class MemoLayerTest(ServiceTestBase):
    """进程内 LRU 兜住「命中 <10 ms」的 SLA：第二次命中不该再碰 SQLite。"""

    def test_second_hit_skips_sqlite(self):
        service = self.make(FakeBackend())
        service.encode_documents(["warm"])
        calls = {"n": 0}
        original = service.cache.get_many

        def counting(keys, expected_dim):
            calls["n"] += 1
            return original(keys, expected_dim)

        service.cache.get_many = counting  # type: ignore[method-assign]
        service.encode_documents(["warm"])
        self.assertEqual(calls["n"], 0, "同进程重复命中应走内存，不查 SQLite")
        self.assertEqual(service.stats()["memo_hits"], 1)

    def test_memo_is_bounded(self):
        service = self.make(FakeBackend(), memo_size=2)
        service.encode_documents(["a", "b", "c", "d"])
        self.assertLessEqual(len(service._memo), 2)  # noqa: SLF001


class RetryTest(ServiceTestBase):
    def test_400_is_retried_in_place(self):
        backend = FakeBackend(fail_times=1)
        matrix = self.make(backend, batch_size=10).encode_documents(["a", "b"])
        self.assertEqual(matrix.shape, (2, VECTOR_SIZE))
        self.assertEqual(backend.calls, [2, 2], "先原地重试一次，成功即止")

    def test_batch_too_large_bisects(self):
        backend = FakeBackend(fail_batch_gt=2)
        matrix = self.make(backend, batch_size=8).encode_documents([f"t{i}" for i in range(8)])
        self.assertEqual(matrix.shape, (8, VECTOR_SIZE))
        # 二分路径上必然先试过 8 和 4（否则说明根本没走二分），
        # 重要的是能收敛到 ≤2 且 8 条文本都拿到了向量。
        self.assertEqual(sorted(set(backend.calls)), [2, 4, 8], f"实际调用 {backend.calls}")
        self.assertEqual(len(set(backend.seen)), 8)
        self.assertIn(2, backend.calls, "必须二分到服务端能接受的批大小")

    def test_backend_error_propagates_when_not_batch_related(self):
        class Broken:
            def embed(self, texts):
                raise EmbeddingBackendError("模型 bge-m3 不存在")

        with self.assertRaises(EmbeddingBackendError) as ctx:
            self.make(Broken()).encode_documents(["x"])
        self.assertIn("bge-m3", str(ctx.exception))


class LatencyTest(ServiceTestBase):
    """ADR 的两条硬指标：P95 降 ≥10×、缓存命中 <10 ms。"""

    def test_p95_latency_drops_at_least_10x(self):
        texts = [f"chunk-{i}" for i in range(40)]
        backend = FakeBackend(delay=0.002)
        service = self.make(backend, batch_size=10)

        def p95(encode):
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                encode(list(texts))
                samples.append(time.perf_counter() - started)
            return float(np.percentile(samples, 95))

        cold = p95(service.encode_documents)
        warm = p95(service.encode_documents)
        self.assertGreaterEqual(cold / max(warm, 1e-9), 10.0,
                                f"P95 只降了 {cold / warm:.1f}×（cold={cold*1000:.1f}ms warm={warm*1000:.2f}ms）")

    def test_cache_hit_latency_under_10ms(self):
        service = self.make(FakeBackend(delay=0.02), batch_size=1)
        service.encode_queries(["what is HNSW?"])
        samples = []
        for _ in range(10):
            started = time.perf_counter()
            service.encode_queries(["what is HNSW?"])
            samples.append((time.perf_counter() - started) * 1000)
        self.assertLess(max(samples), 10.0, f"命中延迟 {max(samples):.1f} ms 超过 10 ms")

    def test_hit_rate_over_95_percent_on_repeated_corpus(self):
        service = self.make(FakeBackend(), batch_size=16)
        texts = [f"doc-{i}" for i in range(100)]
        service.encode_documents(texts)
        service.encode_documents(texts + ["one-more"])
        stats = service.stats()
        # 验收看的是「第二次调用的命中率」：累计值 100/(100+101)=0.50，
        # 拿累计值当验收指标会永远不达标，也不反映缓存是否生效。
        self.assertGreaterEqual(stats["last_pass"]["hit_rate"], 0.95, stats)
        self.assertEqual(stats["last_pass"]["hits"], 100)
        self.assertEqual(stats["last_pass"]["misses"], 1)


class CacheUnitTest(unittest.TestCase):
    def test_keys_are_content_hashes_not_texts(self):
        with TemporaryDirectory() as tmp:
            cache = EmbeddingCache(Path(tmp) / "c.sqlite3")
            key = cache.key_for(kind="doc", model="bge-m3", version="bge-m3-v1", text="hello")
            self.assertEqual(len(key), 64)
            self.assertNotIn("hello", key)
            self.assertEqual(key, cache.key_for(kind="doc", model="bge-m3", version="bge-m3-v1", text="hello"))
            self.assertNotEqual(key, cache.key_for(kind="query", model="bge-m3", version="bge-m3-v1", text="hello"))

    def test_dim_mismatch_on_read_is_miss(self):
        with TemporaryDirectory() as tmp:
            cache = EmbeddingCache(Path(tmp) / "c.sqlite3")
            key = cache.key_for(kind="doc", model="m", version="v", text="t")
            cache.put_many([(key, np.ones(8, dtype=np.float32), "m", "v", "doc", 8)])
            # get_many 只返回命中的键（miss 不返回，而不是返回 None）
            self.assertNotIn(key, cache.get_many([key], expected_dim=1024), "维度不符按 miss 处理")

    def test_purge(self):
        with TemporaryDirectory() as tmp:
            cache = EmbeddingCache(Path(tmp) / "c.sqlite3")
            key = cache.key_for(kind="doc", model="m", version="v", text="t")
            cache.put_many([(key, np.ones(1024, dtype=np.float32), "m", "v", "doc", 1024)])
            self.assertEqual(cache.stats()["rows"], 1)
            cache.purge()
            self.assertEqual(cache.stats()["rows"], 0)


class DefaultsTest(unittest.TestCase):
    def test_model_and_version_defaults(self):
        self.assertEqual(DEFAULT_MODEL, "bge-m3")
        self.assertEqual(DEFAULT_VERSION, "bge-m3-v1")
        self.assertEqual(VECTOR_SIZE, 1024)


if __name__ == "__main__":
    unittest.main()
