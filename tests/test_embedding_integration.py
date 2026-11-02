"""D13 真模型集成测试：只在 `RAG_OLLAMA_CHECK=1` 且 Ollama 可达时运行。

单元测试用的是 `FakeBackend`（能验机制，不能验「bge-m3 真的返回 1024 维有限向量」）。
这里补三件 FakeBackend 验不了的事：
1. 真实维度与数值健康（有限、非零）；
2. 同一文本编码两次的余弦 ≈ 1.0（模型确定性，缓存才有意义）；
3. query 缓存命中延迟 <10 ms（ADR §17.11.1 的硬指标，单条 query 冷调用实测约 2 s）。
"""

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.config import DATA_DIR, OLLAMA_BASE_URL
from src.rag.embedding.service import VECTOR_SIZE, EmbeddingService


def _ollama_ready() -> bool:
    if os.getenv("RAG_OLLAMA_CHECK") != "1":
        return False
    try:
        import requests

        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


@unittest.skipUnless(_ollama_ready(), "需要 RAG_OLLAMA_CHECK=1 且 Ollama 可达")
class RealModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.cache_path = Path(cls._tmp.name) / "embed.sqlite3"
        cls.service = EmbeddingService(cache_path=cls.cache_path)
        md = sorted((DATA_DIR / "md").rglob("*.md"))[:3]
        cls.chunks = []
        for path in md:
            text = path.read_text(encoding="utf-8", errors="replace")
            cls.chunks.append(" ".join(text.split())[:400])
        cls.chunks = [c for c in cls.chunks if c.strip()]

    def test_dimension_and_health(self):
        matrix = self.service.encode_documents(self.chunks)
        self.assertEqual(matrix.shape, (len(self.chunks), VECTOR_SIZE))
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertTrue(np.all(np.abs(matrix).sum(axis=1) > 0))

    def test_same_text_twice_is_near_identical(self):
        text = self.chunks[0]
        first = self.service.encode_documents([text])
        # 换新实例 + 新缓存，确保是「再算一次」而不是读缓存
        with TemporaryDirectory() as tmp:
            again = EmbeddingService(cache_path=Path(tmp) / "fresh.sqlite3").encode_documents([text])
        cos = float(first[0] @ again[0] / (np.linalg.norm(first[0]) * np.linalg.norm(again[0])))
        self.assertGreater(cos, 0.999, f"同文本两次编码余弦只有 {cos:.5f}，模型不确定则缓存不可复用")

    def test_query_cache_hit_under_10ms(self):
        query = "HNSW 的 ef_construct 参数是做什么的？"
        started = time.perf_counter()
        self.service.encode_queries([query])
        cold_ms = (time.perf_counter() - started) * 1000
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            self.service.encode_queries([query])
            samples.append((time.perf_counter() - started) * 1000)
        warm_ms = max(samples)
        self.assertEqual(self.service.stats()["last_pass"]["hits"], 1)
        self.assertLess(warm_ms, 10.0, f"命中延迟 {warm_ms:.1f} ms（冷 {cold_ms:.0f} ms）超过 10 ms")


if __name__ == "__main__":
    unittest.main()
