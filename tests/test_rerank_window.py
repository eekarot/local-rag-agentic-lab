"""重排的「窗口正文」必须真来自父块（N45）。

BUG 实证（2026-11-06）：`retrieve_with_meta` 用 `parents[pid]["text"]` 取窗口正文，
而 `scripts/eval_bench.py` 的 `load_parents` 直接把 SQL 行塞进字典 —— 键是 `content`。
于是评测里**每个候选的窗口都是空串**，cross-encoder 拿到 40 个空串、打分全等、
稳定排序保持原序：**重排从未真正跑过，基线却一直标着「含重排」**。

两处一起修：
1. 取窗口时 `text` / `content` 都认（调用方用哪种 schema 都不会静默失效）；
2. `eval_bench.load_parents` 补上 `text` 别名（与 `store_wiring.load_parent_records` 一致）。

用例的设计：假 encoder 按「窗口里有没有 TARGET」打分 —— 窗口为空时所有分数相等，
顺序不变，用例必红；这也顺带证明「重排确实起作用」这件事本身有守卫。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Recorder:
    """记录喂给 encoder 的 (query, window) 对。"""

    def __init__(self):
        self.pairs = []

    def encode_pairs(self, query, texts):
        import numpy as np

        self.pairs.extend((query, text) for text in texts)
        return np.array([1.0 if "TARGET" in text else 0.0 for text in texts], dtype=np.float32)


class RerankWindowTest(unittest.TestCase):
    def setUp(self):
        from src.rag.ask.pipeline import retrieve_with_meta
        from src.rag.ingestion.tokenizer import TokenCounter
        from src.rag.retrieval.rerank import CrossEncoderReranker, RerankConfig

        self.retrieve = retrieve_with_meta
        self.counter = TokenCounter()
        self.recorder = _Recorder()
        self.reranker = CrossEncoderReranker(config=RerankConfig(enabled=True),
                                             encoder=self.recorder)

    def _parents(self, key):
        return {
            f"par_{i}": {key: ("TARGET 这一段是正确答案。" if i == 3 else "无关内容。"),
                         "content": ("TARGET 这一段是正确答案。" if i == 3 else "无关内容。"),
                         "title": "t", "section_path": "s"}
            for i in range(5)
        }

    def _retriever(self):
        class _Route:
            def __init__(self, ids):
                self.ids = ids

        class _Result:
            def __init__(self):
                self.routes = {"dense_zh": _Route([f"par_{i}" for i in range(5)])}
                self.degraded = None
                self.reranked = False

        class _Retriever:
            # query_en 是 N47 修好后多出来的关键字参数：假实现照收，重点仍是窗口正文
            def retrieve(self, query, k, query_en=None):
                return _Result()

        return _Retriever()

    def test_window_comes_from_content_key(self):
        """只给 `content` 键（评测脚本的 schema）时，窗口不许是空串。"""
        out = self.retrieve("问题", retriever=self._retriever(), parents=self._parents("content"),
                            counter=self.counter, reranker=self.reranker, pool=5, rerank=True)
        windows = [text for _query, text in self.recorder.pairs]
        self.assertTrue(windows, "encoder 没被调用 = 重排根本没跑")
        self.assertTrue(all(windows), f"窗口有空串：{windows}")
        self.assertEqual(out.hits[0].chunk_id, "par_3", "重排没有把 TARGET 那一段提到第一")

    def test_window_comes_from_text_key(self):
        """给 `text` 键（生产链路 schema）时同样成立。"""
        out = self.retrieve("问题", retriever=self._retriever(), parents=self._parents("text"),
                            counter=self.counter, reranker=self.reranker, pool=5, rerank=True)
        self.assertTrue(all(text for _q, text in self.recorder.pairs))
        self.assertEqual(out.hits[0].chunk_id, "par_3")

    def test_rerank_off_keeps_fused_order(self):
        out = self.retrieve("问题", retriever=self._retriever(), parents=self._parents("content"),
                            counter=self.counter, reranker=self.reranker, pool=5, rerank=False)
        self.assertEqual([h.chunk_id for h in out.hits], [f"par_{i}" for i in range(5)])
        self.assertEqual(self.recorder.pairs, [], "关掉重排时不该调用 encoder")


class EvalLoadParentsSchemaTest(unittest.TestCase):
    """`eval_bench.load_parents` 必须提供 `text`，否则重排静默失效。"""

    def test_returns_text_alias(self):
        import sqlite3
        import tempfile

        from scripts.eval_bench import load_parents

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "documents.sqlite3"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE parents (parent_id TEXT, document_id TEXT, content TEXT,"
                         " title TEXT, section_path TEXT, language TEXT)")
            conn.execute("INSERT INTO parents VALUES ('par_1','doc_1','正文','标题','[]','zh-cn')")
            conn.commit()
            conn.close()

            parents = load_parents(path)
            self.assertEqual(parents["par_1"]["text"], "正文")
            self.assertEqual(parents["par_1"]["content"], "正文")
            self.assertEqual(parents["par_1"]["language"], "zh-cn")


if __name__ == "__main__":
    unittest.main()
