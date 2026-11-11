"""C0f：`--window tiles`（让交叉编码器挑窗口）的接线与开关语义。

为什么要有这个开关，而不是直接换掉 `best_window()`：
`best_window()` 靠「与 query 的词面重叠」挑窗口，实测在一个 gold 块里
**随便哪一段都含答案句 = 1.0**，而词面挑中的只有 0.79 —— 错法是「标题命中了 query 词、
正文才是答案」。让 cross-encoder 在粗分段里自己挑，探针四格全赢
（`docs/evidence/c0f-window-strategy-k8s-v1.json`：整体 hit@10 0.8286→0.8571、MRR 0.5116→0.5433）。

但这个改动会多编 1.6× 的段（p50 父块 1.6 段），所以**默认必须还是 `overlap`**，
让 `eval_bench --compare` 用同一套题裁决；裁决之后再谈要不要翻默认。

五条边界：
1. 默认（不传 `window`）= 今日口径，一个字节都不变；
2. `tiles` 时送进编码器的是**段**（数量 > 候选数），不是一候选一段；
3. `tiles` 能改变排序结果（否则等于没接上）；
4. `rerank=False` 时任何口径都不许调编码器；
5. 未知口径要报错，不许静默退回默认（静默退回 = 实验白跑还看不出来）。
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.ask.pipeline import retrieval_window_mode, retrieve_with_meta
from src.rag.retrieval.rerank import RerankConfig


class _Route:
    def __init__(self, ids):
        self.ids = list(ids)
        self.enabled = True


class _Result:
    def __init__(self, ids):
        self.routes = {"dense_zh": _Route(ids)}
        self.degraded = None


class _FixedRetriever:
    """按固定顺序返回父块（RRF 顺序 = 输入顺序）。"""

    def __init__(self, ids):
        self.ids = list(ids)

    def retrieve(self, query, k, query_en=None):
        return _Result(self.ids)


class _SpanEncoder:
    """按「段里有没有 ANSWER 标记」打分：只有包含 ANSWER 的那一段得高分。

    `best_window()` 看的是词面重叠，所以「问题里的词出现在开头、答案在后半段」
    这种最常见的坏 case，它会挑开头；交叉编码器则会挑到后半段。
    """

    def __init__(self):
        self.calls = []

    def encode_pairs(self, query, texts):
        self.calls.append(list(texts))
        return np.asarray([9.0 if "ANSWER" in text else 1.0 for text in texts], dtype=np.float32)


class _FakeReranker:
    def __init__(self, max_length=40):
        self.encoder = _SpanEncoder()
        self.config = RerankConfig(enabled=True, max_length=max_length)
        self._stats = {"fallbacks": 0}

    def stats(self):
        return self._stats


class _FakeCounter:
    max_length = 0

    def count(self, text):
        return max(1, len(text) // 5)      # 5 字符 ≈ 1 token


def _parents() -> dict[str, dict]:
    """两个父块：开头一句含 query 词（像标题），中段是无关内容，答案只在**前者最后一段**。

    文本必须远超窗口预算（fake counter：5 字符 ≈ 1 token、预算 40 token = 200 字符），
    而且中段不许出现 query 词 —— 否则 `best_window()` 的长窗口会把答案一起圈进来，
    两个口径就分不出差别（第一版用例就踩了这个坑）。
    """
    lead = "topology manager 是管节点拓扑的。"
    filler = "这一段照例是与问题无关的字段说明与示例。" * 20
    return {
        "par_a": {"text": lead + filler + "真正的解释在这里 ANSWER。",
                  "content": lead + filler + "真正的解释在这里 ANSWER。",
                  "title": "t", "section_path": "s", "language": "zh-cn"},
        "par_b": {"text": lead + filler, "content": lead + filler,
                  "title": "t", "section_path": "s", "language": "zh-cn"},
    }


class DefaultModeTest(unittest.TestCase):
    """默认口径不能变：所有既有基线都建立在它上面。"""

    def test_default_is_overlap_and_encodes_one_text_per_candidate(self):
        self.assertEqual(retrieval_window_mode({}), "overlap")
        reranker = _FakeReranker()
        outcome = retrieve_with_meta("topology manager 是做什么的？",
                                     retriever=_FixedRetriever(["par_b", "par_a"]),
                                     parents=_parents(), counter=_FakeCounter(),
                                     reranker=reranker, pool=2, k=2)
        self.assertEqual(len(outcome.hits), 2)
        self.assertEqual(len(reranker.encoder.calls[0]), 2,
                         "默认口径是「一候选一段」，送进编码器的条数应等于候选数")
        self.assertEqual(outcome.hits[0].parent_id, "par_b",
                         "默认口径下两个候选都拿不到答案段，顺序保持输入的 RRF 序")


class TilesModeTest(unittest.TestCase):
    """`window="tiles"`：让交叉编码器在段里挑，能把「答案在后半段」的候选提到前面。"""

    def test_tiles_expands_candidates_into_segments(self):
        reranker = _FakeReranker()
        retrieve_with_meta("topology manager 是做什么的？",
                           retriever=_FixedRetriever(["par_b", "par_a"]),
                           parents=_parents(), counter=_FakeCounter(), reranker=reranker,
                           pool=2, k=2, window="tiles")
        sent = reranker.encoder.calls[0]
        self.assertGreater(len(sent), 2, "tiles 口径下送进编码器的是段，数量应多于候选数")
        self.assertTrue(any("ANSWER" in text for text in sent), "答案所在的段必须被送去打分")

    def test_tiles_promotes_the_candidate_whose_later_segment_answers(self):
        reranker = _FakeReranker()
        outcome = retrieve_with_meta("topology manager 是做什么的？",
                                     retriever=_FixedRetriever(["par_b", "par_a"]),
                                     parents=_parents(), counter=_FakeCounter(),
                                     reranker=reranker, pool=2, k=2, window="tiles")
        self.assertEqual(outcome.hits[0].parent_id, "par_a",
                         "答案段属于 par_a，tiles 口径必须把它排到第一")
        self.assertIn("ANSWER", outcome.hits[0].fields["window"],
                      "赢的那一段要留在 fields['window'] 里（诊断与引用都要用）")

    def test_overlap_mode_keeps_the_old_order(self):
        reranker = _FakeReranker()
        outcome = retrieve_with_meta("topology manager 是做什么的？",
                                     retriever=_FixedRetriever(["par_b", "par_a"]),
                                     parents=_parents(), counter=_FakeCounter(),
                                     reranker=reranker, pool=2, k=2, window="overlap")
        self.assertEqual(outcome.hits[0].parent_id, "par_b",
                         "同一个题、同一批候选：overlap 口径挑到的是开头段，所以顺序不变")


class GuardsTest(unittest.TestCase):
    def test_no_encoder_call_when_rerank_disabled(self):
        reranker = _FakeReranker()
        retrieve_with_meta("问题", retriever=_FixedRetriever(["par_b", "par_a"]),
                           parents=_parents(), counter=_FakeCounter(), reranker=reranker,
                           pool=2, k=2, rerank=False, window="tiles")
        self.assertEqual(reranker.encoder.calls, [], "关掉重排就不该有编码调用")

    def test_unknown_window_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            retrieve_with_meta("问题", retriever=_FixedRetriever(["par_b"]),
                               parents=_parents(), counter=_FakeCounter(),
                               reranker=_FakeReranker(), pool=1, k=1, window="tilez")

    def test_env_variable_selects_mode(self):
        self.assertEqual(retrieval_window_mode({"RAG_WINDOW_MODE": "tiles"}), "tiles")
        self.assertEqual(retrieval_window_mode({"RAG_WINDOW_MODE": "Tiles"}), "tiles",
                         "环境变量大小写不该让开关静默失效")
        with self.assertRaises(ValueError):
            retrieval_window_mode({"RAG_WINDOW_MODE": "nope"})


class EvalWindowFlagTest(unittest.TestCase):
    """`eval_bench --window` 必须能把口径传下去，并进指纹（否则 A/B 会被判成可比）。"""

    def test_default_stays_on_overlap(self):
        from scripts.eval_bench import build_parser

        self.assertEqual(build_parser().parse_args([]).window, "overlap")

    def test_window_is_a_comparability_key(self):
        from scripts.eval_bench import COMPARABILITY_KEYS

        self.assertIn("window", COMPARABILITY_KEYS,
                      "换了挑窗口口径就是换了检索，不许被当成「可比」")

    def test_evaluate_forwards_window(self):
        import inspect

        from scripts.eval_bench import evaluate

        self.assertIn("window", inspect.signature(evaluate).parameters)
        source = inspect.getsource(evaluate)
        self.assertIn("window=window", source, "参数必须真的传到 retrieve_with_meta")
