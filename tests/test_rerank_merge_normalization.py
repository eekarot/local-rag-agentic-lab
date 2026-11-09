"""C0e-2：`by_lang` 把两组不同 query 的分数直接混起来排序，量纲不可比。

C0e 的第一版做法是「给每种语言留名额」，实测不赚（14 题规模下逐题也有真有假）：
  * `question/quota3`  MRR 0.5648 → 0.5452（配额单独加就掉分）
  * `by_lang/quota0`   0.5387、`quota3` 0.5416 —— 配额能把 by_lang 抬回来一点，但仍不如基线

真正的问题在 `rerank_by_language()` 的最后一行：`sorted(key=(-rerank_score, ...))`。
中文组的分来自「中文 query × 中文窗口」，英文组来自「英文 query × 英文窗口」——
**cross-encoder 的原始分是「这段像不像在回答这句」，跨不同 query 完全不可比**。
一组的 logits 系统性偏高，就会把另一组整体压掉。

所以这里加一个**合并口径**：先在组内把分数换成可比的量（z 分数 / 组内百分位），再跨组合并。
  * `merge="raw"`      —— 现状（保留，用于回归对照）
  * `merge="zscore"`   —— 组内标准化，保留「差多少」的信息
  * `merge="rank"`     —— 组内百分位，只保留名次（对 logits 尺度完全不敏感）

四条边界：
1. 单组（只有一种语言）时三种口径必须给出**同一个顺序**；
2. 组内只有一个候选 → 归一化不能除以 0；
3. 归一化后仍要保留原始 `rerank_score` 供报告/诊断使用（别把证据抹掉）；
4. 归一化只影响顺序，不影响候选集合。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.ask.pipeline import RerankHit, normalize_group_scores


class _FakeEncoder:
    """直接返回预设分数：让「两组量纲不同」这件事在用例里显形。"""

    def __init__(self, table):
        self.table = table

    def encode_pairs(self, query, texts):
        import numpy as np

        scores = [self.table.get((query, text), 0.0) for text in texts]
        return np.asarray(scores, dtype=np.float32)


def _hits(pairs):
    out = []
    for rank, (pid, score) in enumerate(pairs, start=1):
        item = RerankHit(chunk_id=pid, parent_id=pid, text=pid, score=0.0, rank=rank)
        item.rerank_score = score
        out.append(item)
    return out


class _Counter:
    """假 token 计数器：不装真实 tokenizer，按空白切词即可满足窗口切分的需求。"""

    def count(self, text: str) -> int:
        return max(1, len(str(text).split()))


class NormalizeTest(unittest.TestCase):
    def test_zscore_makes_two_groups_comparable(self):
        # 英文组整体低 10 分，但它组内第一名其实是「很确定」的
        zh = _hits([("z1", 9.0), ("z2", 8.0), ("z3", 0.0)])
        en = _hits([("e1", -1.0), ("e2", -2.0), ("e3", -10.0)])
        normalize_group_scores(zh, merge="zscore")
        normalize_group_scores(en, merge="zscore")
        self.assertGreater(zh[0].fields["norm_score"], 0)
        self.assertGreater(en[0].fields["norm_score"], 0)
        # 原始分数必须还在
        self.assertEqual(zh[0].rerank_score, 9.0)
        self.assertEqual(en[0].rerank_score, -1.0)

    def test_rank_merge_ignores_absolute_scale(self):
        zh = _hits([("z1", 9.0), ("z2", 8.0)])
        en = _hits([("e1", -1.0), ("e2", -2.0)])
        normalize_group_scores(zh, merge="rank")
        normalize_group_scores(en, merge="rank")
        self.assertEqual(zh[0].fields["norm_score"], en[0].fields["norm_score"])
        self.assertGreater(zh[0].fields["norm_score"], en[1].fields["norm_score"])

    def test_raw_merge_uses_original_scores(self):
        zh = _hits([("z1", 9.0), ("z2", 8.0)])
        normalize_group_scores(zh, merge="raw")
        self.assertEqual(zh[0].fields["norm_score"], 9.0)

    def test_single_candidate_group_does_not_divide_by_zero(self):
        zh = _hits([("z1", 3.0)])
        normalize_group_scores(zh, merge="zscore")
        self.assertEqual(zh[0].fields["norm_score"], 0.0)
        normalize_group_scores(zh, merge="rank")
        self.assertEqual(zh[0].fields["norm_score"], 1.0)

    def test_unknown_merge_raises(self):
        with self.assertRaises(ValueError):
            normalize_group_scores(_hits([("z1", 1.0)]), merge="nope")

    def test_single_group_order_unchanged_for_all_merges(self):
        for merge in ("raw", "zscore", "rank"):
            group = _hits([("a", 5.0), ("b", 3.0), ("c", 1.0)])
            normalize_group_scores(group, merge=merge)
            order = [h.parent_id for h in sorted(group, key=lambda h: -h.fields["norm_score"])]
            self.assertEqual(order, ["a", "b", "c"], merge)


class ByLanguageMergeTest(unittest.TestCase):
    """`rerank_by_language(merge=...)` 必须把口径一路传下去，且不改变候选集合。"""

    def _run(self, merge):
        from src.rag.ask.pipeline import rerank_by_language

        parents = {"z1": {"text": "中文 TARGET", "content": "中文 TARGET", "language": "zh-cn"},
                   "z2": {"text": "中文 干扰", "content": "中文 干扰", "language": "zh-cn"},
                   "e1": {"text": "english TARGET", "content": "english TARGET", "language": "en"},
                   "e2": {"text": "english noise", "content": "english noise", "language": "en"}}
        encoder = _FakeEncoder({("中文问句", "中文 TARGET"): 9.0, ("中文问句", "中文 干扰"): 8.0,
                                ("what", "english TARGET"): -1.0,
                                ("what", "english noise"): -2.0})

        class _Reranker:
            from src.rag.retrieval.rerank import RerankConfig

            config = RerankConfig(max_length=256)

        _Reranker.encoder = encoder
        _Reranker.stats = {}
        hits = _hits([("z1", 0.0), ("e1", 0.0), ("z2", 0.0), ("e2", 0.0)])
        for hit in hits:                       # 窗口是从 hit.text 切出来的，必须给真正文
            hit.text = parents[hit.parent_id]["text"]
        ranked = rerank_by_language(hits, parents, query_zh="中文问句", query_en="what",
                                    reranker=_Reranker(), counter=_Counter(), merge=merge)
        return [h.parent_id for h in ranked]

    def test_raw_keeps_english_group_suppressed(self):
        order = self._run("raw")
        self.assertEqual(order[:2], ["z1", "z2"])

    def test_rank_merge_interleaves_groups(self):
        order = self._run("rank")
        self.assertEqual(order[:2], ["z1", "e1"])

    def test_candidate_set_never_changes(self):
        for merge in ("raw", "zscore", "rank"):
            self.assertEqual(sorted(self._run(merge)), ["e1", "e2", "z1", "z2"])


if __name__ == "__main__":
    unittest.main()
