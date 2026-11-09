"""C0e：按语言配额的 top-k。

为什么要做（实测，见 `docs/evidence/retrieval-stage-ceilings-k8s-v1.json`）：

  * 中→英 的漏检里 **18.2% 是「融合 top-10 有、重排打下去」**，同语言格只有 1.9%；
    英→中 是 22.2%。跨语言的重排损失是同语言的十倍。
  * C0d 证明「按块语言选重排 query」能把 中→英 的 MRR 从 0.323 提到 0.471，
    但同时把 中→中 从 0.636 打到 0.508 —— **抬高一种语言就挤掉另一种**（N50）。
  * 所以缺的不是分数尺度，是**名额**：池子里中英混排，top-k 通吃的话，
    一种语言的好候选会把另一种语言的好候选全挤出去。

配额规则（只决定「谁进名单」，不改变名单内的相对顺序）：
  每种语言先按它自己的分数序占住 `min_per_lang` 个名额，剩下的名额再按全局分数序填满。
  这样 MRR / hit@k 的口径一个字都不用变 —— 只是名单换了几个人。

七条边界（都是真会踩的）：
1. `min_per_lang=0` / 单语言池 → 必须与 `hits[:k]` 逐位相同（默认路径零风险）；
2. 配额之和不许超过 k（否则 f(k=10, quota=6×2) 会返回 12 条）；
3. 某语言候选不够配额 → 有多少占多少，剩下的名额照样按全局序填满；
4. 池子比 k 还小 → 全返回，顺序不动；
5. 不许返回重复项、不许改输入列表；
6. 候选块语言读的是 `parents`（列优先 + 正文兜底，N49）——`language` 全 NULL 的库也要能判；
7. 名单内顺序必须仍是原分数序（否则 hit@k 与 rank 分布对不上）。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.ask.pipeline import RerankHit, select_with_language_quota


def _hit(pid: str, rank: int, score: float = 0.0) -> RerankHit:
    item = RerankHit(chunk_id=pid, parent_id=pid, text=pid, score=score, rank=rank)
    item.rerank_score = -float(rank)          # 名次即分数序，方便断言
    return item


def _parents(langs: dict[str, str]) -> dict[str, dict]:
    return {pid: {"text": f"正文 {pid}", "content": f"正文 {pid}", "language": lang}
            for pid, lang in langs.items()}


class QuotaNoOpTest(unittest.TestCase):
    """默认路径（quota=0）必须与现状**逐位相同**，否则所有基线都要重跑。"""

    def test_zero_quota_is_plain_slice(self):
        hits = [_hit(f"p{n}", n) for n in range(1, 6)]
        parents = _parents({f"p{n}": "en" for n in range(1, 6)})
        got = select_with_language_quota(hits, parents, k=3, min_per_lang=0)
        self.assertEqual([h.parent_id for h in got], ["p1", "p2", "p3"])

    def test_single_language_pool_is_plain_slice(self):
        hits = [_hit(f"p{n}", n) for n in range(1, 6)]
        parents = _parents({f"p{n}": "zh-cn" for n in range(1, 6)})
        got = select_with_language_quota(hits, parents, k=3, min_per_lang=2)
        self.assertEqual([h.parent_id for h in got], ["p1", "p2", "p3"])

    def test_pool_smaller_than_k_returns_everything(self):
        hits = [_hit(f"p{n}", n) for n in range(1, 4)]
        parents = _parents({f"p{n}": "zh-cn" for n in range(1, 4)})
        got = select_with_language_quota(hits, parents, k=10, min_per_lang=2)
        self.assertEqual([h.parent_id for h in got], ["p1", "p2", "p3"])

    def test_both_languages_already_inside_k_changes_nothing(self):
        hits = [_hit("z1", 1), _hit("e1", 2), _hit("z2", 3), _hit("e2", 4), _hit("z3", 5)]
        parents = _parents({"z1": "zh-cn", "z2": "zh-cn", "z3": "zh-cn",
                            "e1": "en", "e2": "en"})
        got = select_with_language_quota(hits, parents, k=4, min_per_lang=2)
        self.assertEqual([h.parent_id for h in got], ["z1", "e1", "z2", "e2"])


class QuotaReserveTest(unittest.TestCase):
    """配额真正起作用的情形：一种语言被整段挤出 top-k。"""

    def setUp(self):
        # 中文占满前 4 名，英文从第 5、6、7 名才开始
        self.hits = [_hit("z1", 1), _hit("z2", 2), _hit("z3", 3), _hit("z4", 4),
                     _hit("e1", 5), _hit("e2", 6), _hit("e3", 7), _hit("z5", 8)]
        self.parents = _parents({"z1": "zh-cn", "z2": "zh-cn", "z3": "zh-cn", "z4": "zh-cn",
                                 "z5": "zh-cn", "e1": "en", "e2": "en", "e3": "en"})

    def test_minority_language_gets_reserved_slots(self):
        got = select_with_language_quota(self.hits, self.parents, k=4, min_per_lang=2)
        ids = [h.parent_id for h in got]
        self.assertEqual(ids, ["z1", "z2", "e1", "e2"])

    def test_order_stays_original_score_order(self):
        got = select_with_language_quota(self.hits, self.parents, k=5, min_per_lang=2)
        ranks = [h.rank for h in got]
        self.assertEqual(ranks, sorted(ranks))

    def test_quota_bigger_than_group_takes_what_exists(self):
        parents = _parents({"z1": "zh-cn", "z2": "zh-cn", "z3": "zh-cn", "z4": "zh-cn",
                            "z5": "zh-cn", "e1": "en", "e2": "en", "e3": "en"})
        hits = [_hit("z1", 1), _hit("z2", 2), _hit("z3", 3), _hit("z4", 4),
                _hit("e1", 5), _hit("z5", 6)]
        got = select_with_language_quota(hits, parents, k=5, min_per_lang=3)
        # 英文只有 1 条 → 占 1 个；剩下 4 个按全局序填（z1 z2 z3 z4）
        self.assertEqual([h.parent_id for h in got], ["z1", "z2", "z3", "z4", "e1"])

    def test_quota_sum_over_k_is_capped(self):
        got = select_with_language_quota(self.hits, self.parents, k=3, min_per_lang=4)
        self.assertEqual(len(got), 3)
        self.assertEqual([h.parent_id for h in got], ["z1", "z2", "z3"])

    def test_no_duplicates_and_input_untouched(self):
        before = [h.parent_id for h in self.hits]
        got = select_with_language_quota(self.hits, self.parents, k=6, min_per_lang=2)
        ids = [h.parent_id for h in got]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual([h.parent_id for h in self.hits], before)

    def test_deterministic(self):
        first = [h.parent_id for h in select_with_language_quota(self.hits, self.parents,
                                                                k=6, min_per_lang=2)]
        second = [h.parent_id for h in select_with_language_quota(self.hits, self.parents,
                                                                 k=6, min_per_lang=2)]
        self.assertEqual(first, second)

    def test_unknown_language_parent_falls_back_to_text(self):
        """`language` 列全 NULL 的库（N49）也要能判语言：靠正文。"""
        parents = {"z1": {"text": "这是中文正文", "content": "这是中文正文", "language": None},
                   "e1": {"text": "this is english body", "content": "this is english body",
                          "language": None},
                   "z2": {"text": "还是中文", "content": "还是中文", "language": None}}
        hits = [_hit("z1", 1), _hit("z2", 2), _hit("e1", 3)]
        got = select_with_language_quota(hits, parents, k=2, min_per_lang=1)
        self.assertEqual([h.parent_id for h in got], ["z1", "e1"])


class QuotaOrderTest(unittest.TestCase):
    """重排之后 `rank`（RRF 名次）已经不是输出顺序了 —— 配额不能按它重排（N52）。"""

    def test_output_follows_input_order_not_rrf_rank(self):
        # 输入是「重排后的顺序」：e1 排第一，但它的 RRF 名次是 9
        hits = [_hit("e1", 9), _hit("z1", 1), _hit("z2", 2), _hit("e2", 10)]
        parents = _parents({"z1": "zh-cn", "z2": "zh-cn", "e1": "en", "e2": "en"})
        got = select_with_language_quota(hits, parents, k=4, min_per_lang=1)
        self.assertEqual([h.parent_id for h in got], ["e1", "z1", "z2", "e2"])


class MaxOtherLangTest(unittest.TestCase):
    """C0h：把「最少名额」翻成「对方语言最多占几个」（预登记，先写断言再实现）。

    动机（实测）：by_lang + zscore 让 中→英 的 MRR 从 0.252 抬到 0.409，
    但 中→中 从 0.633 掉到 0.462 —— 中文题的池子里英文块被英文 query 打了高分，
    把中文 gold 挤走。最少名额（quota）在这份数据上从不触发（逐题 0/144），
    因为两种语言本来都有位置；真正要限的是**对方语言占几个**。
    """

    def test_cap_limits_other_language(self):
        # 中文提问：英文块占了前 5 名 → 上限 2 时必须只让 2 个英文进 top-5
        hits = [_hit("e1", 1), _hit("e2", 2), _hit("e3", 3), _hit("e4", 4), _hit("e5", 5),
                _hit("z1", 6), _hit("z2", 7), _hit("z3", 8)]
        parents = _parents({**{f"e{n}": "en" for n in range(1, 6)},
                            **{f"z{n}": "zh-cn" for n in range(1, 4)}})
        got = select_with_language_quota(hits, parents, k=5, min_per_lang=0,
                                         max_other_lang=2, question="中文提问")
        ids = [h.parent_id for h in got]
        self.assertEqual(sum(1 for pid in ids if pid.startswith("e")), 2)
        self.assertEqual(ids[:2], ["e1", "e2"])       # 顺序仍是输入顺序
        self.assertEqual(len(ids), 5)

    def test_cap_zero_is_off(self):
        hits = [_hit("e1", 1), _hit("z1", 2)]
        parents = _parents({"e1": "en", "z1": "zh-cn"})
        got = select_with_language_quota(hits, parents, k=2, min_per_lang=0, question="中文提问")
        self.assertEqual([h.parent_id for h in got], ["e1", "z1"])

    def test_cap_larger_than_k_is_off(self):
        hits = [_hit("e1", 1), _hit("e2", 2), _hit("e3", 3)]
        parents = _parents({"e1": "en", "e2": "en", "e3": "en"})
        got = select_with_language_quota(hits, parents, k=3, min_per_lang=0,
                                         max_other_lang=9, question="中文提问")
        self.assertEqual([h.parent_id for h in got], ["e1", "e2", "e3"])

    def test_question_language_detected_from_text(self):
        """英文提问时「对方语言」= 中文：上限只压中文块。"""
        hits = [_hit("z1", 1), _hit("z2", 2), _hit("z3", 3), _hit("e1", 4), _hit("e2", 5),
                _hit("e3", 6)]
        parents = _parents({"z1": "zh-cn", "z2": "zh-cn", "z3": "zh-cn",
                            "e1": "en", "e2": "en", "e3": "en"})
        got = select_with_language_quota(hits, parents, k=4, min_per_lang=0,
                                         max_other_lang=1, question="what is a pod")
        ids = [h.parent_id for h in got]
        self.assertEqual(sum(1 for pid in ids if pid.startswith("z")), 1)
        self.assertEqual(len(ids), 4)

    def test_cap_and_min_can_combine(self):
        hits = [_hit("e1", 1), _hit("e2", 2), _hit("e3", 3), _hit("e4", 4),
                _hit("z1", 5), _hit("z2", 6), _hit("z3", 7)]
        parents = _parents({**{f"e{n}": "en" for n in range(1, 5)},
                            **{f"z{n}": "zh-cn" for n in (1, 2, 3)}})
        got = select_with_language_quota(hits, parents, k=4, min_per_lang=2,
                                         max_other_lang=1, question="中文提问")
        ids = [h.parent_id for h in got]
        self.assertEqual(len(ids), 4)
        self.assertEqual(sum(1 for pid in ids if pid.startswith("e")), 1)
        self.assertEqual(sum(1 for pid in ids if pid.startswith("z")), 3)   # 上限只压对方语言


class RetrieveWiringTest(unittest.TestCase):
    """`retrieve_with_meta` 拿到 `k` + `lang_quota` 时必须自己做完名额分配。"""

    def _retriever(self):
        class _Route:
            def __init__(self, ids):
                self.ids = list(ids)
                self.enabled = True

        class _Result:
            def __init__(self):
                self.routes = {"dense_zh": _Route(["z1", "z2", "z3", "z4", "e1", "e2"])}
                self.degraded = None

        class _Retriever:
            def retrieve(self, question, k=None, query_en=None):
                return _Result()

        return _Retriever()

    def _run(self, **kwargs):
        from src.rag.ask.pipeline import retrieve_with_meta
        parents = _parents({"z1": "zh-cn", "z2": "zh-cn", "z3": "zh-cn", "z4": "zh-cn",
                            "e1": "en", "e2": "en"})
        return retrieve_with_meta("问句", retriever=self._retriever(), parents=parents,
                                  counter=None, reranker=None, rerank=False, **kwargs)

    def test_quota_applied_when_k_given(self):
        outcome = self._run(k=3, lang_quota=1)
        self.assertEqual([h.parent_id for h in outcome.hits], ["z1", "z2", "e1"])

    def test_k_without_quota_is_plain_head(self):
        outcome = self._run(k=3)
        self.assertEqual([h.parent_id for h in outcome.hits], ["z1", "z2", "z3"])

    def test_pool_semantics_preserved_without_k(self):
        """不传 k 时返回的是整个池子（老行为），调用方自己切。"""
        outcome = self._run()
        self.assertEqual(len(outcome.hits), 6)


class GateFingerprintTest(unittest.TestCase):
    """口径指纹必须认 `lang_quota`：否则换了配额还会说「可比」。"""

    def test_lang_quota_in_comparability(self):
        from scripts.eval_bench import _comparability_reasons

        base = {"k": 10, "pool": 40, "rerank": True, "query_en_mode": "translate",
                "rerank_query_mode": "question", "lang_quota": 0,
                "questions_sha256": "abc", "rows": [{"qid": "a"}]}
        same = dict(base, rows=[{"qid": "a"}])
        self.assertEqual(_comparability_reasons(base, same), [])

        changed = dict(base, lang_quota=3, rows=[{"qid": "a"}])
        reasons = _comparability_reasons(base, changed)
        self.assertTrue(any("lang_quota" in reason for reason in reasons), reasons)


if __name__ == "__main__":
    unittest.main()
