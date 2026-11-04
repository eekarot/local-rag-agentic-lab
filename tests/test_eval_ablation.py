"""D28：消融矩阵 A–E。

A  中文 Query + Dense            → `dense_zh` 单路
B  英文 Query + Dense            → `dense_en` 单路
C  中文 Dense + 英文 Dense       → 两路 Top-k 的**并集候选池**（考察"多路能不能捞回来"）
D  C + 英文 BM25                 → 三路并集候选池
E  D + RRF                       → 三路 RRF 融合，单列表 k 条

C/D 用并集而不是融合，是因为 `RRF` 是 E 的自变量——C/D 若也融合，E 就没有对照了。
"""

import unittest


class _StubRetriever:
    """按固定脚本返回结果的假检索器：gold=gp1，各路的命中情况由脚本决定。"""

    def __init__(self, mapping, k_gold="gp1"):
        self.mapping = mapping      # {qid: {"dense_zh": [...], "dense_en": [...], "lexical_en": [...]}}
        self.k_gold = k_gold
        self.calls = []

    def retrieve(self, query_zh, *, query_en=None, k=10, disable=(), fuse=True):
        from src.rag.retrieval.orchestrator import RetrievalResult
        from src.rag.retrieval.fusion import rrf_fuse_routes
        from src.rag.retrieval.routes import RouteHit, RouteResult

        self.calls.append({"query_zh": query_zh, "query_en": query_en,
                           "k": k, "disable": tuple(disable)})
        script = self.mapping.get(query_zh, {})
        routes = {}
        for name in ("dense_zh", "dense_en", "lexical_en"):
            ids = list(script.get(name, []))
            routes[name] = RouteResult(
                route=name,
                hits=[RouteHit(parent_id=i, score=1.0 / (60 + r), rank=r + 1, route=name)
                      for r, i in enumerate(ids)],
                latency_ms=1.0, enabled=True, query_used=query_zh, candidates=len(ids),
            )
        fused = rrf_fuse_routes({n: r.ids for n, r in routes.items()}, rrf_k=60, top_k=k)
        return RetrievalResult(query_zh=query_zh, query_en=query_en, k=k,
                               routes=routes, fused=fused, debug={"k": k, "rrf_k": 60})


def _questions(*rows):
    return [
        {"qid": qid, "question": qid, "query_en": qid + "-en", "type": "fact",
         "gold_points": {"parent_ids": [gold]}}
        for qid, gold in rows
    ]


class MatrixDefinitionTest(unittest.TestCase):
    """A–E 五格的定义必须能逐格对上预期，不能靠实现里"顺手"。"

    A: 只有 dense_zh
    B: 只有 dense_en
    C: dense_zh ∪ dense_en（并集）→ 用 recall 判断候选池是否含 gold
    D: C ∪ lexical_en
    E: RRF(dense_zh, dense_en, lexical_en) → top-k 里是否含 gold
    """

    def _run(self, mapping, questions, k=4):
        from src.eval_ablation import run_matrix

        retriever = _StubRetriever(mapping)
        return run_matrix(questions, retriever, ks=(k,), cells="ABCDE")

    def test_a_only_uses_chinese_dense(self):
        # gold 只在 dense_en 里 → A 必须不中
        mapping = {"q1": {"dense_zh": ["pX"], "dense_en": ["gp1"], "lexical_en": []}}
        result = self._run(mapping, _questions(("q1", "gp1")))
        self.assertEqual(result["cells"]["A"]["recall"]["4"]["mean"], 0.0)
        self.assertEqual(result["cells"]["B"]["recall"]["4"]["mean"], 1.0)

    def test_c_is_union_of_two_dense_routes(self):
        # gold 在 dense_en，candidate 池 2k 里第 5 位：C 的 recall 用并集池算
        mapping = {"q1": {"dense_zh": ["a", "b", "c", "d"], "dense_en": ["gp1"], "lexical_en": []}}
        result = self._run(mapping, _questions(("q1", "gp1")), k=4)
        self.assertEqual(result["cells"]["C"]["recall"]["4"]["mean"], 1.0)
        self.assertEqual(result["cells"]["C"]["pool_size"]["4"]["mean"], 5.0)

    def test_d_adds_lexical_and_only_lexical_finds_gold(self):
        mapping = {"q1": {"dense_zh": ["a"], "dense_en": ["b"], "lexical_en": ["gp1"]}}
        result = self._run(mapping, _questions(("q1", "gp1")), k=4)
        self.assertEqual(result["cells"]["C"]["recall"]["4"]["mean"], 0.0)
        self.assertEqual(result["cells"]["D"]["recall"]["4"]["mean"], 1.0)

    def test_e_is_rrf_and_respects_k(self):
        # RRF 排序：a=2×(1/61)=0.0328 > b=2×(1/62)=0.0323 > gp1=1/61=0.0164
        # k=3 时 gp1 进榜；k=2 时进不去 → 用 k=3 验证「融合确实捞到了单路命中」
        mapping = {"q1": {"dense_zh": ["a", "b"], "dense_en": ["a", "b"],
                          "lexical_en": ["gp1"]}}
        result = self._run(mapping, _questions(("q1", "gp1")), k=3)
        self.assertEqual(result["cells"]["E"]["recall"]["3"]["mean"], 1.0)
        self.assertEqual(result["cells"]["E"]["list_size"]["3"]["mean"], 3.0)

    def test_cells_are_run_without_hiding_any_route(self):
        """消融必须用同一次检索结果算 A–E，否则「关了再开」会引入不一致。"""
        mapping = {"q1": {"dense_zh": ["gp1"], "dense_en": ["b"], "lexical_en": ["c"]}}
        from src.eval_ablation import run_matrix

        retriever = _StubRetriever(mapping)
        run_matrix(_questions(("q1", "gp1")), retriever, ks=(4,), cells="ABCDE")
        self.assertEqual(len(retriever.calls), 1, "每题只检索一次，五格共用")
        self.assertEqual(retriever.calls[0]["disable"], (), "不许靠 disable 实现 A/B/C/D")


class CellsParsingTest(unittest.TestCase):
    def test_d2_is_parsed_as_one_token(self):
        from src.eval_ablation import parse_cells

        self.assertEqual(parse_cells("ABCDE"), ["A", "B", "C", "D", "E"])
        self.assertEqual(parse_cells("ABCDED2"), ["A", "B", "C", "D", "E", "D2"])
        self.assertEqual(parse_cells("A, E2"), ["A", "E2"])

    def test_unknown_cell_rejected(self):
        from src.eval_ablation import parse_cells

        with self.assertRaises(ValueError):
            parse_cells("ABZ")


class EIsComputedFromRouteListsTest(unittest.TestCase):
    """E 必须由「三路原始列表 + RRF」算出，而不是抄 retriever 的 fused。

    理由：这样 E 与 E2（四路）能在同一次运行里精确配对；否则 `--include-zh-lexical`
    会把 E 也悄悄变成四路，两格的差异就混进了「融合路数」以外的因素。
    """

    def test_e_ignores_retriever_fused(self):
        from src.eval_ablation import run_matrix
        from src.rag.retrieval.orchestrator import RetrievalResult
        from src.rag.retrieval.routes import RouteHit, RouteResult

        class EmptyFusedRetriever:
            def retrieve(self, query_zh, *, query_en=None, k=10, disable=(), fuse=True):
                routes = {
                    name: RouteResult(
                        route=name,
                        hits=[RouteHit(parent_id=item, score=1.0, rank=i + 1, route=name)
                              for i, item in enumerate(ids)],
                        latency_ms=1.0,
                    )
                    for name, ids in {
                        "dense_zh": ["gp1", "a"],
                        "dense_en": ["b"],
                        "lexical_en": ["c"],
                    }.items()
                }
                return RetrievalResult(query_zh=query_zh, query_en=query_en, k=k,
                                       routes=routes, fused=[], debug={"k": k, "rrf_k": 60})

        result = run_matrix(
            _questions(("q1", "gp1")), EmptyFusedRetriever(), ks=(3,), cells="E"
        )
        self.assertEqual(result["cells"]["E"]["recall"]["3"]["mean"], 1.0,
                         "E 自己算 RRF，不该依赖 retriever.fused（空 fused 也要能出结果）")


class MetricsTest(unittest.TestCase):
    def test_ndcg_at_k_binary_gain(self):
        from src.eval_ablation import ndcg_at_k

        # gold 在第 1 位 → 1.0
        self.assertAlmostEqual(ndcg_at_k(["gold"], {"gold"}, 10), 1.0)
        # gold 在第 2 位 → 1/log2(3)
        self.assertAlmostEqual(ndcg_at_k(["x", "gold"], {"gold"}, 10), 1 / 1.5849625007211562)
        # 不在 top-k → 0
        self.assertEqual(ndcg_at_k(["x"], {"gold"}, 10), 0.0)
        # 两个 gold 都在前两位 → 1.0
        self.assertAlmostEqual(ndcg_at_k(["g1", "g2"], {"g1", "g2"}, 10), 1.0)
        # 空 gold → 0（无金标不奖励）
        self.assertEqual(ndcg_at_k(["x"], set(), 10), 0.0)

    def test_ndcg_penalizes_late_gold_relative_to_ideal(self):
        from src.eval_ablation import ndcg_at_k

        late = ndcg_at_k(["a", "b", "gold"], {"gold"}, 10)
        early = ndcg_at_k(["gold", "a", "b"], {"gold"}, 10)
        self.assertLess(late, early)

    def test_latency_percentiles(self):
        from src.eval_ablation import percentile

        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertAlmostEqual(percentile(values, 50), 30.0)
        self.assertAlmostEqual(percentile(values, 95), 48.0)
        self.assertEqual(percentile([], 50), 0.0)


class ReportTest(unittest.TestCase):
    def test_report_reports_translation_cost_separately(self):
        """翻译耗时是独立的常数项（D28 用人工 query_en，不调翻译），必须单列。"""
        from src.eval_ablation import render_report

        payload = {
            "dataset_sha256": "abc",
            "n_questions": 2,
            "ks": [4, 10],
            "cells": {
                name: {
                    "recall": {"4": {"n": 2, "mean": 0.5, "lo": 0.1, "hi": 0.9, "half_width": 0.4}},
                    "mrr": {"10": {"mean": 0.4}},
                    "ndcg": {"10": {"mean": 0.3}},
                    "latency_ms": {"p50": 12.0, "p95": 30.0},
                    "pool_size": {"4": {"mean": 5.0}},
                    "list_size": {"4": {"mean": 4.0}},
                }
                for name in "ABCDE"
            },
            "pairs": [{"a": "A", "b": "E", "k": 10, "p_value": 0.03, "a_only": 1, "b_only": 5}],
            "translation_ms_per_query": 3080.0,
            "translation_source": "D23 实测（冷路径）",
        }
        text = render_report(payload)
        self.assertIn("| 方案 |", text)
        self.assertIn("3080", text)
        self.assertIn("D23 实测", text)
        self.assertIn("0.03", text)

    def test_report_prints_n_and_half_width(self):
        from src.eval_ablation import render_report

        payload = {
            "dataset_sha256": "abc", "n_questions": 83, "ks": [4],
            "cells": {name: {
                "recall": {"4": {"n": 83, "mean": 0.5, "lo": 0.4, "hi": 0.6, "half_width": 0.1}},
                "mrr": {"4": {"mean": 0.4}}, "ndcg": {"4": {"mean": 0.3}},
                "latency_ms": {"p50": 1.0, "p95": 2.0},
                "pool_size": {"4": {"mean": 3.0}}, "list_size": {"4": {"mean": 3.0}},
            } for name in "ABCDE"},
            "pairs": [], "translation_ms_per_query": None, "translation_source": "n/a",
        }
        text = render_report(payload)
        self.assertIn("±10.0", text)
        self.assertIn("83", text)


if __name__ == "__main__":
    unittest.main()
