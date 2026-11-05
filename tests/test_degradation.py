"""检索降级（v0.1.0 新增）：没有 Ollama 时不能整个服务 500。

三层，缺一层就等于没做：

1. **单路容错**（编排层）：`dense_zh` 抛异常不许把 `lexical_en` 一起带走。
   现状是「一路挂全挂」——`src/rag/retrieval/` 里一个 `except` 都没有。
2. **熔断**（embed 能力层）：Ollama 没起时 `embed_texts` 要 **2.9 s** 才失败（requests 重试）。
   不熔断的话，每个请求都白烧 2.9 s——降级比不降级还慢。
3. **标记**（结果层）：`degraded = {reason, missing, fix, routes}` 一路传到
   CLI / `/retrieve` / `/ask` / SSE 的 `sources`。静默降级 = 用户以为「知识库里没有」。

两个必须守住的反面：
- **不能把普通 bug 吞成"降级"**：路由里抛 `TypeError` 也记成 embed 不可用，会让人去重装 Ollama；
- **不能永久降级**：用户 30 s 后起好了 Ollama，下一次请求必须自动恢复（半开探测）。
"""

import time
import unittest

from src.rag.retrieval.degrade import (
    Breaker,
    CircuitOpenError,
    DegradedInfo,
    guarded,
)
from src.rag.retrieval.orchestrator import MultiRouteRetriever
from src.rag.retrieval.routes import DenseRoute, LexicalRoute, RouteHit


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FlakyEmbedder:
    def __init__(self, error=None, dim=16):
        self.error = error
        self.dim = dim
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        if self.error:
            raise self.error
        return [[0.1] * self.dim for _ in texts]


class FakeSearcher:
    """按向量里的「信号」返回固定行；只需要能被 DenseRoute 调。"""

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def __call__(self, vector, k):
        self.calls += 1
        return self.rows[:k]


class FakeLexicalIndex:
    def __init__(self, rows=None, error=None):
        self.rows = rows if rows is not None else [("chk_1", "par_b", 3.0)]
        self.error = error
        self.calls = 0

    def search(self, text, k):
        self.calls += 1
        if self.error:
            raise self.error
        return self.rows[:k]


# ---------------------------------------------------------------- 熔断


class BreakerTest(unittest.TestCase):
    def test_starts_closed(self):
        self.assertFalse(Breaker(clock=FakeClock()).is_open())

    def test_opens_after_failure_and_closes_after_ttl(self):
        clock = FakeClock()
        breaker = Breaker(ttl_s=30, clock=clock)
        breaker.record_failure("无法调用 Ollama API")
        self.assertTrue(breaker.is_open())

        clock.advance(29)
        self.assertTrue(breaker.is_open(), "TTL 内必须保持打开")

        clock.advance(2)
        self.assertFalse(breaker.is_open(), "TTL 过了要允许半开重试（否则永久降级）")

    def test_success_closes_the_breaker(self):
        clock = FakeClock()
        breaker = Breaker(ttl_s=30, clock=clock)
        breaker.record_failure("boom")
        breaker.record_success()
        self.assertFalse(breaker.is_open())
        self.assertEqual(breaker.detail, "")

    def test_keeps_failure_detail_for_the_hint(self):
        breaker = Breaker(clock=FakeClock())
        breaker.record_failure("无法调用 Ollama API：http://localhost:11434/api/embed")
        self.assertIn("Ollama", breaker.detail)


class GuardedTest(unittest.TestCase):
    def test_passes_through_on_success(self):
        embedder = FlakyEmbedder()
        wrapped = guarded(embedder, Breaker(clock=FakeClock()))
        self.assertEqual(wrapped(["x"]), [[0.1] * 16])

    def test_short_circuits_while_open_without_calling(self):
        """熔断的全部意义：第二次请求不许再等 2.9 s。"""
        clock = FakeClock()
        embedder = FlakyEmbedder(error=RuntimeError("无法调用 Ollama API"))
        wrapped = guarded(embedder, Breaker(ttl_s=30, clock=clock))

        with self.assertRaises(RuntimeError):
            wrapped(["x"])
        self.assertEqual(embedder.calls, 1)

        with self.assertRaises(CircuitOpenError):
            wrapped(["x"])
        self.assertEqual(embedder.calls, 1, "熔断期间不许真的发请求")

    def test_recovers_after_ttl(self):
        clock = FakeClock()
        embedder = FlakyEmbedder(error=RuntimeError("boom"))
        breaker = Breaker(ttl_s=30, clock=clock)
        wrapped = guarded(embedder, breaker)

        with self.assertRaises(RuntimeError):
            wrapped(["x"])
        clock.advance(31)
        embedder.error = None                       # 用户把 Ollama 起好了
        self.assertEqual(wrapped(["x"]), [[0.1] * 16])
        self.assertFalse(breaker.is_open())


# ---------------------------------------------------------------- 单路容错


def make_retriever(*, embed=None, lexical=None, breaker=None):
    """两路 dense 共用**一个** embed 能力（真实装配就是这样：一个 bge-m3 服务）。"""
    embed = embed or FlakyEmbedder()
    breaker = breaker or Breaker(ttl_s=30, clock=FakeClock())
    searcher = FakeSearcher([("chk_1", "par_a", 0.9), ("chk_2", "par_b", 0.8)])
    lexical = lexical or FakeLexicalIndex()
    guarded_embed = guarded(embed, breaker, capability="embed_model")
    routes = [
        DenseRoute(name="dense_zh", embedder=guarded_embed, searcher=searcher, input_lang="zh"),
        DenseRoute(name="dense_en", embedder=guarded_embed, searcher=searcher, input_lang="en"),
        LexicalRoute(name="lexical_en", index=lexical, input_lang="en"),
    ]
    return MultiRouteRetriever(routes)


class RouteToleranceTest(unittest.TestCase):
    def test_one_dense_route_failing_does_not_kill_lexical(self):
        retriever = make_retriever(embed=FlakyEmbedder(error=RuntimeError("无法调用 Ollama API")))
        result = retriever.retrieve("快捷键是什么", query_en="shortcut", k=5)

        self.assertEqual(len(result.hits_of("lexical_en")), 1, "lexical 必须照常返回")
        self.assertEqual(result.hits_of("dense_zh"), [])
        self.assertTrue(result.routes["dense_zh"].error, "失败原因要留在这一路上")
        self.assertIsNotNone(result.degraded)
        self.assertEqual(result.degraded.reason, "embed_unavailable")
        self.assertIn("embed_model", result.degraded.missing)
        self.assertIn("ollama", result.degraded.fix.lower())

    def test_fused_still_uses_surviving_routes(self):
        retriever = make_retriever(embed=FlakyEmbedder(error=RuntimeError("x")))
        result = retriever.retrieve("q", query_en="q", k=5)
        self.assertEqual([hit.parent_id for hit in result.fused], ["par_b"])

    def test_generic_bug_is_not_reported_as_missing_model(self):
        """`TypeError` 记成 embed 不可用 → 用户会去重装 Ollama，白折腾。"""
        retriever = make_retriever()
        retriever._by_name["lexical_en"]._index = FakeLexicalIndex(error=TypeError("bad row"))
        result = retriever.retrieve("q", query_en="q", k=5)
        self.assertIsNotNone(result.degraded)
        self.assertEqual(result.degraded.reason, "route_failed")
        self.assertIn("lexical_en", result.degraded.routes)
        self.assertIn("TypeError", result.degraded.detail)

    def test_circuit_open_is_reported_as_missing_model(self):
        clock = FakeClock()
        breaker = Breaker(ttl_s=30, clock=clock)
        embed = FlakyEmbedder(error=RuntimeError("无法调用 Ollama API"))
        retriever = make_retriever(embed=embed, breaker=breaker)

        retriever.retrieve("q", query_en="q", k=5)
        result = retriever.retrieve("q", query_en="q", k=5)
        self.assertEqual(embed.calls, 1, "第二次请求必须被熔断挡住，不许再烧 2.9 s")
        self.assertEqual(result.degraded.reason, "embed_unavailable")
        self.assertIn("ollama", result.degraded.fix.lower())

    def test_all_routes_failing_still_returns_a_result(self):
        retriever = make_retriever(embed=FlakyEmbedder(error=RuntimeError("x")),
                                   lexical=FakeLexicalIndex(error=RuntimeError("bm25 down")))
        result = retriever.retrieve("q", query_en="q", k=5)
        self.assertEqual(result.fused, [])
        self.assertEqual(set(result.degraded.routes), {"dense_zh", "dense_en", "lexical_en"})

    def test_healthy_path_has_no_degraded_marker(self):
        result = make_retriever().retrieve("q", query_en="q", k=5)
        self.assertIsNone(result.degraded)

    def test_debug_carries_degraded_dict(self):
        retriever = make_retriever(embed=FlakyEmbedder(error=RuntimeError("x")))
        result = retriever.retrieve("q", query_en="q", k=5)
        self.assertEqual(result.debug["degraded"]["reason"], "embed_unavailable")

    def test_degraded_info_serializes(self):
        info = DegradedInfo(reason="embed_unavailable", missing=["embed_model"],
                            fix="ollama serve", detail="无法调用 Ollama API", routes=["dense_zh"])
        payload = info.as_dict()
        self.assertEqual(payload["reason"], "embed_unavailable")
        self.assertEqual(payload["routes"], ["dense_zh"])


# ---------------------------------------------------------------- 传到上层


class FakeParents(dict):
    def get(self, key, default=None):
        return {"par_a": {"id": "par_a", "text": "安装要关闭旧版本。", "section_path": ["安装"],
                          "document_id": "doc_1", "source_uri": "md/a.md", "title": "手册"},
                "par_b": {"id": "par_b", "text": "插入时间戳用 Ctrl+Alt+T。", "section_path": ["快捷键"],
                          "document_id": "doc_1", "source_uri": "md/a.md", "title": "手册"}}.get(key, default)


class FakeCounter:
    def count(self, text):
        return max(1, len(text) // 4)


class FakeReranker:
    def __init__(self):
        from src.rag.retrieval.rerank import RerankConfig

        class Encoder:
            def score(self, pairs):
                return [1.0] * len(pairs)

        self.config = RerankConfig(enabled=True)
        self.encoder = Encoder()
        self.stats = {}


class PipelineDegradedTest(unittest.TestCase):
    def _run(self, *, ask=None, rerank=False):
        from src.rag.ask.pipeline import answer_query

        retriever = make_retriever(embed=FlakyEmbedder(error=RuntimeError("无法调用 Ollama API")))
        return answer_query("快捷键是什么", retriever=retriever, parents=FakeParents(),
                            counter=FakeCounter(), reranker=FakeReranker(), ask=ask,
                            query_en="shortcut", k=3, pool=6, budget=2000, rerank=rerank)

    def test_answer_result_exposes_degraded(self):
        result = self._run()
        payload = result.as_dict()
        self.assertEqual(payload["degraded"]["reason"], "embed_unavailable")
        self.assertIn("fix", payload["degraded"])

    def test_stream_sources_event_carries_degraded(self):
        from src.rag.ask.pipeline import answer_query_stream

        retriever = make_retriever(embed=FlakyEmbedder(error=RuntimeError("无法调用 Ollama API")))
        events = list(answer_query_stream(
            "快捷键是什么", retriever=retriever, parents=FakeParents(), counter=FakeCounter(),
            reranker=FakeReranker(), ask=None, query_en="shortcut", k=3, pool=6, budget=2000,
            rerank=False,
        ))
        sources = events[0]["data"]
        self.assertEqual(sources["degraded"]["reason"], "embed_unavailable")
        self.assertEqual(events[-1]["data"]["degraded"]["reason"], "embed_unavailable")

    def test_cli_renders_a_degraded_warning(self):
        from src.rag.ask.pipeline import render_cli

        text = render_cli(self._run())
        self.assertIn("降级", text)
        self.assertIn("ollama", text.lower())


class HttpDegradedTest(unittest.TestCase):
    """降级信息必须出现在 HTTP 契约里：客户端据此提示「结果可能不全」。"""

    def _client(self):
        from fastapi.testclient import TestClient

        from src.rag.api.app import ApiDeps, create_app

        retriever = make_retriever(embed=FlakyEmbedder(error=RuntimeError("无法调用 Ollama API")))
        deps = ApiDeps(
            retriever=retriever, parents=FakeParents(), counter=FakeCounter(),
            reranker=FakeReranker(), ask=None,
        )
        return TestClient(create_app(deps=deps))

    def test_retrieve_returns_degraded_and_is_not_500(self):
        response = self._client().get("/api/v1/retrieve", params={"q": "快捷键", "k": 3, "rerank": "false"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["degraded"]["reason"], "embed_unavailable")
        self.assertIn("ollama", payload["degraded"]["fix"].lower())

    def test_healthy_retrieve_has_null_degraded(self):
        from fastapi.testclient import TestClient

        from src.rag.api.app import ApiDeps, create_app

        deps = ApiDeps(
            retriever=make_retriever(), parents=FakeParents(), counter=FakeCounter(),
            reranker=FakeReranker(), ask=None,
        )
        payload = TestClient(create_app(deps=deps)).get(
            "/api/v1/retrieve", params={"q": "shortcut", "k": 3, "rerank": "false"}).json()
        self.assertIsNone(payload["degraded"])

    def test_openapi_documents_the_degraded_field(self):
        schema = self._client().get("/openapi.json").json()
        self.assertIn("degraded", schema["components"]["schemas"]["RetrieveResponse"]["properties"])
        self.assertIn("degraded", schema["components"]["schemas"]["AskResponse"]["properties"])


if __name__ == "__main__":
    unittest.main()
