"""O3：HTTP 层（FastAPI）。**只用假件测协议与装配**，不连 Qdrant / Ollama / ONNX。

这一层的价值在于「把 `answer_query()` 变成一个别人能调的东西」，所以测试要盯三件事：

1. **契约**：请求字段、响应字段、状态码、`/openapi.json` 里能看到它们；
2. **注入**：`create_app(deps=...)` 能让测试塞假件，也能让生产换真件——
   如果 app 里写死 `from src.ask import build_components`，这条路就断了；
3. **错误语义**：空问题 422、检索崩了 500（带可读 detail，不甩栈），
   以及 `retrieve_only=true` 时**绝不调用模型**（省 53 s，也便于压检索）。
"""

import unittest

from fastapi.testclient import TestClient

from src.rag.api.app import ApiDeps, create_app


class FakeRetriever:
    """返回固定两路命中，用来验证响应里带回了分路计数。"""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query, *, query_en=None, k=20, disable=None, fuse=True):
        self.calls.append((query, k))

        class Route:
            def __init__(self, ids):
                self.ids = ids

        class Result:
            def __init__(self):
                self.routes = {
                    "dense_zh": Route(["par_a", "par_b"]),
                    "dense_en": Route(["par_a"]),
                    "lexical_en": Route([]),
                }

        return Result()


class FakeCounter:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


class FakeEncoder:
    def score(self, pairs):
        return [1.0 + 0.1 * i for i in range(len(pairs))]


class FakeReranker:
    def __init__(self, config=None, encoder=None, boom: bool = False):
        from src.rag.retrieval.rerank import RerankConfig

        self.config = config or RerankConfig(enabled=True)
        self.encoder = encoder or FakeEncoder()
        self.stats: dict = {}
        self.boom = boom
        self.calls = 0

    def score(self, pairs):
        self.calls += 1
        if self.boom:
            raise RuntimeError("ONNX 会话炸了")
        return [1.0 + 0.1 * i for i in range(len(pairs))]


PARENTS = {
    "par_a": {
        "id": "par_a", "document_id": "doc_1", "parent_index": 0,
        "text": "## 安装\n\n星轨笔记的安装包名为 XingguiNote-3.2.msi，安装前要先关闭旧版本。",
        "section_path": ["星轨笔记使用手册", "安装"],
        "source_uri": "md/alpha.md", "title": "星轨笔记使用手册",
    },
    "par_b": {
        "id": "par_b", "document_id": "doc_1", "parent_index": 1,
        "text": "## 快捷键\n\n插入时间戳的快捷键是 Ctrl+Alt+T。",
        "section_path": ["星轨笔记使用手册", "快捷键"],
        "source_uri": "md/alpha.md", "title": "星轨笔记使用手册",
    },
}


class ApiTestBase(unittest.TestCase):
    def make_client(self, *, ask=None, reranker=None, translate=None) -> TestClient:
        deps = ApiDeps(
            retriever=FakeRetriever(),
            parents=PARENTS,
            counter=FakeCounter(),
            reranker=reranker or FakeReranker(),
            ask=ask,
            translate=translate or (lambda question: ("zh", question)),
        )
        return TestClient(create_app(deps=deps))


class OpenApiTest(ApiTestBase):
    def test_openapi_lists_endpoints(self):
        client = self.make_client()
        spec = client.get("/openapi.json").json()
        self.assertIn("/api/v1/ask", spec["paths"])
        self.assertIn("/api/v1/retrieve", spec["paths"])
        self.assertIn("/health", spec["paths"])

    def test_docs_page_available(self):
        client = self.make_client()
        self.assertEqual(client.get("/docs").status_code, 200)


class AskEndpointTest(ApiTestBase):
    def test_ask_returns_answer_and_sources(self):
        client = self.make_client(ask=lambda messages: "插入时间戳是 Ctrl+Alt+T [证据 1]。")
        response = client.post("/api/v1/ask", json={"question": "快捷键是什么？"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("Ctrl+Alt+T", body["answer"])
        self.assertEqual(body["cited"], [1])
        self.assertEqual(body["out_of_range"], [])
        self.assertGreaterEqual(len(body["sources"]), 1)
        source = body["sources"][0]
        for key in ("index", "document_id", "section_path", "source_uri", "title"):
            self.assertIn(key, source)

    def test_ask_reports_route_hits_and_timings(self):
        client = self.make_client(ask=lambda messages: "答案 [证据 1]。")
        body = client.post("/api/v1/ask", json={"question": "快捷键是什么？"}).json()
        self.assertEqual(body["route_hits"]["dense_zh"], 2)
        self.assertEqual(body["route_hits"]["dense_en"], 1)
        self.assertEqual(body["route_hits"]["lexical_en"], 0)
        self.assertIn("retrieve_rerank_ms", body["timings"])
        self.assertIn("generate_ms", body["timings"])

    def test_ask_echoes_question_and_language(self):
        client = self.make_client(ask=lambda messages: "答案 [证据 1]。",
                                 translate=lambda question: ("zh", "what is the shortcut?"))
        body = client.post("/api/v1/ask", json={"question": "快捷键是什么？"}).json()
        self.assertEqual(body["question"], "快捷键是什么？")
        self.assertEqual(body["question_en"], "what is the shortcut?")
        self.assertEqual(body["language"], "zh")

    def test_retrieve_only_does_not_call_model(self):
        called = {"n": 0}

        def ask(messages):           # 一旦被调用就说明链路错了
            called["n"] += 1
            return "不该出现"

        client = self.make_client(ask=ask)
        body = client.post("/api/v1/ask",
                           json={"question": "快捷键是什么？", "retrieve_only": True}).json()
        self.assertEqual(called["n"], 0)
        self.assertFalse(body["llm_called"])
        self.assertEqual(body["answer"], "")
        self.assertGreaterEqual(len(body["blocks"]), 1)

    def test_rerank_false_skips_reranker(self):
        reranker = FakeReranker()
        client = self.make_client(ask=lambda messages: "答案 [证据 1]。", reranker=reranker)
        client.post("/api/v1/ask", json={"question": "快捷键是什么？", "rerank": False})
        self.assertEqual(reranker.calls, 0, "rerank=false 时不该进重排")

    def test_empty_question_is_422(self):
        client = self.make_client(ask=lambda messages: "x")
        self.assertEqual(client.post("/api/v1/ask", json={"question": ""}).status_code, 422)
        self.assertEqual(client.post("/api/v1/ask", json={}).status_code, 422)

    def test_out_of_range_k_is_422(self):
        client = self.make_client(ask=lambda messages: "x")
        self.assertEqual(client.post("/api/v1/ask",
                                     json={"question": "q", "k": 0}).status_code, 422)

    def test_internal_error_is_500_with_readable_detail(self):
        def boom(messages):
            raise RuntimeError("Ollama 挂了：connection refused")

        client = self.make_client(ask=boom)
        response = client.post("/api/v1/ask", json={"question": "快捷键是什么？"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Ollama", response.json()["detail"])


class RetrieveEndpointTest(ApiTestBase):
    def test_retrieve_returns_hits(self):
        client = self.make_client()
        body = client.get("/api/v1/retrieve", params={"q": "快捷键"}).json()
        self.assertEqual(body["query"], "快捷键")
        self.assertGreaterEqual(len(body["hits"]), 1)
        hit = body["hits"][0]
        for key in ("parent_id", "section_path", "score", "rank", "rrf_rank"):
            self.assertIn(key, hit)

    def test_rank_is_position_in_response_not_rrf_order(self):
        """`rank` 必须是 1..n 的位置：重排会打乱 RRF 顺序，客户端按下标画列表才不错位。"""
        client = self.make_client()
        body = client.get("/api/v1/retrieve", params={"q": "快捷键", "k": 2}).json()
        self.assertEqual([hit["rank"] for hit in body["hits"]], [1, 2])
        self.assertTrue(all(isinstance(hit["rrf_rank"], int) for hit in body["hits"]))

    def test_retrieve_debug_includes_route_counts(self):
        client = self.make_client()
        body = client.get("/api/v1/retrieve",
                          params={"q": "快捷键", "debug": "true"}).json()
        self.assertIn("route_hits", body)
        self.assertEqual(body["route_hits"]["dense_zh"], 2)

    def test_retrieve_without_q_is_422(self):
        client = self.make_client()
        self.assertEqual(client.get("/api/v1/retrieve").status_code, 422)

    def test_retrieve_never_calls_model(self):
        called = {"n": 0}
        client = self.make_client(ask=lambda messages: called.__setitem__("n", called["n"] + 1))
        client.get("/api/v1/retrieve", params={"q": "快捷键"})
        self.assertEqual(called["n"], 0)


class HealthEndpointTest(ApiTestBase):
    def test_health_reports_ok_and_counts(self):
        client = self.make_client()
        body = client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["parents"], len(PARENTS))
        self.assertIn("version", body)


if __name__ == "__main__":
    unittest.main()
