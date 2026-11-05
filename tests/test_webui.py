"""`GET /` 的聊天页：**离线可用的零依赖页面**（内网/断网环境也能开）。

只测三件真会坏的事：
1. 页面能拿到、是 `text/html`；
2. 它指向的流式地址**确实存在**（否则打开就是一直转圈）；
3. **不许引任何外部资源**（CDN/字体/图标）——开源项目在内网部署时，一个外链就是白屏。
"""

import unittest

from fastapi.testclient import TestClient

from src.rag.api.app import ApiDeps, create_app


PARENTS = {"par_a": {"id": "par_a", "text": "插入时间戳用 Ctrl+Alt+T。", "section_path": ["快捷键"],
                     "document_id": "doc_1", "source_uri": "md/a.md", "title": "手册"}}


class _Retriever:
    def retrieve(self, query, *, k=20, query_en=None, disable=None, fuse=True):
        class Route:
            ids = ["par_a"]

        class Result:
            routes = {"dense_zh": Route(), "dense_en": Route(), "lexical_en": Route()}
            degraded = None

        return Result()


class _Counter:
    def count(self, text):
        return max(1, len(text) // 4)


class _Reranker:
    def __init__(self):
        from src.rag.retrieval.rerank import RerankConfig

        class Enc:
            def score(self, pairs):
                return [1.0] * len(pairs)

        self.config = RerankConfig(enabled=True)
        self.encoder = Enc()
        self.stats = {}


def make_client():
    deps = ApiDeps(retriever=_Retriever(), parents=PARENTS, counter=_Counter(),
                   reranker=_Reranker(), ask=lambda messages: "答案 [证据 1]。")
    return TestClient(create_app(deps=deps))


class WebUiTest(unittest.TestCase):
    def test_index_is_html(self):
        response = make_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_page_points_at_the_real_stream_endpoint(self):
        html = make_client().get("/").text
        self.assertIn("/api/v1/ask/stream", html)
        # 端点真的在（不是写错的路径）
        paths = make_client().get("/openapi.json").json()["paths"]
        self.assertIn("/api/v1/ask/stream", paths)

    def test_page_has_no_external_dependencies(self):
        html = make_client().get("/").text.lower()
        for marker in ("cdn.", "googleapis", "unpkg", "jsdelivr", "src=\"http", "href=\"http"):
            self.assertNotIn(marker, html, f"页面不许引外部资源（内网会白屏）：{marker}")

    def test_page_shows_degraded_and_stop(self):
        html = make_client().get("/").text
        self.assertIn("降级", html)          # 前端要能显示降级
        self.assertIn("停止", html)          # 断开即取消生成


if __name__ == "__main__":
    unittest.main()
