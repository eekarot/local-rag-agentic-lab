"""O4c：SSE 流式端点（B14 定案）。

服务端已经能流（O4a：`sources → delta → done`），队列也已就位（O4b）。这一步把它们
接到 HTTP 上，并解决三件只有「长连接 + 慢生成」才会暴露的事：

1. **客户端断开必须取消生成**：生成 50 s，用户关页面后如果不取消，那条「1 跑 + 2 排」
   的队列会被僵尸占满 —— 队列会 429 掉所有健康请求，服务看上去"挂了"；
2. **15 s 心跳**：反代（nginx/cloudflared）与浏览器会掐断长时间无数据的连接；
   心跳用 SSE 注释行 `: ping`，不产生事件、不打扰客户端解析；
3. **超时也要保留已发内容**：生成超时 → `error(status=504, partial=已生成部分)`，
   绝不能因为「最后一秒失败」把前面 40 s 的 `sources` 和正文一起丢掉。

事件（五事件，与 `pipeline.STREAM_EVENTS` 一一对应）：

    queued  排队信息（刚才的队列快照，是否要等）——**先发**，别让用户对着空白等
    sources 证据块 + 来源 + degraded —— 4.7 s 就能到
    delta   正文增量
    done    引用体检 + 分段耗时
    error   失败（带 status 与 partial）

HTTP 状态一旦开始流就是 200，所以**失败用事件表达**，不再能改状态码。
"""

import json
import threading
import time
import unittest

from src.rag.api.queue import SingleFlightQueue


PARENTS = {"par_a": {"id": "par_a", "document_id": "doc_1", "parent_index": 0,
                     "text": "## 快捷键\n\n插入时间戳的快捷键是 Ctrl+Alt+T。",
                     "section_path": ["手册", "快捷键"], "source_uri": "md/a.md", "title": "手册"}}


class FakeRetriever:
    def __init__(self, degraded=None):
        self.degraded = degraded

    def retrieve(self, query, *, k=20, query_en=None, disable=None, fuse=True):
        class Route:
            def __init__(self):
                self.ids = ["par_a"]

        class Result:
            routes = {"dense_zh": Route(), "dense_en": Route(), "lexical_en": Route()}

        result = Result()
        result.degraded = self.degraded
        return result


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


class ScriptedAsk:
    """按脚本吐 token；`close_hook` 用来验证「断开 → 上游被关掉」。"""

    def __init__(self, pieces=("插入时间戳是 ", "Ctrl+Alt+T", " [证据 1]。"), delay=0.0,
                 error_after=None, close_hook=None):
        self.pieces = list(pieces)
        self.delay = delay
        self.error_after = error_after
        self.close_hook = close_hook
        self.closed = False
        self.finished = False

    def __call__(self, messages):
        try:
            for index, piece in enumerate(self.pieces):
                if self.error_after is not None and index >= self.error_after:
                    raise RuntimeError("Ollama 连接被重置")
                if self.delay:
                    time.sleep(self.delay)
                yield piece
            self.finished = True
        finally:
            self.closed = True
            if self.close_hook:
                self.close_hook()


def make_client(*, ask=None, queue=None, heartbeat_s=15.0, request_timeout_s=300.0, degraded=None):
    from fastapi.testclient import TestClient

    from src.rag.api.app import ApiDeps, create_app

    deps = ApiDeps(
        retriever=FakeRetriever(degraded=degraded), parents=PARENTS, counter=FakeCounter(),
        reranker=FakeReranker(), ask=ask or ScriptedAsk(),
        ask_queue=queue or SingleFlightQueue(max_inflight=1, max_queue=2, timeout_s=5),
        retrieve_queue=SingleFlightQueue(max_inflight=4, max_queue=2, timeout_s=5),
        heartbeat_s=heartbeat_s, request_timeout_s=request_timeout_s,
    )
    return TestClient(create_app(deps=deps))


def parse_sse(text: str) -> tuple[list[dict], list[str]]:
    """返回 (事件列表, 注释行列表)。事件按 SSE 规范用空行分隔。"""
    events: list[dict] = []
    comments: list[str] = []
    name = None
    data: list[str] = []
    for raw in text.splitlines():
        if raw.startswith(":"):
            comments.append(raw.strip())
            continue
        if raw.startswith("event:"):
            name = raw.split(":", 1)[1].strip()
        elif raw.startswith("data:"):
            data.append(raw.split(":", 1)[1].strip())
        elif raw == "":
            if name is not None:
                events.append({"event": name, "data": json.loads("".join(data))})
            name, data = None, []
    return events, comments


STREAM_URL = "/api/v1/ask/stream"


class StreamBasicsTest(unittest.TestCase):
    def test_content_type_and_event_order(self):
        response = make_client().get(STREAM_URL, params={"q": "快捷键是什么？", "k": 2, "rerank": "false"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertEqual(response.headers.get("x-accel-buffering"), "no", "禁反代缓冲，否则流变一次性")

        events, _ = parse_sse(response.text)
        names = [e["event"] for e in events]
        self.assertEqual(names[0], "queued", "先告诉用户排到没有")
        self.assertEqual(names[1], "sources")
        self.assertIn("delta", names)
        self.assertEqual(names[-1], "done")

    def test_deltas_join_to_the_done_answer(self):
        events, _ = parse_sse(make_client().get(
            STREAM_URL, params={"q": "快捷键", "k": 2, "rerank": "false"}).text)
        text = "".join(e["data"]["text"] for e in events if e["event"] == "delta")
        done = next(e["data"] for e in events if e["event"] == "done")
        self.assertEqual(text, done["answer"])
        self.assertEqual(done["cited"], [1])

    def test_queued_event_reports_queue_state(self):
        events, _ = parse_sse(make_client().get(
            STREAM_URL, params={"q": "快捷键", "k": 2, "rerank": "false"}).text)
        queued = events[0]["data"]
        self.assertEqual(queued["max_inflight"], 1)
        self.assertIn("inflight", queued)
        self.assertIn("waiting", queued)
        self.assertFalse(queued["will_wait"], "队列空时不该说要等")

    def test_sources_carry_degraded(self):
        from src.rag.retrieval.degrade import DegradedInfo

        info = DegradedInfo(reason="embed_unavailable", missing=["embed_model"],
                            fix="ollama serve", detail="x", routes=["dense_zh"])
        events, _ = parse_sse(make_client(degraded=info).get(
            STREAM_URL, params={"q": "快捷键", "k": 2, "rerank": "false"}).text)
        sources = events[1]["data"]
        self.assertEqual(sources["degraded"]["reason"], "embed_unavailable")


class StreamFailureTest(unittest.TestCase):
    def test_midstream_failure_keeps_partial_text(self):
        ask = ScriptedAsk(pieces=("前半段 ", "[证据 1]。", "不该出现"), error_after=2)
        events, _ = parse_sse(make_client(ask=ask).get(
            STREAM_URL, params={"q": "快捷键", "k": 2, "rerank": "false"}).text)
        error = events[-1]
        self.assertEqual(error["event"], "error")
        self.assertEqual(error["data"]["partial"], "前半段 [证据 1]。")
        self.assertTrue(error["data"]["retryable"])
        self.assertIn("连接被重置", error["data"]["detail"])

    def test_queue_full_becomes_error_event_with_status_429(self):
        queue = SingleFlightQueue(max_inflight=1, max_queue=0, timeout_s=0.1, retry_after_s=7)
        holder = queue.acquire()                     # 先占住唯一名额
        try:
            events, _ = parse_sse(make_client(queue=queue).get(
                STREAM_URL, params={"q": "快捷键", "k": 2, "rerank": "false"}).text)
        finally:
            holder.release()
        error = events[-1]
        self.assertEqual(error["event"], "error")
        self.assertEqual(error["data"]["status"], 429, "流里已无法改状态码，用事件表达")
        self.assertEqual(error["data"]["retry_after_s"], 7)

    def test_generation_timeout_keeps_sources_and_reports_504(self):
        ask = ScriptedAsk(pieces=("慢", "慢", "慢"), delay=0.15)
        events, _ = parse_sse(make_client(ask=ask, request_timeout_s=0.25).get(
            STREAM_URL, params={"q": "快捷键", "k": 2, "rerank": "false"}).text)
        names = [e["event"] for e in events]
        self.assertIn("sources", names, "超时前已发的 sources 必须保留")
        error = events[-1]
        self.assertEqual(error["event"], "error")
        self.assertEqual(error["data"]["status"], 504)
        self.assertIn("partial", error["data"])


class StreamHeartbeatTest(unittest.TestCase):
    def test_heartbeat_comment_while_generating(self):
        """反代会掐断长时间无数据的连接；心跳用 `: ping` 注释行（不产生事件）。"""
        ask = ScriptedAsk(pieces=("a", "b", "c", "d"), delay=0.12)
        response = make_client(ask=ask, heartbeat_s=0.1).get(
            STREAM_URL, params={"q": "快捷键", "k": 2, "rerank": "false"})
        events, comments = parse_sse(response.text)
        self.assertTrue(any("ping" in c for c in comments), f"没看到心跳：{comments[:3]}")
        self.assertEqual(events[-1]["event"], "done", "心跳不许干扰事件流")


class StreamDisconnectTest(unittest.TestCase):
    def test_client_disconnect_closes_upstream_generation(self):
        """断开 → `answer_query_stream` 被 close → `chat_stream` 被 close → Ollama 连接释放。

        不这么做，生成会继续烧完（50 s），而队列名额 1+2 很快被僵尸占满。
        """
        closed = threading.Event()
        ask = ScriptedAsk(pieces=[f"tok{i}" for i in range(40)], delay=0.05,
                          close_hook=closed.set)
        client = make_client(ask=ask)

        with client.stream("GET", STREAM_URL,
                           params={"q": "快捷键", "k": 2, "rerank": "false"}) as response:
            seen = 0
            for line in response.iter_lines():
                if line.startswith("event:"):
                    seen += 1
                if seen >= 3:                        # 拿到 queued/sources/第一个 delta 就断开
                    break

        self.assertTrue(closed.wait(5), "客户端断开后必须把上游生成关掉")
        self.assertFalse(ask.finished, "不该让它继续生成完")

        # 名额也要还回去，否则后续请求会被 429
        deadline = time.time() + 5
        while time.time() < deadline:
            if client.get("/health").json()["queue"]["inflight"] == 0:
                return
            time.sleep(0.05)
        self.fail("断开后队列名额没有归还（inflight 一直 > 0）")


if __name__ == "__main__":
    unittest.main()
