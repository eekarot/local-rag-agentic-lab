"""O4a：流式底层——`chat_stream()`（Ollama NDJSON）与 `answer_query_stream()`（事件流）。

为什么流式要做在**编排层**而不是只在 HTTP 层包一层：
生成 p50 53.5 s，如果 SSR/SSE 只是"等完了再一次性吐"，用户体验和同步没区别。
真正的价值是**引用先到**：检索 + 重排 + 组装只要 1–6 s，此时就能告诉用户
"我会从这 N 块里答"，然后再慢慢出正文。

事件协议（HTTP 层 O4c 直接映射成 SSE 的 event 名）：

    {"event": "sources", "data": {...}}      证据块与来源（早发）
    {"event": "delta",   "data": {"text": "…"}}  生成中的增量文本
    {"event": "done",    "data": {...}}      引用体检 + 分段耗时 + token 数
    {"event": "error",   "data": {...}}      生成失败（sources 已经发出去了，不重复发）

硬约束（都被写进用例）：
1. **`sources` 必须在 `delta` 之前**；
2. **没有证据时不调模型**（D33 定案），但仍要发 `sources`（0 块）+ `done`；
3. **只检索不生成**时不产生任何 `delta`；
4. **调用方中途 close() 要能把上游流关掉**——否则 SSE 客户端一断开，
   9B 还在那儿烧 50 s，单飞队列立刻被僵尸请求占满（B14 定案的核心风险）。
"""

import unittest

from src.rag.ask.pipeline import answer_query_stream
from src.ollama_client import chat_stream


# ---------------------------------------------------------------- 假件


class FakeRetriever:
    def retrieve(self, query, *, query_en=None, k=20, disable=None, fuse=True):
        class Route:
            def __init__(self, ids):
                self.ids = ids

        class Result:
            def __init__(self):
                self.routes = {"dense_zh": Route(["par_a", "par_b"]), "dense_en": Route(["par_a"])}

        return Result()


class FakeCounter:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


class FakeEncoder:
    def score(self, pairs):
        return [1.0 + 0.1 * i for i in range(len(pairs))]


class FakeReranker:
    def __init__(self):
        from src.rag.retrieval.rerank import RerankConfig

        self.config = RerankConfig(enabled=True)
        self.encoder = FakeEncoder()
        self.stats: dict = {}


PARENTS = {
    "par_a": {
        "id": "par_a", "document_id": "doc_1", "parent_index": 0,
        "text": "## 安装\n\n星轨笔记安装前要关闭旧版本，否则提示『端口 7801 被占用』。",
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


def collect(gen):
    events = list(gen)
    return events


def stream_answer(question="快捷键是什么？", *, ask=None, k=5, rerank=True, query_en=None):
    return answer_query_stream(
        question,
        retriever=FakeRetriever(),
        parents=PARENTS,
        counter=FakeCounter(),
        reranker=FakeReranker(),
        ask=ask,
        query_en=query_en,
        k=k,
        pool=10,
        budget=2000,
        rerank=rerank,
    )


# ---------------------------------------------------------------- chat_stream


class ChatStreamTest(unittest.TestCase):
    """Ollama 的 `/api/chat` 流式返回是**一行一个 JSON**（NDJSON）。"""

    def test_yields_content_deltas_in_order(self):
        lines = [
            b'{"message": {"content": "\xe4\xbd\xa0"}, "done": false}',
            b'{"message": {"content": "\xe5\xa5\xbd"}, "done": false}',
            b'{"message": {"content": ""}, "done": true, "eval_count": 2}',
        ]
        self.assertEqual(list(chat_stream([{"role": "user", "content": "hi"}], _lines=lines)),
                         ["你", "好"])

    def test_skips_empty_deltas_and_blank_lines(self):
        lines = [
            b'{"message": {"content": ""}, "done": false}',
            b'',
            b'   ',
            b'{"message": {"content": "A"}, "done": false}',
            b'{"done": true}',
        ]
        self.assertEqual(list(chat_stream([{"role": "user", "content": "hi"}], _lines=lines)), ["A"])

    def test_bad_json_line_is_skipped_not_fatal(self):
        """Ollama 偶尔会插入非 JSON 行（例如 502 的 HTML 片段），不能整段崩掉。"""
        lines = [
            b'{"message": {"content": "A"}, "done": false}',
            b'<html>502 Bad Gateway</html>',
            b'{"message": {"content": "B"}, "done": false}',
        ]
        self.assertEqual(list(chat_stream([{"role": "user", "content": "hi"}], _lines=lines)), ["A", "B"])

    def test_error_field_in_stream_raises(self):
        lines = [b'{"error": "model not found"}']
        with self.assertRaises(RuntimeError) as ctx:
            list(chat_stream([{"role": "user", "content": "hi"}], _lines=lines))
        self.assertIn("model not found", str(ctx.exception))

    def test_close_propagates_to_upstream(self):
        """客户端断开 → 关上游响应。不然 9B 会继续烧完 50 s，队列被僵尸占满。"""
        closed = {"n": 0}

        class FakeResponse:
            def iter_lines(self, *args, **kwargs):
                for _ in range(100):
                    yield b'{"message": {"content": "x"}, "done": false}'

            def close(self):
                closed["n"] += 1

        gen = chat_stream([{"role": "user", "content": "hi"}], _response=FakeResponse())
        next(gen)
        gen.close()
        self.assertEqual(closed["n"], 1, "上游响应必须被关掉")


# ---------------------------------------------------------------- answer_query_stream


class AnswerStreamTest(unittest.TestCase):
    def test_sources_before_deltas_before_done(self):
        events = collect(stream_answer(ask=lambda messages: iter(["Ctrl", "+Alt", "+T [证据 1]；"])))
        names = [event["event"] for event in events]
        self.assertEqual(names[0], "sources")
        self.assertEqual(names[-1], "done")
        self.assertIn("delta", names)
        first_delta = names.index("delta")
        self.assertLess(names.index("sources"), first_delta)
        self.assertLess(names.index("done"), len(names))

    def test_deltas_join_to_the_answer(self):
        events = collect(stream_answer(ask=lambda messages: iter(["插入时间戳是 ", "Ctrl+Alt+T", " [证据 1]。"])))
        text = "".join(event["data"]["text"] for event in events if event["event"] == "delta")
        done = events[-1]["data"]
        self.assertEqual(text, done["answer"])

    def test_done_reports_citations_and_timings(self):
        events = collect(stream_answer(ask=lambda messages: iter(["答案 [证据 1]。"])))
        done = events[-1]["data"]
        self.assertEqual(done["cited"], [1])
        self.assertEqual(done["out_of_range"], [])
        self.assertEqual(done["malformed"], [])
        self.assertIn("retrieve_rerank_ms", done["timings"])
        self.assertIn("first_delta_ms", done["timings"])
        self.assertIn("generate_ms", done["timings"])
        self.assertTrue(done["llm_called"])

    def test_sources_carry_block_metadata(self):
        events = collect(stream_answer(ask=lambda messages: iter(["答案 [证据 1]。"])))
        data = events[0]["data"]
        self.assertGreaterEqual(data["blocks"], 1)
        self.assertEqual(data["context_tokens"] > 0, True)
        source = data["sources"][0]
        for key in ("index", "document_id", "section_path", "source_uri", "title"):
            self.assertIn(key, source)
        self.assertIsInstance(source["section_path"], list)

    def test_no_evidence_shortcut_skips_model(self):
        """没有证据块时不调模型（D33），但仍要有 sources + done。"""
        called = {"n": 0}

        def ask(messages):
            called["n"] += 1
            return iter(["不该出现"])

        empty = answer_query_stream(
            "这个问题没有任何证据",
            retriever=FakeRetriever(), parents={}, counter=FakeCounter(),
            reranker=FakeReranker(), ask=ask, k=5, pool=10, budget=2000,
        )
        events = collect(empty)
        self.assertEqual(called["n"], 0)
        self.assertEqual(events[0]["event"], "sources")
        self.assertEqual(events[0]["data"]["blocks"], 0)
        self.assertEqual(events[-1]["event"], "done")
        self.assertTrue(events[-1]["data"]["abstained"])
        self.assertEqual([e for e in events if e["event"] == "delta"], [])

    def test_retrieve_only_emits_no_delta(self):
        events = collect(stream_answer(ask=None))
        self.assertEqual([e for e in events if e["event"] == "delta"], [])
        done = events[-1]["data"]
        self.assertEqual(done["answer"], "")
        self.assertFalse(done["llm_called"])

    def test_generation_error_emits_error_event_with_partial_text(self):
        """生成中途失败：已吐出的半截要留住，并明确告知可重试。"""
        def ask(messages):
            yield "前半段 [证据 1]。"
            raise RuntimeError("Ollama 挂了：connection reset")

        events = collect(stream_answer(ask=ask))
        names = [e["event"] for e in events]
        self.assertIn("error", names)
        error = events[-1]
        self.assertEqual(error["event"], "error")
        self.assertTrue(error["data"]["retryable"])
        self.assertIn("connection reset", error["data"]["detail"])
        self.assertEqual(error["data"]["partial"], "前半段 [证据 1]。")

    def test_consumer_close_stops_generation(self):
        """下游 close() 必须传到上游生成器（SSE 断开时不能白烧算力）。"""
        state = {"closed": False, "produced": 0}

        def ask(messages):
            try:
                for _ in range(1000):
                    state["produced"] += 1
                    yield "x"
            finally:
                state["closed"] = True

        gen = stream_answer(ask=ask)
        next(gen)                      # sources
        next(gen)                      # 一个 delta
        gen.close()
        self.assertTrue(state["closed"], "上游生成器必须收到关闭信号")
        self.assertLess(state["produced"], 1000)


if __name__ == "__main__":
    unittest.main()
