"""O4b 有界队列 + 单飞（B14 定案）。

为什么需要它（实测数字，不是假设）：
- 生成 p50 **53.5 s**，reranker 常驻 **945 MB**，`max_concurrency=1`，可用内存 **5.9 GB**；
- 放开并发不会让 10 个用户各快 10 倍，只会让 **10 个人一起等更久**（CPU 上 9B 是算力独占）；
- 所以策略是「**一次只生成一个，其余排队；排不下就明确拒绝**」，而不是把所有人拖进超时。

四条必须守住的语义：

1. **单飞**：`MAX_INFLIGHT=1` 时永远只有 1 个生成在跑（用并发计数器验证，不看日志）；
2. **有界**：正在跑 1 + 排队 2 = 最多 3 个进来，第 4 个立刻 **429 + `Retry-After`**；
3. **排队超时**：排了 20 s 还没轮到 → **429**（不是 500，也不是无限等）；
4. **`/retrieve` 与 `/ask` 分开限流**：检索是纯 CPU 短任务（1–5 s），不能被排队的生成请求饿死。
"""

import threading
import time
import unittest

from src.rag.api.queue import QueueFullError, QueueTimeoutError, SingleFlightQueue


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def advance(self, seconds):
        with self.lock:
            self.now += seconds


def drain(queue, *, count=1):
    """把队列里排着的 count 个名额依次放掉（模拟生成完成）。"""
    for _ in range(count):
        queue.release()


class QueueUnitTest(unittest.TestCase):
    def test_single_flight_runs_one_at_a_time(self):
        """`max_inflight=1` 时，任何第二个持有者（含同线程）都必须等 —— 这是单飞的定义。"""
        queue = SingleFlightQueue(max_inflight=1, max_queue=2, timeout_s=0.1)
        release = threading.Event()

        with queue.acquire():
            self.assertEqual(queue.snapshot()["inflight"], 1)
            with self.assertRaises(QueueTimeoutError):
                queue.acquire()            # 同线程重入也必须等，等到超时
            release.set()
        self.assertEqual(queue.snapshot()["inflight"], 0)

    def test_rejects_when_queue_is_full(self):
        queue = SingleFlightQueue(max_inflight=1, max_queue=0, timeout_s=5)
        ticket = queue.acquire()
        try:
            with self.assertRaises(QueueFullError) as ctx:
                queue.acquire()
            self.assertIn("1 个在跑", str(ctx.exception), "detail 要说清容量")
            self.assertIn("0 个排队", str(ctx.exception))
        finally:
            ticket.release()
        self.assertEqual(queue.snapshot()["rejected_full"], 1)

    def test_retry_after_defaults_to_expected_service_time_not_timeout(self):
        """实测教训：生成 p50 53.5 s，超时 20 s 的排队者 100% 会 429 → max_queue 形同虚设。"""
        queue = SingleFlightQueue()
        self.assertEqual(queue.timeout_s, 120.0)
        self.assertEqual(queue.snapshot()["retry_after_s"], 60)

    def test_waiter_times_out_instead_of_hanging(self):
        """排队超时必须抛**自己的**异常（→429），不能变成 500，也不能无限等。"""
        clock = FakeClock()
        queue = SingleFlightQueue(max_inflight=1, max_queue=2, timeout_s=20, clock=clock)
        ticket = queue.acquire()
        try:
            with self.assertRaises(QueueTimeoutError) as ctx:
                queue.acquire(blocking=False)      # 非阻塞：立刻超时，测试不用真等 20 s
            self.assertIn("20", str(ctx.exception))
        finally:
            ticket.release()
        self.assertEqual(queue.snapshot()["rejected_timeout"], 1)

    def test_release_lets_a_waiter_in(self):
        queue = SingleFlightQueue(max_inflight=1, max_queue=2, timeout_s=5)
        started = threading.Event()
        done = threading.Event()

        def worker():
            with queue.acquire():
                started.set()
                queue.snapshot()["inflight"]
            done.set()

        with queue.acquire():
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            time.sleep(0.05)
            self.assertFalse(started.is_set(), "单飞时第二个必须等着")
        self.assertTrue(started.wait(2), "放掉名额后必须被唤醒")
        self.assertTrue(done.wait(2))
        thread.join(2)
        self.assertGreater(queue.snapshot()["waited_ms_total"], 0.0, "排队者的等待耗时要被记下来")

    def test_snapshot_counts_served_and_waiting(self):
        queue = SingleFlightQueue(max_inflight=1, max_queue=2, timeout_s=5)
        with queue.acquire():
            self.assertEqual(queue.snapshot()["inflight"], 1)
        snap = queue.snapshot()            # 名额放掉之后才算「服务完成」
        self.assertEqual(snap["served"], 1)
        self.assertEqual(snap["inflight"], 0)
        self.assertEqual(snap["waited_ms_total"], 0.0, "没排过队就不该有等待耗时")

    def test_detail_explains_capacity_and_retry_after(self):
        queue = SingleFlightQueue(max_inflight=1, max_queue=0, timeout_s=20, retry_after_s=20)
        ticket = queue.acquire()
        try:
            with self.assertRaises(QueueFullError):
                queue.acquire()
        finally:
            ticket.release()
        snap = queue.snapshot()
        self.assertEqual(snap["max_inflight"], 1)
        self.assertEqual(snap["max_queue"], 0)
        self.assertEqual(snap["retry_after_s"], 20)


# ---------------------------------------------------------------- HTTP 层


class GatedAsk:
    """可控的生成：卡在 `gate` 上，直到测试放行。用来制造真实的并发。"""

    def __init__(self):
        self.gate = threading.Event()
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, messages):
        with self.lock:
            self.calls += 1
        self.gate.wait(10)
        return "答案 [证据 1]"


PARENTS = {"par_a": {"id": "par_a", "document_id": "doc_1", "parent_index": 0,
                     "text": "## 快捷键\n\n插入时间戳的快捷键是 Ctrl+Alt+T。",
                     "section_path": ["手册", "快捷键"], "source_uri": "md/a.md", "title": "手册"}}


class FakeRetriever:
    def retrieve(self, query, *, k=20, query_en=None, disable=None, fuse=True):
        class Route:
            def __init__(self):
                self.ids = ["par_a"]

        class Result:
            routes = {"dense_zh": Route(), "dense_en": Route(), "lexical_en": Route()}

        return Result()


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


def make_client(*, queue=None, retrieve_queue=None, ask=None):
    from fastapi.testclient import TestClient

    from src.rag.api.app import ApiDeps, create_app

    deps = ApiDeps(
        retriever=FakeRetriever(), parents=PARENTS, counter=FakeCounter(),
        reranker=FakeReranker(), ask=ask or GatedAsk(),
        ask_queue=queue or SingleFlightQueue(max_inflight=1, max_queue=2, timeout_s=5),
        retrieve_queue=retrieve_queue or SingleFlightQueue(max_inflight=4, max_queue=2, timeout_s=5),
    )
    return TestClient(create_app(deps=deps))


BODY = {"question": "插入时间戳的快捷键是什么？", "k": 2, "rerank": False}


class HttpQueueTest(unittest.TestCase):
    def test_second_ask_waits_for_the_first(self):
        """单飞：两个并发 /ask，第二个必须等第一个放行后才返回（都 200）。"""
        ask = GatedAsk()
        client = make_client(ask=ask)
        results = []

        def call():
            response = client.post("/api/v1/ask", json=BODY)
            results.append(response.status_code)

        first = threading.Thread(target=call, daemon=True)
        second = threading.Thread(target=call, daemon=True)
        first.start()
        time.sleep(0.1)
        second.start()
        time.sleep(0.2)
        self.assertEqual(len(results), 0, "两个都还在生成：谁都不该先返回")

        ask.gate.set()
        first.join(10)
        second.join(10)
        self.assertEqual(sorted(results), [200, 200])

    def test_queue_full_returns_429_with_retry_after(self):
        queue = SingleFlightQueue(max_inflight=1, max_queue=0, timeout_s=20, retry_after_s=20)
        ask = GatedAsk()
        client = make_client(queue=queue, ask=ask)
        results = []

        def call():
            results.append(client.post("/api/v1/ask", json=BODY))

        first = threading.Thread(target=call, daemon=True)
        first.start()
        time.sleep(0.15)                     # 让第一个真正占住名额
        second = client.post("/api/v1/ask", json=BODY)

        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.headers.get("Retry-After"), "20")
        self.assertIn("队列已满", second.json()["detail"])
        self.assertIn("Retry-After", second.json()["detail"])

        ask.gate.set()
        first.join(10)

    def test_retrieve_is_not_starved_by_a_long_ask(self):
        """检索是短任务，不能被排队的生成饿死（两条队列分开）。"""
        ask = GatedAsk()
        client = make_client(ask=ask)
        holder = threading.Thread(target=lambda: client.post("/api/v1/ask", json=BODY), daemon=True)
        holder.start()
        time.sleep(0.15)

        started = time.perf_counter()
        response = client.get("/api/v1/retrieve", params={"q": "shortcut", "k": 2, "rerank": "false"})
        elapsed = time.perf_counter() - started
        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed, 2.0, "生成占着队列时，/retrieve 仍必须立刻可用")

        ask.gate.set()
        holder.join(10)

    def test_retrieve_only_does_not_consume_the_ask_slot(self):
        """`retrieve_only=true` 不调模型 → 不该占生成名额（否则一次批量检索能把问答堵死）。"""
        queue = SingleFlightQueue(max_inflight=1, max_queue=0, timeout_s=20)
        ask = GatedAsk()
        client = make_client(queue=queue, ask=ask)

        holder = threading.Thread(target=lambda: client.post("/api/v1/ask", json=BODY), daemon=True)
        holder.start()
        time.sleep(0.15)

        response = client.post("/api/v1/ask", json={**BODY, "retrieve_only": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["llm_called"])

        ask.gate.set()
        holder.join(10)

    def test_health_reports_queue_state(self):
        client = make_client()
        payload = client.get("/health").json()
        self.assertEqual(payload["queue"]["max_inflight"], 1)
        self.assertEqual(payload["queue"]["inflight"], 0)


if __name__ == "__main__":
    unittest.main()
