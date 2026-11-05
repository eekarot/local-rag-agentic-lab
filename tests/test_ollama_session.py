"""Ollama 客户端的三条硬约束：**复用连接**、**不用 `localhost`**、**流式响应必关**。

起因是实测（2026-11-05，Windows 本机）：

    GET http://localhost:11434/api/tags   → 2083 ms
    GET http://127.0.0.1:11434/api/tags   →   49 ms
    GET http://[::1]:11434/api/tags       → 2045 ms  ConnectionError

`localhost` 在 Windows 上先解析到 `::1`，而 Ollama 只监听 `127.0.0.1`；每次**新建连接**
都要先等 IPv6 那一路超时（约 2 s）再回落 IPv4。而 `ollama_client` 原来每条请求都调
`requests.post(...)`（不复用连接），于是**每一次 Ollama 调用都白交 2 s**：

    CLI setup 13.5 s / 翻译层「白花 4 s」/ retrieve 冷 17.9 s / /health 热 4.7 s

单次开销小到没人会怀疑它，但它出现在**每个**请求里。所以这里用用例把两件事钉住：
连接必须复用（线程内共享 Session），默认地址必须是回环字面量 `127.0.0.1`。
"""

import threading
import unittest
from unittest import mock

from src import ollama_client


class SessionReuseTest(unittest.TestCase):
    def test_session_is_reused_within_a_thread(self):
        first = ollama_client._session()
        second = ollama_client._session()
        self.assertIs(first, second, "同一线程必须复用同一个 Session（连接池才有意义）")

    def test_session_is_per_thread(self):
        """requests.Session 不是线程安全的：生成线程、检索线程各自一份。"""
        seen = {}

        def grab(key):
            seen[key] = ollama_client._session()

        threads = [threading.Thread(target=grab, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertIsNot(seen[0], seen[1], "跨线程共享 Session 会在并发下串包")
        self.assertIsNot(seen[0], ollama_client._session())

    def test_post_goes_through_the_session(self):
        calls = []

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": [[0.0] * 1024]}

        class FakeSession:
            def post(self, url, **kwargs):
                calls.append((url, kwargs))
                return FakeResponse()

        with mock.patch.object(ollama_client, "_session", return_value=FakeSession()):
            vectors = ollama_client.embed_texts(["健康检查"])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(calls), 1, "请求必须走 Session（否则每条都新建连接）")
        self.assertIn("/api/embed", calls[0][0])


class BaseUrlTest(unittest.TestCase):
    def test_default_base_url_is_loopback_literal(self):
        from src.config import OLLAMA_BASE_URL

        if OLLAMA_BASE_URL != "http://127.0.0.1:11434":
            self.skipTest(f"环境变量改了地址：{OLLAMA_BASE_URL}")
        self.assertNotIn("localhost", OLLAMA_BASE_URL,
                         "别用 localhost：Windows 上会先撞 ::1，每次新建连接多花约 2 s")

    def test_example_file_documents_the_same_address(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
        self.assertIn("OLLAMA_BASE_URL=http://127.0.0.1:11434", text)


if __name__ == "__main__":
    unittest.main()
