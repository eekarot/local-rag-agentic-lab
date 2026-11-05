"""A1 / O5：`/health` 要能回答「现在到底哪一项坏了、怎么修」。

现状（改之前）的毛病：`/health` 只报「父块加载了几个、队列几个」——它永远说 `ok`，
哪怕 Qdrant 已经停了。用户拿着 `ok` 去排查，方向从一开始就是错的（N17 的同款教训：
**给错药比不给药更浪费时间**）。

四条约定，用例逐条锁死：
1. 探针结果整体进 `checks`，每条带 `fix`；
2. 整体状态取最差项：硬失败 `down` / 可降级 `degraded` / 全绿 `ok`；
3. **HTTP 状态永远是 200**（探活返回 500 会被负载均衡直接摘掉，反而看不到诊断）；
4. 探针自己抛异常也不许 500 —— 那正是最需要诊断信息的时刻。
"""

import unittest

from fastapi.testclient import TestClient

from src.rag.api.app import ApiDeps, create_app
from src.rag.ops.probes import Check, ProbeReport


PARENTS = {"par_a": {"id": "par_a", "text": "文本", "section_path": ["s"], "document_id": "doc_1",
                     "source_uri": "", "title": "t"}}


class _Retriever:
    def retrieve(self, query, *, k=20, query_en=None, disable=None, fuse=True):
        class Result:
            routes = {}
            degraded = None

        return Result()


class _Counter:
    def count(self, text):
        return max(1, len(text) // 4)


class _Reranker:
    """`/health` 只读 `config.enabled`，不需要真模型。"""

    class config:
        enabled = True


def _deps(probe_runner=None):
    return ApiDeps(retriever=_Retriever(), parents=PARENTS, counter=_Counter(),
                   reranker=_Reranker(), ask=lambda messages: "答案。",
                   probe_runner=probe_runner)


def _report(*checks) -> ProbeReport:
    return ProbeReport(list(checks))


ALL_OK = _report(
    Check(name="qdrant", ok=True, detail="rag_chunks_v1：38865 点"),
    Check(name="embed_model", ok=True, detail="bge-m3 可用，维度 1024"),
    Check(name="chat_model", ok=True, detail="qwen3.5:9b 已安装", optional=True),
    Check(name="tokenizer", ok=True, detail="分词器可用"),
    Check(name="reranker", ok=True, detail="已启用"),
    Check(name="parents", ok=True, detail="父块 25411 个"),
)


class HealthProbesTest(unittest.TestCase):
    def test_checks_are_reported_with_fix_commands(self):
        client = TestClient(create_app(deps=_deps(lambda: ALL_OK)))
        body = client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual([c["name"] for c in body["checks"]],
                         ["qdrant", "embed_model", "chat_model", "tokenizer", "reranker", "parents"])
        for check in body["checks"]:
            self.assertIn("fix", check, "每条探针都要带 fix 字段（哪怕是空串）")
            self.assertIn("optional", check)

    def test_blocking_failure_is_down_but_still_http_200(self):
        report = _report(
            Check(name="qdrant", ok=False, detail="连不上 localhost:6333",
                  fix="docker compose up -d qdrant"),
            Check(name="embed_model", ok=True, detail="ok"),
        )
        response = TestClient(create_app(deps=_deps(lambda: report))).get("/health")
        self.assertEqual(response.status_code, 200, "探活不能返回 5xx，否则看不到诊断")
        body = response.json()
        self.assertEqual(body["status"], "down")
        self.assertIn("docker compose up -d qdrant", body["blocking_fixes"])

    def test_optional_failure_is_degraded(self):
        report = _report(
            Check(name="qdrant", ok=True, detail="ok"),
            Check(name="chat_model", ok=False, detail="qwen3.5:9b 未安装",
                  fix="ollama pull qwen3.5:9b", optional=True),
        )
        body = TestClient(create_app(deps=_deps(lambda: report))).get("/health").json()
        self.assertEqual(body["status"], "degraded", "不能生成 ≠ 不能检索")
        self.assertIn("ollama pull qwen3.5:9b", body["degraded_fixes"])
        self.assertEqual(body["blocking_fixes"], [])

    def test_info_failure_does_not_change_status(self):
        report = _report(
            Check(name="qdrant", ok=True, detail="ok"),
            Check(name="data_dir", ok=False, detail="0 个文件", info=True),
        )
        body = TestClient(create_app(deps=_deps(lambda: report))).get("/health").json()
        self.assertEqual(body["status"], "ok", "data/ 为空不影响正在跑的服务")

    def test_probe_crash_does_not_500(self):
        def boom():
            raise RuntimeError("probe blew up")

        response = TestClient(create_app(deps=_deps(boom))).get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "down")
        self.assertIn("probe blew up", body["checks"][0]["detail"])

    def test_legacy_fields_stay_for_existing_clients(self):
        body = TestClient(create_app(deps=_deps(lambda: ALL_OK))).get("/health").json()
        for key in ("version", "parents", "routes", "rerank_enabled", "generation_enabled", "queue"):
            self.assertIn(key, body, f"老字段不能删：{key}")

    def test_without_probe_runner_health_stays_cheap(self):
        """单测里没人注入探针时不能偷偷连 Qdrant/Ollama（那会让 16 个 api 用例变慢变脆）。"""
        body = TestClient(create_app(deps=_deps(None))).get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["checks"], [])


if __name__ == "__main__":
    unittest.main()
