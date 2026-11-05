"""`src/rag/ops/probes.py`：一份探测逻辑，三处复用（`/health`、`doctor`、降级判定）。

为什么必须先抽这一层再写降级/doctor/health：
三处各自实现探测，一定会出现「`/health` 说 ok，`doctor` 说缺词表」的自相矛盾，
而这种矛盾恰好会发生在**用户最需要可信度**的时刻（起不来的时候）。

三个设计约束：

1. **每条失败都要带可复制的修复命令**（`fix`）——"缺少 tokenizer.json"无用，
   `curl -sL -o ... MODELSCOPE_URL` 才有用（B12 定案：不藏失败，给可验证的下一步）；
2. **`optional` 区分硬失败与可降级**：生成模型缺席 → 检索仍可用（降级），
   向量模型/词表/向量库缺席 → 整个检索不可用；
3. **探测函数全部依赖注入**，单测不连 Qdrant / Ollama / HF / 不读真 tokenizer。
"""

import unittest

from src.rag.ops.probes import (
    Check,
    ProbeReport,
    probe_chat_model,
    probe_data_dir,
    probe_embed_model,
    probe_parents,
    probe_qdrant,
    probe_reranker,
    probe_tokenizer,
    run_probes,
)


# ---------------------------------------------------------------- 假件


class FakeEmbedder:
    def __init__(self, dim=1024, error=None):
        self.dim = dim
        self.error = error
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        if self.error:
            raise self.error
        return [[0.01] * self.dim for _ in texts]


class FakeQdrant:
    def __init__(self, collections=None, points=None, error=None):
        self.collections = collections or {}
        self.error = error

    def get_collections(self):
        if self.error:
            raise self.error

        class Item:
            def __init__(self, name):
                self.name = name

        class Resp:
            collections = [Item(name) for name in self.collections]

        return Resp()

    def count(self, collection):
        class Resp:
            def __init__(self, count):
                self.count = count

        return Resp(self.collections.get(collection, 0)) if self.collections.get(collection) else Resp(0)


class FakeChatTags:
    """模拟 `GET /api/tags`：只回模型名列表，不真的生成。"""

    def __init__(self, names):
        self.names = names

    def __call__(self):
        return list(self.names)


class FakeCounter:
    def __init__(self, error=None):
        self.error = error

    def count(self, text):
        if self.error:
            raise self.error
        return len(text)


class FakeReranker:
    def __init__(self, enabled=True, config=None):
        class Config:
            pass

        self.config = config or Config()
        self.config.enabled = enabled


# ---------------------------------------------------------------- 单项探测


class ProbeQdrantTest(unittest.TestCase):
    def test_ok_when_collection_has_points(self):
        check = probe_qdrant(client=FakeQdrant({"rag_chunks_v1": 38865}))
        self.assertTrue(check.ok)
        self.assertIn("38865", check.detail)
        self.assertEqual(check.fix, "")

    def test_missing_collection_tells_where_to_start(self):
        check = probe_qdrant(client=FakeQdrant({}))
        self.assertFalse(check.ok)
        self.assertIn("rag_chunks_v1", check.detail)
        self.assertIn("src.ingest", check.fix)

    def test_empty_collection_is_not_ok(self):
        check = probe_qdrant(client=FakeQdrant({"rag_chunks_v1": 0}))
        self.assertFalse(check.ok)
        self.assertIn("0", check.detail)

    def test_connection_error_is_reported_not_raised(self):
        """探测本身绝不能抛：它就是在"很多东西坏了"的时候被调用的。"""
        check = probe_qdrant(client=FakeQdrant(error=ConnectionError("connection refused")))
        self.assertFalse(check.ok)
        self.assertIn("connection refused", check.detail)
        self.assertIn("docker", check.fix.lower())


class ProbeEmbedModelTest(unittest.TestCase):
    def test_ok_on_expected_dimension(self):
        check = probe_embed_model(embed=FakeEmbedder(dim=1024))
        self.assertTrue(check.ok)
        self.assertIn("1024", check.detail)

    def test_wrong_dimension_is_a_failure_with_why(self):
        """换 768 维模型不会报错、只会让相似度全错 —— 必须在这里抓住。"""
        check = probe_embed_model(embed=FakeEmbedder(dim=768), expected_dim=1024)
        self.assertFalse(check.ok)
        self.assertIn("768", check.detail)
        self.assertIn("1024", check.detail)
        self.assertIn("bge-m3", check.fix)

    def test_ollama_down_gives_pull_hint(self):
        check = probe_embed_model(embed=FakeEmbedder(error=RuntimeError("无法调用 Ollama API")))
        self.assertFalse(check.ok)
        self.assertIn("Ollama", check.detail)
        self.assertIn("ollama pull", check.fix)

    def test_ollama_down_prefers_serve_over_pull(self):
        """服务没起 vs 模型没拉是两件事：前者要先 `ollama serve`，不然 pull 也失败。"""
        check = probe_embed_model(embed=FakeEmbedder(
            error=RuntimeError("无法调用 Ollama API：http://localhost:11434/api/embed")))
        self.assertIn("ollama serve", check.fix)

    def test_embed_model_is_not_optional(self):
        """没有向量模型 = 三路里两路全挂，不是"降级"。"""
        check = probe_embed_model(embed=FakeEmbedder(error=RuntimeError("x")))
        self.assertFalse(check.optional)


class ProbeChatModelTest(unittest.TestCase):
    def test_ok_when_model_installed(self):
        check = probe_chat_model(list_models=FakeChatTags(["qwen3.5:9b", "bge-m3:latest"]), model="qwen3.5:9b")
        self.assertTrue(check.ok)
        self.assertIn("qwen3.5:9b", check.detail)

    def test_latest_tag_still_matches(self):
        check = probe_chat_model(list_models=FakeChatTags(["qwen3.5:9b"]), model="qwen3.5:9b:latest")
        self.assertTrue(check.ok)

    def test_missing_model_gives_pull_command(self):
        check = probe_chat_model(list_models=FakeChatTags(["bge-m3:latest"]), model="llama3:8b")
        self.assertFalse(check.ok)
        self.assertEqual(check.fix, "ollama pull llama3:8b")

    def test_chat_model_is_optional_because_retrieval_survives(self):
        check = probe_chat_model(list_models=FakeChatTags([]), model="llama3:8b")
        self.assertTrue(check.optional)

    def test_ollama_not_running_is_reported(self):
        def boom():
            raise ConnectionError("Connection refused")

        check = probe_chat_model(list_models=boom, model="qwen3.5:9b")
        self.assertFalse(check.ok)
        self.assertIn("Connection refused", check.detail)
        self.assertIn("ollama serve", check.fix)


class ProbeTokenizerTest(unittest.TestCase):
    def test_ok_when_counter_works(self):
        check = probe_tokenizer(counter=FakeCounter())
        self.assertTrue(check.ok)

    def test_missing_tokenizer_keeps_the_curl_from_the_error(self):
        from src.rag.ingestion.errors import TokenizerUnavailable

        check = probe_tokenizer(counter=FakeCounter(error=TokenizerUnavailable(
            "缺少 tokenizer.json\n  curl -sL -o x.json https://example/tokenizer.json")))
        self.assertFalse(check.ok)
        self.assertIn("curl", check.fix)
        self.assertIn("tokenizer.json", check.fix)


class ProbeRerankerTest(unittest.TestCase):
    def test_disabled_reranker_is_ok(self):
        check = probe_reranker(reranker=FakeReranker(enabled=False))
        self.assertTrue(check.ok)
        self.assertIn("未启用", check.detail)

    def test_enabled_reranker_is_ok(self):
        check = probe_reranker(reranker=FakeReranker(enabled=True))
        self.assertTrue(check.ok)


class ProbeDataDirTest(unittest.TestCase):
    def test_reports_file_count(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("x", encoding="utf-8")
            (root / "b.txt").write_text("y", encoding="utf-8")
            check = probe_data_dir(root)
        self.assertTrue(check.ok)
        self.assertIn("2", check.detail)

    def test_empty_dir_points_at_readme(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            check = probe_data_dir(Path(tmp))
        self.assertFalse(check.ok)
        self.assertIn("data/README.md", check.fix)


class ProbeParentsTest(unittest.TestCase):
    def test_ok_with_parents(self):
        self.assertTrue(probe_parents({"par_a": {}}).ok)

    def test_empty_parents_is_not_ok(self):
        check = probe_parents({})
        self.assertFalse(check.ok)
        self.assertIn("父块", check.detail)


# ---------------------------------------------------------------- 汇总


class ReportTest(unittest.TestCase):
    def _run(self, **overrides):
        kwargs = dict(
            qdrant=FakeQdrant({"rag_chunks_v1": 100}),
            embed=FakeEmbedder(),
            list_models=FakeChatTags(["qwen3.5:9b"]),
            model="qwen3.5:9b",
            counter=FakeCounter(),
            reranker=FakeReranker(enabled=True),
            parents={"par_a": {}},
        )
        kwargs.update(overrides)
        return run_probes(**kwargs)

    def test_all_green(self):
        report = self._run()
        self.assertEqual(report.status, "ok")
        self.assertEqual([c.name for c in report.checks],
                         ["qdrant", "embed_model", "chat_model", "tokenizer", "reranker", "parents"])

    def test_missing_chat_model_degrades_not_down(self):
        report = self._run(list_models=FakeChatTags([]))
        self.assertEqual(report.status, "degraded")
        self.assertEqual(report.blocking_fixes, [])
        self.assertTrue(report.degraded_fixes)

    def test_missing_embed_model_is_down(self):
        report = self._run(embed=FakeEmbedder(error=RuntimeError("无法调用 Ollama API")))
        self.assertEqual(report.status, "down")
        self.assertEqual(len(report.blocking_fixes), 1)
        self.assertIn("ollama pull", report.blocking_fixes[0])

    def test_data_dir_not_part_of_serving_health(self):
        """`data/` 为空只影响"再建库"，不影响已经在跑的检索 —— 不进 run_probes 的必查项。"""
        report = self._run()
        self.assertNotIn("data_dir", [c.name for c in report.checks])
        self.assertEqual(report.status, "ok")

    def test_as_dict_is_json_friendly(self):
        import json

        payload = self._run(list_models=FakeChatTags([])).as_dict()
        self.assertEqual(payload["status"], "degraded")
        json.dumps(payload)                                  # 不抛即可
        self.assertIsInstance(payload["checks"][0]["ok"], bool)

    def test_check_is_immutable(self):
        check = Check(name="x", ok=True, detail="d")
        with self.assertRaises(Exception):
            check.ok = False


if __name__ == "__main__":
    unittest.main()
