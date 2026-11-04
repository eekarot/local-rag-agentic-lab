"""D23：Query 翻译服务（严格 JSON + 术语保护 + 缓存 + 失败降级）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.rag.translate.cache import TranslationCache
from src.rag.translate.service import (
    STATUS_CACHED,
    STATUS_OK,
    STATUS_PASSTHROUGH_ERROR,
    STATUS_PASSTHROUGH_TERMS,
    STATUS_SKIPPED,
    TranslateService,
    parse_json_reply,
)
from src.rag.translate.terms import missing_terms, protect_tokens, retention


class TermsTest(unittest.TestCase):
    def test_protect_extracts_models_numbers_and_identifiers(self):
        tokens = protect_tokens("DGX B300 的 HNSW ef_construct 设成 200，显存 288 GB")
        for want in ("DGX B300", "HNSW", "ef_construct", "200", "288"):
            self.assertIn(want, tokens)

    def test_retention_full_and_partial(self):
        self.assertEqual(retention("DGX B300 有几张 GPU？", "How many GPUs does a DGX B300 have?"), 1.0)
        self.assertLess(retention("DGX B300 有几张 GPU？", "How many GPUs does it have?"), 1.0)

    def test_missing_terms_listed(self):
        self.assertEqual(missing_terms("DGX B300 和 HNSW", "about DGX B300"), ["HNSW"])

    def test_whitespace_insensitive(self):
        self.assertEqual(missing_terms("288 GB 显存", "288GB of memory"), [])


class ParseReplyTest(unittest.TestCase):
    def test_json_object(self):
        self.assertEqual(parse_json_reply('{"translation": "What is RAG?"}'), "What is RAG?")

    def test_fenced_json(self):
        self.assertEqual(parse_json_reply('```json\n{"translation": "What is RAG?"}\n```'), "What is RAG?")

    def test_bare_string(self):
        self.assertEqual(parse_json_reply("What is RAG?"), "What is RAG?")

    def test_garbage_returns_empty(self):
        self.assertEqual(parse_json_reply(""), "")
        self.assertEqual(parse_json_reply('{"foo": 1}'), "")
        self.assertEqual(parse_json_reply('{"translation": ""}'), "")


class CacheTest(unittest.TestCase):
    def test_put_get_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = TranslationCache(Path(tmp) / "t.sqlite3", model="m1", version="v1")
            cache.put("什么是 RAG？", "What is RAG?")
            self.assertEqual(cache.get("什么是 RAG？"), "What is RAG?")
            self.assertIsNone(cache.get("别的问法"))
            self.assertEqual(cache.stats()["entries"], 1)

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.sqlite3"
            TranslationCache(path, model="m1", version="v1").put("A", "B")
            self.assertEqual(TranslationCache(path, model="m1", version="v1").get("A"), "B")
            # 换模型/版本就不该命中（否则会把旧模型的译文当新模型的）
            self.assertIsNone(TranslationCache(path, model="m2", version="v1").get("A"))
            self.assertIsNone(TranslationCache(path, model="m1", version="v2").get("A"))


class _Provider:
    """假 provider：记录批次，按脚本返回。"""

    def __init__(self, replies=None, error=None):
        self.replies = replies or {}
        self.error = error
        self.calls = []

    def __call__(self, texts, hints=None):
        self.calls.append(list(texts))
        if self.error:
            raise self.error
        return [self.replies.get(text, f"EN({text})") for text in texts]


class ServiceTest(unittest.TestCase):
    def _service(self, provider, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cache = TranslationCache(Path(tmp.name) / "c.sqlite3", model="fake", version="v1")
        return TranslateService(provider, cache=cache, **kwargs)

    def test_translates_and_reports_ok(self):
        service = self._service(_Provider({"什么是 RAG？": "What is RAG?"}))
        results = service.translate(["什么是 RAG？"])
        self.assertEqual(results[0].text, "What is RAG?")
        self.assertEqual(results[0].status, STATUS_OK)
        self.assertEqual(results[0].source, "什么是 RAG？")

    def test_second_call_hits_cache(self):
        provider = _Provider({"什么是 RAG？": "What is RAG?"})
        service = self._service(provider)
        service.translate(["什么是 RAG？"])
        results = service.translate(["什么是 RAG？"])
        self.assertEqual(results[0].status, STATUS_CACHED)
        self.assertEqual(len(provider.calls), 1)   # 只真正调了一次

    def test_english_query_is_skipped(self):
        provider = _Provider()
        service = self._service(provider)
        results = service.translate(["What is RRF?"])
        self.assertEqual(results[0].status, STATUS_SKIPPED)
        self.assertEqual(results[0].text, "What is RRF?")
        self.assertEqual(provider.calls, [])

    def test_dropped_term_triggers_retry_then_passthrough(self):
        # provider 两次都漏掉 HNSW → 不能把坏译文交给检索，回退原文并记录
        provider = _Provider({"HNSW 是什么？": "What is this?"})
        service = self._service(provider)
        results = service.translate(["HNSW 是什么？"])
        self.assertEqual(results[0].status, STATUS_PASSTHROUGH_TERMS)
        self.assertEqual(results[0].text, "HNSW 是什么？")
        self.assertEqual(results[0].missing, ["HNSW"])
        self.assertEqual(len(provider.calls), 2)   # 带 hint 重试了一次

    def test_term_kept_after_retry_is_ok(self):
        replies = {"HNSW 是什么？": "What is this?"}

        def provider(texts, hints=None, temperature=0.0):
            # 第一次漏术语、重试（换温度）才对
            if temperature > 0:
                return ["What is HNSW?" for _ in texts]
            return [replies.get(text, text) for text in texts]

        service = self._service(provider)
        results = service.translate(["HNSW 是什么？"])
        self.assertEqual(results[0].status, STATUS_OK)
        self.assertEqual(results[0].text, "What is HNSW?")

    def test_timeout_falls_back_to_original(self):
        service = self._service(_Provider(error=TimeoutError("boom")))
        results = service.translate(["什么是 RAG？", "DGX B300 有几张 GPU？"])
        self.assertEqual([r.status for r in results], [STATUS_PASSTHROUGH_ERROR] * 2)
        self.assertEqual([r.text for r in results], ["什么是 RAG？", "DGX B300 有几张 GPU？"])

    def test_timeout_does_not_break_other_items(self):
        # 一批里坏一条不影响其它条，且顺序与输入一致
        class Flaky:
            def __call__(self, texts, hints=None, temperature=0.0):
                if "坏问题" in texts[0]:
                    raise TimeoutError("boom")
                return [f"EN({text})" for text in texts]

        service = self._service(Flaky())
        results = service.translate(["好问题一", "坏问题", "好问题二"])
        self.assertEqual([r.status for r in results],
                         [STATUS_OK, STATUS_PASSTHROUGH_ERROR, STATUS_OK])

    def test_stats_counts_statuses(self):
        service = self._service(_Provider())
        service.translate(["中文问题", "What is RRF?"])
        stats = service.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats[STATUS_OK], 1)
        self.assertEqual(stats[STATUS_SKIPPED], 1)

    def test_empty_input(self):
        self.assertEqual(self._service(_Provider()).translate([]), [])


class RealProviderTest(unittest.TestCase):
    """真跑 Ollama：术语原样保留必须 100%。"""

    @classmethod
    def setUpClass(cls):
        from src.rag.translate.service import ollama_provider

        try:
            provider = ollama_provider()
            provider(["连通性测试"])
            # 必须包 staticmethod：函数挂成类属性后，self.provider 会变成绑定方法，
            # 第一个参数被塞进测试实例 → provider 签名不匹配 → 静默降级成原文
            cls.provider = staticmethod(provider)
        except Exception as exc:   # noqa: BLE001
            raise unittest.SkipTest(f"Ollama 不可用：{exc}")

    def test_terms_kept_on_real_model(self):
        queries = ["DGX B300 的 8192 token 上限是多少？", "HNSW 是什么？"]
        service = TranslateService(self.provider)
        results = service.translate(queries)
        for result in results:
            self.assertEqual(result.missing, [], f"{result.source} → {result.text}")
            # 提示词不许漏进译文：实测拼在 user 消息里时模型会把它当正文翻出来
            self.assertNotIn("verbatim", result.text.lower(), result.text)
        self.assertEqual([r.status for r in results], [STATUS_OK, STATUS_OK])


if __name__ == "__main__":
    unittest.main()
