"""翻译出来的英文 query 必须真的进到 `dense_en` / `lexical_en`（N47）。

BUG 实证（2026-11-06）：`/ask` 与 `/retrieve` 都会先调翻译服务拿到 `query_en`，
但 `retrieve_with_meta()` **根本没有这个参数**，`answer_query()` 也没往下传。
编排层（`orchestrator`）明确写着「`dense_en` / `lexical_en` 吃英文 `query_en`」，
并且缺失时会打日志 `FALLBACK_NOTE: query_en 缺失，回退中文原句（该路结果应视为噪声源）`
—— 也就是说**这两路一直在吃中文原句**：翻译花了 LLM 调用，结果只用在「提示词语言」
和响应里回显的 `question_en` 字段上。

后果不是「少一点效果」，而是**评估口径与生产不一致**：`scripts/eval_bench.py` 压根没翻译，
量的就是「中文原句喂英文索引」。这一轮扩题后做 C0c（中→英），必须先把这个接线修对，
否则「谁拖后腿」的结论会指错人。

用例三条：`retrieve_with_meta` 传下去 / `answer_query` 传下去 / 不给就保持旧行为（回退中文）。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Route:
    def __init__(self, ids):
        self.ids = ids
        self.enabled = True


class _Result:
    def __init__(self):
        self.routes = {"dense_zh": _Route(["par_0"]), "dense_en": _Route(["par_0"])}
        self.degraded = None


class _RecordingRetriever:
    """把每次 `retrieve()` 收到的 (query, query_en) 记下来。"""

    def __init__(self):
        self.calls = []

    def retrieve(self, query, k, query_en=None):
        self.calls.append({"query": query, "query_en": query_en, "k": k})
        return _Result()


class RetrieveWithMetaTest(unittest.TestCase):
    def setUp(self):
        from src.rag.ask.pipeline import retrieve_with_meta
        from src.rag.ingestion.tokenizer import TokenCounter

        self.retrieve = retrieve_with_meta
        self.counter = TokenCounter()
        self.parents = {"par_0": {"text": "正文", "content": "正文", "title": "t",
                                  "section_path": "s"}}

    def test_query_en_is_forwarded(self):
        retriever = _RecordingRetriever()
        self.retrieve("节点失联多久算 NotReady？", retriever=retriever, parents=self.parents,
                      counter=self.counter, reranker=None, pool=1, rerank=False,
                      query_en="How long before a node is marked NotReady?")
        self.assertEqual(retriever.calls[0]["query_en"],
                         "How long before a node is marked NotReady?")

    def test_missing_query_en_still_works(self):
        """不给就是旧行为（编排层会回退中文原句并打 note）—— 不许因为少了参数就崩。"""
        retriever = _RecordingRetriever()
        self.retrieve("问题", retriever=retriever, parents=self.parents,
                      counter=self.counter, reranker=None, pool=1, rerank=False)
        self.assertIsNone(retriever.calls[0]["query_en"])


class AnswerQueryTest(unittest.TestCase):
    def test_answer_query_forwards_query_en_to_retriever(self):
        from src.rag.ask.pipeline import answer_query
        from src.rag.ingestion.tokenizer import TokenCounter

        retriever = _RecordingRetriever()
        answer_query("问题", retriever=retriever,
                     parents={"par_0": {"text": "正文", "content": "正文", "title": "t",
                                        "section_path": "s"}},
                     counter=TokenCounter(), reranker=None, ask=None,
                     query_en="the english query", k=1, pool=1, rerank=False)
        self.assertEqual(retriever.calls[0]["query_en"], "the english query")


class EvalQueryModeTest(unittest.TestCase):
    """`eval_bench --query-en`：三种口径要显式区分，不许含糊。"""

    def setUp(self):
        from scripts.eval_bench import query_for

        self.query_for = query_for
        self.row = {"qid": "k8s-x-001", "question": "中文问题？", "query_en": "English query?"}

    def test_none_mode_uses_the_visible_question(self):
        self.assertEqual(self.query_for(self.row, mode="none"), ("中文问题？", None))

    def test_field_mode_uses_the_handwritten_english_query(self):
        self.assertEqual(self.query_for(self.row, mode="field"),
                         ("中文问题？", "English query?"))

    def test_field_mode_falls_back_when_missing(self):
        row = {"qid": "k8s-n-001", "question": "中文问题？", "query_en": ""}
        self.assertEqual(self.query_for(row, mode="field"), ("中文问题？", None))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.query_for(self.row, mode="auto")


if __name__ == "__main__":
    unittest.main()


class TranslateModeTest(unittest.TestCase):
    """`--query-en translate`：走和 `/ask` 同一套翻译服务，而不是「假装翻译很完美」。

    三档分工（少一档就会含糊）：
      none      = 旧口径 / 翻译缺失的降级态（英文索引吃中文原句）
      field     = 手写 query_en，翻译**上界**（翻译永远正确能到哪）
      translate = 真实翻译（生产口径）
    """

    def setUp(self):
        from scripts.eval_bench import query_for

        self.query_for = query_for
        self.row = {"qid": "k8s-x-001", "question": "中文问题？", "query_en": "English query?"}

    def test_translate_mode_uses_the_translate_fn(self):
        seen = []

        def fake_translate(text):
            seen.append(text)
            return "translated query"

        question, english = self.query_for(self.row, mode="translate", translate=fake_translate)
        self.assertEqual(question, "中文问题？")
        self.assertEqual(english, "translated query")
        self.assertEqual(seen, ["中文问题？"])

    def test_translate_mode_without_fn_is_an_error(self):
        with self.assertRaises(ValueError):
            self.query_for(self.row, mode="translate")

    def test_translate_failure_degrades_to_none_not_to_silent_garbage(self):
        question, english = self.query_for(self.row, mode="translate",
                                           translate=lambda _text: None)
        self.assertEqual(question, "中文问题？")
        self.assertIsNone(english)


class TranslateProviderProtocolTest(unittest.TestCase):
    """`TranslateService` 的 provider 协议是 `(texts: list[str]) -> list[str]`（N48）。

    BUG 实证（2026-11-06）：`app.py` 手写的是 `lambda prompt: chat([...content: prompt...])`。
    服务调用 `provider(texts, hints=...)` → TypeError → 退回 `provider(texts)` →
    `content` 收到一个**列表** → 调用失败 → 每次 4–5 ms 返回空 → 状态 `passthrough_error`，
    `text` 原样回填中文。于是：

    * `/ask` 的 `query_en` 永远是中文原句（编排层还以为拿到了英文）；
    * 「翻译」这一步花了代码、花了缓存表，**一次都没真正翻过**。

    这类 bug 的共性是「组件单独都能跑通，接口名字对得上，但协议对不上」，
    所以用例直接钉协议：provider 必须收到**字符串列表**、必须带 SYSTEM_PROMPT、
    必须返回**同长度**列表。
    """

    def test_provider_from_chat_follows_the_protocol(self):
        from src.rag.translate.service import SYSTEM_PROMPT, TranslateService, provider_from_chat

        calls = []

        def fake_chat(messages, **kwargs):
            calls.append((messages, kwargs))
            return '{"translation": "how long before a node is NotReady"}'

        service = TranslateService(provider=provider_from_chat(fake_chat))
        results = service.translate(["节点失联多久算 NotReady？"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "ok", f"降级了：{results[0].as_dict()}")
        self.assertEqual(results[0].text, "how long before a node is NotReady")

        messages, kwargs = calls[0]
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertIsInstance(messages[1]["content"], str, "content 必须是字符串，不能是列表")
        self.assertIn("NotReady", messages[1]["content"])
        self.assertEqual(kwargs.get("fmt"), "json")

    def test_multiple_texts_keep_one_reply_per_text(self):
        from src.rag.translate.service import TranslateService, provider_from_chat

        def fake_chat(messages, **kwargs):
            return '{"translation": "english version"}'

        service = TranslateService(provider=provider_from_chat(fake_chat))
        results = service.translate(["第一问？", "第二问？"])
        self.assertEqual([r.status for r in results], ["ok", "ok"])

    def test_api_and_eval_build_the_provider_with_the_shared_factory(self):
        """AST 守卫：不许再手写 `lambda prompt:` —— 那是这个 bug 的形状。"""
        import ast

        for path in ("src/rag/api/app.py", "scripts/eval_bench.py"):
            tree = ast.parse(Path(ROOT, path).read_text(encoding="utf-8"))
            providers = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "TranslateService":
                    for kw in node.keywords:
                        if kw.arg == "provider":
                            providers.append(kw.value)
            self.assertTrue(providers, f"{path} 里没找到 TranslateService(provider=...)")
            for value in providers:
                self.assertIsInstance(value, ast.Call, f"{path}：provider 必须是工厂调用")
                name = getattr(value.func, "id", getattr(value.func, "attr", ""))
                self.assertIn(name, {"provider_from_chat", "ollama_provider"},
                              f"{path}：provider 必须由共享工厂构造，实际是 {name}")


class TranslateStatsTest(unittest.TestCase):
    """基线里必须留下「这次真的翻了几条」—— 翻译降级成 passthrough 时口径就变了。

    这不是锦上添花：`passthrough_error` 时 `text` 会**回填中文原文**，
    此时 `--query-en translate` 悄悄退化成 `--query-en none`，
    基线数字的含义跟着变。不记下来，下次就没人知道基线是哪一档。
    """

    def test_adapter_exposes_service_stats(self):
        from src.rag.translate.service import TranslateService, provider_from_chat
        from scripts.eval_bench import build_translate_fn

        adapter = build_translate_fn(chat=lambda messages, **kwargs: '{"translation": "x"}')
        self.assertTrue(callable(adapter))
        self.assertTrue(hasattr(adapter, "stats"), "adapter 必须能报 stats")
        # 直接构造一遍，确认 stats 的形状是 service 给的
        service = TranslateService(provider=provider_from_chat(lambda m, **k: "{}"))
        self.assertIn("success_rate", service.stats())

    def test_stats_count_real_translations(self):
        from scripts.eval_bench import build_translate_fn

        adapter = build_translate_fn(chat=lambda messages, **kwargs: '{"translation": "hello"}')
        self.assertEqual(adapter("中文问题？"), "hello")
        stats = adapter.stats()
        self.assertEqual(stats["ok"] + stats["cached"], 1)
        self.assertEqual(stats["passthrough_error"], 0)
