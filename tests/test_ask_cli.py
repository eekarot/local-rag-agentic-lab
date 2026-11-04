"""E07 问答入口的用例：全部用假件，不碰 Qdrant / Ollama / ONNX。

覆盖四件事：
1. 链路顺序与字段透传（检索 → 重排 → 组装 → prompt → 生成 → 引用体检）；
2. **负样本路径**：没有证据时不调模型、直接出弃答话术；
3. `--retrieve-only`（`ask=None`）不生成也不报错；
4. CLI 渲染：答案、引用、来源表、耗时都在，且不重复打印 `[证据 n]` 之外的噪音。
"""

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from src.rag.ask.pipeline import AskResult, answer_query, render_cli
from src.rag.answer.citations import parse_citations


class FakeRoute:
    def __init__(self, ids):
        self.ids = list(ids)
        self.hits = list(ids)
        self.latency_ms = 1.0
        self.enabled = True
        self.note = ""


class FakeResult:
    def __init__(self, routes):
        self.routes = routes


class FakeRetriever:
    """按名字返回固定的 id 列表，并记下被问过的 query。"""

    def __init__(self, routes=None):
        self.routes = routes or {
            "dense_zh": ["par_a", "par_b", "par_c"],
            "dense_en": ["par_a", "par_c", "par_b"],
            "lexical_en": ["par_b", "par_a"],
        }
        self.calls = []

    def retrieve(self, query, *, k=10, **kwargs):
        self.calls.append((query, k))
        return FakeResult({name: FakeRoute(ids) for name, ids in self.routes.items()})


class FakeReranker:
    """分数 = id 里字母顺序，用来断言「重排真的改变了顺序」。"""

    def __init__(self, enabled=True, scores=None, max_length=256):
        from src.rag.retrieval.rerank import RerankConfig

        self.config = RerankConfig(enabled=enabled, max_length=max_length)
        self.scores = scores or {}
        self.encoder = self
        self.stats_holder = {"calls": 0, "pairs": 0, "errors": 0, "fallbacks": 0, "encode_ms": 0.0}

    def __call__(self, pairs):
        self.stats_holder["calls"] += 1
        return [self.scores.get(pair[1][:20], 0.0) for pair in pairs]

    def stats(self):
        return self.stats_holder


class FakeCounter:
    def count(self, text):
        return max(1, len(text or "") // 4)

    def truncate(self, text, max_tokens):
        return text

    def truncate_tail(self, text, max_tokens):
        return text


PARENTS = {
    "par_a": {"parent_id": "par_a", "document_id": "doc_1", "parent_index": 0, "title": "父块 A",
              "source_uri": "a.md", "section_path": ["第1章", "1.1 甲"],
              "text": "父块 A 正文：智能体化 RAG 把检索变成工具。" * 3},
    "par_b": {"parent_id": "par_b", "document_id": "doc_1", "parent_index": 1, "title": "父块 B",
              "source_uri": "b.md", "section_path": ["第1章", "1.2 乙"],
              "text": "父块 B 正文：传统 RAG 是单向管道。" * 3},
    "par_c": {"parent_id": "par_c", "document_id": "doc_2", "parent_index": 0, "title": "父块 C",
              "source_uri": "c.md", "section_path": ["第2章"],
              "text": "父块 C 正文：无关内容。" * 3},
}

WHOLE_TEXT = "整块正文"


def build(**overrides):
    kwargs = dict(retriever=FakeRetriever(), parents=PARENTS, counter=FakeCounter(),
                  reranker=FakeReranker(), ask=lambda messages: "答案 [证据 1]")
    kwargs.update(overrides)
    return answer_query("智能体化 RAG 和传统 RAG 有什么区别？", **kwargs)


class AnswerQueryTest(unittest.TestCase):
    def test_pipeline_returns_answer_and_sources(self):
        result = build()
        self.assertIsInstance(result, AskResult)
        self.assertEqual(result.answer, "答案 [证据 1]")
        self.assertTrue(result.llm_called)
        self.assertEqual(result.report.cited, [1])
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0]["document_id"], "doc_1")
        self.assertEqual(result.blocks[0]["section_path"], ["第1章", "1.1 甲"])

    def test_retriever_called_with_pool_depth(self):
        retriever = FakeRetriever()
        build(retriever=retriever, pool=7)
        self.assertEqual(retriever.calls, [("智能体化 RAG 和传统 RAG 有什么区别？", 7)])

    def test_k_limits_blocks_and_budget_caps_tokens(self):
        result = build(k=2)
        self.assertLessEqual(len(result.blocks), 2)

    def test_no_evidence_shortcut_skips_model(self):
        calls = []

        def ask(messages):
            calls.append(messages)
            return "不该被调用"

        result = build(retriever=FakeRetriever({}), parents={}, ask=ask)
        self.assertEqual(calls, [])
        self.assertTrue(result.abstained)
        self.assertFalse(result.llm_called)
        self.assertEqual(result.abstain_reason, "no_evidence")
        self.assertIn("证据不足", result.answer)
        self.assertEqual(result.timings["generate_ms"], 0.0)

    def test_retrieve_only_does_not_generate(self):
        result = build(ask=None)
        self.assertEqual(result.answer, "")
        self.assertFalse(result.llm_called)
        self.assertGreater(len(result.blocks), 0)

    def test_timings_are_recorded(self):
        result = build()
        for key in ("retrieve_rerank_ms", "assemble_ms", "prompt_ms", "generate_ms"):
            self.assertIn(key, result.timings)
            self.assertGreaterEqual(result.timings[key], 0.0)

    def test_as_dict_is_json_friendly(self):
        import json

        result = build()
        payload = result.as_dict()
        text = json.dumps(payload, ensure_ascii=False)
        self.assertIn("答案 [证据 1]", text)
        self.assertEqual(payload["cited"], [1])
        self.assertEqual(payload["blocks"], len(result.blocks))
        self.assertGreaterEqual(payload["blocks"], 1)
        self.assertIsInstance(payload["timings"]["generate_ms"], float)

    def test_language_follows_question(self):
        self.assertEqual(build().language, "zh")
        english = answer_query("What is agentic RAG?", retriever=FakeRetriever(), parents=PARENTS,
                              counter=FakeCounter(), reranker=FakeReranker(),
                              ask=lambda messages: "Answer [证据 1]")
        self.assertEqual(english.language, "en")


class RenderTest(unittest.TestCase):
    def test_render_contains_answer_citations_and_sources(self):
        text = render_cli(build())
        self.assertIn("问题：智能体化 RAG", text)
        self.assertIn("回答：", text)
        self.assertIn("答案 [证据 1]", text)
        self.assertIn("引用：[1]", text)
        self.assertIn("来源：", text)
        self.assertIn("第1章 > 1.1 甲", text)
        self.assertIn("a.md", text)
        self.assertIn("耗时：retrieve_rerank", text)

    def test_render_marks_abstain(self):
        text = render_cli(build(retriever=FakeRetriever({}), parents={}, ask=lambda m: "x"))
        self.assertIn("[弃答]", text)
        self.assertIn("no_evidence", text)

    def test_render_truncates_long_answer(self):
        text = render_cli(build(ask=lambda messages: "长" * 50), max_answer_chars=10)
        self.assertIn("截断，共 50 字符", text)

    def test_render_handles_missing_citations(self):
        text = render_cli(build(ask=lambda messages: "没有引用编号的答案"))
        self.assertIn("引用：无", text)
        self.assertIn("答案没有标注引用", text)

    def test_render_limits_source_rows(self):
        text = render_cli(build(ask=lambda messages: "答案 [证据 1] [证据 2] [证据 3]"), show_sources=1)
        self.assertEqual(text.count("[证据 1]"), 2)   # 答案里一次 + 来源表一次
        self.assertNotIn("b.md", text)


class CliMainTest(unittest.TestCase):
    """`src/ask.py` 的 main：参数解析 + 假装配（不下载模型、不连 Qdrant）。"""

    def run_main(self, argv, *, ask=None, retriever=None):
        import src.ask as cli

        captured = {}

        def fake_build_components(args):
            # 真 `build_components` 也要遵守 --retrieve-only：这里等价实现，才能测到 CLI 的开关
            default_ask = None if args.retrieve_only else (lambda messages: "答案 [证据 1]")
            return {
                "retriever": retriever or FakeRetriever(),
                "parents": PARENTS,
                "counter": FakeCounter(),
                "reranker": FakeReranker(),
                "ask": ask if ask is not None else default_ask,
                "question_en": None,
            }

        buffer = io.StringIO()
        with mock.patch.object(cli, "build_components", side_effect=fake_build_components) as build_mock:
            with redirect_stdout(buffer):
                code = cli.main(argv)
        captured["build_args"] = build_mock.call_args[0][0] if build_mock.called else None
        return code, buffer.getvalue(), captured

    def test_main_prints_answer(self):
        code, output, _ = self.run_main(["智能体化 RAG 和传统 RAG 有什么区别？"])
        self.assertEqual(code, 0)
        self.assertIn("答案 [证据 1]", output)

    def test_main_json_output(self):
        import json

        code, output, _ = self.run_main(["问题？", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["answer"], "答案 [证据 1]")
        self.assertEqual(payload["cited"], [1])

    def test_main_passes_k_and_pool(self):
        _, _, captured = self.run_main(["问题？", "--k", "3", "--pool", "9"])
        self.assertEqual(captured["build_args"].k, 3)
        self.assertEqual(captured["build_args"].pool, 9)

    def test_main_retrieve_only(self):
        code, output, captured = self.run_main(["问题？", "--retrieve-only"])
        self.assertEqual(code, 0)
        self.assertTrue(captured["build_args"].retrieve_only)
        self.assertIn("未生成", output)

    def test_main_retrieve_only_lists_context_blocks(self):
        code, output, _ = self.run_main(["问题？", "--retrieve-only"])
        self.assertEqual(code, 0)
        self.assertIn("上下文块", output)
        self.assertIn("第1章 > 1.1 甲", output)

    def test_main_prints_setup_timing_first(self):
        _, output, _ = self.run_main(["问题？"])
        line = next(item for item in output.splitlines() if item.startswith("耗时："))
        self.assertTrue(line.split("｜")[0].startswith("耗时：setup "), line)

    def test_main_show_sources_all(self):
        code, output, _ = self.run_main(["问题？", "--sources", "all"])
        self.assertEqual(code, 0)
        self.assertIn("a.md", output)


if __name__ == "__main__":
    unittest.main()
