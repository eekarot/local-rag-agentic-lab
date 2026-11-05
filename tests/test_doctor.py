"""`python -m src.doctor`：一条命令自检 4 项外部依赖 + 索引，每项给可复制的修复命令。

**最重要的用例是最后那几个**：词表缺失时，`doctor` 仍然必须把 Qdrant / Ollama / 父块
的结果报出来。真实场景就是「词表没下 → 用户只看得到一条与词表无关的堆栈」，
而这恰恰是 doctor 存在的理由。所以装配必须是**逐项惰性**的，任何一件炸掉只影响它自己那一行。
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from src.rag.ops.doctor import DoctorDeps, main, run


class FakeCounter:
    def __init__(self, error=None):
        self.error = error

    def count(self, text):
        if self.error:
            raise self.error
        return len(text)


class FakeClient:
    def __init__(self, count=100, error=None):
        self.count_value = count
        self.error = error

    def get_collections(self):
        if self.error:
            raise self.error

        class Item:
            name = "rag_chunks_v1"

        class Resp:
            collections = [Item()]

        return Resp()

    def count(self, collection):
        class Resp:
            count = self.count_value

        return Resp()


class FakeReranker:
    def __init__(self, enabled=True):
        class Config:
            pass

        self.config = Config()
        self.config.enabled = enabled


_DATA_DIR = Path(tempfile.mkdtemp(prefix="doctor_data_"))
(_DATA_DIR / "sample.md").write_text("# 样例" + chr(10) + chr(10) + "用于 probe_data_dir 的最小语料。", encoding="utf-8")


def make_deps(**overrides):
    """默认全绿；测试按需把某一项换成坏的。"""
    values = dict(
        client_factory=lambda: FakeClient(),
        embed=lambda texts: [[0.0] * 1024 for _ in texts],
        list_models=lambda: ["qwen3.5:9b", "bge-m3:latest"],
        model="qwen3.5:9b",
        embed_model="bge-m3",
        counter_factory=lambda: FakeCounter(),
        reranker_factory=lambda: FakeReranker(),
        parents_factory=lambda: {"par_a": {}, "par_b": {}},
        data_dir=_DATA_DIR,
    )
    values.update(overrides)
    return DoctorDeps(**values)


class RunTest(unittest.TestCase):
    def test_all_green(self):
        report = run(make_deps())
        self.assertEqual(report.status, "ok")
        self.assertEqual(len(report.checks), 7, "6 项必查 + data_dir")

    def test_broken_tokenizer_does_not_hide_other_checks(self):
        from src.rag.ingestion.errors import TokenizerUnavailable

        def broken_counter():
            raise TokenizerUnavailable("缺少 tokenizer.json\n  curl -sL -o x http://example")

        report = run(make_deps(counter_factory=broken_counter))
        names = {c.name for c in report.checks}
        self.assertEqual(len(names), 7, "七项必须都出现，不能一坏就停")
        tokenizer = report.of("tokenizer")
        self.assertFalse(tokenizer.ok)
        self.assertIn("curl", tokenizer.fix)
        self.assertEqual(report.status, "down")

    def test_broken_qdrant_client_still_reports_other_checks(self):
        def broken_client():
            raise ConnectionError("connection refused")

        report = run(make_deps(client_factory=broken_client))
        self.assertFalse(report.of("qdrant").ok)
        self.assertTrue(report.of("embed_model").ok)
        self.assertTrue(report.of("parents").ok)

    def test_broken_parents_factory_is_reported(self):
        def boom():
            raise FileNotFoundError("documents.sqlite3 不存在")

        report = run(make_deps(parents_factory=boom))
        parents = report.of("parents")
        self.assertFalse(parents.ok)
        self.assertIn("documents.sqlite3", parents.detail)

    def test_data_dir_is_reported_separately_not_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run(make_deps(data_dir=Path(tmp)))
        self.assertIn("data_dir", {c.name for c in report.checks})
        self.assertEqual(report.status, "ok", "空 data/ 不影响已在跑的检索")


class CliTest(unittest.TestCase):
    def test_human_output_lists_every_line_and_the_header(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main([], deps=make_deps())
        out = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("[ok]", out)
        for name in ("qdrant", "embed_model", "chat_model", "tokenizer", "reranker", "parents", "data_dir"):
            self.assertIn(name, out)

    def test_down_returns_nonzero_for_scripting(self):
        buffer, err = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(err):
            code = main([], deps=make_deps(counter_factory=lambda: FakeCounter(error=RuntimeError("缺"))))
        self.assertEqual(code, 1)

    def test_degraded_returns_zero_but_warns(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main([], deps=make_deps(list_models=lambda: ["bge-m3:latest"]))
        self.assertEqual(code, 0, "只是不能生成，退出码不该是失败")
        self.assertIn("degraded", buffer.getvalue())

    def test_json_mode_keeps_stdout_pure(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["--json"], deps=make_deps())
        payload = json.loads(buffer.getvalue())          # stdout 必须是纯 JSON
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["checks"]), 7)

    def test_fix_commands_are_printed_inline(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main([], deps=make_deps(list_models=lambda: []))
        out = buffer.getvalue()
        self.assertIn("修复：ollama pull qwen3.5:9b", out)


if __name__ == "__main__":
    unittest.main()
