"""`python -m src.rag.ingestion.parser` 的语料级验收命令测试。

D02/D03 的验收都是在**真实语料**上算比率（section_path 非空率 100%、
空 block 数 0），所以需要一个能一行跑完的命令，而不是靠肉眼看单测。
"""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag.ingestion.parser.cli import main

MARKDOWN = "# 标题\n\n正文段落。\n\n```python\n# 注释不是标题\nx = 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
PLAIN = "第一段。\n\n第二段。\n"


def run(argv):
    out = io.StringIO()
    with redirect_stdout(out):
        code = main(argv)
    return code, out.getvalue()


class CheckCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a.md").write_text(MARKDOWN, encoding="utf-8")
        (self.root / "b.txt").write_text(PLAIN, encoding="utf-8")
        (self.root / "skip.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_check_reports_rates_over_directory(self):
        code, out = run(["check", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("文件 2", out, "只应解析已注册后缀，跳过 .json")
        self.assertIn("section_path 非空率 100.00%", out)
        self.assertIn("空 block 0", out)

    def test_check_reports_block_type_breakdown(self):
        code, out = run(["check", str(self.root)])
        self.assertEqual(code, 0)
        for block_type in ("heading", "paragraph", "code", "table"):
            self.assertIn(block_type, out)

    def test_check_on_missing_directory_returns_2(self):
        code, out = run(["check", str(self.root / "nope")])
        self.assertEqual(code, 2)
        self.assertIn("目录不存在", out)

    def test_check_warns_when_nothing_parsed(self):
        empty = self.root / "empty"
        empty.mkdir()
        (empty / "note.bin").write_text("x", encoding="utf-8")
        code, out = run(["check", str(empty)])
        self.assertEqual(code, 2)
        self.assertIn("没有可解析的文件", out)


class DumpCommandTest(unittest.TestCase):
    def test_dump_prints_one_json_per_block(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.md"
            path.write_text(MARKDOWN, encoding="utf-8")
            code, out = run(["dump", str(path)])
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in out.splitlines() if line.strip()]
            self.assertEqual([r["block_type"] for r in rows], ["heading", "paragraph", "code", "table"])
            self.assertEqual(rows[0]["text"], "标题")

    def test_dump_unsupported_suffix_returns_2(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.bin"
            path.write_text("x", encoding="utf-8")
            code, out = run(["dump", str(path)])
            self.assertEqual(code, 2)
            self.assertIn("不支持", out)


if __name__ == "__main__":
    unittest.main()
