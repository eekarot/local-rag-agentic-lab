"""`scripts/chunk_quality.py` 的行为测试（G4 / 方案 P2）。

这个命令面向的是「索引自己文档、没有任何金标」的用户，所以它的**失败方式**和
成功一样重要：缺词表不许甩堆栈、目录为空不算错误、未知指标要报可选值、
`--strict` 要与阈值一致。这几条在这里各有一个用例。
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._tokenizer_fixture import test_tokenizer_path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "chunk_quality.py"

# 必须长到超过子块上限（500 token）：短段落根本切不开，BI 永远等于 1，就测不出差别
LONG_PARAGRAPH = "这一段故意写长一点，好让子块在段落中间被切开，从而让结构完整度低于 1。" * 24

DOC_A = """# 甲文档

第一段，交代背景，内容短一些。

## 小节

{long}

第二段，结尾。
""".format(long=LONG_PARAGRAPH)

DOC_B = """# 乙文档

只有一段很短的内容。
"""


def run(args, *, env_extra=None, timeout=180):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["RAG_TOKENIZER_PATH"] = test_tokenizer_path()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


class ChunkQualityCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        (self.dir / "a.md").write_text(DOC_A, encoding="utf-8")
        (self.dir / "b.md").write_text(DOC_B, encoding="utf-8")
        self.json_path = self.dir / "out" / "report.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_reports_metrics_and_writes_json(self):
        result = run(["--dir", str(self.dir), "--json", str(self.json_path)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("体检 2 篇", result.stdout)
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["docs"], 2)
        self.assertEqual(len(payload["per_doc"]), 2)
        self.assertIsNotNone(payload["sc_mean"])
        self.assertIsNotNone(payload["bi_mean"])
        self.assertEqual(payload["metrics"], ["sc", "bi"])

    def test_plain_markdown_has_no_text_loss(self):
        """体检同时是不变量检查：正常语料不许报丢字。"""
        result = run(["--dir", str(self.dir), "--json", str(self.json_path)])
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["gap_docs"], 0)
        self.assertNotIn("丢了字", result.stdout)

    def test_strict_exit_code_follows_threshold(self):
        """阈值 1.0 时必然有文档不合格（长段落会被从中切断）→ 退出码 1。"""
        strict = run(["--dir", str(self.dir), "--threshold", "1.0", "--strict"])
        self.assertEqual(strict.returncode, 1, strict.stdout)
        lenient = run(["--dir", str(self.dir), "--threshold", "0.0", "--strict"])
        self.assertEqual(lenient.returncode, 0, lenient.stdout)

    def test_size_compliance_uses_upper_bound(self):
        """`--max-tokens` 压到 100 时几乎全部越界：SC 要跟着降下来。"""
        result = run(["--dir", str(self.dir), "--max-tokens", "100", "--json", str(self.json_path)])
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertLess(payload["sc_mean"], 1.0)

    def test_missing_tokenizer_prints_fix_not_traceback(self):
        result = run(
            ["--dir", str(self.dir)],
            env_extra={"RAG_TOKENIZER_PATH": str(self.dir / "无此词表.json")},
        )
        self.assertEqual(result.returncode, 1)
        combined = result.stdout + result.stderr
        self.assertIn("词表", combined)
        self.assertIn("curl", combined, "缺词表要给一条可复制的下载命令")
        self.assertIn("bge-m3", combined)
        self.assertNotIn("Traceback", combined, "缺依赖不许甩堆栈")

    def test_empty_dir_is_not_an_error(self):
        empty = self.dir / "empty"
        empty.mkdir()
        result = run(["--dir", str(empty)])
        self.assertEqual(result.returncode, 0)
        self.assertIn("空目录不算错误", result.stdout)

    def test_unknown_metric_lists_choices(self):
        result = run(["--dir", str(self.dir), "--metrics", "sc,nope"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("sc, bi, gap", result.stderr)

    def test_missing_dataset_points_at_fetch(self):
        result = run(["--dataset", "no-such-dataset"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("fetch_dataset.py", result.stderr)


if __name__ == "__main__":
    unittest.main()
