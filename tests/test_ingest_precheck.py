"""`ingest --precheck` 的行为测试（G4 的一部分 / 方案 P2）。

体检的价值在于**建库之前**给出结论：建库几百到几千秒，体检只解析+切分。
所以这里守两件事：

1. `--precheck-only` **完全不碰 Qdrant**（把地址指到死端口也必须绿）——
   否则「先体检再说」这条路对还没起服务的用户就是堵死的；
2. `--precheck` 是**闸门**：报告必须先打印出来，然后才轮到建库，并且
   `--precheck-strict` 不合格时直接中止（退出码 1）。
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._tokenizer_fixture import test_tokenizer_path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ingest.py"

LONG = "长段落用来触发切分，故意写长一点好让子块在段落中间被切开。" * 40
DOC = f"# 甲\n\n{LONG}\n"


def run(args, *, env_extra=None, timeout=180):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "RAG_TOKENIZER_PATH": test_tokenizer_path(),
            "QDRANT_URL": "http://127.0.0.1:1",     # 死端口：证明体检不连库
            "OLLAMA_BASE_URL": "http://127.0.0.1:1",
        }
    )
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


class IngestPrecheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        (self.dir / "a.md").write_text(DOC, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_precheck_only_never_touches_qdrant(self):
        result = run(["--dir", str(self.dir), "--precheck-only"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[chunk_quality]", result.stdout)
        self.assertNotIn("Qdrant", result.stdout + result.stderr)

    def test_precheck_only_reports_low_block_integrity(self):
        result = run(["--dir", str(self.dir), "--precheck-only"])
        self.assertIn("bi=", result.stdout)
        self.assertIn("被切断", result.stdout)

    def test_strict_aborts_with_exit_code(self):
        result = run(["--dir", str(self.dir), "--precheck-only", "--precheck-strict"])
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("体检不合格", result.stderr)

    def test_strict_passes_when_threshold_is_zero(self):
        result = run(
            ["--dir", str(self.dir), "--precheck-only", "--precheck-strict", "--precheck-threshold", "0"]
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_precheck_runs_before_ingest_and_then_stops_at_qdrant(self):
        """`--precheck`（不是 only）：报告先出，然后才轮到连库失败。"""
        result = run(["--dir", str(self.dir), "--precheck"])
        combined = result.stdout + result.stderr
        self.assertIn("[chunk_quality]", combined)
        self.assertIn("Qdrant", combined)
        self.assertLess(
            combined.index("[chunk_quality]"), combined.index("Qdrant"), "体检报告必须排在建库之前"
        )
        self.assertEqual(result.returncode, 1)

    def test_empty_dir_is_reported_but_not_fatal(self):
        empty = self.dir / "empty"
        empty.mkdir()
        result = run(["--dir", str(empty), "--precheck-only"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("空目录不算错误", result.stdout)


if __name__ == "__main__":
    unittest.main()
