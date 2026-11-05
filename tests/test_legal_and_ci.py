"""A4：法务与 CI 的**可验证部分**（许可证文件、NOTICE 归属、workflow 不联网）。

这些不是「文档检查」摆样子——每一条都对应一个真会出事的场景：

- `LICENSE` 缺全文 → 别人不敢用（B11 定案的 Apache-2.0 是**专利授权条款**撑起来的，
  只写一句「Apache-2.0」不产生任何授权）；
- `NOTICE` 漏掉 PyMuPDF 的 AGPL → 有人以为整个仓库只有一个许可证，把它塞进闭源产品；
- `.github/workflows/tests.yml` 里偷偷装了慢依赖 / 起了服务 → CI 从 1 分钟变 10 分钟，
  然后被禁用（那才是真正的「没有 CI」）。
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LicenseFileTest(unittest.TestCase):
    def test_license_is_the_full_apache_text(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0, January 2004", text)
        for clause in ("Grant of Patent License", "Redistribution", "Limitation of Liability",
                       "AS IS", "APPENDIX"):
            self.assertIn(clause, text, f"LICENSE 缺条款：{clause}")
        self.assertGreater(len(text), 9000, "这不像 Apache-2.0 全文（约 11 KB）")

    def test_notice_covers_runtime_assets_and_the_agpl_exception(self):
        notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
        for name in ("bge-m3", "mmarco-mMiniLMv2", "Qwen"):
            self.assertIn(name, notice, f"NOTICE 漏了运行时下载的模型：{name}")
        self.assertIn("AGPL", notice)
        self.assertIn("PyMuPDF", notice)
        self.assertIn("不携带任何语料", notice, "B13 定案要在法务上写明")

    def test_pymupdf_is_not_a_hard_dependency(self):
        runtime = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        optional = (ROOT / "requirements-pdf.txt").read_text(encoding="utf-8")
        self.assertNotRegex(runtime, r"(?m)^pymupdf", "PyMuPDF 是 AGPL，不能作为默认依赖")
        self.assertRegex(optional, r"(?m)^pymupdf")
        self.assertIn("AGPL", optional)


class ContributingTest(unittest.TestCase):
    def test_contributing_states_the_non_negotiables(self):
        text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        for must in ("Apache-2.0", "RAG_RUN_SLOW=1", "<60 s", "RED", "NOTICE"):
            self.assertIn(must, text, f"CONTRIBUTING 缺关键约定：{must}")


class WorkflowTest(unittest.TestCase):
    """CI 只许跑默认套件：装了慢依赖或起了服务，就等于把 CI 提前判死刑。"""

    def setUp(self):
        raw = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        # 只看**生效的行**：注释里说明「我们不这么做」是好事，不该被判违规
        self.text = chr(10).join(line for line in raw.splitlines()
                                 if not line.lstrip().startswith("#"))

    def test_runs_the_default_suite(self):
        self.assertIn("python -m unittest discover -s tests", self.text)

    def test_never_enables_slow_tests(self):
        self.assertNotIn("RAG_RUN_SLOW: 1", self.text)
        self.assertNotRegex(self.text, r"RAG_RUN_SLOW=1")

    def test_does_not_install_agpl_pdf_stack(self):
        # 注释里提到文件名是允许的，装它不行
        self.assertNotRegex(self.text, r"(?m)^\s*pip install[^\n]*requirements-pdf")
        self.assertNotRegex(self.text, r"(?m)^\s*-\s*run:.*requirements-pdf\.txt$")

    def test_points_at_dead_ports_so_network_use_fails_loudly(self):
        """万一有人加了联网用例，CI 会立刻红，而不是「恰好有服务」而蒙混过关。"""
        self.assertIn("QDRANT_URL: http://127.0.0.1:1", self.text)
        self.assertIn("OLLAMA_BASE_URL: http://127.0.0.1:1", self.text)

    def test_runs_on_three_platforms(self):
        for os_name in ("ubuntu-latest", "windows-latest", "macos-latest"):
            self.assertIn(os_name, self.text)


class ChangelogTest(unittest.TestCase):
    """发版时最容易漂的就是版本号：代码说 0.1.0、CHANGELOG 还写着 0.0.x。

    所以让用例去比：`src/rag/api/app.py` 的 `API_VERSION` 必须出现在 CHANGELOG 里。
    """

    def test_changelog_covers_the_current_api_version(self):
        from src.rag.api.app import API_VERSION

        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"## v{API_VERSION}", text,
                      f"CHANGELOG 没有 v{API_VERSION} 这一节（改了版本号就补一节）")

    def test_changelog_says_what_is_not_included(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        for must in ("不包含", "已知限制", "语料"):
            self.assertIn(must, text, f"发版说明必须写清边界：缺 {must}")


if __name__ == "__main__":
    unittest.main()
