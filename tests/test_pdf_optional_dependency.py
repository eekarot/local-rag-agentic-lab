"""PDF 解析依赖必须是**可选**的 —— 因为它不是 Apache-2.0。

`pymupdf` 的许可证是 **AGPL-3.0 / 商业双授权**（本机元数据实测：
`Dual Licensed - GNU AFFERO GPL 3.0 or Arti…`）。AGPL 是网络服务级 copyleft：
把一个 AGPL 库作为**硬依赖**分发，会让整个项目在法务上变成 AGPL，与本仓库
Apache-2.0（B11 定案：让别人能随便用）直接冲突。

所以三件事必须成立，本文件用用例锁住：

1. `import src.rag.ingestion.parser.pdf_parser` **不许**因为缺 pymupdf 就炸
   —— 只解析 md/txt 的用户完全不该被牵连；
2. 真的去解析 PDF 时，报错要说明「装什么」+「许可证是什么」；
3. 缺依赖时 md/txt 的解析与建库照常可用（由 `_load_pymupdf()` 的惰性边界保证）。
"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class OptionalImportTest(unittest.TestCase):
    def test_module_imports_without_pymupdf(self):
        """子进程里把 pymupdf 屏蔽掉，模块仍必须能 import。"""
        code = (
            "import sys;"
            "sys.modules['pymupdf'] = None;"
            "sys.modules['fitz'] = None;"
            "import src.rag.ingestion.parser.pdf_parser as m;"
            "print('ok', hasattr(m, 'PdfParser'))"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f"缺 pymupdf 时 import 就炸了：{result.stderr[-400:]}")
        self.assertIn("ok True", result.stdout)

    def test_parsing_pdf_without_pymupdf_gives_actionable_error(self):
        from src.rag.ingestion import errors
        from src.rag.ingestion.parser import pdf_parser

        with mock.patch.object(pdf_parser, "_PYMUPDF", None), \
                mock.patch.dict(sys.modules, {"pymupdf": None, "fitz": None}):
            with self.assertRaises(errors.IngestionError) as ctx:
                pdf_parser.load_pymupdf()
        message = str(ctx.exception)
        self.assertIn("pymupdf", message)
        self.assertIn("pip install", message, "必须告诉用户装什么")
        self.assertIn("AGPL", message, "必须把许可证说出来：它不是 Apache-2.0")

    def test_pymupdf_reports_its_license_in_notice(self):
        """NOTICE 必须写明这条，否则别人以为整个仓库只有一个许可证。"""
        notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("PyMuPDF", notice)
        self.assertIn("AGPL", notice)


if __name__ == "__main__":
    unittest.main()
