"""A3 / E03：测试分层必须**由用例守护**，否则下一轮改动就会悄悄破掉。

发布门槛写着「默认套件 <60 s 且不联网」，而这件事最容易被无声破坏：
某天有人在 `test_xxx.py` 顶部加一行 `SERVER = _server_available()`，全套件就多 3 s
并重新依赖 Qdrant；他不会觉得这算「改坏了」，因为没有任何东西报警。

这里用 AST 只查一件最要命的事：**导入期**有没有在探真实服务（那会让「跳过」也变慢、
让默认套件重新依赖外部进程）。函数体内的调用不管——那些才是用假件该覆盖的。
"""

import ast
import os
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent

#: 导入期出现这些调用，就等于「一 import 就连服务」
NETWORK_NAMES = {
    "make_client", "get_collections", "installed_models", "embed_texts",
    "ollama_provider", "Tokenizer", "from_file",
}

#: 已经挂过 @slow 的模块（列出来是为了防止有人悄悄把标记删掉）
SLOW_MODULES = {
    "test_search.py", "test_indexer.py", "test_qdrant_store.py",
    "test_embedding_text_report.py", "test_translate.py",
    "test_token_counter_integration.py", "test_recursive_splitter_real.py",
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return ""


def import_time_network_calls(tree: ast.Module) -> list[str]:
    """只看**模块级**（不在函数/类里）的调用。"""
    hits: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            name = _call_name(child)
            if name in NETWORK_NAMES:
                hits.append(f"{name}() 行 {getattr(child, 'lineno', '?')}")
    return hits


class ImportTimeProbeTest(unittest.TestCase):
    def test_no_module_probes_a_real_service_on_import(self):
        offenders = []
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            source = path.read_text(encoding="utf-8")
            hits = import_time_network_calls(ast.parse(source))
            if not hits:
                continue
            # 允许：调用被 run_slow() 短路 / 被 mock 包住 / 属于慢用例模块
            if "run_slow" in source or path.name in SLOW_MODULES:
                continue
            offenders.append(f"{path.name}: {', '.join(hits)}")
        self.assertEqual(offenders, [],
                         "导入期就探真实服务会让『跳过』也变慢并重新依赖外部进程：" + "；".join(offenders))

    def test_slow_modules_keep_their_marker(self):
        missing = [name for name in SLOW_MODULES
                   if "tests._slow import" not in (TESTS_DIR / name).read_text(encoding="utf-8")]
        self.assertEqual(missing, [], f"这些模块的 @slow 标记被删了：{missing}")


class SwitchTest(unittest.TestCase):
    def test_run_slow_reads_both_names(self):
        from tests import _slow

        for name in ("RAG_RUN_SLOW", "RAG_OLLAMA_CHECK"):
            with mock.patch.dict(os.environ, {name: "1"}, clear=False):
                self.assertTrue(_slow.run_slow(), f"{name}=1 必须生效（兼容老文档）")
        with mock.patch.dict(os.environ, {"RAG_RUN_SLOW": "0", "RAG_OLLAMA_CHECK": "0"}, clear=False):
            self.assertFalse(_slow.run_slow())

    def test_hint_tells_how_to_run_them(self):
        from tests import _slow

        self.assertIn("RAG_RUN_SLOW=1", _slow.SLOW_HINT)
        self.assertIn("不联网", _slow.SLOW_HINT)


if __name__ == "__main__":
    unittest.main()
