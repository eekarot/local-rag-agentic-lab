"""O1：语料目录是空的（或 `--only` 全落空）时必须给可复制指引，而不是静默跑完。

开源场景下这是**最高频的第一次失败**：克隆仓库 → `data/` 里没有文档（我们不上传语料）
→ `python scripts/ingest.py` 打印「发现 0 个文件」→ 用户不知道下一步。

两条铁律：
1. 空目录**不是错误**：`failed == 0`、退出码 0，只是没活干；
2. 指引必须说清三件事：**该放哪**、**支持什么格式**、**放完跑哪条命令**。

另外「空目录」与「`--only` 没匹配」是两件不同的事，不能共用一句话：
前者要教怎么放文件，后者要教路径怎么写（相对 `--dir`）。
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.rag.ingest.pipeline import (
    IngestPipeline,
    empty_corpus_hint,
    missing_only_keys_hint,
)
from src.rag.ingest.state import IngestState

from tests._tokenizer_fixture import ensure_tokenizer

# 干净克隆里没有 17 MB 真词表（A5 逮到 37 个 ERROR）：自检时挂上替身
ensure_tokenizer()


class FakeEmbedder:
    def encode_documents(self, texts):
        return np.zeros((len(texts), 1024), dtype=np.float32)


class FakeIndexer:
    def index_document(self, document, parents, children, *, vectors=None):
        return {"upserted": len(children), "deleted_stale": 0}


class FakeStore:
    def replace_document(self, document, parents, children):
        return {"documents": 1, "parents": len(parents), "children": len(children)}


class EmptyCorpusHintTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.state = IngestState(self.root / "state.sqlite3")
        self.addCleanup(self.state.close)
        self.logs: list[str] = []

    def make_pipeline(self) -> IngestPipeline:
        return IngestPipeline(
            data_dir=self.data,
            state=self.state,
            store=FakeStore(),
            indexer=FakeIndexer(),
            embedder=FakeEmbedder(),
            log=self.logs.append,
        )

    def joined(self) -> str:
        return "\n".join(self.logs)

    # -- 空目录 ---------------------------------------------------------

    def test_empty_dir_still_succeeds(self):
        """空目录不是错误：没有失败文档，也没有崩。"""
        summary = self.make_pipeline().run()
        self.assertEqual(summary["discovered"], 0)
        self.assertEqual(summary["failed"], 0)

    def test_empty_dir_prints_where_to_put_files(self):
        summary = self.make_pipeline().run()
        text = self.joined()
        self.assertIn(str(self.data), text, "指引里必须出现实际扫描的目录")
        self.assertIn("data/", text, "要给出「放进 data/」的具体写法")

    def test_empty_dir_prints_supported_formats(self):
        self.make_pipeline().run()
        text = self.joined()
        for pattern in ("*.md", "*.txt", "*.pdf"):
            self.assertIn(pattern, text)

    def test_empty_dir_prints_next_command(self):
        self.make_pipeline().run()
        self.assertIn("scripts/ingest.py", self.joined(), "必须给出可复制的下一条命令")

    def test_empty_dir_exposes_hint_in_summary(self):
        """`--json` 消费者也要能拿到指引（日志会被 --json 吞掉）。"""
        summary = self.make_pipeline().run()
        self.assertIsInstance(summary.get("hint"), list)
        self.assertGreaterEqual(len(summary["hint"]), 3)

    def test_dry_run_on_empty_dir_also_hints(self):
        """`--dry-run` 是新用户第一个会跑的命令，它提前 return，指引不能漏。"""
        summary = self.make_pipeline().run(dry_run=True)
        self.assertIn("scripts/ingest.py", self.joined())
        self.assertIsInstance(summary.get("hint"), list)

    # -- 有文件时不能误报 -----------------------------------------------

    def test_non_empty_corpus_has_no_hint(self):
        (self.data / "note.md").write_text("# 标题\n\n正文讲 HNSW。\n", encoding="utf-8")
        summary = self.make_pipeline().run()
        self.assertGreaterEqual(summary["discovered"], 1)
        self.assertFalse(summary.get("hint"), "有文件时不该刷「怎么放文件」的指引")
        self.assertNotIn("没有可导入的文件", self.joined())

    # -- `--only` 落空是另一件事 ----------------------------------------

    def test_only_keys_miss_explains_path_format(self):
        (self.data / "note.md").write_text("# 标题\n\n正文讲 HNSW。\n", encoding="utf-8")
        summary = self.make_pipeline().run(only_keys=["md/nope.md"])
        text = self.joined()
        self.assertEqual(summary["discovered"], 1)
        self.assertIn("--only", text)
        self.assertIn("md/nope.md", text, "要回显用户写的 key，便于对照")
        self.assertNotIn("把文档放进", text, "这不是「没放文件」，别给错药")

    # -- 纯函数 ---------------------------------------------------------

    def test_empty_corpus_hint_mentions_all_patterns(self):
        lines = empty_corpus_hint(self.data, ("**/*.md", "**/*.txt", "**/*.pdf"))
        text = "\n".join(lines)
        for pattern in ("**/*.md", "**/*.txt", "**/*.pdf"):
            self.assertIn(pattern, text)

    def test_missing_only_keys_hint_lists_keys(self):
        lines = missing_only_keys_hint(["md/a.md", "pdf/b.pdf"])
        text = "\n".join(lines)
        self.assertIn("md/a.md", text)
        self.assertIn("pdf/b.pdf", text)
        self.assertIn("相对", text, "要说明 key 是相对 --dir 的路径")


def run_ingest(argv, env_extra):
    """在子进程里跑 `scripts/ingest.py`（唯一能证明「干净环境」的方式）。"""
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=120)

class NoTokenizerYetTest(unittest.TestCase):
    """干净克隆的**第一条命令**不能炸（A5 最后一步在真·干净环境里逮到的）。

    场景：git clone 完、没下 17 MB 词表，直接按 README 跑
    `python scripts/ingest.py --dir data`。原实现先建管线（`TokenCounter()` 在这里初始化），
    于是用户在「空目录引导」之前先吃到一屏 `TokenizerUnavailable` 堆栈 ——
    明明知道该说什么（该放哪 / 支持什么格式 / 下词表的 curl），却说成了崩溃。

    另一条同类：`python -m src.rag.ingestion.tokenizer count "x"` 也直接抛堆栈，
    而这正是 README 第 3 步让人跑的**验证**命令。
    """

    def _run(self, argv, env_extra):
        return run_ingest(argv, env_extra)

    def test_empty_dir_without_tokenizer_still_gives_the_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(
                ["scripts/ingest.py", "--dir", tmp, "--dry-run"],
                {"RAG_TOKENIZER_PATH": str(Path(tmp) / "nope.json")},
            )
        combined = result.stdout + result.stderr
        self.assertNotIn("Traceback", combined, f"第一条命令就崩了：{combined[-500:]}")
        self.assertEqual(result.returncode, 0, "空目录不是错误（B13/E07 定案 exit 0）")
        self.assertIn("没有可导入的文件", combined, "必须告诉用户「该放哪」")
        self.assertIn("curl", combined, "必须给下词表的命令，而不是让他自己猜")

    def test_tokenizer_cli_without_tokenizer_explains_how_to_get_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(
                ["-m", "src.rag.ingestion.tokenizer", "count", "测试"],
                {"RAG_TOKENIZER_PATH": str(Path(tmp) / "nope.json")},
            )
        combined = result.stdout + result.stderr
        self.assertNotIn("Traceback", combined, f"诊断命令不该抛堆栈：{combined[-400:]}")
        self.assertNotEqual(result.returncode, 0, "拿不到 token 数就该返回非零")
        self.assertIn("curl", combined, "必须给下载命令")
        self.assertIn("RAG_TOKENIZER_PATH", combined, "必须给替代做法")


class PlaceholderAndTokenizerGateTest(unittest.TestCase):
    """`data/` 里只有我们自己的 README.md / .gitkeep 时，等于空语料。

    实测（A5 真·干净环境）：`data/README.md` 命中了 `**/*.md`，于是「空目录引导」
    判定为「有文件」，接着建管线 → 缺词表 → 一屏 TokenizerUnavailable 堆栈。
     rookie 看到的是崩溃，不是「该放哪」。

    另一半：**有文件但没词表**时，报错也不能是堆栈 —— 该说的话（下载命令）异常里都有。
    """

    def _run(self, argv, env_extra):
        return run_ingest(argv, env_extra)

    def test_readme_only_dir_counts_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "README.md").write_text("放置规则说明", encoding="utf-8")
            Path(tmp, ".gitkeep").write_text("", encoding="utf-8")
            result = self._run(["scripts/ingest.py", "--dir", tmp, "--dry-run"],
                               {"RAG_TOKENIZER_PATH": str(Path(tmp) / "nope.json")})
        combined = result.stdout + result.stderr
        self.assertNotIn("Traceback", combined, combined[-400:])
        self.assertEqual(result.returncode, 0)
        self.assertIn("没有可导入的文件", combined)

    def test_missing_tokenizer_with_files_is_a_message_not_a_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.md").write_text("# 标题" + chr(10) + "正文", encoding="utf-8")
            result = self._run(["scripts/ingest.py", "--dir", tmp],
                               {"RAG_TOKENIZER_PATH": str(Path(tmp) / "nope.json")})
        combined = result.stdout + result.stderr
        self.assertNotIn("Traceback", combined, f"有文件但缺词表时报错也不该是堆栈：{combined[-400:]}")
        self.assertNotEqual(result.returncode, 0, "缺依赖要返回非零")
        # 哪一步先失败取决于环境：本机没起 Qdrant 时先说 Qdrant（N34），
        # 起了 Qdrant 才会走到词表。两种都必须是「说得清 + 可复制」，不许是堆栈。
        self.assertTrue("curl" in combined or "docker compose" in combined,
                        f"报错必须给出可复制的下一步：{combined[-400:]}")


class QdrantDownGateTest(unittest.TestCase):
    """**最典型的顺序错误**：README 第 2 步（起 Qdrant）没做就跑第 5 步（建库）。

    实测（真·干净克隆发布前验收）：这种情况会甩出一屏 httpx 重试链 + qdrant-client
    的 4 层 "During handling of the above exception"，最后才是
    `ResponseHandlingException: [WinError 10061]`。用户看到的第一行反而是
    `UserWarning: Failed to obtain server version`（N35），真正的病因埋在中间。

    要求：一条原因 + 两条可复制命令，退出码非零，不许有堆栈、不许有 warning。
    """

    def _run(self, argv, env_extra):
        return run_ingest(argv, env_extra)

    def test_no_qdrant_is_a_message_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.md").write_text("# 标题" + chr(10) + "正文", encoding="utf-8")
            env = {"QDRANT_URL": "http://127.0.0.1:1"}      # 死端口：CI 与本机都一样
            result = self._run(["scripts/ingest.py", "--dir", tmp], env)

        combined = result.stdout + result.stderr
        self.assertNotIn("Traceback", combined, f"没起库不该是堆栈：{combined[-400:]}")
        self.assertNotIn("Failed to obtain server version", combined, "那条 warning 会把真病因埋掉")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("docker compose up -d qdrant", combined, "必须给出起库命令")
        self.assertIn("curl -s localhost:6333", combined, "必须给出探活命令")


if __name__ == "__main__":
    unittest.main()
