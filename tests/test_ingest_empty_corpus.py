"""O1：语料目录是空的（或 `--only` 全落空）时必须给可复制指引，而不是静默跑完。

开源场景下这是**最高频的第一次失败**：克隆仓库 → `data/` 里没有文档（我们不上传语料）
→ `python scripts/ingest.py` 打印「发现 0 个文件」→ 用户不知道下一步。

两条铁律：
1. 空目录**不是错误**：`failed == 0`、退出码 0，只是没活干；
2. 指引必须说清三件事：**该放哪**、**支持什么格式**、**放完跑哪条命令**。

另外「空目录」与「`--only` 没匹配」是两件不同的事，不能共用一句话：
前者要教怎么放文件，后者要教路径怎么写（相对 `--dir`）。
"""

import unittest
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


if __name__ == "__main__":
    unittest.main()
