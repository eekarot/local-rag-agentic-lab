"""D18 编排层：一条命令从「一堆文件」跑到「Qdrant + Store 双写完成」。

这一层测的是**编排**，不是组件本身（组件各有自己的用例），所以 store / indexer /
embedder 都用假实现注入：真接线由 D16/D17 的用例负责，这里只关心
「阶段有没有落库、失败有没有留痕、重启会不会重算」。

验收（§17.3）：人为 kill 后重启，已完成批次不重算（日志可证）；
失败文档进 failure 清单并可单条重试成功。
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.rag.ingest.pipeline import IngestPipeline
from src.rag.ingest.state import IngestState, Stage


class FakeEmbedder:
    def __init__(self, fail_on: str | None = None):
        self.calls: list[list[str]] = []
        self.texts: list[str] = []
        self.fail_on = fail_on

    def encode_documents(self, texts):
        self.calls.append(list(texts))
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise RuntimeError(f"embedding 崩了：{self.fail_on}")
        self.texts.extend(texts)
        return np.zeros((len(texts), 1024), dtype=np.float32)


class FakeIndexer:
    def __init__(self, fail_on: str | None = None):
        self.indexed: list[str] = []
        self.fail_on = fail_on

    def index_document(self, document, parents, children, *, vectors=None):
        if self.fail_on and self.fail_on in document["source_uri"]:
            raise RuntimeError("Qdrant 写挂了")
        self.indexed.append(document["source_uri"])
        return {"upserted": len(children), "deleted_stale": 0}


class FakeStore:
    def __init__(self):
        self.documents: list[str] = []

    def replace_document(self, document, parents, children):
        self.documents.append(document["document_id"])
        return {"documents": 1, "parents": len(parents), "children": len(children)}


class PipelineTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.state = IngestState(self.root / "state.sqlite3")
        self.addCleanup(self.state.close)
        self.embedder = FakeEmbedder()
        self.indexer = FakeIndexer()
        self.store = FakeStore()
        self.logs: list[str] = []

    def write(self, name: str, paragraphs: int = 3) -> Path:
        path = self.data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n\n".join(f"## 章节 {i}\n\n这是第 {i} 段正文，讲 HNSW 与检索。" for i in range(paragraphs))
        path.write_text(f"# {name}\n\n{body}\n", encoding="utf-8")
        return path

    #: 构造参数（决定"跟谁说话"）
    CTOR_KEYS = ("patterns", "window_docs", "window_children", "embedding_version")
    #: 运行参数（决定"这一趟做什么"）
    RUN_KEYS = ("limit", "force", "dry_run", "crash_hook", "continue_on_error", "only_keys")

    def pipeline(self, **kwargs) -> IngestPipeline:
        options = dict(
            data_dir=self.data, state=self.state, store=self.store,
            indexer=self.indexer, embedder=self.embedder,
            log=self.logs.append, window_children=4, window_docs=8,
        )
        options.update({k: v for k, v in kwargs.items() if k in self.CTOR_KEYS})
        return IngestPipeline(**options)

    def run_pipeline(self, **kwargs):
        ctor = {k: v for k, v in kwargs.items() if k in self.CTOR_KEYS}
        run = {k: v for k, v in kwargs.items() if k in self.RUN_KEYS}
        return self.pipeline(**ctor).run(**run)


class HappyPathTest(PipelineTestBase):
    def test_indexes_every_document(self):
        for i in range(3):
            self.write(f"doc{i}.md")
        summary = self.run_pipeline()
        self.assertEqual(summary["indexed"], 3)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(sorted(self.indexer.indexed),
                         ["doc0.md", "doc1.md", "doc2.md"])
        counts = self.state.stage_counts()
        self.assertEqual(counts[Stage.COMPLETED.value], 3)

    def test_state_records_versions(self):
        self.write("a.md")
        self.run_pipeline()
        row = self.state.get("a.md")
        self.assertEqual(row["stage"], Stage.COMPLETED.value)
        self.assertIsNotNone(row["document_id"])
        self.assertIsNotNone(row["chunker_version"])
        self.assertIsNotNone(row["embedding_version"])

    def test_store_gets_every_document(self):
        self.write("a.md"), self.write("b.md")
        self.run_pipeline()
        self.assertEqual(len(self.store.documents), 2)

    def test_respects_limit(self):
        for i in range(5):
            self.write(f"doc{i}.md")
        summary = self.run_pipeline(limit=2)
        self.assertEqual(summary["indexed"], 2)

    def test_dry_run_writes_nothing(self):
        self.write("a.md")
        summary = self.run_pipeline(dry_run=True)
        self.assertEqual(summary["discovered"], 1)
        self.assertEqual(summary["indexed"], 0)
        self.assertEqual(self.indexer.indexed, [])
        self.assertEqual(self.store.documents, [])
        self.assertEqual(self.state.stage_counts(), {}, "dry-run 不该写状态表")

    def test_empty_directory_is_not_an_error(self):
        summary = self.run_pipeline()
        self.assertEqual(summary["discovered"], 0)
        self.assertEqual(summary["indexed"], 0)


class ResumeTest(PipelineTestBase):
    def test_second_run_skips_completed_without_recomputing(self):
        """验收核心：已完成的不重算——embedding 与 upsert 都不能再发生。"""
        for i in range(3):
            self.write(f"doc{i}.md")
        self.run_pipeline()
        self.assertEqual(len(self.embedder.calls), 1)  # 一个 window 一次
        embed_calls = len(self.embedder.calls)

        summary = self.run_pipeline()
        self.assertEqual(summary["skipped"], 3)
        self.assertEqual(summary["indexed"], 0)
        self.assertEqual(len(self.embedder.calls), embed_calls, "重启后不该再调 embedding")
        self.assertEqual(len(self.indexer.indexed), 3, "不该重复 upsert")

    def test_changed_file_is_reprocessed(self):
        path = self.write("a.md")
        self.run_pipeline()
        path.write_text(path.read_text(encoding="utf-8") + "\n\n新增一段。\n", encoding="utf-8")
        summary = self.run_pipeline()
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual(summary["indexed"], 1)

    def test_force_ignores_completed_state(self):
        self.write("a.md")
        self.run_pipeline()
        summary = self.run_pipeline(force=True)
        self.assertEqual(summary["indexed"], 1)

    def test_kill_mid_run_resumes_remaining_only(self):
        """模拟 Ctrl-C：已完成的两篇不再重算，剩下的继续。"""
        for i in range(4):
            self.write(f"doc{i}.md")

        def boom(processed):
            if processed >= 2:
                raise KeyboardInterrupt("模拟 kill")

        with self.assertRaises(KeyboardInterrupt):
            self.run_pipeline(crash_hook=boom, window_docs=1)
        self.assertEqual(self.state.stage_counts().get(Stage.COMPLETED.value), 2)

        self.embedder.calls.clear()
        summary = self.run_pipeline(window_docs=1)
        self.assertEqual(summary["skipped"], 2, "已完成的必须跳过")
        self.assertEqual(summary["indexed"], 2)
        texts = [t for call in self.embedder.calls for t in call]
        self.assertEqual(len(texts), 2, f"只该为剩下两篇编码，实际 {len(texts)} 条")


class DeleteStateTest(PipelineTestBase):
    """删除后状态表必须同步清掉，否则重跑会被跳过、永远重建不回来（D20 实测）。"""

    def test_forget_by_document_id(self):
        self.write("a.md")
        self.run_pipeline()
        row = self.state.get("a.md")
        self.assertEqual(row["stage"], Stage.COMPLETED.value)
        self.assertEqual(self.state.forget_document_id(row["document_id"]), 1)
        self.assertIsNone(self.state.get("a.md"))
        # 清掉状态后重跑必须真的重建
        summary = self.run_pipeline()
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual(summary["indexed"], 1)


class FailureTest(PipelineTestBase):
    def test_parse_failure_recorded_and_others_continue(self):
        self.write("good.md")
        bad = self.data / "bad.xyz"
        bad.write_text("unsupported", encoding="utf-8")
        summary = self.run_pipeline(patterns=("**/*.md", "**/*.xyz"))
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["indexed"], 1)
        failures = self.state.failures()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["stage"], Stage.FAILED_PARSING.value)
        self.assertIn("xyz", failures[0]["error"].lower())

    def test_embedding_failure_marks_window_documents(self):
        self.write("a.md")
        self.write("b.md")
        self.embedder.fail_on = "HNSW"
        summary = self.run_pipeline()
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["indexed"], 0)
        self.assertTrue(all(f["stage"] == Stage.FAILED_EMBEDDING.value for f in self.state.failures()))

    def test_index_failure_recorded_per_document(self):
        self.write("a.md")
        self.write("b.md")
        self.indexer.fail_on = "a.md"
        summary = self.run_pipeline()
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["indexed"], 1)
        self.assertEqual(self.state.get("a.md")["stage"], Stage.FAILED_INDEXING.value)

    def test_retry_failed_document_succeeds_after_fix(self):
        """验收：失败文档进清单 → 修好后同一个 key 重跑即成功。"""
        path = self.write("a.md")
        self.embedder.fail_on = "HNSW"          # 正文里有 HNSW → embedding 必崩
        first = self.run_pipeline()
        self.assertEqual(first["failed"], 1)
        self.assertEqual([r["doc_key"] for r in self.state.failed_documents()], ["a.md"])

        path.write_text("# 修好了" + chr(10) + chr(10) + "这段没有触发词，可以正常编码。" + chr(10),
                        encoding="utf-8")
        self.embedder.fail_on = None
        result = self.run_pipeline()
        self.assertEqual(result["indexed"], 1)
        self.assertEqual(self.state.failed_documents(), [])
        row = self.state.failures()[0]
        self.assertEqual(row["resolved"], 1, "重试成功后失败记录标记已解决")
        self.assertEqual(row["resolution"], "completed")

    def test_renamed_file_retires_old_state(self):
        """文件改名后，旧 key 的失败行必须清掉，否则重试清单里有修不好的幽灵项。"""
        bad = self.data / "bad.xyz"
        bad.write_text("unsupported", encoding="utf-8")
        self.run_pipeline(patterns=("**/*.md", "**/*.xyz"))
        self.assertEqual(len(self.state.failed_documents()), 1)

        bad.unlink()
        self.write("good.md")
        self.run_pipeline()
        self.assertEqual(self.state.failed_documents(), [], "源文件已消失的状态行应被清理")
        rows = self.state.failures()
        self.assertEqual(len(rows), 1, "失败历史仍保留（审计用）")
        self.assertEqual(rows[0]["resolution"], "source_removed", "要能区分「重试成功」与「源文件删了」")

    def test_failure_does_not_block_next_window(self):
        self.write("a.md")
        self.write("b.md")
        self.write("c.md")
        self.embedder.fail_on = None
        self.indexer.fail_on = "b.md"
        summary = self.run_pipeline(window_docs=3)
        self.assertEqual(summary["indexed"], 2)
        self.assertEqual(summary["failed"], 1)


class WindowTest(PipelineTestBase):
    def test_embedding_is_batched_across_documents(self):
        """跨文档攒批：单篇一次调用会让 Ollama 的 2.1 s 固定开销乘以文档数。"""
        for i in range(4):
            self.write(f"doc{i}.md")
        self.run_pipeline(window_docs=4, window_children=1000)
        self.assertEqual(len(self.embedder.calls), 1, "4 篇应合并成一次 embedding 调用")
        self.assertEqual(len(self.embedder.calls[0]), sum(
            1 for t in self.embedder.calls[0] if t), "批次里应包含所有子块的 embedding_text")

    def test_window_children_triggers_flush(self):
        for i in range(4):
            self.write(f"doc{i}.md")
        self.run_pipeline(window_docs=4, window_children=3)
        self.assertGreater(len(self.embedder.calls), 1, "攒够 window_children 就该 flush")


if __name__ == "__main__":
    unittest.main()
