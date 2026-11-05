"""D18：导入状态机 + 失败清单 + 断点恢复。

方案 §10.1 的状态序列（照抄，不自己发明）：

    PENDING → PARSING → CHUNKING → EMBEDDING → INDEXING → COMPLETED
    失败态：FAILED_PARSING / FAILED_CHUNKING / FAILED_EMBEDDING / FAILED_INDEXING

验收（§17.3）：人为 kill 后重启，已完成批次不重算（日志可证）；失败文档进 failure
清单并可单条重试成功。

「不重算」凭什么成立（三个机制叠加，缺一不可）：
1. 状态表把每个文档的 `stage` + `document_version` + `chunker_version` +
   `embedding_version` 持久化，重启后据此跳过；
2. 即使状态被中断打断（停在 EMBEDDING），embedding 缓存让重算近乎免费；
3. 即使向量要重写，point id 由 chunk_id 决定，upsert 是覆盖不是新增。
"""

import json
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag.ingest.state import IngestState, Stage


class StageTest(unittest.TestCase):
    def test_stage_names_match_spec(self):
        self.assertEqual(
            [s.value for s in Stage],
            ["PENDING", "PARSING", "CHUNKING", "EMBEDDING", "INDEXING", "COMPLETED",
             "FAILED_PARSING", "FAILED_CHUNKING", "FAILED_EMBEDDING", "FAILED_INDEXING"],
        )

    def test_failed_stage_maps_from_stage(self):
        self.assertEqual(Stage.FAILED_PARSING.for_stage(Stage.PARSING), Stage.FAILED_PARSING)
        self.assertEqual(Stage.FAILED_INDEXING.for_stage(Stage.INDEXING), Stage.FAILED_INDEXING)
        with self.assertRaises(KeyError):
            Stage.FAILED_PARSING.for_stage(Stage.COMPLETED)


class StateTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = IngestState(Path(self._tmp.name) / "state.sqlite3")
        self.addCleanup(self.state.close)


class TransitionTest(StateTestBase):
    def test_records_stage_timestamps(self):
        self.state.begin("a.md")
        self.assertEqual(self.state.get("a.md")["stage"], Stage.PARSING.value)
        self.assertIsNotNone(self.state.get("a.md")["started_at"])
        self.state.advance("a.md", Stage.CHUNKING)
        self.state.advance("a.md", Stage.EMBEDDING)
        self.state.advance("a.md", Stage.INDEXING)
        self.state.complete("a.md", document_id="doc_x", document_version="sha256:1",
                            chunker_version="c1", embedding_version="e1")
        row = self.state.get("a.md")
        self.assertEqual(row["stage"], Stage.COMPLETED.value)
        self.assertIsNotNone(row["finished_at"])
        self.assertEqual(row["document_id"], "doc_x")

    def test_failure_records_stage_error_and_attempt(self):
        self.state.begin("a.md")
        self.state.advance("a.md", Stage.CHUNKING)
        self.state.fail("a.md", Stage.CHUNKING, "切分出错：编码不是 utf-8")
        row = self.state.get("a.md")
        self.assertEqual(row["stage"], Stage.FAILED_CHUNKING.value)
        self.assertEqual(row["attempts"], 1)
        self.assertIn("utf-8", row["error"])

        failures = self.state.failures()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["doc_key"], "a.md")
        self.assertEqual(failures[0]["stage"], Stage.FAILED_CHUNKING.value)

    def test_failure_history_is_kept_across_retry(self):
        self.state.begin("a.md")
        self.state.fail("a.md", Stage.EMBEDDING, "第一次失败")
        self.state.begin("a.md")  # 重试
        self.state.complete("a.md", document_version="v1")
        rows = self.state.failures()
        self.assertEqual(len(rows), 1, "失败历史要留痕，不能因为重试成功就抹掉")
        self.assertEqual(rows[0]["resolved"], 1, "重试成功后应标记为已解决")
        self.assertEqual(self.state.get("a.md")["stage"], Stage.COMPLETED.value)


class ResumeTest(StateTestBase):
    def test_is_done_requires_matching_versions(self):
        self.state.begin("a.md")
        self.state.complete("a.md", document_version="v1", chunker_version="c1", embedding_version="e1")
        self.assertTrue(self.state.is_done("a.md", document_version="v1",
                                           chunker_version="c1", embedding_version="e1"))
        self.assertFalse(self.state.is_done("a.md", document_version="v2",
                                            chunker_version="c1", embedding_version="e1"),
                         "文档改过就必须重做")
        self.assertFalse(self.state.is_done("a.md", document_version="v1",
                                            chunker_version="c2", embedding_version="e1"),
                         "换切分口径必须重做")
        self.assertFalse(self.state.is_done("a.md", document_version="v1",
                                            chunker_version="c1", embedding_version="e2"),
                         "换 embedding 版本必须重做")

    def test_failed_document_is_not_done(self):
        self.state.begin("a.md")
        self.state.fail("a.md", Stage.EMBEDDING, "爆了")
        self.assertFalse(self.state.is_done("a.md", document_version="v1",
                                            chunker_version="c1", embedding_version="e1"))

    def test_unknown_document_is_not_done(self):
        self.assertFalse(self.state.is_done("nope.md"))

    def test_interrupted_stage_is_not_done(self):
        """kill 在 EMBEDDING 阶段：重启必须接着做，不能当完成。"""
        self.state.begin("a.md")
        self.state.advance("a.md", Stage.EMBEDDING)
        self.assertFalse(self.state.is_done("a.md", document_version="v1",
                                            chunker_version="c1", embedding_version="e1"))
        self.assertEqual(self.state.stage_counts()[Stage.EMBEDDING.value], 1)


class QueryTest(StateTestBase):
    def test_stage_counts(self):
        for key, stage in (("a.md", Stage.COMPLETED), ("b.md", Stage.COMPLETED),
                           ("c.md", Stage.FAILED_EMBEDDING), ("d.md", Stage.PENDING)):
            self.state.begin(key)
            if stage is Stage.COMPLETED:
                self.state.complete(key, document_version="v1")
            elif stage is Stage.FAILED_EMBEDDING:
                self.state.fail(key, Stage.EMBEDDING, "x")
        counts = self.state.stage_counts()
        self.assertEqual(counts[Stage.COMPLETED.value], 2)
        self.assertEqual(counts[Stage.FAILED_EMBEDDING.value], 1)

    def test_pending_documents_listed_for_retry(self):
        self.state.begin("a.md")
        self.state.fail("a.md", Stage.PARSING, "坏文件")
        self.state.begin("b.md")
        self.state.complete("b.md", document_version="v1")
        pending = self.state.failed_documents()
        self.assertEqual([row["doc_key"] for row in pending], ["a.md"])

    def test_document_ids_for_cleanup(self):
        """删除文档时要拿到所有相关 document_id（Qdrant + Store 都要清）。"""
        self.state.begin("a.md")
        self.state.complete("a.md", document_id="doc_1", document_version="v1")
        self.state.begin("b.md")
        self.state.complete("b.md", document_id="doc_2", document_version="v1")
        self.assertEqual(set(self.state.all_document_ids()), {"doc_1", "doc_2"})


class PruneTest(StateTestBase):
    def test_prune_marks_failures_of_removed_sources(self):
        """源文件消失后，未解决的失败记录必须被标记，否则重试清单永远有幽灵项。"""
        base = Path(self._tmp.name) / "corpus"
        base.mkdir()
        (base / "keep.md").write_text("x", encoding="utf-8")
        self.state.begin("keep.md")
        self.state.fail("keep.md", Stage.EMBEDDING, "爆了")
        self.state.begin("gone.md")
        self.state.fail("gone.md", Stage.PARSING, "坏文件")

        removed = self.state.prune_missing(base)
        self.assertEqual(removed, 1, "只该清掉 gone.md 的状态行")
        resolutions = {row["doc_key"]: row["resolution"] for row in self.state.failures()}
        self.assertIsNone(resolutions["keep.md"])
        self.assertEqual(resolutions["gone.md"], "source_removed")
        self.assertEqual([r["doc_key"] for r in self.state.failed_documents()], ["keep.md"])

    def test_prune_also_handles_failures_without_state_rows(self):
        """状态行被删了但失败记录还在（真实踩到的情形）。"""
        base = Path(self._tmp.name) / "corpus"
        base.mkdir()
        self.state.begin("gone.md")
        self.state.fail("gone.md", Stage.PARSING, "坏文件")
        self.state.reset("gone.md")          # 状态行没了
        self.state.prune_missing(base)
        self.assertEqual(self.state.failures()[0]["resolution"], "source_removed")
        self.assertEqual(self.state.failures(unresolved_only=True), [])


class PersistenceTest(unittest.TestCase):
    def test_survives_restart(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite3"
            first = IngestState(path)
            first.begin("a.md")
            first.complete("a.md", document_version="v1", chunker_version="c1", embedding_version="e1")
            first.close()
            second = IngestState(path)
            try:
                self.assertTrue(second.is_done("a.md", document_version="v1",
                                               chunker_version="c1", embedding_version="e1"))
            finally:
                second.close()


class CollectionSwitchTest(unittest.TestCase):
    """换集合 = 换一整套库；状态表不能把旧集合的「已完成」带过去。

    踩坑（A5 干净克隆验收，接着 RAG_COLLECTION 之后）：用 `RAG_COLLECTION=新集合`
    建库，`ingest` 却报 `skipped=N` —— 状态表的键只有 `doc_key`，不看集合，
    于是新集合里几乎没有点，而输出看起来「一切正常」。这类静默半成品比报错危险得多。
    """

    def setUp(self):
        from src.rag.ingest.state import IngestState

        self.IngestState = IngestState
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "state.sqlite3"

    def tearDown(self):
        self.tmp.cleanup()

    def _mark_done(self, state, key="a.md"):
        state.begin(key)
        state.complete(key, document_version="sha256:x", chunker_version="v1",
                       embedding_version="bge-m3-v1")

    def test_same_collection_keeps_progress(self):
        state = self.IngestState(self.path, collection="rag_a")
        self._mark_done(state)
        again = self.IngestState(self.path, collection="rag_a")
        self.assertFalse(again.collection_switched)
        self.assertTrue(again.is_done("a.md", document_version="sha256:x",
                                      chunker_version="v1", embedding_version="bge-m3-v1"))

    def test_switching_collection_invalidates_progress(self):
        state = self.IngestState(self.path, collection="rag_a")
        self._mark_done(state)
        switched = self.IngestState(self.path, collection="rag_b")
        self.assertTrue(switched.collection_switched, "换集合必须让管线知道要重建")
        self.assertFalse(switched.is_done("a.md", document_version="sha256:x",
                                          chunker_version="v1", embedding_version="bge-m3-v1"),
                         "旧集合的完成记录不能算到新集合头上")

    def test_switching_back_also_rebuilds(self):
        """A → B → A：旧记录已被清掉，回到 A 也得重建（否则会漏文档）。"""
        self.IngestState(self.path, collection="rag_a")
        self._mark_done(self.IngestState(self.path, collection="rag_a"))
        self.IngestState(self.path, collection="rag_b")
        back = self.IngestState(self.path, collection="rag_a")
        self.assertTrue(back.collection_switched)
        self.assertFalse(back.is_done("a.md", document_version="sha256:x",
                                      chunker_version="v1", embedding_version="bge-m3-v1"))

    def test_without_collection_argument_behaves_as_before(self):
        """老调用（不传 collection）不能因为这次改动而清库。"""
        state = self.IngestState(self.path)
        self._mark_done(state)
        plain = self.IngestState(self.path)
        self.assertFalse(plain.collection_switched)
        self.assertTrue(plain.is_done("a.md", document_version="sha256:x",
                                      chunker_version="v1", embedding_version="bge-m3-v1"))

    def test_legacy_state_is_adopted_not_cleared(self):
        """升级前建的老库没有 meta：认领给当前集合，不能把进度清掉（否则全量重跑）。"""
        legacy = self.IngestState(self.path)          # 老调用：不传 collection，不写 meta
        self._mark_done(legacy)
        adopted = self.IngestState(self.path, collection="rag_chunks_v1")
        self.assertFalse(adopted.collection_switched)
        self.assertTrue(adopted.collection_adopted, "应该认领，而不是当作新库")
        self.assertTrue(adopted.is_done("a.md", document_version="sha256:x",
                                       chunker_version="v1", embedding_version="bge-m3-v1"))


if __name__ == "__main__":
    unittest.main()
