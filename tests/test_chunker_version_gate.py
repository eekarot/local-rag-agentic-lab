"""入库闸门必须同时认「父块版本」和「子块版本」（N43）。

BUG 实证（2026-11-06）：`IngestPipeline` 的跳过判定只用 `self.builder.chunker_version`
= `PARENT_CHUNKER_VERSION`。而这一轮改的是**子块**的重叠语义（含 overlap 的块不得超过上限），
version 没进闸门，于是：

> 已有库升级到新代码后跑 `ingest`，所有 `COMPLETED` 文档被 `[skip]`，
> 旧语义的子块留在库里、新文档用新语义 —— **一份库混着两套切分语义，而且看不出来**。

两条修法一起做才有用：
1. 闸门用「父+子」复合版本（本文件守）；
2. 子块语义一改就把 `CHILD_CHUNKER_VERSION` 升一档（升版会让**子块** id 全变，
   但**父块 id 不变** —— 所以 `gold_chunks`（父块）仍有效，这正是我们要的粒度）。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.ingest.state import IngestState  # noqa: E402
from src.rag.ingestion import parent_builder as pb  # noqa: E402


class FingerprintTest(unittest.TestCase):
    def test_fingerprint_contains_both_versions(self):
        fingerprint = pb.ingest_chunker_version()
        self.assertIn(pb.PARENT_CHUNKER_VERSION, fingerprint)
        self.assertIn(pb.CHILD_CHUNKER_VERSION, fingerprint)

    def test_fingerprint_differs_from_parent_alone(self):
        """闸门值不能等于父块版本本身 —— 否则子块改动照样不会触发重建。"""
        self.assertNotEqual(pb.ingest_chunker_version(), pb.PARENT_CHUNKER_VERSION)

    def test_child_version_was_bumped_for_the_overlap_change(self):
        """重叠语义改过就必须升版，否则 v1 同时代表两套行为。"""
        self.assertNotEqual(pb.CHILD_CHUNKER_VERSION, "recursive_child_v1")

    def test_bumping_child_version_changes_the_gate(self):
        original = pb.CHILD_CHUNKER_VERSION
        before = pb.ingest_chunker_version()
        try:
            pb.CHILD_CHUNKER_VERSION = "recursive_child_vTEST"
            self.assertNotEqual(pb.ingest_chunker_version(), before)
        finally:
            pb.CHILD_CHUNKER_VERSION = original


class OldStateRowDoesNotSkipTest(unittest.TestCase):
    """核心行为：旧库里那条 `chunker_version=recursive_parent_child_v1` 的记录，升级后不许再被跳过。"""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.state = IngestState(Path(self._tmp.name) / "state.sqlite3")
        self.state.begin("a.md")
        self.state.complete(
            "a.md",
            document_version="v1",
            chunker_version=pb.PARENT_CHUNKER_VERSION,   # 升级前写进去的旧值
            embedding_version="bge-m3-v1",
        )

    def tearDown(self):
        self.state.close()
        self._tmp.cleanup()

    def test_old_row_is_not_done_under_new_gate(self):
        self.assertFalse(
            self.state.is_done(
                "a.md",
                document_version="v1",
                chunker_version=pb.ingest_chunker_version(),
                embedding_version="bge-m3-v1",
            ),
            "旧记录必须被判定为「不完整」，否则子块语义混库",
        )

    def test_same_gate_still_skips(self):
        """闸门本身不能失效：同一份新值要能正常跳过（否则每次都全量重建）。"""
        self.state.complete(
            "a.md",
            document_version="v1",
            chunker_version=pb.ingest_chunker_version(),
            embedding_version="bge-m3-v1",
        )
        self.assertTrue(
            self.state.is_done(
                "a.md",
                document_version="v1",
                chunker_version=pb.ingest_chunker_version(),
                embedding_version="bge-m3-v1",
            )
        )


class ParentIdStaysStableTest(unittest.TestCase):
    """升子块版本时：子块 id 变、父块 id 不变（所以 `gold_chunks` 不会被打破）。"""

    def test_parent_ids_ignore_child_version(self):
        import tempfile

        from src.rag.ingestion.parser.markdown_parser import MarkdownParser
        from src.rag.ingestion.tokenizer import TokenCounter

        counter = TokenCounter()
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "a.md"
            body = ("这是第 %d 段。" + "用来凑长度的一句话。" * 30 for i in range(3))
            doc.write_text("# 标题" + chr(10) + chr(10) + (chr(10) * 2).join(body), encoding="utf-8")
            blocks = MarkdownParser().parse(str(doc))
        builder = pb.ParentBuilder(counter=counter)
        parents = builder.build(blocks, document_id="doc", document_version="v1")
        import src.rag.ingestion.parent_builder as module

        old_ids = [p.parent_id for p in parents]
        original = module.CHILD_CHUNKER_VERSION
        try:
            module.CHILD_CHUNKER_VERSION = "recursive_child_vTEST"
            changed_children = pb.build_children(parents, counter=counter)
        finally:
            module.CHILD_CHUNKER_VERSION = original
        same_parents = pb.build_children(parents, counter=counter)

        self.assertEqual([p.parent_id for p in parents], old_ids)
        self.assertNotEqual([c.chunk_id for c in changed_children], [c.chunk_id for c in same_parents])
        self.assertEqual([c.parent_id for c in changed_children], [c.parent_id for c in same_parents])


if __name__ == "__main__":
    unittest.main()
