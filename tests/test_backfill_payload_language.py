"""`scripts/backfill_payload_language.py` 的用例（N61 存量部分）。

这个脚本会**写数据库**，所以每条防线都要有用例：

1. 只补缺的 —— payload 里已有 `language` 的点不许被覆盖（否则一旦判错就再也回不去）；
2. Store 里查不到 / 那一列是 NULL 的，**不猜也不写**，只计数；
3. `--dry-run` 一个点都不许动；
4. 幂等 —— 同一份数据跑第二遍应该是 0 变更。

不碰真 Qdrant：`plan()` 是纯读、`apply_patch()` 只调 `set_payload`，两者都能用假客户端测。
"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_payload_language import apply_patch, plan  # noqa: E402


class FakePoint:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class FakeQdrant:
    """够用就行的 Qdrant 替身：scroll 分页 + set_payload 记账。"""

    def __init__(self, points, page=2):
        self.points = list(points)
        self.page = page
        self.written: list[tuple[str, dict]] = []

    def scroll(self, collection, limit=None, offset=None, with_payload=True, with_vectors=False):
        start = int(offset or 0)
        end = min(start + min(limit or self.page, self.page), len(self.points))
        nxt = str(end) if end < len(self.points) else None
        return self.points[start:end], nxt

    def set_payload(self, collection, payload, points, wait=True):
        for point_id in points:
            self.written.append((str(point_id), dict(payload)))


class StoreFixture:
    """造一个只有 parents 表的最小 Store（走真 DocumentStore，别手搓 schema）。"""

    def __init__(self, tmp: Path, rows, *, document_language: str | None = "zh"):
        from src.rag.store.document_store import DocumentStore

        self.store = DocumentStore(tmp / "documents.sqlite3")
        document = {"document_id": "doc_1", "source_uri": "a.md", "file_name": "a.md",
                    "file_type": "md", "title": "a", "language": document_language,
                    "document_version": "sha256:x", "parser_version": "md_v1",
                    "created_at": None, "status": "active"}
        parents = [{"parent_id": pid, "document_id": "doc_1", "document_version": "sha256:x",
                    "language": lang, "title": "a", "section_path": ["a"], "content": "正文",
                    "token_count": 1, "source_start": 0, "source_end": 2,
                    "chunker_version": "recursive_parent_child_v1", "parent_index": i}
                   for i, (pid, lang) in enumerate(rows)]
        children = [{"chunk_id": f"chk_{i}", "parent_id": pid, "document_id": "doc_1",
                     "document_version": "sha256:x", "content_hash": "sha256:y",
                     "token_count": 1, "source_start": 0, "source_end": 2,
                     "embedding_version": "bge-m3-v1", "status": "active"}
                    for i, (pid, _lang) in enumerate(rows)]
        self.store.replace_document(document, parents, children)
        self.path = tmp / "documents.sqlite3"


def _load(store_path: Path):
    from scripts.backfill_payload_language import load_parent_languages

    return load_parent_languages(store_path)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_only_points_missing_language_are_patched(self):
        fixture = StoreFixture(self.tmp, [("par_1", "zh"), ("par_2", "en")])
        client = FakeQdrant([
            FakePoint("p1", {"parent_id": "par_1"}),
            FakePoint("p2", {"parent_id": "par_2", "language": "en"}),   # 已有值：不许动
        ])
        patch, stats, _ = plan(client, "c", _load(fixture.path))
        self.assertEqual(patch, [("p1", "zh")])
        self.assertEqual(stats["already_has_language"], 1)
        self.assertEqual(stats["to_patch"], 1)

    def test_document_level_language_is_the_fallback(self):
        """父块那一列是 NULL 时退到文档级（与读侧 `load_parent_records()` 同一个口径）。"""
        fixture = StoreFixture(self.tmp, [("par_1", None)], document_language="en")
        client = FakeQdrant([FakePoint("p1", {"parent_id": "par_1"})])
        patch, _, _ = plan(client, "c", _load(fixture.path))
        self.assertEqual(patch, [("p1", "en")])

    def test_unknown_parent_and_null_language_are_skipped_not_guessed(self):
        # 文档级也判不出来 → 这一条才是「真的拿不到语言」
        fixture = StoreFixture(self.tmp, [("par_1", None)], document_language=None)
        client = FakeQdrant([
            FakePoint("p1", {"parent_id": "par_1"}),      # Store 里 language 是 NULL
            FakePoint("p2", {"parent_id": "par_ghost"}),  # Store 里没这条父块
        ])
        patch, stats, _ = plan(client, "c", _load(fixture.path))
        self.assertEqual(patch, [], "查不到就查不到，不许猜一个语言写进去")
        self.assertEqual(stats["null_language"], 1)
        self.assertEqual(stats["unknown_parent"], 1)

    def test_dry_run_is_the_same_read_only_pass(self):
        """`--dry-run` 与真跑共用 `plan()`：跑一遍下来 `set_payload` 一次都不许发生。"""
        fixture = StoreFixture(self.tmp, [("par_1", "zh")])
        client = FakeQdrant([FakePoint("p1", {"parent_id": "par_1"})])
        plan(client, "c", _load(fixture.path))
        self.assertEqual(client.written, [])

    def test_apply_is_idempotent(self):
        fixture = StoreFixture(self.tmp, [("par_1", "zh"), ("par_2", "zh")])
        client = FakeQdrant([
            FakePoint("p1", {"parent_id": "par_1"}),
            FakePoint("p2", {"parent_id": "par_2"}),
        ])
        languages = _load(fixture.path)
        patch, _, _ = plan(client, "c", languages)
        self.assertEqual(apply_patch(client, "c", patch), 2)

        # 模拟 Qdrant 侧已经写进去了：第二遍应当无事可做
        for point in client.points:
            point.payload = {**point.payload, "language": languages[point.payload["parent_id"]]}
        patch2, stats2, _ = plan(client, "c", languages)
        self.assertEqual(patch2, [])
        self.assertEqual(stats2["to_patch"], 0)

    def test_apply_batches_by_language(self):
        fixture = StoreFixture(self.tmp, [("par_1", "zh"), ("par_2", "en")])
        client = FakeQdrant([
            FakePoint("p1", {"parent_id": "par_1"}),
            FakePoint("p2", {"parent_id": "par_2"}),
        ])
        patch, _, _ = plan(client, "c", _load(fixture.path))
        apply_patch(client, "c", patch, batch=1)
        self.assertEqual(sorted(client.written),
                         [("p1", {"language": "zh"}), ("p2", {"language": "en"})])


if __name__ == "__main__":
    unittest.main()
