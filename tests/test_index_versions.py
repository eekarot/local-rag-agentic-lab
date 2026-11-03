"""D21：索引版本化——换 embedding 模型必须换 collection。

验收（§17.3）：「换 embedding 模型后脚本拒绝写入旧 collection 并提示新建
（**负向测试必须存在**）」。

为什么必须有这道闸：向量维度相同（比如都是 1024 维的两个模型）时，把新模型的
向量写进旧 collection **不会报任何错**，检索结果只是慢慢变差——这是最难查的一类
静默故障。所以版本信息要单独记，写入前先对账。

版本记录放 SQLite 而不是 Qdrant：Qdrant 的 collection 只记「维度/距离/量化」
这类引擎参数，没有放 `embedding_version` 这种业务元数据的地方（塞进 payload
的哨兵点更糟：它会被检索命中）。
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag.store.index_versions import (
    CollectionVersionMismatch,
    IndexVersions,
    collection_name_for,
)


class CollectionNameTest(unittest.TestCase):
    def test_slug_from_embedding_version(self):
        self.assertEqual(collection_name_for("bge-m3-v2"), "rag_chunks_bge_m3_v2")
        self.assertEqual(collection_name_for("BAAI/bge-m3"), "rag_chunks_baai_bge_m3")

    def test_default_version_maps_to_existing_collection(self):
        """老名字 `rag_chunks_v1` 必须还能对上，不能让已有数据失效。"""
        self.assertEqual(collection_name_for("bge-m3-v1", default="rag_chunks_v1"), "rag_chunks_v1")


class VersionTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.versions = IndexVersions(Path(self._tmp.name) / "versions.sqlite3")
        self.addCleanup(self.versions.close)
        self.spec = dict(
            collection="rag_chunks_v1",
            embedding_model="bge-m3",
            embedding_version="bge-m3-v1",
            vector_size=1024,
            chunker_version="recursive_child_v1",
        )

    def register(self, **overrides):
        return self.versions.register(**{**self.spec, **overrides})


class RegisterTest(VersionTestBase):
    def test_register_then_check(self):
        self.register()
        self.versions.assert_compatible("rag_chunks_v1", embedding_model="bge-m3",
                                       embedding_version="bge-m3-v1", vector_size=1024)
        row = self.versions.get("rag_chunks_v1")
        self.assertEqual(row["chunker_version"], "recursive_child_v1")
        self.assertEqual(row["status"], "active")

    def test_register_is_idempotent(self):
        self.register()
        self.register()
        self.assertEqual(len(self.versions.list()), 1)

    def test_unknown_collection_is_adopted(self):
        """先有 collection（D15 建的）后有版本表：第一次写入时自动登记。"""
        row = self.versions.ensure_registered(**self.spec)
        self.assertEqual(row["embedding_version"], "bge-m3-v1")


class MismatchTest(VersionTestBase):
    def test_embedding_version_change_is_rejected(self):
        self.register()
        with self.assertRaises(CollectionVersionMismatch) as ctx:
            self.versions.assert_compatible("rag_chunks_v1", embedding_model="bge-m3",
                                            embedding_version="bge-m3-v2", vector_size=1024)
        message = str(ctx.exception)
        self.assertIn("bge-m3-v2", message)
        self.assertIn("rag_chunks_bge_m3_v2", message, "报错必须直接给出可用的新 collection 名")

    def test_model_change_is_rejected(self):
        self.register()
        with self.assertRaises(CollectionVersionMismatch) as ctx:
            self.versions.assert_compatible("rag_chunks_v1", embedding_model="nomic-embed-text",
                                            embedding_version="bge-m3-v1", vector_size=1024)
        self.assertIn("nomic-embed-text", str(ctx.exception))

    def test_vector_size_change_is_rejected(self):
        self.register()
        with self.assertRaises(CollectionVersionMismatch) as ctx:
            self.versions.assert_compatible("rag_chunks_v1", embedding_model="bge-m3",
                                            embedding_version="bge-m3-v1", vector_size=768)
        self.assertIn("768", str(ctx.exception))

    def test_same_model_but_new_chunker_is_allowed(self):
        """换切分口径不需要换 collection：chunk_id 变了，但向量空间没变。"""
        self.register()
        self.versions.assert_compatible("rag_chunks_v1", embedding_model="bge-m3",
                                       embedding_version="bge-m3-v1", vector_size=1024,
                                       chunker_version="recursive_child_v2")

    def test_mismatch_does_not_overwrite_registry(self):
        self.register()
        with self.assertRaises(CollectionVersionMismatch):
            self.versions.register(**{**self.spec, "embedding_version": "bge-m3-v2"})
        self.assertEqual(self.versions.get("rag_chunks_v1")["embedding_version"], "bge-m3-v1")


class LifecycleTest(VersionTestBase):
    def test_retire_and_activate(self):
        self.register()
        self.versions.retire("rag_chunks_v1")
        self.assertEqual(self.versions.get("rag_chunks_v1")["status"], "retired")
        self.assertEqual(self.versions.active(), [])
        self.versions.activate("rag_chunks_v1")
        self.assertEqual([r["collection"] for r in self.versions.active()], ["rag_chunks_v1"])

    def test_retired_collection_refuses_writes(self):
        self.register()
        self.versions.retire("rag_chunks_v1")
        with self.assertRaises(CollectionVersionMismatch) as ctx:
            self.versions.assert_compatible("rag_chunks_v1", embedding_model="bge-m3",
                                           embedding_version="bge-m3-v1", vector_size=1024)
        self.assertIn("retired", str(ctx.exception).lower())

    def test_persists_across_instances(self):
        self.register()
        again = IndexVersions(self.versions.path)
        try:
            self.assertIsNotNone(again.get("rag_chunks_v1"))
        finally:
            again.close()


if __name__ == "__main__":
    unittest.main()
