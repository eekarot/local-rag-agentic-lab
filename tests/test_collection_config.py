"""集合名必须能用环境变量换 —— 否则「第二个项目」无处安放。

A5（干净克隆验收）实测撞出来的：把仓库 clone 到临时目录、按 README 走
「放文档 → 建库」，`python scripts/ingest.py --dir data` 写进的却是**正式集合**
`rag_chunks_v1`（因为集合名是代码里的常量），于是正式库从 38,865 点变成 38,871 点。
更糟的是 `src/ask.py` 与 HTTP 层连 `--collection` 都没有，两个语料库根本无法共存。

这不是「测试环境不干净」的小事，而是部署问题：同一台机器上一套 Qdrant 服务
常要放多个知识库（测试 / 预发 / 生产），撞库意味着**检索到别人的文档**，
而且没有任何报错，只有答案变怪。
"""

import os
import unittest
from unittest import mock

from src.rag.store import qdrant_store


class DefaultCollectionTest(unittest.TestCase):
    def test_default_is_the_historical_name(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_COLLECTION", None)
            self.assertEqual(qdrant_store.default_collection(), "rag_chunks_v1")

    def test_env_var_wins(self):
        with mock.patch.dict(os.environ, {"RAG_COLLECTION": "rag_demo_x1"}):
            self.assertEqual(qdrant_store.default_collection(), "rag_demo_x1")

    def test_blank_env_falls_back(self):
        """空字符串不能变成「集合名为空」——那会直接建出一个无名集合。"""
        with mock.patch.dict(os.environ, {"RAG_COLLECTION": "   "}):
            self.assertEqual(qdrant_store.default_collection(), "rag_chunks_v1")


class WiringUsesItTest(unittest.TestCase):
    """`build_default_retriever` 不传 collection 时必须服从环境变量。"""

    def test_retriever_defaults_to_env_collection(self):
        import inspect

        from src.rag.retrieval import store_wiring

        source = inspect.getsource(store_wiring.build_default_retriever)
        self.assertIn("default_collection()", source,
                      "装配层还在用常量 COLLECTION_NAME，换集合后检索会打到旧库")
        self.assertNotIn("collection or COLLECTION_NAME", source)

    def test_ingest_cli_default_follows_env(self):
        import importlib

        import scripts.ingest as ingest

        with mock.patch.dict(os.environ, {"RAG_COLLECTION": "rag_demo_x1"}):
            importlib.reload(ingest)
            try:
                args = ingest.build_parser().parse_args(["--dir", "data", "--dry-run"])
                self.assertEqual(args.collection, "rag_demo_x1")
            finally:
                os.environ.pop("RAG_COLLECTION", None)
                importlib.reload(ingest)


    def test_doctor_checks_the_configured_collection(self):
        from src.rag.ops import probes

        source = (probes.__file__)
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("default_collection", text,
                      "探针还写死 rag_chunks_v1，换集合后 /health 会报假故障")


if __name__ == "__main__":
    unittest.main()
