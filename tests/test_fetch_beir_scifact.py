"""导入 BEIR–SciFact 的转换层（只做格式归一，不改写正文）。

数据集事实（实查，见 `datasets/scifact-v1/sources.json` / `NOTICE`）：

  * 原始 SciFact（Wadden et al., EMNLP 2020）release：`corpus.jsonl` 5,183 篇摘要、
    `claims_train.jsonl` 809 条、`claims_dev.jsonl` 300 条、`claims_test.jsonl` 300 条（**无标注**）
  * BEIR（Thakur et al., NeurIPS 2021）把它收成一个零样本检索集：queries 1,109（train 809 + dev 300），
    test qrels 339 条 —— 与我们从 `claims_dev` 的 `cited_doc_ids` 反推出来的 **339 条完全一致**
  * 许可证（`allenai/scifact/LICENSE.md` 原文）：claims 与 evidence 标注 **CC BY 4.0**，
    corpus 摘要属 S2ORC、**ODC-By 1.0**

这些用例守三件事：
1. 只做格式归一 —— 正文一个字都不许改（B13 那套纪律）；
2. qrels 必须从 `cited_doc_ids` 反推，且总数要对得上 339（口径错了会静默变成另一个基准）；
3. `claims_test` 没有标注，不许被当成可评测集（否则会算出「0 命中」的假指标）。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_beir_scifact import (build_doc_markdown, derive_doc_qrels, load_jsonl,
                                        write_qrels_tsv)


class DocMarkdownTest(unittest.TestCase):
    def test_title_becomes_heading_and_abstract_is_joined_verbatim(self):
        doc = {"doc_id": 4983, "title": "A Title.", "structured": False,
               "abstract": ["First sentence.", "Second sentence."]}
        text = build_doc_markdown(doc)
        self.assertEqual(text, "# A Title.\n\nFirst sentence. Second sentence.\n")

    def test_missing_title_falls_back_to_doc_id(self):
        text = build_doc_markdown({"doc_id": 7, "title": "", "abstract": ["Only sentence."]})
        self.assertTrue(text.startswith("# doc 7"))

    def test_string_abstract_accepted(self):
        """BEIR 打包版把 abstract 合并成一个字符串；两种形态都要能吃。"""
        text = build_doc_markdown({"doc_id": 1, "title": "T", "abstract": "One. Two."})
        self.assertEqual(text, "# T\n\nOne. Two.\n")

    def test_no_text_is_rewritten(self):
        weird = 'Gène 表达（p<0.05）— "quoted" 100% ✓'
        text = build_doc_markdown({"doc_id": 1, "title": "T", "abstract": [weird]})
        self.assertIn(weird, text)


class QrelsTest(unittest.TestCase):
    def _claims(self):
        return [
            {"id": 1, "claim": "c1", "cited_doc_ids": [10, 11]},
            {"id": 2, "claim": "c2", "cited_doc_ids": [20]},
            {"id": 3, "claim": "c3", "cited_doc_ids": []},          # NOINFO：没有相关文档
            {"id": 4, "claim": "c4"},                               # 字段缺失
        ]

    def test_derive_doc_qrels(self):
        qrels = derive_doc_qrels(self._claims())
        self.assertEqual(qrels, {"1": ["10", "11"], "2": ["20"]})

    def test_derive_from_evidence_when_top_level_missing(self):
        claims = [{"id": 9, "claim": "c", "evidence": {"x": [{"cited_doc_ids": [5]}]}}]
        self.assertEqual(derive_doc_qrels(claims), {"9": ["5"]})

    def test_write_tsv_is_beir_shape(self):
        path = Path(self._tmp()) / "test.tsv"
        write_qrels_tsv(path, {"1": ["10", "11"]})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(lines[0], "query-id\tcorpus-id\tscore")
        self.assertEqual(lines[1:], ["1\t10\t1", "1\t11\t1"])

    def _tmp(self):
        import tempfile

        return tempfile.mkdtemp(prefix="scifact-qrels-")


class LoadJsonlTest(unittest.TestCase):
    def test_load_jsonl_returns_rows(self):
        import tempfile

        path = Path(tempfile.mkdtemp(prefix="scifact-jsonl-")) / "x.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
        self.assertEqual(load_jsonl(path), [{"a": 1}, {"a": 2}])


class QrelFileNameTest(unittest.TestCase):
    """BEIR 把 dev 当 test 发布：`qrels/test.tsv` 装的其实是 claims_dev 的判断。"""

    def test_dev_split_is_written_as_test_tsv(self):
        from scripts.fetch_beir_scifact import QREL_FILES, SPLITS

        self.assertEqual(SPLITS["dev"], "claims_dev.jsonl")
        self.assertEqual(QREL_FILES["dev"], "test.tsv")
        self.assertEqual(QREL_FILES["train"], "train.tsv")


class TestSplitIsNotEvaluatedTest(unittest.TestCase):
    """`claims_test` 没有标注 —— 用它评测只会得到假的 0 命中。"""

    def test_test_split_has_no_labels(self):
        from scripts.fetch_beir_scifact import TEST_SPLIT_NOTE, SPLITS

        self.assertIn("无标注", TEST_SPLIT_NOTE)
        self.assertIn("dev", SPLITS)          # 可评测的是 dev（BEIR 的 test.tsv）
        self.assertNotIn("test", SPLITS)


if __name__ == "__main__":
    unittest.main()
