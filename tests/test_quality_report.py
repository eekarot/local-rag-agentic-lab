"""D11 Chunking 质量报告与 A–E 对照（方案 §11.1）。

D11 验收：30–50 篇 × 8 项人工抽查表 + A–E 的 chunk 数 / token 分布 / 人工评分对照表。

A–E 五种方案（§11.1）：
  A 固定长度 | B Recursive | C Recursive + Overlap
  D Parent-Child + Recursive + Overlap | E Semantic + Parent-Child

E 依赖语义切分，方案里写明「第一版关闭」，所以报告里 E 必须是
**显式的 not_implemented**，不能拿 B/C 的数字冒充。
"""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config import DATA_DIR
from src.rag.ingestion.parser.markdown_parser import MarkdownParser
from src.rag.ingestion.quality_report import (
    CHECKLIST_ITEMS,
    VARIANTS,
    build_report,
    cli_main,
    run_all_variants,
    run_variant,
)
from src.rag.ingestion.tokenizer import DEFAULT_TOKENIZER_PATH, TokenCounter

VOCAB_AVAILABLE = DEFAULT_TOKENIZER_PATH.exists()


class CharCounter:
    def count(self, text):
        return len(text)

    def count_batch(self, texts):
        return [len(t) for t in texts]

    def truncate(self, text, max_tokens):
        return text[:max_tokens]

    def truncate_tail(self, text, max_tokens):
        return text[-max_tokens:] if max_tokens > 0 else ""


COUNTER = CharCounter()

DOC = """# 检索基础

第一段正文，讲检索的基本概念。第二句话在这里。

## 向量检索

向量检索把文本映射到向量空间。相似度用余弦计算。

| 模型 | 维度 |
|---|---|
| nomic | 768 |
| bge-m3 | 1024 |

```
def score(x):
    return x * 2
```

## 关键词检索

BM25 用词频和逆文档频率打分。它不需要训练。
"""


def documents():
    blocks = MarkdownParser().parse("doc.md", text=DOC)
    return [("doc.md", blocks)]


class VariantSpecTest(unittest.TestCase):
    def test_five_variants_are_declared_in_order(self):
        self.assertEqual([variant.key for variant in VARIANTS], ["A", "B", "C", "D", "E"])

    def test_semantic_variant_is_marked_not_implemented(self):
        semantic = next(variant for variant in VARIANTS if variant.key == "E")
        self.assertEqual(semantic.status, "not_implemented")

    def test_eight_checklist_items(self):
        self.assertEqual(len(CHECKLIST_ITEMS), 8, "§11.1 列了 8 项人工抽查")


class RunVariantTest(unittest.TestCase):
    def test_fixed_length_variant_produces_chunks(self):
        metrics = run_variant("A", documents(), counter=COUNTER, fixed_chars=60)
        self.assertEqual(metrics.status, "ok")
        self.assertGreater(metrics.chunks, 1)
        self.assertIsNone(metrics.parents, "A 没有 Parent 概念")

    def test_recursive_variant_respects_token_limit(self):
        metrics = run_variant("B", documents(), counter=COUNTER, child_max_tokens=80)
        self.assertLessEqual(metrics.tokens_max, 80)

    def test_overlap_variant_reports_in_range_ratio(self):
        metrics = run_variant("C", documents(), counter=COUNTER, child_max_tokens=80)
        self.assertIsNotNone(metrics.overlap_in_range_ratio)
        self.assertGreaterEqual(metrics.overlap_in_range_ratio, 0.0)

    def test_parent_child_variant_reports_parents_and_ratio(self):
        metrics = run_variant("D", documents(), counter=COUNTER)
        self.assertGreaterEqual(metrics.parents, 1)
        self.assertIsNotNone(metrics.child_per_parent)

    def test_not_implemented_variant_returns_status_only(self):
        metrics = run_variant("E", documents(), counter=COUNTER)
        self.assertEqual(metrics.status, "not_implemented")
        self.assertIsNone(metrics.chunks)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            run_variant("Z", documents(), counter=COUNTER)

    def test_token_statistics_are_ordered(self):
        metrics = run_variant("D", documents(), counter=COUNTER)
        self.assertLessEqual(metrics.tokens_p50, metrics.tokens_p90)
        self.assertLessEqual(metrics.tokens_p90, metrics.tokens_max)


class StructuralMetricsTest(unittest.TestCase):
    def test_sentence_cut_ratio_is_computed(self):
        """A 固定长度会拦腰切断句子，ratio 应该明显高于 D。"""
        fixed = run_variant("A", documents(), counter=COUNTER, fixed_chars=40)
        parent_child = run_variant("D", documents(), counter=COUNTER)
        self.assertGreater(fixed.sentence_cut_ratio, parent_child.sentence_cut_ratio)

    def test_table_and_code_intact_ratios_are_reported(self):
        metrics = run_variant("D", documents(), counter=COUNTER)
        self.assertIsNotNone(metrics.table_intact_ratio)
        self.assertIsNotNone(metrics.code_intact_ratio)

    def test_heading_link_ratio_is_reported_for_parent_child(self):
        metrics = run_variant("D", documents(), counter=COUNTER)
        self.assertIsNotNone(metrics.heading_link_ratio)
        self.assertAlmostEqual(metrics.heading_link_ratio, 1.0, places=2)

    def test_child_traceable_ratio_is_reported(self):
        metrics = run_variant("D", documents(), counter=COUNTER)
        self.assertEqual(metrics.child_traceable_ratio, 1.0)


class SentenceCutMetricTest(unittest.TestCase):
    """回归：以换行结尾的块不算「句子被切断」。

    曾经用 `piece.rstrip()` 后再判 `endswith("\n")`，永远为假，
    结果以换行结尾的列表项全被误算成切断，指标虚高。
    """

    def test_newline_terminated_chunk_is_not_a_cut(self):
        from src.rag.ingestion.quality_report import _ends_cleanly

        newline = chr(10)
        self.assertTrue(_ends_cleanly("列表项一" + newline))
        self.assertTrue(_ends_cleanly("一句话。"))
        self.assertTrue(_ends_cleanly("补一句" + newline + "  "), "行尾空格不影响判断")
        self.assertFalse(_ends_cleanly("半句话没有结尾"))
        self.assertFalse(_ends_cleanly("用逗号结尾，"))


class ReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = run_all_variants(documents(), counter=COUNTER, fixed_chars=60)
        cls.report = build_report(
            cls.metrics,
            per_document_rows=[{"document": "doc.md", "chars": len(DOC), "blocks": 8, "parents": 3, "children": 5}],
            corpus_label="合成样例",
            versions={"chunker_version": "v1"},
        )

    def test_report_has_variant_table_with_five_rows(self):
        for key in ("A", "B", "C", "D", "E"):
            self.assertIn(f"| {key} |", self.report, f"报告缺方案 {key} 的行")

    def test_report_marks_semantic_variant_unimplemented(self):
        self.assertIn("未实现", self.report)

    def test_report_has_eight_checklist_rows(self):
        for item in CHECKLIST_ITEMS:
            self.assertIn(item.name, self.report)

    def test_report_has_manual_score_column(self):
        self.assertIn("人工评分", self.report)

    def test_report_contains_per_document_table(self):
        self.assertIn("doc.md", self.report)

    def test_report_is_deterministic_apart_from_timestamp(self):
        again = build_report(
            run_all_variants(documents(), counter=COUNTER, fixed_chars=60),
            per_document_rows=[{"document": "doc.md", "chars": len(DOC), "blocks": 8, "parents": 3, "children": 5}],
            corpus_label="合成样例",
            versions={"chunker_version": "v1"},
        )
        strip = lambda text: "\n".join(  # noqa: E731
            line for line in text.splitlines() if not line.startswith("> 生成时间")
        )
        self.assertEqual(strip(self.report), strip(again))


class CliTest(unittest.TestCase):
    def test_cli_writes_report_file(self):
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "quality.md"
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli_main(["--dir", str(Path(tmp) / "empty"), "--out", str(out_path)])
            self.assertEqual(code, 2, "目录不存在时应返回 2")
            self.assertIn("目录不存在", out.getvalue())


@unittest.skipUnless(VOCAB_AVAILABLE, f"缺少真词表 {DEFAULT_TOKENIZER_PATH}")
class RealCorpusReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        counter = TokenCounter()
        paths = sorted(DATA_DIR.rglob("*.md"))[:12]
        cls.docs = []
        for path in paths:
            cls.docs.append((path.relative_to(DATA_DIR).as_posix(), MarkdownParser().parse(str(path))))
        cls.metrics = run_all_variants(cls.docs, counter=counter)

    def test_all_implemented_variants_produce_numbers(self):
        for metric in self.metrics:
            if metric.status != "ok":
                continue
            self.assertIsNotNone(metric.chunks)
            self.assertGreaterEqual(metric.chunks, 0)

    def test_report_generates_on_real_corpus(self):
        report = build_report(
            self.metrics,
            per_document_rows=[{"document": name, "chars": 0, "blocks": 0, "parents": 0, "children": 0} for name, _ in self.docs],
            corpus_label="真实语料 12 篇",
            versions={"chunker_version": "v1"},
        )
        self.assertIn("| A |", report)
        print("\n" + "\n".join(line for line in report.splitlines() if line.startswith("| ") and "方案" not in line)[:1200])

    def test_parent_child_reduces_sentence_cuts(self):
        by_key = {metric.key: metric for metric in self.metrics}
        self.assertLess(by_key["D"].sentence_cut_ratio, by_key["A"].sentence_cut_ratio)


if __name__ == "__main__":
    unittest.main()
