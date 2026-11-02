"""D09 Chunk Validator 的单元测试。

D09 验收：校验报告必须含四项数字 —— 空 chunk 0、超限 0、重复文本占比、
边界完整率；失败样例落 `*.invalid.jsonl`。
另有一条：报告要带「多 gold 收敛统计」（同一 gold_span 落在几个 chunk 里的分布）。

校验的意义在于**先于向量化发现问题**：往 Qdrant 里灌了脏 chunk，后面
所有指标都会被污染，而且删起来比拦下来贵得多。
"""

import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config import DATA_DIR
from src.rag.ingestion.ids import make_document_id
from src.rag.ingestion.parent_builder import Parent, ParentBuilder, build_children
from src.rag.ingestion.parser.markdown_parser import MarkdownParser
from src.rag.ingestion.tokenizer import DEFAULT_TOKENIZER_PATH, TokenCounter
from src.rag.ingestion.validator import ValidationReport, validate

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


def unique_text(n_pieces=400, unit="第{}句正文。"):
    """非周期性文本：每个切片内容都不同，避免「不同位置切片却文本相同」的假重复。

    教训：用 "正文"*300 这类周期串做样本时，content[200:400] 与 content[0:200]
    完全一样，重复率会被算成 0.75 而不是 0.25——样本本身的错，不是实现错。
    """
    return "".join(unit.format(i) for i in range(n_pieces))


def make_parent(index=0, content="父块内容：" + "内容" * 100, source_start=0):
    return Parent(
        parent_index=index,
        content=content,
        token_count=len(content),
        section_path=["标题"],
        source_start=source_start,
        source_end=source_start + len(content),
        document_id="doc_1",
        document_version="sha256:v",
        title="标题",
        parent_id="par_" + "a" * 32,
    )


def make_children(parent, texts):
    """用真实 Child 结构造样本，保证字段完整。"""
    from src.rag.ingestion.parent_builder import Child

    children = []
    offset = 0
    for index, text in enumerate(texts):
        children.append(
            Child(
                parent_index=parent.parent_index,
                child_index=index,
                content=text,
                token_count=len(text),
                section_path=list(parent.section_path),
                source_start=(parent.source_start or 0) + offset,
                source_end=(parent.source_start or 0) + offset + len(text),
                document_id=parent.document_id,
                document_version=parent.document_version,
                title=parent.title,
                parent_id=parent.parent_id,
                chunk_id="chk_" + f"{index:032x}",
            )
        )
        offset += len(text)
    return children


def clean_case():
    parent = make_parent(content="第一部分内容。" + "填充" * 50 + "第二部分内容。" + "填充" * 50)
    part1 = parent.content[:150]
    part2 = parent.content[150:]
    return parent, make_children(parent, [part1, part2])


class CleanCaseTest(unittest.TestCase):
    def test_clean_case_reports_zero_problems(self):
        parent, children = clean_case()
        report = validate([parent], children, counter=COUNTER, child_max_tokens=500, parent_max_tokens=2000)
        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.empty_chunks, 0)
        self.assertEqual(report.over_limit_chunks, 0)
        self.assertEqual(report.duplicate_ratio, 0.0)
        self.assertEqual(report.boundary_valid_ratio, 1.0)

    def test_report_exposes_the_four_required_numbers(self):
        parent, children = clean_case()
        payload = validate([parent], children, counter=COUNTER).to_dict()
        for key in ("empty_chunks", "over_limit_chunks", "duplicate_ratio", "boundary_valid_ratio"):
            self.assertIn(key, payload, f"报告缺 {key}")

    def test_counts_are_reported(self):
        parent, children = clean_case()
        report = validate([parent], children, counter=COUNTER)
        self.assertEqual(report.total_parents, 1)
        self.assertEqual(report.total_children, 2)


class EmptyAndOverLimitTest(unittest.TestCase):
    def test_empty_chunk_is_flagged(self):
        parent = make_parent()
        children = make_children(parent, [parent.content[:50], "   ", parent.content[50:150]])
        report = validate([parent], children, counter=COUNTER, child_max_tokens=500)
        self.assertEqual(report.empty_chunks, 1)
        self.assertFalse(report.ok)
        self.assertIn("empty", report.issues[0].reasons)

    def test_over_limit_chunk_is_flagged(self):
        parent = make_parent(content=unique_text())
        children = make_children(parent, [parent.content[:400], parent.content[400:800]])
        report = validate([parent], children, counter=COUNTER, child_max_tokens=300)
        self.assertEqual(report.over_limit_chunks, 2, "两块都是 400 token，上限 300 → 两块都超限")
        self.assertIn("over_limit", report.issues[0].reasons)

    def test_parent_over_limit_is_flagged(self):
        parent = make_parent(content=unique_text(1000))
        parent.token_count = len(parent.content)
        report = validate([parent], [], counter=COUNTER, parent_max_tokens=1000)
        self.assertEqual(report.over_limit_parents, 1)

    def test_token_count_mismatch_is_a_metadata_error(self):
        parent, children = clean_case()
        children[0].token_count = 9999
        report = validate([parent], children, counter=COUNTER, child_max_tokens=5000)
        self.assertEqual(report.metadata_errors, 1)
        self.assertIn("token_count_mismatch", report.issues[0].reasons)


class DuplicateTest(unittest.TestCase):
    def test_duplicate_ratio_measures_repeated_text(self):
        parent = make_parent(content=unique_text())
        repeated = parent.content[:200]
        children = make_children(
            parent, [repeated, repeated, parent.content[200:400], parent.content[400:600]]
        )
        report = validate([parent], children, counter=COUNTER, child_max_tokens=5000)
        self.assertAlmostEqual(report.duplicate_ratio, 0.25, places=3, msg="4 块里 1 块是重复")

    def test_whitespace_differences_still_count_as_duplicate(self):
        parent = make_parent(content=unique_text())
        children = make_children(
            parent,
            [parent.content[:100], "  " + parent.content[:100] + "  ", parent.content[100:200]],
        )
        report = validate([parent], children, counter=COUNTER, child_max_tokens=5000)
        self.assertAlmostEqual(report.duplicate_ratio, 1 / 3, places=3)


class BoundaryTest(unittest.TestCase):
    def test_boundary_valid_ratio_drops_when_content_not_in_parent(self):
        parent, children = clean_case()
        children[1].content = "这段内容根本不在 parent 里"
        children[1].token_count = len(children[1].content)
        report = validate([parent], children, counter=COUNTER, child_max_tokens=5000)
        self.assertAlmostEqual(report.boundary_valid_ratio, 0.5, places=3)
        self.assertIn("boundary_broken", report.issues[0].reasons)

    def test_offset_mismatch_is_detected(self):
        parent, children = clean_case()
        children[1].source_start = children[1].source_start + 7
        report = validate([parent], children, counter=COUNTER, child_max_tokens=5000)
        self.assertAlmostEqual(report.boundary_valid_ratio, 0.5, places=3)

    def test_overlapping_children_are_valid(self):
        """overlap 让相邻 child 共享内容，这不算边界错误。"""
        parent = make_parent(content="句子内容。" * 200)
        children = make_children(
            parent,
            [parent.content[:300], parent.content[240:540], parent.content[540:840]],
        )
        report = validate([parent], children, counter=COUNTER, child_max_tokens=5000)
        self.assertEqual(report.boundary_valid_ratio, 1.0)


class MultiDocumentTest(unittest.TestCase):
    """回归：parent_index 是文档内序号，跨文档校验不能串台。

    真实场景（40 篇一起校验）曾把边界完整率算成 7.45%，原因就是只按 index
    关联 parent —— A 文档的 child 全被挂到 B 文档的 parent 0 上。
    """

    def test_same_parent_index_in_two_documents_does_not_cross_link(self):
        first_parent, first_children = clean_case()
        second_parent, second_children = clean_case()
        second_parent.document_id = "doc_2"
        second_parent.parent_id = "par_" + "b" * 32
        for index, child in enumerate(second_children):
            child.document_id = "doc_2"
            child.parent_id = second_parent.parent_id
            child.chunk_id = "chk_" + f"{index + 100:032x}"

        report = validate(
            [first_parent, second_parent],
            first_children + second_children,
            counter=COUNTER,
            child_max_tokens=5000,
        )
        self.assertEqual(report.boundary_valid_ratio, 1.0, "跨文档校验不应串台")
        self.assertEqual(report.orphan_children, 0)


class ParentKeyCollisionTest(unittest.TestCase):
    """同一个 (document_id, parent_index) 下有两个不同 parent_id → 必须报出来。"""

    def test_colliding_parent_key_is_reported(self):
        first, first_children = clean_case()
        second, second_children = clean_case()
        second.parent_id = "par_" + "c" * 32  # 同 document_id、同 parent_index，但是不同父块
        report = validate(
            [first, second], first_children + second_children, counter=COUNTER, child_max_tokens=5000
        )
        reasons = [reason for issue in report.issues for reason in issue.reasons]
        self.assertIn("duplicate_parent_key", reasons)


class LinkageTest(unittest.TestCase):
    def test_orphan_child_is_flagged(self):
        parent, children = clean_case()
        children[0].parent_index = 99
        report = validate([parent], children, counter=COUNTER, child_max_tokens=5000)
        self.assertEqual(report.orphan_children, 1)
        self.assertIn("orphan_parent", report.issues[0].reasons)

    def test_missing_section_path_is_metadata_error(self):
        parent, children = clean_case()
        children[0].section_path = []
        report = validate([parent], children, counter=COUNTER, child_max_tokens=5000)
        self.assertEqual(report.metadata_errors, 1)
        self.assertIn("section_path_empty", report.issues[0].reasons)


class InvalidFileTest(unittest.TestCase):
    def test_invalid_rows_are_written_to_jsonl(self):
        parent, children = clean_case()
        children[0].content = "   "
        report = validate([parent], children, counter=COUNTER, child_max_tokens=5000)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.invalid.jsonl"
            report.write_invalid(path)
            self.assertTrue(path.exists())
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), len(report.issues))
            self.assertIn("empty", rows[0]["reasons"])
            self.assertIn("token_count_mismatch", rows[0]["reasons"], "样本 token_count 字段未同步，应一并报出")
            for key in ("target", "id", "reasons", "detail"):
                self.assertIn(key, rows[0])

    def test_no_invalid_file_when_report_is_clean(self):
        parent, children = clean_case()
        report = validate([parent], children, counter=COUNTER, child_max_tokens=5000)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.invalid.jsonl"
            report.write_invalid(path)
            self.assertFalse(path.exists(), "没有问题时不应产生 invalid 文件")


class GoldConvergenceTest(unittest.TestCase):
    """D09 附加项：同一 gold_span 落在几个 chunk 里的分布。

    B02 重切分后「多 gold 收敛」会变多（chunk 更大、更可能同时命中多个 gold），
    这个统计是判断「hit@k 与 full@k 是否还值得分开报」的依据。
    """

    def test_histogram_counts_chunks_per_gold_span(self):
        parent = make_parent(content="甲结论。" + "填充" * 50 + "乙结论。" + "填充" * 50 + "丙结论。")
        third = len(parent.content) // 3
        children = make_children(
            parent,
            [parent.content[:third], parent.content[third: 2 * third], parent.content[2 * third:]],
        )
        report = validate(
            [parent],
            children,
            counter=COUNTER,
            child_max_tokens=5000,
            gold_spans=["甲结论。", "乙结论。", "丙结论。", "根本不存在的结论。"],
        )
        histogram = report.gold_convergence["histogram"]
        self.assertEqual(histogram["0"], 1)
        self.assertEqual(histogram["1"], 3)
        self.assertEqual(histogram["2"], 0)
        self.assertEqual(report.gold_convergence["spans_total"], 4)

    def test_span_found_in_two_chunks_is_counted_as_converged(self):
        """结论句正好横跨第 1、2 块的边界（overlap 场景），应被算作 2 个 chunk 命中。"""
        body = unique_text(200)
        marker = "跨边界的关键结论。"
        content = body[:380] + marker + body[380:]
        parent = make_parent(content=content)
        children = make_children(
            parent,
            [content[100:400], content[350:650], content[650:950]],  # 第 2 块与第 1 块重叠
        )
        report = validate(
            [parent], children, counter=COUNTER, child_max_tokens=5000, gold_spans=[marker]
        )
        self.assertEqual(report.gold_convergence["histogram"]["2"], 1, report.gold_convergence)
        self.assertGreater(report.gold_convergence["converged_ratio"], 0)

    def test_no_gold_spans_means_no_section(self):
        parent, children = clean_case()
        report = validate([parent], children, counter=COUNTER)
        self.assertIsNone(report.gold_convergence)


@unittest.skipUnless(VOCAB_AVAILABLE, f"缺少真词表 {DEFAULT_TOKENIZER_PATH}")
class RealCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = TokenCounter()
        builder = ParentBuilder(counter=cls.counter)
        cls.parents, cls.children = [], []
        for path in sorted(DATA_DIR.rglob("*.md"))[:40]:
            blocks = MarkdownParser().parse(str(path))
            # 每篇文档必须有独立 document_id：parent_index 是文档内序号
            # source_uri 必须是语料根下的相对路径：NVIDIA 语料里有多个同名 overview.md
            document_id = make_document_id(path.relative_to(DATA_DIR).as_posix())
            parents = builder.build(blocks, document_id=document_id, document_version="sha256:x")
            cls.parents.extend(parents)
            cls.children.extend(build_children(parents, counter=cls.counter))

    def test_clean_corpus_has_no_empty_or_over_limit_chunks(self):
        report = validate(
            self.parents, self.children, counter=self.counter, child_max_tokens=500, parent_max_tokens=2000
        )
        print(
            f"\n  [D09] parent {report.total_parents} / child {report.total_children}："
            f"空 {report.empty_chunks}、超限 {report.over_limit_chunks}、"
            f"重复占比 {report.duplicate_ratio:.2%}、边界完整率 {report.boundary_valid_ratio:.2%}"
        )
        self.assertEqual(report.empty_chunks, 0)
        self.assertEqual(report.over_limit_chunks, 0)
        self.assertEqual(report.boundary_valid_ratio, 1.0)

    def test_gold_spans_from_real_text_are_located(self):
        spans = [child.content[:30].strip() for child in self.children[:20] if len(child.content) > 40]
        report = validate(
            self.parents,
            self.children,
            counter=self.counter,
            child_max_tokens=500,
            gold_spans=spans,
        )
        self.assertIsNotNone(report.gold_convergence)
        self.assertGreater(report.gold_convergence["spans_found"], 0)


if __name__ == "__main__":
    unittest.main()
