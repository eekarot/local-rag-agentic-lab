"""切分的不变量：**不许丢字**、**overlap 不许顶破上限**（G1 / G2，方案 P1.1 + P1.2）。

这两条原来都没有用例守着（N40）：

* `test_children_cover_every_parent` 只断言「每个父块至少有一个子块」——
  一个字符都不许丢这件事，从来没被验证过。子块之间少一个字符，答案就会在
  切断处说不完整，而用例一个都不会红。
* 「含 overlap 的子块不超过 `child_max_tokens`」也从未写死。实现里靠
  `split_max = child_max_tokens - overlap_tokens` 恰好做到，但没人钉住：
  改一个数就会静默超限，然后检索质量漂移没人知道。

口径（与方案 §4-A 一致）：
* 覆盖率在**父块内容坐标**里算：`child.source_start - parent.source_start`。
  不直接用文档坐标，因为块间分隔符会被归一化（`\\n\\n\\n` → `\\n\\n`），
  文档跨度与内容长度本来就有已知偏差（见 `SpanDriftTest` 的实测上界）。
* overlap 语义选定为 **`chunk_size`**：**含** overlap 的子块也必须 ≤ `child_max_tokens`。
"""

import unittest

from src.config import DATA_DIR
from src.rag.ingestion.block import DocumentBlock
from src.rag.ingestion.parent_builder import ParentBuilder, ParentConfig, build_children
from src.rag.ingestion.parser.markdown_parser import MarkdownParser
from src.rag.ingestion.quality_metrics import find_coverage_gaps
from src.rag.ingestion.tokenizer import DEFAULT_TOKENIZER_PATH, TokenCounter
from tests._tokenizer_fixture import ensure_tokenizer

VOCAB_AVAILABLE = DEFAULT_TOKENIZER_PATH.exists() or ensure_tokenizer() is not None

SAMPLE_MD = """# 全文标题

开篇段落，交代背景。

## 第一章 检索

检索的第一段，内容长一点，好让切分有得切。

检索的第二段，再长一点，重复几次同样的句子以凑出足够 token。

### 1.1 向量检索

向量检索的正文，同样长一点，便于触发子块切分。

向量检索的第二段，还是那句：长一点、长一点、再长一点。

## 第二章 生成

生成的正文。
"""


class _BuildMixin:
    counter: TokenCounter

    @classmethod
    def setUpClass(cls):
        ensure_tokenizer()
        cls.counter = TokenCounter()

    def _build(self, text: str = SAMPLE_MD, config: ParentConfig | None = None):
        blocks = MarkdownParser().parse("<sample>", text)
        builder = ParentBuilder(counter=self.counter, config=config)
        parents = builder.build(
            blocks,
            document_id="doc_sample",
            document_version="v1",
            title="样例",
            language="zh-cn",
        )
        children = build_children(
            parents,
            counter=self.counter,
            child_target_tokens=(config or ParentConfig()).child_target_tokens,
            child_max_tokens=(config or ParentConfig()).child_max_tokens,
            overlap_tokens=(config or ParentConfig()).child_overlap_tokens,
        )
        return parents, children


class NoTextLossTest(_BuildMixin, unittest.TestCase):
    """子块必须**无缝**铺满父块内容：缺口 0。"""

    def test_children_tile_parent_without_gap(self):
        parents, children = self._build()
        by_parent = {}
        for child in children:
            by_parent.setdefault(child.parent_id, []).append(child)
        self.assertEqual(len(parents), len(by_parent), "有父块没有任何子块")

        for parent in parents:
            spans = [
                (child.source_start - parent.source_start, child.source_end - parent.source_start)
                for child in by_parent[parent.parent_id]
                if child.source_start is not None
            ]
            self.assertTrue(spans, f"父块 {parent.parent_index} 没有带偏移的子块")
            self.assertEqual(
                find_coverage_gaps(spans, end=len(parent.content)),
                [],
                f"父块 {parent.parent_index} 子块之间/末尾丢了字",
            )

    def test_child_content_matches_its_own_span(self):
        """偏移不只是「覆盖」，还得**对得上**：切出来的字符串就是原文那一段。"""
        parents, children = self._build()
        index = {parent.parent_id: parent for parent in parents}
        for child in children:
            parent = index[child.parent_id]
            start = child.source_start - parent.source_start
            end = child.source_end - parent.source_start
            self.assertEqual(
                parent.content[start:end],
                child.content,
                f"child {child.parent_index}/{child.child_index} 的偏移与内容不符",
            )

    def test_short_document_still_covered(self):
        parents, children = self._build(text="# 只有标题\n\n一句话。\n")
        self.assertEqual(len(parents), 1)
        parent = parents[0]
        spans = [(c.source_start - parent.source_start, c.source_end - parent.source_start) for c in children]
        self.assertEqual(find_coverage_gaps(spans, end=len(parent.content)), [])


def _long_markdown(paragraphs: int = 20, sentence: str = "这一句用来把父块填满，好让子块切到接近上限。") -> str:
    """造一篇「段落够长」的文档：短样例切不到上限，压不出超限 bug。

    段落必须长到能把 `split_max` 填满，否则「含 overlap 是否超限」根本测不出来：
    实测短段落（~35 字）时子块只有 30–40 token，怎么重叠都不会超。
    """
    lines = ["# 长文标题", ""]
    for index in range(paragraphs):
        if index % 5 == 0:
            lines.extend([f"## 第 {index // 5 + 1} 节", ""])
        lines.extend([sentence * 2 + f"（第 {index} 段）", ""])
    return "\n".join(lines)


class OverlapWithinLimitTest(_BuildMixin, unittest.TestCase):
    """含 overlap 的子块也必须 ≤ `child_max_tokens`（语义选 `chunk_size`）。"""

    def test_default_config_respects_limit(self):
        config = ParentConfig()
        _, children = self._build(text=_long_markdown(), config=config)
        self.assertTrue(children)
        for child in children:
            self.assertLessEqual(
                child.token_count,
                config.child_max_tokens,
                f"子块超上限：{child.token_count} > {config.child_max_tokens}",
            )

    def test_tight_limit_still_respected(self):
        """把上限压到 120、overlap 60：实现必须按「留出 overlap 余量」切。"""
        config = ParentConfig(
            max_tokens=2000,
            min_tokens=0,
            child_target_tokens=120,
            child_max_tokens=120,
            child_overlap_tokens=60,
        )
        _, children = self._build(text=_long_markdown(), config=config)
        self.assertTrue(children)
        for child in children:
            self.assertLessEqual(child.token_count, 120, f"子块超上限：{child.token_count} > 120")

    def test_overlap_must_be_smaller_than_limit(self):
        """配置阶段就挡住：overlap ≥ 上限时切不出内容，不许等到建库才发现。"""
        with self.assertRaises(ValueError):
            ParentConfig(child_target_tokens=100, child_max_tokens=100, child_overlap_tokens=100)
        with self.assertRaises(ValueError):
            build_children([], counter=self.counter, child_max_tokens=100, overlap_tokens=100)


class SpanDriftTest(_BuildMixin, unittest.TestCase):
    """记录「文档跨度 ≠ 内容长度」这个已知归一化偏差，防止它变大。

    实测（基准库 7,108 父块）：差值范围 −14…+237，中位 +4 —— 主要来自块间空行被
    归一化成 `\\n\\n`，少数来自合并时跳过的块间内容。**它不影响父块内容坐标下的定位**，
    但会让 `source_start/end` 在文档坐标里偏几段 —— 属于已知限制，不是本次要修的。
    """

    def test_single_block_parent_has_no_drift(self):
        parents, _ = self._build()
        for parent in parents:
            if len(parent.block_indexes) <= 1:
                span = parent.source_end - parent.source_start
                self.assertEqual(span, len(parent.content), f"单块父块 {parent.parent_index} 出现跨度偏差")

    def test_drift_is_bounded(self):
        parents, _ = self._build()
        for parent in parents:
            drift = (parent.source_end - parent.source_start) - len(parent.content)
            self.assertLessEqual(abs(drift), 64, f"父块 {parent.parent_index} 跨度偏差 {drift} 超出预期上限")


class RealCorpusGapTest(unittest.TestCase):
    """真实语料上的同一不变量（缺词表/缺 data 时跳过）。"""

    @classmethod
    def setUpClass(cls):
        if not VOCAB_AVAILABLE:
            raise unittest.SkipTest("缺词表")
        cls.counter = TokenCounter()
        cls.files = sorted(DATA_DIR.rglob("*.md"))[:8]
        if not cls.files:
            raise unittest.SkipTest("data/ 下没有 Markdown")

    def test_no_gap_on_real_documents(self):
        builder = ParentBuilder(counter=self.counter)
        checked = 0
        for path in self.files:
            blocks = MarkdownParser().parse(str(path))
            parents = builder.build(blocks, document_id="doc", document_version="v1")
            if not parents:
                continue
            children = build_children(parents, counter=self.counter)
            by_parent = {}
            for child in children:
                by_parent.setdefault(child.parent_id, []).append(child)
            for parent in parents:
                spans = [
                    (c.source_start - parent.source_start, c.source_end - parent.source_start)
                    for c in by_parent.get(parent.parent_id, [])
                    if c.source_start is not None
                ]
                if not spans:
                    continue
                self.assertEqual(
                    find_coverage_gaps(spans, end=len(parent.content)),
                    [],
                    f"{path.name} 父块 {parent.parent_index} 丢字",
                )
                checked += 1
        self.assertGreater(checked, 0, "没检查到任何父块")


if __name__ == "__main__":
    unittest.main()
