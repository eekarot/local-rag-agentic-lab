"""D08 Parent Builder 的单元测试。

D08 验收：`chunk→parent` 关联成功率 == 100%；parent token ≤ 2000 超限数 == 0。

构建规则来自方案 §2.3：
  1. 章节/标题层级优先作为 Parent 边界
  2. 同章节连续段落可合并
  3. 超长时在子标题或段落边界拆
  4. 太短时可与同一父章节下相邻段落合并
  5. 不跨明显不同的章节主题合并

Child 的 overlap 属于 D07，本轮 Child 只做「不重叠切分 + 关联 parent_id」，
但关联关系必须 100% 成立——否则 D09/D12 全都站不住。
"""

import unittest
from pathlib import Path

from src.config import DATA_DIR
from src.rag.ingestion.block import DocumentBlock
from src.rag.ingestion.parent_builder import ParentBuilder, ParentConfig, build_children
from src.rag.ingestion.parser.markdown_parser import MarkdownParser
from src.rag.ingestion.tokenizer import DEFAULT_TOKENIZER_PATH, TokenCounter

VOCAB_AVAILABLE = DEFAULT_TOKENIZER_PATH.exists()

SAMPLE_MD = """# 全文标题

开篇段落，交代背景。

## 第一章 检索

检索的第一段。

检索的第二段。

### 1.1 向量检索

向量检索的正文。

### 1.2 关键词检索

关键词检索的正文。

## 第二章 切分

切分的第一段。

切分的第二段。
"""


def make_blocks(specs, *, path="doc.md"):
    """specs: [(block_type, text, section_path, heading_level)]"""
    blocks = []
    for index, spec in enumerate(specs):
        block_type, text, section_path = spec[0], spec[1], spec[2]
        heading_level = spec[3] if len(spec) > 3 else None
        blocks.append(
            DocumentBlock(
                block_id=f"doc#{index:04d}",
                block_type=block_type,
                text=text,
                heading_level=heading_level,
                section_path=section_path,
                source_start=index * 100,
                source_end=index * 100 + len(text),
            )
        )
    return blocks


def builder(min_tokens=200, target_tokens=1500, max_tokens=2000):
    """结构类断言要显式关掉合并（min_tokens=0）。

    FakeCounter 是「1 字符 = 1 token」，样例文档整篇才 110 token；默认
    min_tokens=200 会让规则 4 把整篇并成一个 parent，结构断言就全都失去意义。
    合并行为本身由 MergeShortTest 单独钉住。
    """
    return ParentBuilder(
        counter=_FakeCounter(),
        config=ParentConfig(target_tokens=target_tokens, max_tokens=max_tokens, min_tokens=min_tokens),
    )


class _FakeCounter:
    """1 字符 = 1 token，便于精确构造长度边界。"""

    def count(self, text):
        return len(text)

    def count_batch(self, texts):
        return [len(t) for t in texts]

    def truncate(self, text, max_tokens):
        return text[:max_tokens]


class SectionBoundaryTest(unittest.TestCase):
    def test_one_section_becomes_one_parent(self):
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        parents = ParentBuilder(counter=_FakeCounter()).build(blocks)
        self.assertTrue(parents)
        for parent in parents:
            self.assertTrue(parent.section_path, "parent 必须带 section_path")
            self.assertTrue(parent.content.strip())

    def test_heading_text_stays_inside_its_parent(self):
        """「不要把标题和正文分离」：标题文字必须出现在自己的 parent 里。"""
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        parents = builder(min_tokens=0).build(blocks)
        by_heading = {p.section_path[-1]: p for p in parents}
        self.assertIn("第一章 检索", by_heading)
        self.assertTrue(by_heading["第一章 检索"].content.startswith("第一章 检索"))
        self.assertIn("检索的第一段", by_heading["第一章 检索"].content)

    def test_parents_do_not_mix_top_level_chapters(self):
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        parents = ParentBuilder(counter=_FakeCounter()).build(blocks)
        chapters = {p.section_path[0] for p in parents}
        self.assertEqual(chapters, {"全文标题"})
        for parent in parents:
            self.assertNotIn("第二章", parent.content if parent.section_path[1:2] == ["第一章 检索"] else "")

    def test_subsection_gets_its_own_parent(self):
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        parents = builder(min_tokens=0).build(blocks)
        paths = [tuple(p.section_path) for p in parents]
        self.assertIn(("全文标题", "第一章 检索", "1.1 向量检索"), paths)
        self.assertIn(("全文标题", "第一章 检索", "1.2 关键词检索"), paths)

    def test_parent_index_is_sequential(self):
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        parents = ParentBuilder(counter=_FakeCounter()).build(blocks)
        self.assertEqual([p.parent_index for p in parents], list(range(len(parents))))


class SizeLimitTest(unittest.TestCase):
    def test_no_parent_exceeds_max_tokens(self):
        blocks = make_blocks(
            [("heading", "章 一", ["章 一"], 1)] + [("paragraph", "正文" * 200, ["章 一"]) for _ in range(6)]
        )
        config = ParentConfig(target_tokens=1000, max_tokens=2000, min_tokens=100)
        parents = ParentBuilder(counter=_FakeCounter(), config=config).build(blocks)
        self.assertGreater(len(parents), 1, "超长章节必须被拆成多个 parent")
        for parent in parents:
            self.assertLessEqual(parent.token_count, 2000)

    def test_split_happens_at_block_boundary(self):
        block_texts = [f"段落{i}：" + "内容" * 300 for i in range(6)]
        blocks = make_blocks([("heading", "章 一", ["章 一"], 1)] + [("paragraph", t, ["章 一"]) for t in block_texts])
        config = ParentConfig(target_tokens=1000, max_tokens=1200, min_tokens=100)
        parents = ParentBuilder(counter=_FakeCounter(), config=config).build(blocks)
        for parent in parents[1:]:  # 第一个含标题
            self.assertTrue(
                any(parent.content.endswith(text) for text in block_texts),
                "parent 不能在 block 中间截断",
            )

    def test_short_sibling_sections_are_merged(self):
        """规则 4：太短的兄弟小节合并进同一父章节的相邻 parent，且合并后仍不超限。"""
        blocks = make_blocks(
            [
                ("heading", "大章", ["大章"], 1),
                ("paragraph", "主体内容" * 20, ["大章"]),
                ("heading", "小节甲", ["大章", "小节甲"], 2),
                ("paragraph", "甲" * 30, ["大章", "小节甲"]),
                ("heading", "小节乙", ["大章", "小节乙"], 2),
                ("paragraph", "乙" * 30, ["大章", "小节乙"]),
            ]
        )
        config = ParentConfig(target_tokens=500, max_tokens=800, min_tokens=200)
        parents = ParentBuilder(counter=_FakeCounter(), config=config).build(blocks)
        self.assertLess(len(parents), 4, "过短的小节应被合并，否则 parent 碎成一片")
        for parent in parents:
            self.assertLessEqual(parent.token_count, 800)

    def test_merge_never_crosses_top_level_chapter(self):
        blocks = make_blocks(
            [
                ("heading", "第一章", ["第一章"], 1),
                ("paragraph", "短" * 10, ["第一章"]),
                ("heading", "第二章", ["第二章"], 1),
                ("paragraph", "短" * 10, ["第二章"]),
            ]
        )
        config = ParentConfig(target_tokens=500, max_tokens=800, min_tokens=400)
        parents = ParentBuilder(counter=_FakeCounter(), config=config).build(blocks)
        for parent in parents:
            self.assertFalse(
                "第一章" in parent.content and "第二章" in parent.content,
                "不允许跨顶级章节合并",
            )


class DefaultMergeBehaviourTest(unittest.TestCase):
    """默认 min_tokens=200 时，过短章节会被并进同一父章节——行为要钉住，不能靠感觉。"""

    def test_default_config_merges_tiny_document_into_one_parent(self):
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        parents = ParentBuilder(counter=_FakeCounter()).build(blocks)
        self.assertEqual(len(parents), 1, "整篇只有 110 token，默认配置应并成一个 parent")
        self.assertEqual(parents[0].section_path, ["全文标题"], "合并后章节取共同前缀")
        self.assertIn("第二章 切分", parents[0].content)

    def test_merge_can_be_disabled(self):
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        parents = builder(min_tokens=0).build(blocks)
        self.assertGreater(len(parents), 3, "min_tokens=0 时每个章节都应独立成 parent")


class MetadataTest(unittest.TestCase):
    def test_parent_carries_source_span_and_counts(self):
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        parents = builder(min_tokens=0).build(blocks)
        parent = next(p for p in parents if p.section_path[-1] == "第一章 检索")
        self.assertIsNotNone(parent.source_start)
        self.assertIsNotNone(parent.source_end)
        self.assertLessEqual(parent.source_start, parent.source_end)
        self.assertEqual(parent.token_count, len(parent.content))
        self.assertEqual(parent.chunker_version, ParentBuilder.chunker_version)

    def test_document_metadata_is_propagated(self):
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        parents = ParentBuilder(counter=_FakeCounter()).build(
            blocks, document_id="doc_abc", document_version="sha256:deadbeef", title="全文标题", language="zh"
        )
        for parent in parents:
            self.assertEqual(parent.document_id, "doc_abc")
            self.assertEqual(parent.document_version, "sha256:deadbeef")
            self.assertEqual(parent.title, "全文标题")
            self.assertEqual(parent.language, "zh")

    def test_empty_blocks_yield_no_parents(self):
        self.assertEqual(ParentBuilder(counter=_FakeCounter()).build([]), [])

    def test_build_is_deterministic(self):
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        builder = ParentBuilder(counter=_FakeCounter())
        first = [p.to_dict() for p in builder.build(blocks)]
        second = [p.to_dict() for p in builder.build(blocks)]
        self.assertEqual(first, second)


class ChildLinkageTest(unittest.TestCase):
    def _build(self):
        blocks = MarkdownParser().parse("doc.md", text=SAMPLE_MD)
        builder = ParentBuilder(counter=_FakeCounter())
        parents = builder.build(blocks, document_id="doc_abc", document_version="sha256:x")
        children = build_children(parents, counter=_FakeCounter(), child_max_tokens=120, child_target_tokens=100)
        return parents, children

    def test_every_child_links_to_exactly_one_parent(self):
        """D08 验收：chunk→parent 关联成功率 == 100%。"""
        parents, children = self._build()
        parent_ids = {p.parent_index for p in parents}
        self.assertTrue(children)
        for child in children:
            self.assertIn(child.parent_index, parent_ids)

    def test_child_content_is_contained_in_its_parent(self):
        parents, children = self._build()
        by_index = {p.parent_index: p for p in parents}
        for child in children:
            self.assertIn(child.content.strip(), by_index[child.parent_index].content)

    def test_child_token_limit_is_respected(self):
        _, children = self._build()
        for child in children:
            self.assertLessEqual(child.token_count, 120)

    def test_child_section_path_is_inherited(self):
        parents, children = self._build()
        by_index = {p.parent_index: p for p in parents}
        for child in children:
            self.assertEqual(child.section_path, by_index[child.parent_index].section_path)

    def test_child_index_is_sequential_within_parent(self):
        _, children = self._build()
        seen = {}
        for child in children:
            expected = seen.get(child.parent_index, 0)
            self.assertEqual(child.child_index, expected)
            seen[child.parent_index] = expected + 1

    def test_children_cover_every_parent(self):
        parents, children = self._build()
        covered = {child.parent_index for child in children}
        for parent in parents:
            self.assertIn(parent.parent_index, covered, f"parent {parent.parent_index} 没有 child")


@unittest.skipUnless(VOCAB_AVAILABLE, f"缺少真词表 {DEFAULT_TOKENIZER_PATH}")
class RealCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = TokenCounter()
        cls.files = sorted(DATA_DIR.rglob("*.md"))[:15]

    def test_real_documents_respect_limits_and_linkage(self):
        builder = ParentBuilder(counter=self.counter)
        total_parents = total_children = over_limit = orphans = 0
        for path in self.files:
            blocks = MarkdownParser().parse(str(path))
            parents = builder.build(blocks, document_id="doc_x", document_version="sha256:x")
            children = build_children(parents, counter=self.counter)
            total_parents += len(parents)
            total_children += len(children)
            over_limit += sum(1 for p in parents if p.token_count > builder.config.max_tokens)
            parent_indexes = {p.parent_index for p in parents}
            orphans += sum(1 for c in children if c.parent_index not in parent_indexes)
        print(
            f"\n  [D08] {len(self.files)} 篇真实文档 → parent {total_parents}，child {total_children}，"
            f"超限 {over_limit}，孤儿 child {orphans}"
        )
        self.assertEqual(over_limit, 0, "parent token 超限数必须为 0")
        self.assertEqual(orphans, 0, "chunk→parent 关联必须 100%")

    def test_parent_token_distribution_is_reported(self):
        builder = ParentBuilder(counter=self.counter)
        tokens = []
        for path in self.files:
            for parent in builder.build(MarkdownParser().parse(str(path))):
                tokens.append(parent.token_count)
        tokens.sort()
        print(
            f"  [D08] parent token：p50={tokens[len(tokens)//2]} p90={tokens[int(len(tokens)*0.9)]} max={tokens[-1]}"
        )
        self.assertLessEqual(tokens[-1], 2000)


if __name__ == "__main__":
    unittest.main()
