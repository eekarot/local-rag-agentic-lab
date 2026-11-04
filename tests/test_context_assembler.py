"""D32 Context Assembler 用例。

要拦住的四类真实问题（都来自 §17 差距清单第 2057 行）：

1. **同一段话进两次**：top-4 里同文档多 chunk 占 79.6%，其中相邻重叠块 22.9%；
2. **父块相邻却当两条**：相邻父块自带 50–70 token 重叠（D07 的 overlap），
   直接拼进 context 等于把同一句话喂两遍；
3. **预算失控**：`format_context()` 是「全量拼接」，10 个父块就 7,000+ token，
   没有上限，模型看一半丢一半；
4. **来源不可定位**：拼出来的文本里没有 `document_id / section_path`，答案层没法给引用。

所以组装器必须同时做到：去重、相邻合并（合并时剥掉重复尾巴）、6000 token 上限、
每条带来源字段，并且**能把「省了多少」量出来**（`adjacent_overlap_ratio`）。
"""

import unittest

from src.rag.context.assembler import (
    DEFAULT_BUDGET,
    AssembledContext,
    ContextBlock,
    adjacent_overlap_ratio,
    assemble,
    document_fragmentation,
    merge_parent_texts,
)


class FakeCounter:
    """4 字符 ≈ 1 token；truncate 按字符切，够验证预算逻辑。"""

    def count(self, text):
        return max(0, len(text) // 4)

    def truncate(self, text, max_tokens):
        return text[: max_tokens * 4]

    def truncate_tail(self, text, max_tokens):
        return text[-max_tokens * 4:] if max_tokens else ""


def _parent(pid, doc, index, text, section=("A",), source="file.md"):
    return {
        "id": pid,
        "parent_id": pid,
        "document_id": doc,
        "parent_index": index,
        "content": text,
        "text": text,
        "section_path": list(section),
        "title": doc,
        "source_uri": source,
    }


def _hits(*pairs):
    return [{"parent_id": pid, "score": score, "rerank_score": None} for pid, score in pairs]


class MergeTest(unittest.TestCase):
    def test_overlapping_tail_is_stripped_once(self):
        left = "前面的话。" * 20 + "重复的接缝句这一句在两边都有。" * 3
        right = "重复的接缝句这一句在两边都有。" * 3 + "后面的话。" * 20
        merged = merge_parent_texts([left, right])
        self.assertEqual(merged.count("重复的接缝句这一句在两边都有。"), 3)

    def test_no_overlap_just_joins_with_blank_line(self):
        self.assertEqual(merge_parent_texts(["甲。" * 10, "乙。" * 10]).count("甲。"), 10)
        self.assertIn("\n\n", merge_parent_texts(["甲。" * 10, "乙。" * 10]))

    def test_single_text_returned_as_is(self):
        self.assertEqual(merge_parent_texts(["单块"]), "单块")


class AssembleDedupeTest(unittest.TestCase):
    def setUp(self):
        self.counter = FakeCounter()

    def test_same_parent_twice_kept_once(self):
        parents = {"p1": _parent("p1", "d1", 0, "正文一。" * 20)}
        ctx = assemble(_hits(("p1", 0.9), ("p1", 0.5)), parents, counter=self.counter)
        self.assertEqual(len(ctx.blocks), 1)
        self.assertEqual(ctx.stats["duplicate_parent"], 1)

    def test_identical_text_in_different_parents_deduped(self):
        text = "完全一样的正文。" * 20
        parents = {"p1": _parent("p1", "d1", 0, text), "p2": _parent("p2", "d2", 0, text)}
        ctx = assemble(_hits(("p1", 0.9), ("p2", 0.8)), parents, counter=self.counter)
        self.assertEqual(len(ctx.blocks), 1)
        self.assertEqual(ctx.stats["duplicate_text"], 1)

    def test_near_duplicate_versions_dropped(self):
        # 真实场景：多版本文档只改了一两个词。注意别用「同一句重复 30 遍」造数据——
        # shingle 是集合，重复会把集合压小，Jaccard 反而失真
        base = "说明：" + "".join(f"第{i}行内容各不相同的数据与结论。" + chr(10) for i in range(40))
        variant = base.replace("说明：", "声明：", 1)
        parents = {"p1": _parent("p1", "d1", 0, base), "p2": _parent("p2", "d2", 0, variant)}
        ctx = assemble(_hits(("p1", 0.9), ("p2", 0.8)), parents, counter=self.counter)
        self.assertEqual(len(ctx.blocks), 1)
        self.assertEqual(ctx.stats["near_duplicate_text"], 1)

    def test_short_texts_are_not_near_duplicate_checked(self):
        parents = {"p1": _parent("p1", "d1", 0, "短句一。"), "p2": _parent("p2", "d2", 0, "短句一。改")}
        ctx = assemble(_hits(("p1", 0.9), ("p2", 0.8)), parents, counter=self.counter)
        self.assertEqual(ctx.stats["near_duplicate_text"], 0)

    def test_dedupe_can_be_disabled(self):
        text = "完全一样的正文。" * 20
        parents = {"p1": _parent("p1", "d1", 0, text), "p2": _parent("p2", "d2", 0, text)}
        ctx = assemble(_hits(("p1", 0.9), ("p2", 0.8)), parents, counter=self.counter, dedupe=False)
        self.assertEqual(len(ctx.blocks), 2)


class AssembleMergeTest(unittest.TestCase):
    def setUp(self):
        self.counter = FakeCounter()

    def test_adjacent_parents_merged_into_one_block(self):
        seam = "接缝那句话在两边都有。" * 4      # 44 字符，过 MIN_SEAM_CHARS 的信任线
        parents = {
            "p1": _parent("p1", "d1", 0, "第一段正文。" * 15 + seam),
            "p2": _parent("p2", "d1", 1, seam + "第二段正文。" * 15),
        }
        ctx = assemble(_hits(("p1", 0.9), ("p2", 0.8)), parents, counter=self.counter)
        self.assertEqual(len(ctx.blocks), 1)
        self.assertEqual(ctx.stats["merged"], 1)
        self.assertEqual(ctx.blocks[0].parent_ids, ["p1", "p2"])
        self.assertEqual(ctx.blocks[0].text.count("接缝那句话在两边都有。"), 4)

    def test_non_adjacent_parents_not_merged(self):
        parents = {
            "p1": _parent("p1", "d1", 0, "第一段。" * 20),
            "p2": _parent("p2", "d1", 5, "第五段。" * 20),
        }
        ctx = assemble(_hits(("p1", 0.9), ("p2", 0.8)), parents, counter=self.counter)
        self.assertEqual(len(ctx.blocks), 2)
        self.assertEqual(ctx.stats["merged"], 0)

    def test_different_documents_not_merged(self):
        parents = {
            "p1": _parent("p1", "d1", 0, "甲文正文。" * 20),
            "p2": _parent("p2", "d2", 1, "乙文正文。" * 20),
        }
        ctx = assemble(_hits(("p1", 0.9), ("p2", 0.8)), parents, counter=self.counter)
        self.assertEqual(len(ctx.blocks), 2)

    def test_merge_can_be_disabled(self):
        parents = {
            "p1": _parent("p1", "d1", 0, "甲。" * 20),
            "p2": _parent("p2", "d1", 1, "乙。" * 20),
        }
        ctx = assemble(_hits(("p1", 0.9), ("p2", 0.8)), parents, counter=self.counter,
                       merge_adjacent=False)
        self.assertEqual(len(ctx.blocks), 2)


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.counter = FakeCounter()

    def test_default_budget_is_6000(self):
        self.assertEqual(DEFAULT_BUDGET, 6000)

    def test_total_tokens_never_exceeds_budget(self):
        text = "很长的正文内容。" * 4000          # 32,000 字符 ≈ 8,000 token
        parents = {"p1": _parent("p1", "d1", 0, text)}
        ctx = assemble(_hits(("p1", 0.9)), parents, counter=self.counter, budget=1000)
        self.assertLessEqual(ctx.tokens, 1000)
        self.assertTrue(ctx.blocks[0].truncated)

    def test_blocks_beyond_budget_are_dropped_and_recorded(self):
        parents = {f"p{i}": _parent(f"p{i}", f"d{i}", 0, f"正文{i}。" * 100) for i in range(1, 6)}
        hits = _hits(*[(f"p{i}", 1.0 - i / 10) for i in range(1, 6)])
        ctx = assemble(hits, parents, counter=self.counter, budget=120)
        self.assertLess(len(ctx.blocks), 5)
        self.assertTrue(ctx.dropped)
        self.assertTrue(all(item["reason"] == "budget" for item in ctx.dropped))

    def test_rank_order_preserved(self):
        parents = {f"p{i}": _parent(f"p{i}", f"d{i}", 0, f"正文{i}。" * 50) for i in range(1, 4)}
        ctx = assemble(_hits(("p3", 0.9), ("p1", 0.5), ("p2", 0.1)), parents, counter=self.counter)
        self.assertEqual([block.parent_ids[0] for block in ctx.blocks], ["p3", "p1", "p2"])


class SourceFieldTest(unittest.TestCase):
    def test_every_block_carries_source_fields(self):
        parents = {"p1": _parent("p1", "d1", 0, "正文。" * 50, section=("第1章", "1.2 小节"))}
        ctx = assemble([{"parent_id": "p1", "score": 0.7}], parents, counter=FakeCounter())
        block = ctx.blocks[0]
        self.assertEqual(block.document_id, "d1")
        self.assertEqual(block.parent_ids, ["p1"])
        self.assertEqual(block.section_path, ["第1章", "1.2 小节"])
        self.assertEqual(block.source_uri, "file.md")
        self.assertEqual(block.index, 1)
        self.assertIn("document_id", block.as_dict())

    def test_missing_parent_is_recorded_not_crashed(self):
        ctx = assemble([{"parent_id": "nope", "score": 0.7}], {}, counter=FakeCounter())
        self.assertEqual(ctx.blocks, [])
        self.assertEqual(ctx.stats["missing_parent"], 1)


class OverlapMetricTest(unittest.TestCase):
    """验收指标：上下文内「同文档相邻重叠块」占比，要从 22.9% 降到 ≤5%。

    判据有两条，命中任一即算「相邻重叠」：
    ① 同文档且 `parent_index` 相接；② 同文档且两段正文在接缝处有明显重叠（序号缺失时的退路，
       当前库里 `parent_index` 全为 NULL，见 N6）。
    """

    def _block(self, index, text, doc="d1", parent_indexes=()):
        return ContextBlock(index=index, document_id=doc, parent_ids=[f"p{index}"],
                            parent_indexes=list(parent_indexes), chunk_ids=[], section_path=[],
                            source_uri="", title="", text=text, tokens=1, score=1.0)

    def test_consecutive_indexes_count_as_overlap(self):
        blocks = [self._block(1, "甲", parent_indexes=[3]), self._block(2, "乙", parent_indexes=[4])]
        self.assertEqual(adjacent_overlap_ratio(blocks), 1.0)

    def test_seam_overlap_counts_even_without_indexes(self):
        seam = "接缝那句话在两边都有。" * 4
        blocks = [self._block(1, "前文。" * 20 + seam), self._block(2, seam + "后文。" * 20)]
        self.assertEqual(adjacent_overlap_ratio(blocks), 1.0)

    def test_far_apart_indexes_do_not_count(self):
        blocks = [self._block(1, "甲", parent_indexes=[124]), self._block(2, "乙", parent_indexes=[104])]
        self.assertEqual(adjacent_overlap_ratio(blocks), 0.0)

    def test_unrelated_blocks_of_same_doc_do_not_count(self):
        blocks = [self._block(1, "甲。"), self._block(2, "乙。")]
        self.assertEqual(adjacent_overlap_ratio(blocks), 0.0)

    def test_merged_block_counts_as_one(self):
        parents = {
            "p1": _parent("p1", "d1", 0, "甲。" * 20),
            "p2": _parent("p2", "d1", 1, "乙。" * 20),
        }
        ctx = assemble(_hits(("p1", 1.0), ("p2", 0.9)), parents, counter=FakeCounter())
        self.assertEqual(adjacent_overlap_ratio(ctx.blocks), 0.0)

    def test_empty_is_zero(self):
        self.assertEqual(adjacent_overlap_ratio([]), 0.0)

    def test_document_fragmentation(self):
        blocks = [self._block(1, "甲", doc="d1"), self._block(2, "乙", doc="d1"),
                  self._block(3, "丙", doc="d2")]
        self.assertAlmostEqual(document_fragmentation(blocks), 1 - 2 / 3)
        self.assertEqual(document_fragmentation([]), 0.0)


class RenderTest(unittest.TestCase):
    def test_render_numbers_blocks_and_shows_sources(self):
        parents = {"p1": _parent("p1", "d1", 0, "正文。" * 50, section=("第1章",))}
        ctx = assemble([{"parent_id": "p1", "score": 0.7}], parents, counter=FakeCounter())
        rendered = ctx.render()
        self.assertIn("[证据 1]", rendered)
        self.assertIn("d1", rendered)
        self.assertIn("第1章", rendered)
        self.assertIsInstance(ctx, AssembledContext)

    def test_render_is_empty_for_no_blocks(self):
        self.assertEqual(AssembledContext(blocks=[], tokens=0, dropped=[], stats={}).render(), "")


if __name__ == "__main__":
    unittest.main()
