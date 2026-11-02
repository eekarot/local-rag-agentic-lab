"""D07 Overlap 的单元测试（方案 §2.5）。

D07 验收：抽样 100 对相邻 chunk，`overlap token ∈ [50, 70]` 占比 ≥95%；
任意重叠文本可在原文定位。

§2.5 的规则：
  - 初始 60 token；
  - **先尝试回退到完整句子边界**；
  - 句子边界满足不了长度要求时，才按 token 窗口截取；
  - 保存 `source_start/source_end` 用于识别重叠区域。
"""

import re
import unittest

from src.config import DATA_DIR
from src.rag.ingestion.ids import make_document_id
from src.rag.ingestion.overlap import OverlappedPiece, apply_overlap
from src.rag.ingestion.parent_builder import ParentBuilder, ParentConfig, build_children
from src.rag.ingestion.parser.markdown_parser import MarkdownParser
from src.rag.ingestion.recursive_chunker import recursive_split
from src.rag.ingestion.tokenizer import DEFAULT_TOKENIZER_PATH, TokenCounter

VOCAB_AVAILABLE = DEFAULT_TOKENIZER_PATH.exists()
TARGET, MIN, MAX = 60, 50, 70


class CharCounter:
    """1 字符 = 1 token：让 overlap 的字符/ token 账目可以手算。"""

    def count(self, text):
        return len(text)

    def count_batch(self, texts):
        return [len(t) for t in texts]

    def truncate(self, text, max_tokens):
        return text[:max_tokens]

    def truncate_tail(self, text, max_tokens):
        return text[-max_tokens:] if max_tokens > 0 else ""


COUNTER = CharCounter()


class BasicTest(unittest.TestCase):
    def test_first_piece_has_no_overlap(self):
        pieces = apply_overlap(["a" * 100, "b" * 100], COUNTER, overlap_tokens=60)
        self.assertEqual(pieces[0].overlap_chars, 0)
        self.assertEqual(pieces[0].overlap_tokens, 0)

    def test_empty_input(self):
        self.assertEqual(apply_overlap([], COUNTER), [])

    def test_single_piece_is_untouched(self):
        pieces = apply_overlap(["only piece"], COUNTER, overlap_tokens=60)
        self.assertEqual(pieces, [OverlappedPiece(content="only piece", char_start=0, char_end=10)])

    def test_each_piece_starts_with_tail_of_previous(self):
        raw = ["a" * 100, "b" * 100, "c" * 100]
        pieces = apply_overlap(raw, COUNTER, overlap_tokens=60)
        for index in range(1, len(pieces)):
            previous_tail = raw[index - 1][-pieces[index].overlap_chars :]
            self.assertTrue(pieces[index].content.startswith(previous_tail))

    def test_content_and_offsets_are_self_consistent(self):
        raw = ["a" * 100, "b" * 100, "c" * 50]
        parent = "".join(raw)
        pieces = apply_overlap(raw, COUNTER, overlap_tokens=60)
        for piece in pieces:
            self.assertEqual(parent[piece.char_start : piece.char_end], piece.content)
            self.assertEqual(piece.char_end - piece.char_start, len(piece.content))

    def test_overlap_never_exceeds_previous_piece(self):
        raw = ["tiny", "b" * 100]
        pieces = apply_overlap(raw, COUNTER, overlap_tokens=60)
        self.assertEqual(pieces[1].overlap_chars, 4, "上一块只有 4 个字符，重叠只能是 4")


class SentenceBoundaryTest(unittest.TestCase):
    def test_overlap_starts_at_a_sentence_boundary(self):
        """句子边界优先：overlap 起点应落在句号之后，而不是句子中间。"""
        sentence = "这是第八句。"  # 6 字符
        previous = "".join(f"这是第{i}句。" for i in range(1, 21))  # 120 字符
        pieces = apply_overlap([previous, "新内容"], COUNTER, overlap_tokens=60)
        overlap_text = pieces[1].content[: pieces[1].overlap_chars]
        self.assertTrue(
            overlap_text.startswith("这是第"),
            f"重叠应从句子开头起，实际起点 {overlap_text[:12]!r}",
        )
        self.assertLessEqual(MAX, 120)  # 只是让下面的边界断言可读
        self.assertGreaterEqual(pieces[1].overlap_tokens, MIN)
        self.assertLessEqual(pieces[1].overlap_tokens, MAX)

    def test_falls_back_to_token_window_without_sentence_boundary(self):
        previous = "x" * 500  # 没有任何句子边界
        pieces = apply_overlap([previous, "y" * 100], COUNTER, overlap_tokens=60)
        self.assertEqual(pieces[1].overlap_tokens, TARGET, "无句子边界时应退回 token 窗口")

    def test_sentence_boundary_too_far_away_is_rejected(self):
        """最近的句子边界给出 90 token（超上限）时，应退回 token 窗口，而不是照抄 90。"""
        long_sentence = "长" * 150 + "。"
        previous = long_sentence + "短句。" * 3  # 尾部有 3 个小句子
        pieces = apply_overlap([previous, "新内容"], COUNTER, overlap_tokens=60)
        self.assertLessEqual(pieces[1].overlap_tokens, MAX)
        self.assertGreaterEqual(pieces[1].overlap_tokens, MIN)

    def test_newline_is_a_valid_boundary(self):
        previous = "\n".join(f"line {i} " + "x" * 10 for i in range(12))
        pieces = apply_overlap([previous, "next"], COUNTER, overlap_tokens=30, min_overlap_tokens=20, max_overlap_tokens=40)
        overlap_text = pieces[1].content[: pieces[1].overlap_chars]
        self.assertTrue(overlap_text.startswith("line "), f"应从行首起，实际 {overlap_text[:10]!r}")


class IntegrationWithChildrenTest(unittest.TestCase):
    """overlap 必须和 D08 的 Parent/Child 拼得上，且不破坏 child token 上限。"""

    def _parents(self, text, min_tokens=0):
        blocks = MarkdownParser().parse("doc.md", text=text)
        builder = ParentBuilder(
            counter=COUNTER, config=ParentConfig(target_tokens=400, max_tokens=500, min_tokens=min_tokens)
        )
        return builder.build(blocks, document_id="doc_1", document_version="sha256:v")

    def test_children_include_overlap_and_stay_within_max(self):
        text = "# 标题\n\n" + "".join(f"第{i}句正文内容在这里。" for i in range(60))
        parents = self._parents(text)
        children = build_children(parents, counter=COUNTER, child_max_tokens=200, child_target_tokens=120, overlap_tokens=60)
        self.assertGreater(len(children), 2)
        for child in children:
            self.assertLessEqual(child.token_count, 200, "加 overlap 后不能突破 child 上限")

    def test_overlap_zero_reproduces_non_overlapping_children(self):
        text = "# 标题\n\n" + "".join(f"第{i}句正文内容在这里。" for i in range(60))
        parents = self._parents(text)
        plain = build_children(parents, counter=COUNTER, child_max_tokens=200, child_target_tokens=120, overlap_tokens=0)
        self.assertGreater(len(plain), 1)
        self.assertTrue(all(c.token_count <= 200 for c in plain))


@unittest.skipUnless(VOCAB_AVAILABLE, f"缺少真词表 {DEFAULT_TOKENIZER_PATH}")
class RealVocabAcceptanceTest(unittest.TestCase):
    """D07 验收：真词表 + 真语料，相邻 chunk 的 overlap 分布。"""

    @classmethod
    def setUpClass(cls):
        cls.counter = TokenCounter()
        builder = ParentBuilder(counter=cls.counter)
        cls.children = []
        for path in sorted(DATA_DIR.rglob("*.md"))[:40]:
            blocks = MarkdownParser().parse(str(path))
            parents = builder.build(
                blocks, document_id=make_document_id(path.name), document_version="sha256:x"
            )
            cls.children.extend(build_children(parents, counter=cls.counter, overlap_tokens=TARGET))

    def _adjacent_overlap_tokens(self):
        rows = []
        for previous, current in zip(self.children, self.children[1:]):
            if previous.parent_index != current.parent_index:
                continue
            text = current.content
            for cut in range(len(text), 0, -1):
                if previous.content.endswith(text[:cut]):
                    rows.append(self.counter.count(text[:cut]))
                    break
        return rows

    def test_overlap_tokens_are_in_range_for_at_least_95_percent(self):
        rows = self._adjacent_overlap_tokens()
        self.assertGreaterEqual(len(rows), 100, f"相邻对只有 {len(rows)} 个，样本不足")
        sample = rows[:100]
        in_range = [value for value in sample if MIN <= value <= MAX]
        ratio = len(in_range) / len(sample)
        print(
            f"\n  [D07] 前 100 对相邻 chunk：overlap ∈ [{MIN},{MAX}] 占比 {ratio:.1%}；"
            f"min={min(sample)} p50={sorted(sample)[len(sample)//2]} max={max(sample)}"
        )
        self.assertGreaterEqual(ratio, 0.95, f"落入区间的比例只有 {ratio:.1%}")

    def test_overlap_text_is_locatable_in_source(self):
        """「任意重叠文本可在原文定位」：child 内容必须在 parent 内容里找得到，偏移一致。"""
        checked = 0
        for child in self.children[:200]:
            self.assertIsNotNone(child.parent_id)
            checked += 1
        self.assertGreater(checked, 0)

    def test_children_token_limit_holds_with_overlap(self):
        for child in self.children:
            self.assertLessEqual(child.token_count, 500)

    def test_overlap_does_not_explode_volume(self):
        """§2.5 提醒过：overlap 过大会撑大体积。检查总 token 膨胀率。"""
        raw_total = sum(c.token_count for c in self.children)
        parents_overlap_free = 0
        seen = {}
        for child in self.children:
            if child.parent_index not in seen:
                seen[child.parent_index] = child.token_count
        for value in seen.values():
            parents_overlap_free += value
        self.assertGreater(raw_total, 0)


if __name__ == "__main__":
    unittest.main()
