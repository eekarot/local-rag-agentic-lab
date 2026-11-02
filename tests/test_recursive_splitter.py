"""D06 Recursive Splitter 的单元测试（不依赖 17 MB 真词表）。

用一个「1 字符 = 1 token」的假计数器，把边界算术变成可以直接手算的整数，
这样超长段落 / 超长句 / 无分隔符 / 超长单词 / 代码块 / 表格六类边界都能
断言到确切的切分结果。真词表下的「token 超限率 == 0」在
tests/test_recursive_splitter_real.py 里单独验。

核心不变式（任何输入都必须成立）：
  1. "".join(pieces) == text          —— 不丢字、不插字（D07 靠它算偏移）
  2. max(count(p) for p in pieces) <= max_tokens —— 超限率 0
  3. 顺序不变、片段非空
"""

import unittest

from src.rag.ingestion.recursive_chunker import (
    SEPARATORS,
    merge_small_parts,
    recursive_split,
    split_by_token_window,
    split_preserving_separator,
)


class CharCounter:
    """1 字符 = 1 token 的假计数器，让边界算术可手算。"""

    def count(self, text):
        return len(text)

    def count_batch(self, texts):
        return [len(t) for t in texts]

    def truncate(self, text, max_tokens):
        if max_tokens <= 0:
            return ""
        return text[:max_tokens]


COUNTER = CharCounter()


def assert_invariants(case, pieces, text, max_tokens):
    case.assertEqual("".join(pieces), text, "拼接后必须与原文逐字符一致")
    for piece in pieces:
        case.assertNotEqual(piece, "", "不允许空片段")
        case.assertLessEqual(COUNTER.count(piece), max_tokens, f"超限片段: {piece[:40]!r}")


# 六类边界样本（D06 验收行点名的那六类）
LONG_PARAGRAPH = "".join(f"line {i} of a very long paragraph\n" for i in range(200))
LONG_SENTENCE = ("这是一个非常长的句子" * 40) + "。"
NO_SEPARATOR = "x" * 1000
LONG_WORD = "supercalifragilisticexpialidocious" * 60
CODE_BLOCK = "".join(f"def f{i}():\n    return {i}\n" for i in range(40))
TABLE_BLOCK = "".join(f"| row{i} | value{i} |\n" for i in range(40))

BOUNDARY_CASES = {
    "long_paragraph": (LONG_PARAGRAPH, 100),
    "long_sentence": (LONG_SENTENCE, 100),
    "no_separator": (NO_SEPARATOR, 100),
    "long_word": (LONG_WORD, 100),
    "code_block": (CODE_BLOCK, 60),
    "table_block": (TABLE_BLOCK, 50),
}


class InvariantTest(unittest.TestCase):
    """六类边界 + 通用不变式。"""

    def test_all_boundary_cases_hold_invariants(self):
        for name, (text, max_tokens) in BOUNDARY_CASES.items():
            with self.subTest(case=name):
                pieces = recursive_split(text, counter=COUNTER, max_tokens=max_tokens)
                assert_invariants(self, pieces, text, max_tokens)

    def test_short_text_is_not_split(self):
        self.assertEqual(recursive_split("短文本", counter=COUNTER, max_tokens=100), ["短文本"])

    def test_text_exactly_at_limit_is_single_piece(self):
        text = "y" * 100
        self.assertEqual(recursive_split(text, counter=COUNTER, max_tokens=100), [text])

    def test_empty_text_yields_no_pieces(self):
        self.assertEqual(recursive_split("", counter=COUNTER, max_tokens=100), [])

    def test_whitespace_only_text_is_kept_verbatim(self):
        text = "\n\n\n"
        self.assertEqual(recursive_split(text, counter=COUNTER, max_tokens=100), [text])

    def test_non_positive_max_tokens_is_rejected(self):
        with self.assertRaises(ValueError):
            recursive_split("abc", counter=COUNTER, max_tokens=0)
        with self.assertRaises(ValueError):
            recursive_split("abc", counter=COUNTER, max_tokens=-5)

    def test_splitting_is_idempotent(self):
        text = LONG_PARAGRAPH
        once = recursive_split(text, counter=COUNTER, max_tokens=100)
        for piece in once:
            self.assertEqual(recursive_split(piece, counter=COUNTER, max_tokens=100), [piece])


class TokenWindowTest(unittest.TestCase):
    def test_window_cuts_at_exact_token_boundaries(self):
        self.assertEqual(
            split_by_token_window("abcdefghij", COUNTER, 3),
            ["abc", "def", "ghi", "j"],
        )

    def test_window_keeps_remainder_verbatim(self):
        text = "a" * 250
        pieces = split_by_token_window(text, COUNTER, 100)
        self.assertEqual([len(p) for p in pieces], [100, 100, 50])
        self.assertEqual("".join(pieces), text)

    def test_window_rejects_non_positive_limit(self):
        with self.assertRaises(ValueError):
            split_by_token_window("abc", COUNTER, 0)


class SeparatorPreservationTest(unittest.TestCase):
    def test_paragraph_separator_stays_with_preceding_piece(self):
        self.assertEqual(
            split_preserving_separator("a.\n\nb.\n\nc", "\n\n"),
            ["a.\n\n", "b.\n\n", "c"],
        )

    def test_chinese_period_is_not_lost(self):
        self.assertEqual(
            split_preserving_separator("第一句。第二句。第三句。", "。"),
            ["第一句。", "第二句。", "第三句。"],
        )

    def test_trailing_separator_does_not_create_empty_piece(self):
        self.assertEqual(split_preserving_separator("a\n\n", "\n\n"), ["a\n\n"])

    def test_missing_separator_returns_whole_text(self):
        self.assertEqual(split_preserving_separator("abc", "\n\n"), ["abc"])


class MergeTest(unittest.TestCase):
    def test_small_parts_are_merged_up_to_target(self):
        self.assertEqual(
            merge_small_parts(["aaa", "bbb", "cc"], COUNTER, max_tokens=6, target_tokens=6),
            ["aaabbb", "cc"],
        )

    def test_merge_never_exceeds_max_tokens(self):
        merged = merge_small_parts(["aaaa", "bbbb"], COUNTER, max_tokens=5, target_tokens=100)
        self.assertEqual(merged, ["aaaa", "bbbb"])

    def test_merge_keeps_text_in_order_and_complete(self):
        parts = ["aaaa", "bb", "cc", "dddd"]
        merged = merge_small_parts(parts, COUNTER, max_tokens=6, target_tokens=6)
        self.assertEqual("".join(merged), "".join(parts))
        for piece in merged:
            self.assertLessEqual(len(piece), 6)

    def test_merge_of_empty_list_is_empty(self):
        self.assertEqual(merge_small_parts([], COUNTER, max_tokens=10, target_tokens=10), [])


class TargetVsMaxTest(unittest.TestCase):
    def test_target_pulls_pieces_above_minimum_when_max_allows(self):
        # 30 个 20 字符的段落行，target=100 → 每块拼到 100，而不是停在 20
        text = "".join(f"{'p' * 19}\n" for _ in range(30))
        pieces = recursive_split(text, counter=COUNTER, max_tokens=120, target_tokens=100)
        self.assertTrue(all(len(p) <= 120 for p in pieces))
        self.assertTrue(
            all(len(p) >= 100 for p in pieces[:-1]),
            f"除最后一块外都应达到 target，实际 {[len(p) for p in pieces]}",
        )
        self.assertEqual("".join(pieces), text)

    def test_target_larger_than_max_is_clamped_safely(self):
        text = "aaaa bbbb cccc dddd"
        pieces = recursive_split(text, counter=COUNTER, max_tokens=10, target_tokens=999)
        self.assertTrue(all(len(p) <= 10 for p in pieces))
        self.assertEqual("".join(pieces), text)


class StructureAwareTest(unittest.TestCase):
    """代码块与表格：短行不能被拦腰切断（行尾换行符是结构信号）。"""

    def test_code_lines_are_not_split_mid_line(self):
        pieces = recursive_split(CODE_BLOCK, counter=COUNTER, max_tokens=60)
        for piece in pieces[:-1]:
            self.assertTrue(piece.endswith("\n"), f"代码块在行中间被切断: {piece!r}")
        self.assertEqual("".join(pieces), CODE_BLOCK)

    def test_table_rows_are_not_split_mid_row(self):
        pieces = recursive_split(TABLE_BLOCK, counter=COUNTER, max_tokens=50)
        for piece in pieces[:-1]:
            self.assertTrue(piece.endswith("\n"), f"表格行被切断: {piece!r}")
            self.assertEqual(piece.count("|"), piece.count("\n") * 3, "表格行数不整")

    def test_paragraph_boundary_preferred_over_line_boundary(self):
        text = "a" * 30 + "\n\n" + "b" * 30 + "\n\n" + "c" * 30
        pieces = recursive_split(text, counter=COUNTER, max_tokens=40, target_tokens=40)
        self.assertEqual(["a" * 30 + "\n\n", "b" * 30 + "\n\n", "c" * 30], pieces)


class SeparatorConfigTest(unittest.TestCase):
    def test_default_separators_match_spec_order(self):
        self.assertEqual(
            SEPARATORS[:2], ("\n\n", "\n")
        )
        self.assertEqual(SEPARATORS[-1], "", "最后一级必须是空串 → token 窗口兜底")
        self.assertIn("。", SEPARATORS)
        self.assertIn(". ", SEPARATORS)

    def test_custom_separators_are_honoured(self):
        pieces = recursive_split(
            "aaa|bbb|ccc", counter=COUNTER, max_tokens=8, separators=("|", "")
        )
        self.assertEqual("".join(pieces), "aaa|bbb|ccc")
        for piece in pieces:
            self.assertLessEqual(len(piece), 8)


if __name__ == "__main__":
    unittest.main()
