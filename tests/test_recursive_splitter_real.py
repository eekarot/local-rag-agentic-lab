"""D06 的真词表验收：token 超限率必须为 0。

D06 验收行要求「`token 超限率 == 0`」，这只有用 BGE-M3 真词表才算数——
假计数器证明不了 tokenizer 在窗口边界上的行为（重新 tokenize 前缀可能多出 token）。

真词表缺失时整体跳过；不需要 Ollama（纯本地 CPU）。
"""

import io
import json
import random
import unittest
from contextlib import redirect_stdout

from src.config import STORAGE_DIR
from src.rag.ingestion.recursive_chunker import cli_main, recursive_split
from src.rag.ingestion.tokenizer import DEFAULT_TOKENIZER_PATH, TokenCounter
from tests._slow import run_slow, slow

VOCAB_AVAILABLE = DEFAULT_TOKENIZER_PATH.exists()
INDEX_PATH = STORAGE_DIR / "index.json"

# 方案 §2.3 的初始参数
CHILD_TARGET = 400
CHILD_MAX = 500
MIN_CHILD = 80


def load_real_chunks(n, seed=20261102):
    """从现有索引里抽样真实 chunk（混合中英文）。"""
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    texts = [record["text"] for record in payload["records"]]
    texts.sort(key=len)
    # 分成三段抽样：短/中/长，保证覆盖「不用切」「刚好要切」「切很多段」
    third = len(texts) // 3
    rng = random.Random(seed)
    buckets = (texts[:third], texts[third : 2 * third], texts[2 * third :])
    per_bucket = max(1, n // 3)
    return [rng.choice(bucket) for bucket in buckets for _ in range(per_bucket)]


def load_real_documents(n=40, chunks_per_doc=8, seed=7):
    """把相邻真实 chunk 拼成「长文档」，用来压到真正需要切分的路径。

    现有索引是「500 字符」切的，单条最大也只有 300 多 token，全部低于
    500 token 的 child 上限——这本身就是 B02（按 token 重切分）的理由，
    但也意味着单条 chunk 无法验证切分逻辑，必须拼长。
    """
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    texts = [record["text"] for record in payload["records"]]
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        start = rng.randrange(0, len(texts) - chunks_per_doc)
        docs.append("\n\n".join(texts[start : start + chunks_per_doc]))
    return docs


@unittest.skipUnless(VOCAB_AVAILABLE, f"缺少真词表 {DEFAULT_TOKENIZER_PATH}")
@slow
class RealVocabSplitterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = TokenCounter()
        cls.chunks = load_real_chunks(300)

    def test_no_piece_exceeds_child_max_on_real_chunks(self):
        """300 条真实 chunk：超限率必须为 0。"""
        over = []
        for text in self.chunks:
            pieces = recursive_split(
                text,
                counter=self.counter,
                max_tokens=CHILD_MAX,
                target_tokens=CHILD_TARGET,
            )
            for piece in pieces:
                if self.counter.count(piece) > CHILD_MAX:
                    over.append((self.counter.count(piece), piece[:60]))
        self.assertEqual(over, [], f"超限片段 {len(over)} 个：{over[:3]}")

    def test_pieces_rejoin_to_original_text(self):
        """拼接必须能逐字符还原，否则 D07 的 source_start/end 一定错。"""
        for text in self.chunks:
            with self.subTest(chunk=text[:40]):
                pieces = recursive_split(
                    text, counter=self.counter, max_tokens=CHILD_MAX, target_tokens=CHILD_TARGET
                )
                self.assertEqual("".join(pieces), text)

    def test_long_chunks_are_actually_split(self):
        """拼出来的长文档必须真的被切成多段，否则上面的用例等于没测。

        顺带记录一条 B02 的事实：现有索引（500 字符切分）里最长的 chunk
        也只有几百 token，全部低于 500 token 的 child 上限。
        """
        stored = [self.counter.count(t) for t in self.chunks]
        print(
            f"\n  [D06 实测] 现有索引抽样 {len(stored)} 条 chunk 的 token 数："
            f"p50={sorted(stored)[len(stored) // 2]} max={max(stored)}"
        )
        docs = load_real_documents(40, chunks_per_doc=8)
        split_docs = 0
        for doc in docs:
            pieces = recursive_split(
                doc, counter=self.counter, max_tokens=CHILD_MAX, target_tokens=CHILD_TARGET
            )
            self.assertEqual("".join(pieces), doc, "拼接必须还原原文")
            self.assertTrue(all(self.counter.count(p) <= CHILD_MAX for p in pieces))
            if len(pieces) > 1:
                split_docs += 1
        self.assertGreater(split_docs, 20, f"只有 {split_docs}/40 个长文档被切开，用例没压到切分路径")

    def test_no_separator_blob_uses_token_window(self):
        """无任何分隔符的长串（base64 / 长 ID / 连写文本）走 token 窗口兜底。"""
        blob = "A1b2C3d4" * 800  # 6400 字符，无空格、无标点、无换行
        self.assertGreater(self.counter.count(blob), CHILD_MAX)
        pieces = recursive_split(blob, counter=self.counter, max_tokens=CHILD_MAX)
        self.assertGreater(len(pieces), 1)
        self.assertEqual("".join(pieces), blob)
        self.assertTrue(all(self.counter.count(p) <= CHILD_MAX for p in pieces))

    def test_overlong_single_word_is_split(self):
        word = "supercalifragilisticexpialidocious" * 400
        pieces = recursive_split(word, counter=self.counter, max_tokens=CHILD_MAX)
        self.assertEqual("".join(pieces), word)
        self.assertTrue(all(self.counter.count(p) <= CHILD_MAX for p in pieces))

    def test_real_chinese_long_paragraph_is_split_on_sentence_boundary(self):
        """中文超长段落应优先在句号处断，而不是在字符中间断。"""
        text = "知识库检索需要先切分再向量化，否则语义会被稀释。" * 120
        pieces = recursive_split(
            text, counter=self.counter, max_tokens=CHILD_MAX, target_tokens=CHILD_TARGET
        )
        self.assertGreater(len(pieces), 1)
        for piece in pieces[:-1]:
            self.assertTrue(piece.rstrip().endswith("。"), f"未在句号处切分: {piece[-30:]!r}")

    def test_fragment_ratio_is_reported(self):
        """统计碎块率（< min_child token 的片段占比），供 D09 Validator 设阈值参考。"""
        total = fragments = 0
        for text in self.chunks:
            pieces = recursive_split(
                text, counter=self.counter, max_tokens=CHILD_MAX, target_tokens=CHILD_TARGET
            )
            total += len(pieces)
            fragments += sum(1 for p in pieces if self.counter.count(p) < MIN_CHILD)
        ratio = fragments / total if total else 0.0
        print(f"\n  [D06 实测] 片段总数 {total}，碎块(<{MIN_CHILD} token) {fragments}，占比 {ratio:.1%}")
        self.assertLess(ratio, 0.10, f"碎块率 {ratio:.1%} 过高，merge 策略需要调整")


@slow
class SplitterCliTest(unittest.TestCase):
    """验收命令要能当一行命令跑，期望值写死在断言里。"""

    def run_cli(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(argv)
        return code, out.getvalue()

    def test_split_command_prints_each_piece(self):
        code, out = self.run_cli(["split", "abcdefghij", "--max-tokens", "3"])
        self.assertEqual(code, 0)
        lines = [line for line in out.splitlines() if line.strip()]
        self.assertGreater(len(lines), 1, "3 token 上限下应被切成多段")
        pieces, tokens = [], []
        for line in lines:
            head, piece = line.split("tok  ", 1)
            tokens.append(int(head.split()[-1]))
            pieces.append(piece)
        self.assertEqual("".join(pieces), "abcdefghij", "各段拼接必须还原原文")
        self.assertTrue(all(t <= 3 for t in tokens), f"存在超限段: {tokens}")

    def test_check_command_reports_zero_over_limit(self):
        code, out = self.run_cli(["check", "--n", "300"])
        self.assertEqual(code, 0)
        self.assertIn("超限率 0.00%", out)
        self.assertIn("还原失败 0", out)

    def test_check_command_rejects_unknown_index(self):
        code, out = self.run_cli(["check", "--n", "5", "--index", "nope/missing.json"])
        self.assertEqual(code, 2)
        self.assertIn("找不到索引", out)


if __name__ == "__main__":
    unittest.main()
