"""重建库之后，题集里的两套 gold id 必须能重算（N42/N43 的配套工具）。

为什么需要它：
- `gold_chunks` 是**父块** id（`par_*`，检索粒度）；切分参数一变，父块划分变了，它就全废；
- `gold_points` 是**子块** id（`chk_*`，向量点）；重叠/边界一变，子块 id 也全废。

题集里另存着一份 `gold_spans`（**语料原文**，出题时就钉死），所以「重算」是可能的：
拿 span 回当前库里找「包含它的子块」，再取这些子块的 parent。

三条纪律：
1. **负样本不动**（它们的正确行为是弃答，本来就没有 gold）；
2. span 定位不到时**保留原值并报出来**，不许静默清空 —— 清空等于把「题坏了」伪装成「题少了」；
3. 其他字段原样保留、行序不变（题集是人工维护的，重排会让 diff 无法 review）。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _chunk(chunk_id, parent_id, text, uri="zh-cn/a.md"):
    return {"chunk_id": chunk_id, "parent_id": parent_id, "text": text,
            "source_uri": uri, "document_id": "doc_1", "lang": "zh-cn"}


class RemapGoldTest(unittest.TestCase):
    def setUp(self):
        from scripts.build_bench_questions import remap_gold

        self.remap = remap_gold
        self.chunks = [
            _chunk("chk_a", "par_1", "第一句在甲块里。第二句也在甲块里。"),
            _chunk("chk_b", "par_1", "第二句也在甲块里。第三句在乙块里。"),
            _chunk("chk_c", "par_2", "另一篇的句子。", uri="en/b.md"),
        ]

    def test_resolves_parents_and_points(self):
        question = {"qid": "q1", "source_path": "zh-cn/a.md", "type": "single_hop",
                    "gold_spans": ["第一句在甲块里。"], "gold_chunks": [], "gold_points": []}
        parents, points, problems = self.remap(question, self.chunks)
        self.assertEqual(points, ["chk_a"])
        self.assertEqual(parents, ["par_1"])
        self.assertEqual(problems, [])

    def test_span_in_two_chunks_maps_to_both(self):
        question = {"qid": "q2", "source_path": "zh-cn/a.md", "type": "multi_hop",
                    "gold_spans": ["第二句也在甲块里。"], "gold_chunks": [], "gold_points": []}
        parents, points, problems = self.remap(question, self.chunks)
        self.assertEqual(points, ["chk_a", "chk_b"])
        self.assertEqual(parents, ["par_1"])
        self.assertEqual(problems, [])

    def test_duplicate_text_on_another_page_still_counts(self):
        """**默认口径是全库匹配**：别的页面上写着同一段话，也算检出了。

        为什么不限制页面：旧题集的 gold 就是这么造的，改它等于换了衡量的东西。
        实测（2026-11-06）限制到 `source_path` 会把这 3 道题判成「检索错了」，
        可它们检回的孪生页面上确实写着同一段话 —— `k8s-e-033` 会从命中直接变不命中。
        """
        chunks = self.chunks + [_chunk("chk_d", "par_9", "第一句在甲块里。", uri="en/other.md")]
        question = {"qid": "q3", "source_path": "zh-cn/a.md", "type": "single_hop",
                    "gold_spans": ["第一句在甲块里。"], "gold_chunks": [], "gold_points": []}
        _, points, _ = self.remap(question, chunks)
        self.assertEqual(points, ["chk_a", "chk_d"])

    def test_same_page_flag_is_stricter(self):
        chunks = self.chunks + [_chunk("chk_d", "par_9", "第一句在甲块里。", uri="en/other.md")]
        question = {"qid": "q3", "source_path": "zh-cn/a.md", "type": "single_hop",
                    "gold_spans": ["第一句在甲块里。"], "gold_chunks": [], "gold_points": []}
        _, points, _ = self.remap(question, chunks, same_page_only=True)
        self.assertEqual(points, ["chk_a"])

    def test_negative_is_left_alone(self):
        question = {"qid": "q4", "source_path": "zh-cn/a.md", "type": "negative",
                    "gold_spans": [], "gold_chunks": ["par_old"], "gold_points": ["chk_old"]}
        parents, points, problems = self.remap(question, self.chunks)
        self.assertEqual((parents, points, problems), (["par_old"], ["chk_old"], []))

    def test_missing_span_keeps_old_ids_and_reports(self):
        question = {"qid": "q5", "source_path": "zh-cn/a.md", "type": "single_hop",
                    "gold_spans": ["这句话在库里根本不存在。"],
                    "gold_chunks": ["par_old"], "gold_points": ["chk_old"]}
        parents, points, problems = self.remap(question, self.chunks)
        self.assertEqual((parents, points), (["par_old"], ["chk_old"]))
        self.assertTrue(problems and "定位不到" in problems[0])

    def test_question_without_spans_is_reported_not_silently_cleared(self):
        question = {"qid": "q6", "source_path": "zh-cn/a.md", "type": "single_hop",
                    "gold_spans": [], "gold_chunks": ["par_old"], "gold_points": ["chk_old"]}
        parents, points, problems = self.remap(question, self.chunks)
        self.assertEqual((parents, points), (["par_old"], ["chk_old"]))
        self.assertTrue(problems and "gold_spans" in problems[0])


class CliContractTest(unittest.TestCase):
    def test_refresh_ids_subcommand_exists(self):
        from scripts.build_bench_questions import build_parser

        args = build_parser().parse_args(["refresh-ids", "--dry-run"])
        self.assertTrue(args.dry_run)
        self.assertFalse(args.same_page, "默认必须与原出题口径一致（全库匹配）")
        self.assertEqual(args.cmd, "refresh-ids")

    def test_help_mentions_span_remap(self):
        from scripts.build_bench_questions import build_parser

        help_text = build_parser().format_help()
        self.assertIn("refresh-ids", help_text)


if __name__ == "__main__":
    unittest.main()
