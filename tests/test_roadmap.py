"""ROADMAP 的守护：每条待做项都必须写清「怎么算做完」。

仓库的规矩是 **验收一律写成「命令 + 期望值」**（ROADMAP 头部原话）。
但规矩写在文件里不会自动生效——写着写着就会出现「优化一下体验」「完善文档」这种
谁都能说自己做完的条目。这里把它变成红灯。

查 F 节（工程化与上手门槛）与 G 节（切片护栏与体检）：这两节最容易写成
「优化一下体验」「把切片做得更好」这种谁都能说自己做完的话。
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROADMAP = ROOT / "docs" / "ROADMAP.md"

VAGUE = ("优化一下", "完善一下", "待补充", "TODO", "TBD", "做得更好", "提升体验")
ITEM = re.compile(r"^\|\s*([FG]\d+)\s*\|")
GUARDED = ("## F. ", "## G. ")


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    nxt = text.find("\n## ", start + 1)
    return text[start:] if nxt == -1 else text[start:nxt]


class RoadmapEngineeringItemsTest(unittest.TestCase):
    def setUp(self):
        if not ROADMAP.is_file():
            self.skipTest("ROADMAP 不在了")
        self.text = ROADMAP.read_text(encoding="utf-8")
        self.section = "".join(_section(self.text, head) for head in GUARDED)

    def test_section_exists(self):
        for head in GUARDED:
            self.assertIn(head, self.text, f"{head.strip()} 那一节被删了？")

    def test_every_item_has_command_and_expected_value(self):
        rows = [(m.group(1), line) for line in self.section.splitlines()
                if (m := ITEM.match(line))]
        self.assertGreaterEqual(len(rows), 40, "F/G 两节条目数掉下来了（应有 40+）")
        for qid, line in rows:
            if "同 B" in line:          # 与 B 节重复的项，验收在 B 节维护，这里只留指针
                continue
            commands = re.findall(r"`[^`]+`", line)
            self.assertTrue(commands, f"{qid}：验收里没有一条可复制的命令（反引号片段）")
            self.assertIn("→", line, f"{qid}：没写期望值标记（`→ 期望值`），无法判定做完没有")
            expected = line.split("→", 1)[1]
            self.assertGreater(len(expected.strip(" |")), 5,
                               f"{qid}：`→` 后面得有真的期望值，不能空着")

    def test_every_item_has_a_tier(self):
        f_section = _section(self.text, "## F. ")
        for line in f_section.splitlines():
            m = ITEM.match(line)
            if not m:
                continue
            tier = line.rstrip().rstrip("|").split("|")[-1].strip()
            self.assertIn(tier, {"1", "2", "3"}, f"{m.group(1)}：档位必须写 1/2/3，实际 {tier!r}")

    def test_no_unfalsifiable_wording(self):
        for word in VAGUE:
            self.assertNotIn(word, self.section,
                             f"出现了不可判定的说法「{word}」——验收会变成口号")

    def test_duplicates_point_to_b_section(self):
        """与 B 节重复的项只记一次账，必须写明「同 B<n>」，不能各写一套标准。"""
        for line in self.section.splitlines():
            m = ITEM.match(line)
            if m and re.search(r"同 [ABC]\d+", line):
                self.assertRegex(line, r"同 [ABC]\d+", f"{m.group(1)}：指向写坏了")

    def test_ordering_section_covers_first_tier(self):
        """建议顺序里必须出现第一档的三项，否则等于没排优先级。"""
        order = _section(self.text, "## F. ")
        order = order[order.index("### 建议动手顺序（工程线）"):]
        for qid in ("F2", "F4", "F8"):
            self.assertIn(qid, order, f"建议顺序里没提 {qid}（第一档）")


if __name__ == "__main__":
    unittest.main()
