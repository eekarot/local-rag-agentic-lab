"""CI 工作流自身也要有守卫（2026-11-06 踩过）。

事故：为了「失败时把红在哪写进 annotation」，我在 `run: |` 里写了一句
`tr '\\n' '|'`，落到文件里变成一个**真实的换行** —— YAML 的块标量在零缩进行处提前结束，
整个 workflow 变成非法 YAML。GitHub 的表现是：run 立刻失败、**jobs 列表为空**，
而 annotation 只有一句 `Process completed with exit code 1.`。

也就是说：**CI 坏了但没有任何测试会红**，只能靠人去翻 run 详情。所以要两条守卫：

1. 能装 pyyaml 就真解析一遍（结构对不对）；
2. 不能装也不放过「块标量被截断」这个具体形状：`run: |` 之后的第一条命令缩进就是块缩进，
   后面每一行非空行都必须**至少**这么缩进，否则说明块已经断了。

第 2 条不依赖任何第三方库，正是这次出事的形状。
"""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


class WorkflowShapeTest(unittest.TestCase):
    def setUp(self):
        self.text = WORKFLOW.read_text(encoding="utf-8")
        self.lines = self.text.splitlines()

    def test_block_scalars_are_not_truncated(self):
        problems = []
        for index, line in enumerate(self.lines):
            if not re.match(r"^\s*(run|script):\s*\|\s*$", line):
                continue
            indent = len(line) - len(line.lstrip())
            body_indent = None
            for offset in range(index + 1, len(self.lines)):
                candidate = self.lines[offset]
                if not candidate.strip():
                    continue
                current = len(candidate) - len(candidate.lstrip())
                if current <= indent:            # 回到 key 缩进 = 块结束（正常出口）
                    break
                if body_indent is None:
                    body_indent = current
                elif current < body_indent:
                    problems.append(f"{WORKFLOW.name}:{offset + 1} 缩进比块首行浅（块可能已被截断）")
                    break
        self.assertEqual(problems, [], "\n".join(problems))

    def test_run_blocks_have_no_unindented_lines(self):
        """块内出现零缩进行 = 这个块已经被切断（本次事故的原始形状）。"""
        bad = []
        inside = False
        indent = 0
        for offset, line in enumerate(self.lines, start=1):
            if re.match(r"^\s*(run|script):\s*\|\s*$", line):
                inside = True
                indent = len(line) - len(line.lstrip())
                continue
            if inside:
                if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                    inside = False
                    continue
                if line.strip() and (len(line) - len(line.lstrip())) == 0:
                    bad.append(f"第 {offset} 行没有缩进：{line[:40]!r}")
        self.assertEqual(bad, [], "\n".join(bad))

    def test_parses_with_pyyaml_when_available(self):
        try:
            import yaml
        except ImportError:                    # dev 依赖里没有 pyyaml 时跳过（第 2 条仍然生效）
            self.skipTest("没装 pyyaml：跳过结构解析，形状守卫仍然在跑")
        payload = yaml.safe_load(self.text)
        steps = payload["jobs"]["default-suite"]["steps"]
        names = [step.get("name") for step in steps]
        self.assertIn("默认套件（不联网，期望 <60 s）", names)
        self.assertIn("法务与清单完整性", names)


if __name__ == "__main__":
    unittest.main()
