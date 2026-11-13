"""架构说明文档的守护：数字必须与基线/证据逐位一致，表格不能塌。

`docs/architecture.md` 是「这系统由哪些部件组成」的唯一总览。它最容易烂在两处：

1. **数字漂移**：基线一重立，文档里的 `hit@10 0.8571` 就变成历史遗迹，
   而读者没法判断这是旧数还是当前数。所以这里直接从基线 JSON 里取真值，
   反过来要求文档里出现的就是这个格式化结果 —— 数字变了，红灯会告诉你该改文档。
2. **表格塌列**：单元格里出现没转义的 `|` 会让整张表在渲染时错位，
   而纯文本读起来一切正常（人眼看不出来）。

只查断言得到的东西，不查措辞。
"""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "architecture.md"
BASELINE = ROOT / "docs" / "baseline-k8s-v1.json"
BEIR_BASELINE = ROOT / "docs" / "evidence" / "beir-scifact-v1-baseline.json"
C0B = ROOT / "docs" / "evidence" / "c0b-rerank-pool-ablation.json"

#: 本机绝对路径 / 个人信息：入库文件里一律不许出现
LOCAL_PATHS = ("C:/", "C:\\", "/Users/", "Desktop", "@gmail", "@qq.com")


def _cells(line: str) -> list[str]:
    protected = line.strip().replace("\\|", "\x00")
    cells = protected.split("|")
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1].strip() == "":
        cells = cells[:-1]
    return cells


class ArchitectureDocTest(unittest.TestCase):
    def setUp(self):
        if not DOC.is_file():
            self.skipTest("架构说明还没写")
        self.text = DOC.read_text(encoding="utf-8")

    def test_three_sections_exist(self):
        for head in ("## 1. 整体架构", "## 2. 测试、基准", "## 3. 未来优化方向"):
            self.assertIn(head, self.text, f"{head} 那一节被删了？")

    def test_k8s_baseline_numbers_match_json(self):
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        overall = baseline["overall"]
        for metric, label in (("hit@10", "hit@10"), ("full@10", "full@10"),
                              ("recall@10", "recall@10"), ("mrr", "MRR")):
            value = overall[metric]
            self.assertIn(f"**{value:.4f}**", self.text,
                          f"文档里的 {label} 与基线 {value:.4f} 不一致（基线重立了？改文档）")
        latency = baseline["latency_ms"]
        for key in ("p50", "p90"):
            self.assertIn(f"{latency[key]:.1f}", self.text,
                          f"文档里的延迟 {key}={latency[key]:.1f} ms 与基线不一致")

    def test_scifact_numbers_match_evidence(self):
        data = json.loads(BEIR_BASELINE.read_text(encoding="utf-8"))
        overall = data["overall"]
        for metric in ("ndcg@10", "recall@10", "hit@10", "mrr@10"):
            self.assertIn(f"{overall[metric]:.4f}", self.text,
                          f"SciFact 的 {metric}={overall[metric]:.4f} 没写对（或证据更新了）")

    def test_rerank_pool_ablation_numbers_match_evidence(self):
        data = json.loads(C0B.read_text(encoding="utf-8"))
        hit = data["configs"]["rerank40"]["hit@10"]
        norerank = data["configs"]["norerank"]["hit@10"]
        self.assertIn(f"{hit:.4f}", self.text, "pool=40 的 hit@10 与证据不一致")
        self.assertIn(f"{norerank:.4f}", self.text, "关重排的 hit@10 与证据不一致")

    def test_tables_do_not_collapse(self):
        """每张表的每一行列数必须与表头一致（单元格里的 `|` 要写成 `\\|`）。"""
        blocks, current = [], []
        for line in self.text.splitlines():
            if line.lstrip().startswith("|"):
                current.append(line)
            else:
                if current:
                    blocks.append(current)
                current = []
        if current:
            blocks.append(current)

        self.assertGreaterEqual(len(blocks), 10, "架构文档里的表格少了（应有 10+ 张）")
        for block in blocks:
            counts = [len(_cells(line)) for line in block]
            self.assertEqual(len(set(counts)), 1,
                             f"表格列数不一致：表头 {counts[0]} 列 vs {counts}：{block[0][:70]}")

    def test_no_local_paths_or_personal_info(self):
        for needle in LOCAL_PATHS:
            self.assertNotIn(needle, self.text, f"文档里出现了本机路径/个人信息：{needle}")

    def test_commands_are_runnable_shapes(self):
        """文档里给出的命令必须是能直接粘进终端的形状（不许留 `...` 占位）。"""
        for command in re.findall(r"`(?:python|pip|git|curl) [^`]+`", self.text):
            self.assertNotIn("...", command, f"命令里留了占位符：{command}")


if __name__ == "__main__":
    unittest.main()
