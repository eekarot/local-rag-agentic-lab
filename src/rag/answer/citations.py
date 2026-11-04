"""D33 引用解析：把答案文本里的 `[证据 n]` 变成可核对的结构。

为什么要单独解析而不是让模型自己保证：

- 模型会写出 `[证据 7]` 而上下文只有 5 条（越界引用）→ 必须计数，计数进回归门禁；
- 模型会写成 `【证据 3】`（全角）或 `[证据3]`（无空格）→ 都要认，否则「解析成功率」是假的；
- 模型会写 `[证据 二]` 这种没法解析的 → 要单独记 `malformed`，不能和「没引用」混为一谈。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: 正例：`[证据 3]` / `[证据3]` / `【证据 3】` / `[证据 3,4]` / `[证据 3、4]`
CITATION_PATTERN = re.compile(
    r"[\[【]\s*证据\s*([0-9][0-9\s,，、]*)\s*[\]】]"
)

#: 反例：括号里写了「证据」但不是数字 → 记 malformed
MALFORMED_PATTERN = re.compile(r"[\[【]\s*证据\s*([^\]】]*)\s*[\]】]")

ABSTAIN_PATTERNS = (
    "证据不足", "知识库未提供", "知识库中没有", "无法从证据",
    "insufficient evidence", "no evidence", "not enough evidence",
    "cannot be determined from the provided",
)


@dataclass
class CitationReport:
    """一次答案文本的引用体检结果。"""

    cited: list[int] = field(default_factory=list)          # 去重后、按出现顺序
    occurrences: int = 0                                    # 引用出现次数（含重复）
    out_of_range: list[int] = field(default_factory=list)    # 越界（> k 或 < 1）
    malformed: list[str] = field(default_factory=list)       # 括号里有「证据」但不是数字
    abstained: bool = False
    text_length: int = 0

    @property
    def has_citation(self) -> bool:
        return bool(self.cited) or bool(self.out_of_range)

    @property
    def parse_rate(self) -> float:
        """解析成功率 = 成功解析的引用次数 / 所有「像引用」的片段数。

        没写引用的答案这里算 1.0（没有可失败的对象），是否引用由 `has_citation` 表达。
        """
        total = self.occurrences + len(self.malformed)
        return 1.0 if total == 0 else self.occurrences / total

    @property
    def precision(self) -> float:
        """引用精确率（自动口径的一部分）：落在范围内的引用 / 全部解析出的引用。"""
        if not self.occurrences:
            return 0.0
        in_range = self.occurrences - len(self.out_of_range)
        return max(0.0, in_range / self.occurrences)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cited": list(self.cited),
            "occurrences": self.occurrences,
            "out_of_range": list(self.out_of_range),
            "malformed": list(self.malformed),
            "abstained": self.abstained,
            "has_citation": self.has_citation,
            "parse_rate": round(self.parse_rate, 4),
            "precision": round(self.precision, 4),
        }


def parse_citations(text: str, *, k: int | None = None) -> CitationReport:
    """解析引用。`k` 给了就顺便判越界（`n > k` 或 `n < 1`）。"""
    body = text or ""
    report = CitationReport(text_length=len(body))

    for match in CITATION_PATTERN.finditer(body):
        numbers = [int(piece) for piece in re.findall(r"\d+", match.group(1))]
        if not numbers:
            report.malformed.append(match.group(0))
            continue
        for number in numbers:
            report.occurrences += 1
            if number not in report.cited and not (k is not None and (number < 1 or number > k)):
                report.cited.append(number)
            if k is not None and (number < 1 or number > k):
                report.out_of_range.append(number)

    for match in MALFORMED_PATTERN.finditer(body):
        if not CITATION_PATTERN.fullmatch(match.group(0)):
            report.malformed.append(match.group(0))

    lowered = body.lower()
    report.abstained = any(pattern in lowered for pattern in ABSTAIN_PATTERNS)
    return report


def resolve_sources(report: CitationReport, blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """引用编号 → 来源块（`blocks` 是 `AssembledContext.blocks`，`index` 从 1 开始）。"""
    by_index = {int(block["index"]): dict(block) for block in blocks if "index" in block}
    return [by_index[number] for number in report.cited if number in by_index]


def citation_summary(reports: Iterable[CitationReport]) -> dict[str, Any]:
    """一批答案的引用体检汇总（D35 的自动指标之一）。"""
    items = list(reports)
    if not items:
        return {"n": 0, "parse_rate": 0.0, "out_of_range": 0, "no_citation": 0,
                "abstain_rate": 0.0, "mean_precision": 0.0}
    return {
        "n": len(items),
        "parse_rate": round(sum(item.parse_rate for item in items) / len(items), 4),
        "out_of_range": sum(len(item.out_of_range) for item in items),
        "no_citation": sum(1 for item in items if not item.has_citation),
        "abstain_rate": round(sum(1 for item in items if item.abstained) / len(items), 4),
        "mean_precision": round(sum(item.precision for item in items) / len(items), 4),
    }
