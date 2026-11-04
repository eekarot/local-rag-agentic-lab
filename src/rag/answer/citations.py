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

#: 结论型信号：只在这些词出现在**开头或结尾**时才算弃答（结论通常在这两处）
ABSTAIN_STRONG = (
    "知识库未提供", "知识库中没有", "知识库中未提供", "未提供足够", "没有提供足够", "没有足够证据",
    "无法从证据", "无法回答", "无法找到", "无法确认", "无法确定", "不足以回答",
    "insufficient evidence", "not enough evidence", "cannot answer", "cannot find",
    "cannot determine", "cannot be determined from the provided",
)

#: 弱信号：**只认一句话式的极简弃答**（开头就是它、且整段很短）
ABSTAIN_WEAK = ("证据不足", "no evidence")

#: 兼容旧名（外部按整组匹配的地方仍然可用）
ABSTAIN_PATTERNS = ABSTAIN_STRONG + ABSTAIN_WEAK

#: 弱信号当作弃答的最大答案长度（字符）：超过就认为是在正常作答中引用该词
ABSTAIN_TERSE_CHARS = 40

#: 结论区：首尾各取几个句子参与强信号匹配。
#: 实测 40 条答案：edge=3 是拐点（2 会漏掉「末尾倒数第三句下结论」的 eval-058，
#: 4 会把正文里转述的 eval-015 误判），所以取 3。
ABSTAIN_EDGE_SENTENCES = 3

_SENTENCE_END = "。！？!?;" + chr(10)


def _edge_text(text: str, sentences: int = ABSTAIN_EDGE_SENTENCES) -> str:
    """取答案的「首 N 句 + 末 N 句」：结论在这两处，正文中间的多是转述（N12）。"""
    parts = [piece for piece in re.split("[" + re.escape(_SENTENCE_END) + "]", text) if piece.strip()]
    if len(parts) <= 2 * sentences:
        return text
    return " ".join(parts[:sentences] + parts[-sentences:])


def detect_abstain(text: str, *, terse_chars: int = ABSTAIN_TERSE_CHARS) -> bool:
    """模型是否在弃答。

    强信号（"无法找到"/"知识库中未提供"…）只看**首尾两句**——完整答案的正文里经常出现
    "无法确定『哪家公司？』" 这类转述（实测 eval-015 被误判）。
    弱信号（"证据不足"）只认极简回答的开头，因为它在长文里几乎总是被引内容。
    """
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    if any(pattern in _edge_text(lowered) for pattern in ABSTAIN_STRONG):
        return True
    if len(lowered) > terse_chars:
        return False
    head = lowered.strip().lstrip("，。:：、,.:;；!！?？")
    return any(head.startswith(pattern) for pattern in ABSTAIN_WEAK)


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

    report.abstained = detect_abstain(body)
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
