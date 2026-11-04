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

#: 正例：`[证据 3]` / `[证据3]` / `【证据 3】` / `[证据 3,4]` / `[证据 3、4]` / `[证据 1-7]`
#: 区间写法是实测出来的（D33b 40 题里 eval-137 写成 `[证据 1-7]`），不认就会被记成畸形引用。
CITATION_PATTERN = re.compile(
    r"[\[【]\s*证据\s*([0-9][0-9\s,，、\-–—~至]*)\s*[\]】]"
)

#: 区间引用：`1-7` / `1–7` / `1~7` / `1至7`
_RANGE_PATTERN = re.compile(r"(\d+)\s*[-–—~至]\s*(\d+)")

#: 一次最多展开多少条：防止 `[证据 1-9999]` 把报告撑爆
MAX_RANGE_SPAN = 20

#: 反例：括号里写了「证据」但不是数字 → 记 malformed
MALFORMED_PATTERN = re.compile(r"[\[【]\s*证据\s*([^\]】]*)\s*[\]】]")

#: 「否定 + 缺失」配对：否定词后面**紧跟（≤6 字内）**缺失谓词，才算在声称"证据里没有"。
#:
#: 不用固定短语清单，是因为真实回答的措辞发散得厉害：同一批 20 条负样本里就出现了
#: 未包含 / 未找到 / 未提供 / 未直接提供 / 无法查询到 / 无法提供 六种写法。
#: 也不能只判「同一句里既有否定又有缺失词」——"根据**提供**的知识库…循环永远**无法**结束"
#: 会因此误判；位置关系（否定在前、谓词紧随）才是"缺失声明"的证据。
#: 注意：**不能放单字 "不"**——「证据不足时改写查询」会被判成「不 + 查询」，实测误伤
_NEGATION = ("没有", "无法", "不能", "未", "不存在", "cannot", "without", "not ", "no ")
_ABSENCE = ("提供", "包含", "找到", "查到", "查询", "回答", "提及", "说明", "收录",
            "确定", "确认", "evidence", "information", "answer", "mention", "contain",
            "provide", "found")
_ABSENCE_PATTERN = re.compile(
    "(" + "|".join(re.escape(token) for token in _NEGATION) + ")"
    "[^。！？；;!?|]{0,6}?"
    "(" + "|".join(re.escape(token) for token in _ABSENCE) + ")",
    re.IGNORECASE,
)

#: 弱信号：**只认一句话式的极简弃答**（开头就是它、且整段很短）
ABSTAIN_WEAK = ("证据不足", "no evidence")

#: 兼容旧名（外部按整组匹配的地方仍然可用）
ABSTAIN_PATTERNS = _NEGATION + _ABSENCE + ABSTAIN_WEAK

#: 弱信号当作弃答的最大答案长度（字符）：超过就认为是在正常作答中引用该词
ABSTAIN_TERSE_CHARS = 40

#: 只看**开头几句**：弃答是"开口就拒"，正文中后段的信息缺口声明不算（N12/N13）
ABSTAIN_EDGE_SENTENCES = 3

_SENTENCE_END = "。！？!?;" + chr(10)


def _sentences(text: str) -> list[str]:
    return [piece for piece in re.split("[" + re.escape(_SENTENCE_END) + "]", text) if piece.strip()]


def opening_sentences(text: str, sentences: int = ABSTAIN_EDGE_SENTENCES) -> list[str]:
    """答案的**开头 N 句**（逐句判断，不能先拼起来再判——拼完句号就没了，
    "…无法结束。设置最大步数…" 这种会被当成"否定 + 缺失"同一句，直接误判）。

    不再取末尾 N 句：末尾补一句「知识库未提供具体算法原理」是完整答案的常见写法，
    取末尾会把 eval-006 这类答完的题误判成弃答。
    """
    return _sentences(text)[:sentences]


def claims_absence(text: str) -> bool:
    """这段话是否在声称「证据里没有 X」：**否定词在前、缺失谓词紧随（≤6 字）**。"""
    return bool(_ABSENCE_PATTERN.search(text or ""))


def detect_abstain(text: str, *, terse_chars: int = ABSTAIN_TERSE_CHARS) -> bool:
    """模型是否在弃答。

    判据（N12 修正后定稿）：**开头 3 句里有"否定 + 缺失"**，或者整段很短且开头就是
    "证据不足"。反例都写成用例了：
    - 「模型反复认为证据不足并不断搜索」→ 正文转述，不是弃答；
    - 「如果 chunk 太小，无法确定『哪家公司？』」→ 正文举例，且在第 4 句之后。
    """
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    if any(claims_absence(sentence) for sentence in opening_sentences(lowered)):
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


def _expand_numbers(raw: str) -> list[int] | None:
    """`3,4` / `1-7` / `1、2、5` → 编号列表。区间跨度超过 `MAX_RANGE_SPAN` 返回 None。"""
    numbers: list[int] = []
    consumed = raw
    for match in _RANGE_PATTERN.finditer(raw):
        start, end = int(match.group(1)), int(match.group(2))
        if end < start or end - start > MAX_RANGE_SPAN:
            return None
        numbers.extend(range(start, end + 1))
        consumed = consumed.replace(match.group(0), " ")
    numbers.extend(int(piece) for piece in re.findall(r"\d+", consumed))
    return numbers


def parse_citations(text: str, *, k: int | None = None) -> CitationReport:
    """解析引用。`k` 给了就顺便判越界（`n > k` 或 `n < 1`）。"""
    body = text or ""
    report = CitationReport(text_length=len(body))

    for match in CITATION_PATTERN.finditer(body):
        numbers = _expand_numbers(match.group(1))
        if numbers is None:                       # 区间跨度离谱 → 记畸形，不展开
            report.malformed.append(match.group(0))
            continue
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
