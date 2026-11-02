"""D09 Chunk Validator：把脏 chunk 拦在向量化之前（方案 §2.1 的 Validator 环节）。

六类检查，对应 D09 行里的「长度/空/重复/边界/元数据」：

| 检查 | 判据 | 为什么值得单独一条 |
|---|---|---|
| empty | `content.strip()` 为空 | 空块会拿到一个「什么都像」的向量，污染 top-k |
| over_limit | 重新数 token 后超过上限 | 只信 `token_count` 字段会被上游 bug 蒙混过去 |
| duplicate | 归一化（折叠空白）后与前面的块完全相同 | 重复块挤占 top-k 名额（真实的 top-4 同文档 79.6% 已经很难看） |
| boundary | child 内容必须能在 parent 内容里、且落在记录的偏移上 | overlap 让「子块是父块的连续切片」这条不变量变得更难保证 |
| metadata | id / section_path / token_count 字段自洽 | 下游 Qdrant payload 依赖这些字段，缺了就是运行时炸 |
| linkage | `parent_index` 必须存在 | 孤儿 child 会让 context expansion（D32）拿不到父块 |

报告口径：四项数字（空、超限、重复占比、边界完整率）是 D09 的验收项；
`gold_convergence` 是附加项，用来观察 B02 重切分后 `hit@k` 与 `full@k` 的收敛。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .parent_builder import Child, Parent
from .tokenizer import TokenCounter

WHITESPACE_RE = re.compile(r"\s+")

#: 校验严重级别：这些一律算「不通过」
BLOCKING_REASONS = frozenset(
    {"empty", "over_limit", "boundary_broken", "orphan_parent", "metadata_invalid"}
)


@dataclass
class Issue:
    target: str  # "child" / "parent"
    id: str
    reasons: list[str]
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"target": self.target, "id": self.id, "reasons": list(self.reasons), "detail": dict(self.detail)}


@dataclass
class ValidationReport:
    total_parents: int = 0
    total_children: int = 0
    empty_chunks: int = 0
    over_limit_chunks: int = 0
    over_limit_parents: int = 0
    duplicate_chunks: int = 0
    boundary_broken: int = 0
    metadata_errors: int = 0
    orphan_children: int = 0
    issues: list[Issue] = field(default_factory=list)
    gold_convergence: dict | None = None

    # ---- 四项验收数字 ----

    @property
    def duplicate_ratio(self) -> float:
        return self.duplicate_chunks / self.total_children if self.total_children else 0.0

    @property
    def boundary_valid_ratio(self) -> float:
        if not self.total_children:
            return 1.0
        return 1 - self.boundary_broken / self.total_children

    @property
    def ok(self) -> bool:
        return not any(set(issue.reasons) & BLOCKING_REASONS for issue in self.issues)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "total_parents": self.total_parents,
            "total_children": self.total_children,
            "empty_chunks": self.empty_chunks,
            "over_limit_chunks": self.over_limit_chunks,
            "over_limit_parents": self.over_limit_parents,
            "duplicate_chunks": self.duplicate_chunks,
            "duplicate_ratio": round(self.duplicate_ratio, 4),
            "boundary_broken": self.boundary_broken,
            "boundary_valid_ratio": round(self.boundary_valid_ratio, 4),
            "metadata_errors": self.metadata_errors,
            "orphan_children": self.orphan_children,
            "issues": len(self.issues),
            "gold_convergence": self.gold_convergence,
        }

    def write_invalid(self, path: str | Path) -> int:
        """失败样例落 `*.invalid.jsonl`（没有问题时**不生成**文件，避免假象）。"""
        if not self.issues:
            return 0
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for issue in self.issues:
                handle.write(json.dumps(issue.to_dict(), ensure_ascii=False) + "\n")
        return len(self.issues)


def validate(
    parents: Sequence[Parent],
    children: Sequence[Child],
    *,
    counter: TokenCounter,
    child_max_tokens: int = 500,
    parent_max_tokens: int = 2000,
    gold_spans: Sequence[str] | None = None,
) -> ValidationReport:
    report = ValidationReport(total_parents=len(parents), total_children=len(children))
    # 键必须是 (document_id, parent_index)：parent_index 是**文档内**序号，
    # 跨文档批量校验时只用 index 会把 A 文档的 child 关联到 B 文档的 parent。
    parents_by_key: dict[tuple, Parent] = {}
    for parent in parents:
        key = (parent.document_id, parent.parent_index)
        existing = parents_by_key.get(key)
        if existing is not None and existing.parent_id != parent.parent_id:
            # 同一 (document_id, parent_index) 挂着两个不同父块：说明 document_id 不唯一
            # （只拿文件名当 source_uri 时，语料里的多个 index.md / overview.md 就会撞）。
            report.issues.append(
                Issue(
                    target="parent",
                    id=parent.parent_id or f"parent#{parent.parent_index}",
                    reasons=["duplicate_parent_key"],
                    detail={"document_id": parent.document_id, "parent_index": parent.parent_index},
                )
            )
        parents_by_key[key] = parent
    seen_texts: dict[str, str] = {}

    for parent in parents:
        reasons: list[str] = []
        detail: dict = {}
        if not parent.content.strip():
            reasons.append("empty")
        actual = counter.count(parent.content)
        if actual > parent_max_tokens:
            report.over_limit_parents += 1
            reasons.append("over_limit")
            detail["tokens"] = actual
        if actual != parent.token_count:
            report.metadata_errors += 1
            reasons.append("token_count_mismatch")
            detail["token_count_field"] = parent.token_count
            detail["token_count_actual"] = actual
        if not parent.section_path:
            report.metadata_errors += 1
            reasons.append("section_path_empty")
        if not parent.parent_id:
            report.metadata_errors += 1
            reasons.append("missing_parent_id")
        if reasons:
            report.issues.append(
                Issue(target="parent", id=parent.parent_id or f"parent#{parent.parent_index}", reasons=reasons, detail=detail)
            )

    for child in children:
        reasons: list[str] = []
        detail: dict = {}
        content = child.content

        if not content.strip():
            report.empty_chunks += 1
            reasons.append("empty")

        actual_tokens = counter.count(content)
        if actual_tokens > child_max_tokens:
            report.over_limit_chunks += 1
            reasons.append("over_limit")
            detail["tokens"] = actual_tokens
        if actual_tokens != child.token_count:
            report.metadata_errors += 1
            reasons.append("token_count_mismatch")
            detail["token_count_field"] = child.token_count
            detail["token_count_actual"] = actual_tokens

        normalized = WHITESPACE_RE.sub(" ", content).strip()
        if normalized:
            if normalized in seen_texts:
                report.duplicate_chunks += 1
                reasons.append("duplicate")
                detail["duplicate_of"] = seen_texts[normalized]
            else:
                seen_texts[normalized] = child.chunk_id or f"child#{len(seen_texts)}"

        parent = parents_by_key.get((child.document_id, child.parent_index))
        if parent is None:
            report.orphan_children += 1
            reasons.append("orphan_parent")
        else:
            if not content.strip():
                pass  # 空块已单独计过，不重复算边界错误
            else:
                positions = _find_all(parent.content, content)
                if not positions:
                    report.boundary_broken += 1
                    reasons.append("boundary_broken")
                    detail["reason"] = "content_not_in_parent"
                elif child.source_start is not None and parent.source_start is not None:
                    expected = {parent.source_start + pos for pos in positions}
                    if child.source_start not in expected:
                        report.boundary_broken += 1
                        reasons.append("boundary_broken")
                        detail["reason"] = "offset_mismatch"
                        detail["source_start"] = child.source_start
                        detail["expected"] = sorted(expected)[:5]
                elif child.source_end is not None and child.source_start is not None:
                    if child.source_end - child.source_start != len(content):
                        report.boundary_broken += 1
                        reasons.append("boundary_broken")
                        detail["reason"] = "span_length_mismatch"

        if not child.section_path:
            report.metadata_errors += 1
            reasons.append("section_path_empty")
        if not child.chunk_id:
            report.metadata_errors += 1
            reasons.append("missing_chunk_id")
        if not child.parent_id:
            report.metadata_errors += 1
            reasons.append("missing_parent_id")

        if reasons:
            report.issues.append(
                Issue(target="child", id=child.chunk_id or f"child#{child.parent_index}/{child.child_index}", reasons=reasons, detail=detail)
            )

    if gold_spans is not None:
        report.gold_convergence = gold_convergence(children, gold_spans)
    return report


def gold_convergence(children: Sequence[Child], gold_spans: Iterable[str]) -> dict:
    """统计每个 gold_span 被几个 chunk 命中（0 / 1 / 2 / 3+）。

    B02 重切分让 chunk 变大，多 gold 会更容易落进同一 chunk——这个分布是
    判断「hit@k 与 full@k 是否还值得分开报」的直接依据。
    """
    normalized_children = [WHITESPACE_RE.sub(" ", child.content) for child in children]
    histogram = {"0": 0, "1": 0, "2": 0, "3+": 0}
    per_span: list[dict] = []
    for span in gold_spans:
        needle = WHITESPACE_RE.sub(" ", span).strip()
        if not needle:
            continue
        hits = sum(1 for text in normalized_children if needle in text)
        bucket = "3+" if hits >= 3 else str(hits)
        histogram[bucket] += 1
        per_span.append({"span": span, "chunks": hits})
    total = sum(histogram.values())
    converged = histogram["2"] + histogram["3+"]
    return {
        "spans_total": total,
        "spans_found": total - histogram["0"],
        "histogram": histogram,
        "converged_ratio": round(converged / total, 4) if total else 0.0,
        "per_span": per_span,
    }


def _find_all(haystack: str, needle: str) -> list[int]:
    """所有出现位置（用于兼容 overlap 造成的重复出现）。"""
    positions: list[int] = []
    start = 0
    while True:
        found = haystack.find(needle, start)
        if found < 0:
            return positions
        positions.append(found)
        start = found + 1


__all__ = [
    "BLOCKING_REASONS",
    "Issue",
    "ValidationReport",
    "gold_convergence",
    "validate",
]
