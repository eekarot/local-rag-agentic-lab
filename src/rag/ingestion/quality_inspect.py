"""切片体检的编排层（G4 / 方案 P2）：把「解析 → 切分 → 算指标」串起来，**不写库**。

分成「指标纯函数」+「编排」两层的原因：

* `quality_metrics.py` 只认整数区间，谁都能测（包括别人的切分器）；
* 本模块负责「拿到语料、跑我们自己的切分、汇总」，给两个入口共用：
  `scripts/chunk_quality.py`（独立体检）与 `scripts/ingest.py --precheck`（建库前一道闸）。

一条纪律：**体检永远不写库、不改状态表**。它跑得比建库快两个数量级，
所以可以在任何语料上随手跑；一旦它开始写东西，就没人敢在正式环境跑了。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .parent_builder import ParentBuilder, ParentConfig, build_children
from .parser.markdown_parser import MarkdownParser
from .quality_metrics import (
    DEFAULT_TOLERANCE_CHARS,
    block_integrity,
    find_coverage_gaps,
    size_compliance,
)
from .tokenizer import TokenCounter


@dataclass
class DocReport:
    """单篇文档的体检结果。"""

    name: str
    parents: int = 0
    children: int = 0
    tokens_max: int = 0
    sc: float | None = None
    bi: float | None = None
    gap_parents: int = 0
    broken_blocks: int = 0
    block_total: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parents": self.parents,
            "children": self.children,
            "tokens_max": self.tokens_max,
            "sc": self.sc,
            "bi": self.bi,
            "gap_parents": self.gap_parents,
            "broken_blocks": self.broken_blocks,
            "block_total": self.block_total,
        }


@dataclass
class Summary:
    docs: int = 0
    children: int = 0
    sc_mean: float | None = None
    bi_mean: float | None = None
    bi_below_threshold: int = 0
    gap_docs: int = 0
    threshold: float = 0.95
    tolerance_chars: int = DEFAULT_TOLERANCE_CHARS
    reports: list[DocReport] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """有没有「不合格」或「丢字」。丢字永远算不合格（那是不变量，不是评分）。"""
        return not (self.bi_below_threshold or self.gap_docs)

    def to_dict(self) -> dict:
        return {
            "docs": self.docs,
            "children": self.children,
            "sc_mean": self.sc_mean,
            "bi_mean": self.bi_mean,
            "bi_below_threshold": self.bi_below_threshold,
            "gap_docs": self.gap_docs,
            "threshold": self.threshold,
            "tolerance_chars": self.tolerance_chars,
            "per_doc": [report.to_dict() for report in self.reports],
        }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def display_name(path: Path, root: Path | None) -> str:
    """报告里用相对路径：同名文件很常见（基准数据集有几十个 `_index.md`）。"""
    if root is not None:
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def inspect_document(
    path: Path,
    counter: TokenCounter,
    config: ParentConfig | None = None,
    *,
    tolerance_chars: int = DEFAULT_TOLERANCE_CHARS,
    min_tokens: int = 0,
    max_tokens: int | None = None,
    name: str | None = None,
) -> DocReport:
    """解析 → 切分 → 算三个口径（纯内存，**不写库**）。

    ``min_tokens`` 默认 0：切分器只保证「不超过 `child_max_tokens`」，从没承诺下限
    （最后一块天然偏短）。想要「切太碎」的告警就显式传一个下限。
    """
    config = config or ParentConfig()
    blocks = MarkdownParser().parse(str(path))
    report = DocReport(name=name or path.name)
    if not blocks:
        return report

    parents = ParentBuilder(counter=counter, config=config).build(
        blocks, document_id="doc", document_version="v1"
    )
    if not parents:
        return report
    children = build_children(
        parents,
        counter=counter,
        child_target_tokens=config.child_target_tokens,
        child_max_tokens=config.child_max_tokens,
        overlap_tokens=config.child_overlap_tokens,
    )
    if not children:
        return report

    report.parents = len(parents)
    report.children = len(children)
    report.tokens_max = max(child.token_count for child in children)
    report.sc = size_compliance(
        [child.token_count for child in children],
        min_tokens=min_tokens,
        max_tokens=config.child_max_tokens if max_tokens is None else max_tokens,
    )

    block_spans = [(block.source_start, block.source_end) for block in blocks]
    report.bi = block_integrity(
        [(child.source_start, child.source_end) for child in children],
        block_spans,
        tolerance_chars=tolerance_chars,
    )
    report.block_total = sum(1 for left, right in block_spans if left is not None and right is not None)
    if report.bi is not None and report.block_total:
        report.broken_blocks = int(round((1 - report.bi) * report.block_total))

    by_parent: dict[str, list[tuple[int | None, int | None]]] = {}
    for child in children:
        by_parent.setdefault(child.parent_id or "", []).append((child.source_start, child.source_end))
    for parent in parents:
        if parent.source_start is None:
            continue
        spans = [
            (start - parent.source_start, end - parent.source_start)
            for start, end in by_parent.get(parent.parent_id or "", [])
            if start is not None and end is not None
        ]
        if not spans:
            continue
        if find_coverage_gaps(spans, end=len(parent.content)):
            report.gap_parents += 1
    return report


def inspect_paths(
    paths: list[Path],
    counter: TokenCounter,
    config: ParentConfig | None = None,
    *,
    threshold: float = 0.95,
    tolerance_chars: int = DEFAULT_TOLERANCE_CHARS,
    min_tokens: int = 0,
    max_tokens: int | None = None,
    display_root: Path | None = None,
) -> Summary:
    """批量体检；``paths`` 由调用方决定顺序（脚本里已按相对路径排序）。"""
    summary = Summary(threshold=threshold, tolerance_chars=tolerance_chars)
    sc_values: list[float] = []
    bi_values: list[float] = []
    for path in paths:
        report = inspect_document(
            path,
            counter,
            config,
            tolerance_chars=tolerance_chars,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            name=display_name(path, display_root),
        )
        summary.reports.append(report)
        summary.docs += 1
        summary.children += report.children
        if report.sc is not None:
            sc_values.append(report.sc)
        if report.bi is not None:
            bi_values.append(report.bi)
            if report.bi < threshold:
                summary.bi_below_threshold += 1
        if report.gap_parents:
            summary.gap_docs += 1
    summary.sc_mean = _mean(sc_values)
    summary.bi_mean = _mean(bi_values)
    return summary


def format_report(summary: Summary, *, limit: int = 10) -> list[str]:
    """人看的一小段摘要。最后一行的提醒是**故意的**：别把结构指标当质量分贴出去。"""
    lines = [
        f"[chunk_quality] 体检 {summary.docs} 篇 · 子块 {summary.children} 个",
        f"  口径：sc=尺寸合规 bi=结构完整（容差 {summary.tolerance_chars} 字符） gap=丢字检查（不变量）",
        f"  sc 均值 {summary.sc_mean} · bi 均值 {summary.bi_mean} · "
        f"bi < {summary.threshold} 的文档 {summary.bi_below_threshold} 篇 · 丢字文档 {summary.gap_docs} 篇",
    ]
    weak = [r for r in summary.reports if r.bi is not None and r.bi < summary.threshold]
    if weak:
        lines.append(f"  结构完整性偏低的文档（前 {limit} 篇）：")
        for report in sorted(weak, key=lambda item: (item.bi or 0, item.name))[:limit]:
            lines.append(
                f"    {report.name}: bi={report.bi:.3f} 被切断 {report.broken_blocks}/{report.block_total} 块"
                f"（{report.children} 个子块，最大 {report.tokens_max} token）"
            )
    gaps = [r for r in summary.reports if r.gap_parents]
    if gaps:
        lines.append("  丢了字的文档（**这是 bug，不是评分**）：")
        for report in gaps[:limit]:
            lines.append(f"    {report.name}: {report.gap_parents} 个父块的子块没有铺满")
    lines.append("  提醒：sc/bi 只说明结构，**不预测答对率**；检索质量看 scripts/eval_bench.py")
    return lines


__all__ = [
    "DocReport",
    "Summary",
    "display_name",
    "format_report",
    "inspect_document",
    "inspect_paths",
]
