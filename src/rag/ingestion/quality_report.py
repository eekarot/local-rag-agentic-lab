"""D11 Chunking 质量报告 + A–E 五种切分对照（方案 §11.1）。

报告要回答的问题只有一个：**换切分方案，到底换到了什么**。所以每个方案都
用同一套可复算的指标，而不是「看起来更好」：

| 方案 | 定义 | 用来回答 |
|---|---|---|
| A | 固定长度窗口 | 基线：完全不看结构会长什么样 |
| B | Recursive（D06） | 结构分隔符能救回多少 |
| C | Recursive + Overlap（D07） | overlap 是不是真在句子边界上 |
| D | Parent-Child + Recursive + Overlap（D08） | 两层结构 + 标题归属的净收益 |
| E | Semantic + Parent-Child | 需语义切分，第一版关闭 → **显式 not_implemented** |

8 项人工抽查（§11.1）里，7 项能自动算，剩下的人工评分列留空给评测者填——
能自动算的绝不靠眼看，算不出来的绝不假装算过。
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .block import DocumentBlock
from .ids import make_document_id
from .overlap import (
    DEFAULT_MAX_OVERLAP_TOKENS,
    DEFAULT_MIN_OVERLAP_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    apply_overlap,
)
from .parent_builder import ParentBuilder, ParentConfig, build_children
from .recursive_chunker import recursive_split
from .tokenizer import TokenCounter, default_counter
from .validator import validate

SENTENCE_END_CHARS = "。！？；!?;."
MIN_CHILD_TOKENS = 80

#: 「句子是否被不合理切断」只在散文区统计。
#: 代码块/表格本来就不该按句子结尾判断——把 `# Answer 'y' if prompted...`、
#: `.../<key>.pub` 这种结尾算成"句子被切断"会把指标污染到无法比较
#: （实测 D 方案因此虚高到 77%，比不做结构的 B 还差）。代码/表格的完整性
#: 由 code_intact_ratio / table_intact_ratio 单独负责。
PROSE_BLOCK_TYPES = ("heading", "paragraph", "list", "quote")


@dataclass
class VariantSpec:
    key: str
    name: str
    status: str  # "ok" / "not_implemented"
    note: str = ""


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("A", "固定长度", "ok", "字符窗口，不看任何结构"),
    VariantSpec("B", "Recursive", "ok", "按分隔符优先级递归切分（D06）"),
    VariantSpec("C", "Recursive + Overlap", "ok", "D06 + 句子边界回退重叠（D07）"),
    VariantSpec("D", "Parent-Child + Recursive + Overlap", "ok", "D08 两层结构 + 标题归属"),
    VariantSpec("E", "Semantic + Parent-Child", "not_implemented", "语义切分第一版关闭（方案 §2.6）"),
)


@dataclass
class ChecklistItem:
    key: str
    name: str
    metric: str
    direction: str = "higher"  # higher 越好 / lower 越好


CHECKLIST_ITEMS: tuple[ChecklistItem, ...] = (
    ChecklistItem("heading_link", "标题是否和正文关联", "heading_link_ratio"),
    ChecklistItem("sentence_cut", "句子是否被不合理切断", "sentence_cut_ratio", "lower"),
    ChecklistItem("table_intact", "表格是否保留行列关系", "table_intact_ratio"),
    ChecklistItem("code_intact", "代码块是否完整", "code_intact_ratio"),
    ChecklistItem("parent_section", "Parent 是否覆盖正确章节", "parent_section_purity"),
    ChecklistItem("child_trace", "Child 能否回溯 Parent", "child_traceable_ratio"),
    ChecklistItem("overlap_sane", "Overlap 是否过大", "overlap_in_range_ratio"),
    ChecklistItem("empty_dup", "是否出现空/重复 Chunk", "duplicate_ratio", "lower"),
)


@dataclass
class VariantMetrics:
    key: str
    name: str
    status: str
    documents: int = 0
    chunks: int | None = None
    tokens_p50: int | None = None
    tokens_p90: int | None = None
    tokens_max: int | None = None
    empty_ratio: float | None = None
    duplicate_ratio: float | None = None
    sentence_cut_ratio: float | None = None
    overlap_in_range_ratio: float | None = None
    table_intact_ratio: float | None = None
    code_intact_ratio: float | None = None
    heading_link_ratio: float | None = None
    parent_section_purity: float | None = None
    child_traceable_ratio: float | None = None
    parents: int | None = None
    child_per_parent: float | None = None
    fragment_ratio: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- 各方案

def run_variant(
    key: str,
    documents: Sequence[tuple[str, list[DocumentBlock]]],
    *,
    counter: TokenCounter,
    fixed_chars: int = 500,
    child_target_tokens: int = 400,
    child_max_tokens: int = 500,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_overlap_tokens: int = DEFAULT_MIN_OVERLAP_TOKENS,
    max_overlap_tokens: int = DEFAULT_MAX_OVERLAP_TOKENS,
) -> VariantMetrics:
    spec = next((variant for variant in VARIANTS if variant.key == key), None)
    if spec is None:
        raise ValueError(f"未知方案 {key!r}，可选：{[v.key for v in VARIANTS]}")
    if spec.status != "ok":
        return VariantMetrics(key=spec.key, name=spec.name, status=spec.status, notes=[spec.note])

    if key == "A":
        metrics = _metrics_from_chunks(spec, documents, _fixed_windows(documents, fixed_chars), counter)
    elif key in ("B", "C"):
        metrics = _metrics_from_chunks(
            spec,
            documents,
            _recursive_document_chunks(
                documents, counter, key, child_target_tokens, child_max_tokens,
                overlap_tokens, min_overlap_tokens, max_overlap_tokens,
            ),
            counter,
        )
    else:
        metrics = _parent_child_metrics(
            spec, documents, counter, child_target_tokens, child_max_tokens,
            overlap_tokens, min_overlap_tokens, max_overlap_tokens,
        )

    # 句子切断指标改在「散文区」上重算一遍（分块统计仍用全文）
    metrics.sentence_cut_ratio = _prose_sentence_cut_ratio(
        key, documents, counter, fixed_chars, child_target_tokens, child_max_tokens,
        overlap_tokens, min_overlap_tokens, max_overlap_tokens,
    )
    return metrics
    return _parent_child_metrics(
        spec, documents, counter, child_target_tokens, child_max_tokens, overlap_tokens, min_overlap_tokens, max_overlap_tokens
    )


def run_all_variants(
    documents: Sequence[tuple[str, list[DocumentBlock]]],
    *,
    counter: TokenCounter,
    **kwargs,
) -> list[VariantMetrics]:
    return [run_variant(spec.key, documents, counter=counter, **kwargs) for spec in VARIANTS]


def _document_text(blocks: Sequence[DocumentBlock]) -> str:
    return "\n\n".join(block.text for block in blocks)


def _fixed_windows(documents, chars: int) -> list[tuple[str, list[str]]]:
    result = []
    for name, blocks in documents:
        text = _document_text(blocks)
        result.append((name, [text[i : i + chars] for i in range(0, len(text), chars)] if text else []))
    return result


def _recursive_document_chunks(
    documents, counter, key, target, max_tokens, overlap_tokens, min_overlap, max_overlap
) -> list[tuple[str, list[str]]]:
    result = []
    for name, blocks in documents:
        text = _document_text(blocks)
        if key == "B":
            pieces = recursive_split(text, counter=counter, max_tokens=max_tokens, target_tokens=target)
        else:
            raw = recursive_split(
                text, counter=counter, max_tokens=max_tokens - overlap_tokens, target_tokens=max(1, target - overlap_tokens)
            )
            pieces = [
                piece.content
                for piece in apply_overlap(
                    raw,
                    counter,
                    overlap_tokens=overlap_tokens,
                    min_overlap_tokens=min_overlap,
                    max_overlap_tokens=max_overlap,
                )
            ]
        result.append((name, pieces))
    return result


def _metrics_from_chunks(
    spec: VariantSpec,
    documents: Sequence[tuple[str, list[DocumentBlock]]],
    chunks_by_document: Sequence[tuple[str, list[str]]],
    counter: TokenCounter,
) -> VariantMetrics:
    token_counts: list[int] = []
    empty = duplicates = 0
    cuts = boundaries = 0
    seen: Counter[str] = Counter()

    for name, pieces in chunks_by_document:
        for index, piece in enumerate(pieces):
            tokens = counter.count(piece)
            token_counts.append(tokens)
            if not piece.strip():
                empty += 1
            seen[" ".join(piece.split())] += 1
            if index < len(pieces) - 1:
                boundaries += 1
                if not piece.rstrip().endswith(tuple(SENTENCE_END_CHARS)) and not piece.rstrip().endswith("\n"):
                    cuts += 1

    duplicates = sum(count - 1 for count in seen.values())
    metrics = VariantMetrics(
        key=spec.key,
        name=spec.name,
        status="ok",
        documents=len(documents),
        chunks=len(token_counts),
        **(_token_stats(token_counts)),
        empty_ratio=round(empty / len(token_counts), 4) if token_counts else 0.0,
        duplicate_ratio=round(duplicates / len(token_counts), 4) if token_counts else 0.0,
        sentence_cut_ratio=round(cuts / boundaries, 4) if boundaries else 0.0,
        fragment_ratio=round(sum(1 for t in token_counts if t < MIN_CHILD_TOKENS) / len(token_counts), 4) if token_counts else 0.0,
        notes=[spec.note] if spec.note else [],
    )
    metrics.table_intact_ratio = _block_intact_ratio(documents, chunks_by_document, "table")
    metrics.code_intact_ratio = _block_intact_ratio(documents, chunks_by_document, "code")
    if spec.key == "C":
        metrics.overlap_in_range_ratio = _overlap_in_range_ratio(chunks_by_document, counter, DEFAULT_MIN_OVERLAP_TOKENS, DEFAULT_MAX_OVERLAP_TOKENS)
    return metrics


def _token_stats(counts: list[int]) -> dict:
    if not counts:
        return {"tokens_p50": 0, "tokens_p90": 0, "tokens_max": 0}
    ordered = sorted(counts)
    return {
        "tokens_p50": ordered[len(ordered) // 2],
        "tokens_p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "tokens_max": ordered[-1],
    }


def _block_intact_ratio(
    documents: Sequence[tuple[str, list[DocumentBlock]]],
    chunks_by_document: Sequence[tuple[str, list[str]]],
    block_type: str,
) -> float | None:
    """表格/代码块有多少比例被完整收在**同一个** chunk 里。"""
    total = intact = 0
    for (name, blocks), (_, pieces) in zip(documents, chunks_by_document):
        structural = [block.text for block in blocks if block.block_type == block_type]
        if not structural:
            continue
        joined = "\n".join(pieces)
        for text in structural:
            total += 1
            if text.strip() and text in joined:
                intact += 1
    return round(intact / total, 4) if total else None


def _overlap_in_range_ratio(
    chunks_by_document: Sequence[tuple[str, list[str]]],
    counter: TokenCounter,
    minimum: int,
    maximum: int,
) -> float | None:
    values: list[int] = []
    for _, pieces in chunks_by_document:
        for previous, current in zip(pieces, pieces[1:]):
            for cut in range(len(current), 0, -1):
                if previous.endswith(current[:cut]):
                    values.append(counter.count(current[:cut]))
                    break
    if not values:
        return None
    return round(sum(1 for value in values if minimum <= value <= maximum) / len(values), 4)


def _parent_child_metrics(
    spec: VariantSpec,
    documents,
    counter,
    child_target,
    child_max,
    overlap_tokens,
    min_overlap,
    max_overlap,
) -> VariantMetrics:
    builder = ParentBuilder(
        counter=counter,
        config=ParentConfig(
            child_target_tokens=child_target,
            child_max_tokens=child_max,
            child_overlap_tokens=overlap_tokens,
        ),
    )
    parents_all, children_all = [], []
    chunks_by_document = []
    heading_linked = heading_total = 0
    parent_sections_ok = 0

    for name, blocks in documents:
        document_id = make_document_id(name)
        parents = builder.build(blocks, document_id=document_id, document_version="sha256:probe")
        children = build_children(
            parents,
            counter=counter,
            child_target_tokens=child_target,
            child_max_tokens=child_max,
            overlap_tokens=overlap_tokens,
        )
        parents_all.extend(parents)
        children_all.extend(children)
        chunks_by_document.append((name, [child.content for child in children]))

        by_index = {parent.parent_index: parent for parent in parents}
        for child in children:
            parent = by_index[child.parent_index]
            heading = child.section_path[-1] if child.section_path else ""
            if heading:
                heading_total += 1
                if heading in parent.content:
                    heading_linked += 1
        for parent in parents:
            section_prefix = parent.section_path
            if all(
                blocks[index].section_path[: len(section_prefix)] == section_prefix
                for index in parent.block_indexes
            ):
                parent_sections_ok += 1

    report = validate(
        parents_all,
        children_all,
        counter=counter,
        child_max_tokens=child_max,
        parent_max_tokens=builder.config.max_tokens,
    )
    token_counts = [child.token_count for child in children_all]
    metrics = VariantMetrics(
        key=spec.key,
        name=spec.name,
        status="ok",
        documents=len(documents),
        chunks=len(children_all),
        parents=len(parents_all),
        child_per_parent=round(len(children_all) / len(parents_all), 2) if parents_all else None,
        **(_token_stats(token_counts)),
        empty_ratio=round(report.empty_chunks / len(children_all), 4) if children_all else 0.0,
        duplicate_ratio=round(report.duplicate_ratio, 4),
        overlap_in_range_ratio=_overlap_in_range_ratio(
            chunks_by_document, counter, min_overlap, max_overlap
        ),
        heading_link_ratio=round(heading_linked / heading_total, 4) if heading_total else None,
        parent_section_purity=round(parent_sections_ok / len(parents_all), 4) if parents_all else None,
        child_traceable_ratio=round(1 - report.orphan_children / len(children_all), 4) if children_all else None,
        fragment_ratio=round(sum(1 for t in token_counts if t < MIN_CHILD_TOKENS) / len(token_counts), 4) if token_counts else 0.0,
        notes=[spec.note] if spec.note else [],
    )
    metrics.table_intact_ratio = _block_intact_ratio(documents, chunks_by_document, "table")
    metrics.code_intact_ratio = _block_intact_ratio(documents, chunks_by_document, "code")
    metrics.notes.append(
        f"校验：空 {report.empty_chunks}、超限 {report.over_limit_chunks}、边界完整率 {report.boundary_valid_ratio:.2%}"
    )
    return metrics


def _ends_cleanly(piece: str) -> bool:
    """块是否在「干净边界」上结束：句子结束标点，或换行（只去掉行尾空格）。

    坑：先 `rstrip()` 会把换行剥掉，再判断 `endswith("\\n")` 永远为假，
    于是所有以换行结尾的块都被算成「句子被切断」。
    """
    right = piece.rstrip(" \t")
    return right.endswith(chr(10)) or right.endswith(tuple(SENTENCE_END_CHARS))


def _prose_documents(documents: Sequence[tuple[str, list[DocumentBlock]]]) -> list[tuple[str, list[DocumentBlock]]]:
    """只保留散文类 block（标题/段落/列表/引用），用于「句子是否被切断」。"""
    result = []
    for name, blocks in documents:
        prose = [block for block in blocks if block.block_type in PROSE_BLOCK_TYPES]
        if prose:
            result.append((name, prose))
    return result


def _prose_sentence_cut_ratio(
    key, documents, counter, fixed_chars, child_target, child_max, overlap_tokens, min_overlap, max_overlap
) -> float:
    prose = _prose_documents(documents)
    if not prose:
        return 0.0
    if key == "A":
        chunks = _fixed_windows(prose, fixed_chars)
    elif key in ("B", "C"):
        chunks = _recursive_document_chunks(
            prose, counter, key, child_target, child_max, overlap_tokens, min_overlap, max_overlap
        )
    else:
        builder = ParentBuilder(
            counter=counter,
            config=ParentConfig(
                child_target_tokens=child_target,
                child_max_tokens=child_max,
                child_overlap_tokens=overlap_tokens,
            ),
        )
        # 按 parent 分组：§2.3 规定 Child 不跨 Parent，所以跨 parent 的接缝
        # 是章节边界而不是「句子被切断」，不能算进去。
        chunks = []
        for name, blocks in prose:
            parents = builder.build(blocks, document_id=make_document_id(name), document_version="sha256:probe")
            children = build_children(
                parents, counter=counter, child_target_tokens=child_target,
                child_max_tokens=child_max, overlap_tokens=overlap_tokens,
            )
            grouped: dict[int, list[str]] = {}
            for child in children:
                grouped.setdefault(child.parent_index, []).append(child.content)
            for pieces in grouped.values():
                chunks.append((name, pieces))

    cuts = boundaries = 0
    for _, pieces in chunks:
        for piece in pieces[:-1]:
            boundaries += 1
            if not _ends_cleanly(piece):
                cuts += 1
    return round(cuts / boundaries, 4) if boundaries else 0.0


# --------------------------------------------------------------------- 报告

def build_report(
    metrics: Sequence[VariantMetrics],
    *,
    per_document_rows: Sequence[dict],
    corpus_label: str,
    versions: dict | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# Chunking 质量报告（方案 §11.1）")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 语料：{corpus_label}")
    if versions:
        lines.append("> 版本：" + "，".join(f"{key}={value}" for key, value in versions.items()))
    lines.append("")

    lines.append("## 1. A–E 方案对照")
    lines.append("")
    lines.append("| 方案 | 定义 | chunk 数 | token p50 | p90 | max | 空占比 | 重复占比 | 句子被切断 | 碎块(<80) | Parent 数 | child/parent |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for metric in metrics:
        if metric.status != "ok":
            lines.append(
                f"| {metric.key} | {metric.name} | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {metric.key} | {metric.name} | {metric.chunks} | {metric.tokens_p50} | {metric.tokens_p90} | "
            f"{metric.tokens_max} | {_pct(metric.empty_ratio)} | {_pct(metric.duplicate_ratio)} | "
            f"{_pct(metric.sentence_cut_ratio)} | {_pct(metric.fragment_ratio)} | "
            f"{_dash(metric.parents)} | {_dash(metric.child_per_parent)} |"
        )
    lines.append("")
    lines.append("未实现的方案：")
    for metric in metrics:
        if metric.status != "ok":
            lines.append(f"- **{metric.key} {metric.name}**：{metric.notes[0] if metric.notes else ''}")
    lines.append("")

    lines.append("## 2. 八项人工抽查（自动指标 + 人工评分）")
    lines.append("")
    header = "| 抽查项 | 指标 | 方向 | " + " | ".join(metric.key for metric in metrics) + " | 人工评分 |"
    lines.append(header)
    lines.append("|---|---|---|" + "---|" * len(metrics) + "---|")
    for item in CHECKLIST_ITEMS:
        cells = []
        for metric in metrics:
            value = getattr(metric, item.metric, None) if metric.status == "ok" else None
            cells.append(_dash(value) if not isinstance(value, float) else _pct(value))
        lines.append(
            f"| {item.name} | `{item.metric}` | {'越高越好' if item.direction == 'higher' else '越低越好'} | "
            + " | ".join(cells)
            + " |  |"
        )
    lines.append("")

    lines.append("## 3. 逐文档明细")
    lines.append("")
    lines.append("| 文档 | 字符数 | block | parent | child |")
    lines.append("|---|---|---|---|---|")
    for row in per_document_rows:
        lines.append(
            f"| {row.get('document', '')} | {row.get('chars', 0)} | {row.get('blocks', 0)} | "
            f"{row.get('parents', 0)} | {row.get('children', 0)} |"
        )
    lines.append("")

    lines.append("## 4. 读法")
    lines.append("")
    lines.append("- **句子被切断** 越低越好：A 固定长度应该最高，D 应该最低（结构分隔符 + 句子边界回退）。")
    lines.append("- **碎块(<80 token)** 高说明 `merge` 没吃饱或 `min_child` 太严，是 D09 校验要拦的目标。")
    lines.append("- **child/parent** 接近 1 说明 Child 没比 Parent 更细，两级结构只剩体积成本——这是调 `target_tokens`/`min_tokens` 的信号。")
    lines.append("- **人工评分** 列留空：自动指标只能筛掉明显坏掉的方案，语义完整性仍要人看 30–50 篇。")
    lines.append("")
    return "\n".join(lines)


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{value:.2%}" if isinstance(value, float) else str(value)


def _dash(value) -> str:
    return "—" if value is None else f"{value:g}" if isinstance(value, float) else str(value)


# --------------------------------------------------------------------- CLI

def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.rag.ingestion.quality_report")
    parser.add_argument("--dir", required=True, help="语料目录（递归扫描支持的后缀）")
    parser.add_argument("--limit", type=int, default=40, help="最多处理多少篇（D11 要求 30–50 篇）")
    parser.add_argument("--out", default=None, help="报告输出路径（markdown）")
    parser.add_argument("--json", default=None, help="同时输出 JSON 指标")
    parser.add_argument("--fixed-chars", type=int, default=500, help="方案 A 的字符窗口")
    args = parser.parse_args(argv)

    root = Path(args.dir)
    if not root.exists():
        print(f"目录不存在：{root}")
        return 2

    from .parser import get_parser, register_builtin_parsers

    register_builtin_parsers(replace=True)
    paths = [path for path in sorted(root.rglob("*.md"))][: args.limit]
    if not paths:
        print(f"目录里没有 .md 文件：{root}")
        return 2

    counter = default_counter()
    documents = []
    rows = []
    for path in paths:
        name = path.relative_to(root).as_posix()
        blocks = get_parser(str(path)).parse(str(path))
        documents.append((name, blocks))
        rows.append(
            {
                "document": name,
                "chars": sum(len(block.text) for block in blocks),
                "blocks": len(blocks),
                "parents": 0,
                "children": 0,
            }
        )

    metrics = run_all_variants(documents, counter=counter, fixed_chars=args.fixed_chars)
    # 逐文档明细补上 D 方案的 parent/child（只有它有两级结构）
    builder = ParentBuilder(counter=counter)
    for row, (name, blocks) in zip(rows, documents):
        parents = builder.build(blocks, document_id=make_document_id(name), document_version="sha256:probe")
        children = build_children(parents, counter=counter)
        row["parents"], row["children"] = len(parents), len(children)

    report = build_report(
        metrics,
        per_document_rows=rows,
        corpus_label=f"{root} 前 {len(documents)} 篇",
        versions={"chunker_version": ParentBuilder.chunker_version},
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"报告已写入 {args.out}")
    else:
        print(report)

    if args.json:
        import json

        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps([metric.to_dict() for metric in metrics], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"指标已写入 {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(cli_main())
