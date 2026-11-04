"""D32 Context Assembler：把排名候选变成「能进 prompt 的上下文」。

`format_context()`（`src/rag.py`）是 demo 级实现：全量拼接、无去重、无预算。
实测 top-4 里同文档多 chunk 占 **79.6%**、相邻重叠块占 **22.9%**，
10 个父块轻松 7,000+ token。这个模块把四件事一次做掉：

1. **去重** —— 同一父块重复出现（多路命中）只留一次；不同父块正文完全相同（多版本文档）也只留一次；
2. **邻接合并** —— 同文档 `parent_index` 相邻的块合并成一块，合并时剥掉 D07 留下的接缝重叠；
3. **Token 预算** —— 默认 6000（方案 §3.6 的 `max_context_tokens`），超预算就截最后一块、再超就丢尾部并留痕；
4. **来源字段** —— 每块带 `document_id / parent_ids / chunk_ids / section_path / source_uri`，答案层才能给可点击的引用。

组装顺序 = 排名顺序（不做重排），这样「上下文顺序」和「检索指标」是同一个东西，D34 的 F/G 才能对齐比较。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: 方案 §3.6：上下文 token 上限
DEFAULT_BUDGET = 6000

#: 合并相邻父块时，从后一块开头剥掉的重叠长度上限（字符）。D07 的 overlap 是 50–70 token，
#: 中文 1 token ≈ 1.6 字符，所以 400 字符足够覆盖，同时不会误删正常重复的开头。
MAX_SEAM_CHARS = 400

#: 接缝最短匹配长度：太短（如 8 个字）会误判，实测 24 字符已能稳定命中重叠段
MIN_SEAM_CHARS = 24


def _path_list(raw: Any) -> list[str]:
    """`section_path` 的规范化：list 直接用；JSON 文本解析；其余退化成空列表。

    上游（`load_parent_records`）已经会解析，这里再写一次是因为**字符串被 `list()`
    拆成字符是静默失败**——不报错，只是 prompt 里多出几百个孤立字符（N11）。
    """
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return []
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    return []


@dataclass
class ContextBlock:
    """上下文里的一块。`parent_ids` 可能是多个（邻接合并的结果）。"""

    index: int
    document_id: str
    parent_ids: list[str]
    chunk_ids: list[str]
    section_path: list[str]
    source_uri: str
    title: str
    text: str
    tokens: int
    score: float = 0.0
    rerank_score: float | None = None
    truncated: bool = False
    parent_indexes: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "document_id": self.document_id,
            "parent_ids": list(self.parent_ids),
            "parent_indexes": list(self.parent_indexes),
            "chunk_ids": list(self.chunk_ids),
            "section_path": list(self.section_path),
            "source_uri": self.source_uri,
            "title": self.title,
            "text": self.text,
            "tokens": self.tokens,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "truncated": self.truncated,
        }

    def header(self) -> str:
        path = " > ".join(self.section_path) if self.section_path else ""
        parts = [f"[证据 {self.index}]", f"document={self.document_id}"]
        if self.title:
            parts.append(f"title={self.title}")
        if path:
            parts.append(f"section={path}")
        if self.source_uri:
            parts.append(f"source={self.source_uri}")
        return " ".join(parts)


@dataclass
class AssembledContext:
    blocks: list[ContextBlock]
    tokens: int
    dropped: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """拼成 prompt 里那段「证据」文本。编号就是答案层要引用的 `[证据 n]`。"""
        return "\n\n".join(f"{block.header()}\n{block.text}" for block in self.blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "blocks": [block.as_dict() for block in self.blocks],
            "dropped": list(self.dropped),
            "stats": dict(self.stats),
        }


#: 近重复判定：5-gram 的 Jaccard 阈值。多版本文档只改一两行，用精确相等的 hash 抓不到
NEAR_DUP_JACCARD = 0.85
#: 太短的文本不做近重复判定（Jaccard 在小样本上抖动大）
NEAR_DUP_MIN_CHARS = 200


def _shingles(text: str, size: int = 5) -> set[str]:
    compact = " ".join(text.split())
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[i:i + size] for i in range(len(compact) - size + 1)}


def _is_near_duplicate(left: str, right: str, threshold: float = NEAR_DUP_JACCARD) -> bool:
    if min(len(left), len(right)) < NEAR_DUP_MIN_CHARS:
        return False
    a, b = _shingles(left), _shingles(right)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= threshold


def document_fragmentation(blocks: Sequence["ContextBlock"]) -> float:
    """同文档碎片占比 = 1 − 独立文档数 / 块数（0 表示每个块都来自不同文档）。"""
    if not blocks:
        return 0.0
    return 1.0 - len({block.document_id for block in blocks}) / len(blocks)


def _content_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()


def _longest_suffix_prefix(left: str, right: str, *, max_chars: int = MAX_SEAM_CHARS) -> int:
    """`left` 的后缀与 `right` 的前缀最长公共长度（只看到 `max_chars`）。"""
    limit = min(len(left), len(right), max_chars)
    for size in range(limit, MIN_SEAM_CHARS - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def merge_parent_texts(texts: Sequence[str]) -> str:
    """相邻父块合并：剥掉接缝处重复的那段，用空行连接。"""
    if not texts:
        return ""
    merged = texts[0]
    for text in texts[1:]:
        seam = _longest_suffix_prefix(merged, text)
        merged = merged + "\n\n" + text[seam:] if seam else merged + "\n\n" + text
    return merged


def adjacent_overlap_ratio(blocks: Sequence[ContextBlock]) -> float:
    """「同文档相邻块」占比 —— D32 的验收指标（改造前 22.9%，目标 ≤5%）。

    口径：**参与「同文档相邻对」的块数 / 总块数**（一对相邻块两块都算）。
    合并之后这类块本来就不该存在，所以它衡量的是「合并有没有真的生效」。
    """
    if not blocks:
        return 0.0
    tainted = 0
    for index, block in enumerate(blocks):
        neighbour_before = blocks[index - 1] if index > 0 else None
        neighbour_after = blocks[index + 1] if index + 1 < len(blocks) else None
        if any(other is not None and _is_adjacent(other, block)
               for other in (neighbour_before, neighbour_after)):
            tainted += 1
    return tainted / len(blocks)


def _has_seam(left: str, right: str) -> bool:
    """两段正文在接缝处有 ≥MIN_SEAM_CHARS 的重叠（谁在前都算）。"""
    return (_longest_suffix_prefix(left, right) >= MIN_SEAM_CHARS
            or _longest_suffix_prefix(right, left) >= MIN_SEAM_CHARS)


def _is_adjacent(left: "ContextBlock", right: "ContextBlock") -> bool:
    """判定「同文档相邻重叠块」。

    优先用 `parent_index` 相接；序号缺失（当前库里 **全部为 NULL**，见 N6）时
    退化成**数据驱动**判据：两段正文在接缝处有明显重叠（D07 的 overlap 留下的尾巴）。
    后者其实更贴近指标本意——它数的是「同一句话被喂了两遍」。
    """
    if left.document_id != right.document_id:
        return False
    if left.parent_indexes and right.parent_indexes:
        # 必须**恰好**相接：`right.min == left.max + 1`（或反序）。
        # 写成 `right.min <= left.max + 1` 会把「同一文档里隔着 20 个父块的两块」
        # 也算成重叠，实测把组装后的重叠占比虚报到 17.9%。
        return (min(right.parent_indexes) == max(left.parent_indexes) + 1
                or min(left.parent_indexes) == max(right.parent_indexes) + 1)
    return _has_seam(left.text, right.text)


def _as_hit(hit: Any) -> dict[str, Any]:
    if isinstance(hit, Mapping):
        return dict(hit)
    return {
        "parent_id": getattr(hit, "parent_id", None) or getattr(hit, "id", None),
        "score": getattr(hit, "score", 0.0),
        "rerank_score": getattr(hit, "rerank_score", None),
    }


def assemble(
    hits: Iterable[Any],
    parents: Mapping[str, Mapping[str, Any]],
    *,
    counter,
    budget: int = DEFAULT_BUDGET,
    dedupe: bool = True,
    merge_adjacent: bool = True,
) -> AssembledContext:
    """把排名候选组装成上下文。

    - `hits`：排名顺序的候选（dict 或带 `parent_id/score` 的对象），`parent_id` 必须在 `parents` 里；
    - `parents`：`parent_id → {document_id, content|text, parent_index, section_path, title, source_uri}`；
    - `counter`：要 `count(text)` 与 `truncate(text, max_tokens)`（`TokenCounter` 满足）。
    """
    stats = {
        "candidates": 0,
        "missing_parent": 0,
        "duplicate_parent": 0,
        "duplicate_text": 0,
        "near_duplicate_text": 0,
        "merged": 0,
        "truncated": 0,
        "dropped": 0,
    }

    # 1) 去重：同父块、同正文都只留第一个（排名更靠前的那个）
    seen_parents: set[str] = set()
    seen_text: set[str] = set()
    seen_text_samples: list[str] = []
    ordered: list[dict[str, Any]] = []
    for raw in hits:
        hit = _as_hit(raw)
        stats["candidates"] += 1
        parent_id = hit.get("parent_id")
        record = parents.get(parent_id) if parent_id else None
        if record is None:
            stats["missing_parent"] += 1
            continue
        text = record.get("content") or record.get("text") or ""
        if dedupe:
            if parent_id in seen_parents:
                stats["duplicate_parent"] += 1
                continue
            digest = _content_hash(text)
            if digest in seen_text:
                stats["duplicate_text"] += 1
                continue
            if any(_is_near_duplicate(text, kept) for kept in seen_text_samples):
                stats["near_duplicate_text"] += 1
                continue
            seen_text_samples.append(text)
            seen_text.add(digest)
            seen_parents.add(parent_id)
        ordered.append({"parent_id": parent_id, "record": record, "text": text, "hit": hit})

    # 2) 邻接合并：**按文档分组**再按 parent_index 排，相接的并成一块。
    #    只按「排名相邻」合并是不够的——同一文档的两个相邻父块常常分居第 3 名和第 9 名，
    #    那样几乎永远合不上（实测合并次数 0），重叠块该在还是在。
    #    合并后的块落在**组内最好名次**的位置上，这样上下文顺序仍然是检索顺序。
    groups: list[list[dict[str, Any]]] = []
    if merge_adjacent:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for rank, item in enumerate(ordered):
            item["rank"] = rank
            buckets.setdefault(str(item["record"].get("document_id") or ""), []).append(item)
        for bucket in buckets.values():
            bucket.sort(key=lambda it: (it["record"].get("parent_index") is None,
                                        it["record"].get("parent_index") or 0, it["rank"]))
            pending: list[dict[str, Any]] = []
            for item in bucket:
                if pending:
                    previous_index = pending[-1]["record"].get("parent_index")
                    current_index = item["record"].get("parent_index")
                    by_index = (isinstance(previous_index, int) and isinstance(current_index, int)
                                and current_index == previous_index + 1)
                    by_seam = _has_seam(merge_parent_texts([it["text"] for it in pending]),
                                        item["text"])
                    if by_index or by_seam:
                        pending.append(item)
                        stats["merged"] += 1
                        continue
                    groups.append(pending)
                pending = [item]
            if pending:
                groups.append(pending)
        groups.sort(key=lambda group: min(item["rank"] for item in group))
    else:
        for rank, item in enumerate(ordered):
            item["rank"] = rank
            groups.append([item])

    # 3) Token 预算：截最后一块、丢尾部
    blocks: list[ContextBlock] = []
    dropped: list[dict[str, Any]] = []
    used = 0
    for group in groups:
        text = merge_parent_texts([item["text"] for item in group])
        tokens = counter.count(text)
        truncated = False
        if used + tokens > budget:
            remaining = budget - used
            if remaining <= 0:
                dropped.append({"parent_ids": [item["parent_id"] for item in group],
                                "reason": "budget", "tokens": tokens})
                stats["dropped"] += 1
                continue
            text = counter.truncate(text, remaining)
            tokens = counter.count(text)
            truncated = True
            stats["truncated"] += 1
        head = group[0]
        blocks.append(ContextBlock(
            index=len(blocks) + 1,
            document_id=str(head["record"].get("document_id") or ""),
            parent_ids=[item["parent_id"] for item in group],
            parent_indexes=[int(item["record"].get("parent_index")) for item in group
                            if isinstance(item["record"].get("parent_index"), int)],
            chunk_ids=[item["parent_id"] for item in group],
            section_path=_path_list(head["record"].get("section_path")),
            source_uri=str(head["record"].get("source_uri") or ""),
            title=str(head["record"].get("title") or ""),
            text=text,
            tokens=tokens,
            score=float(head["hit"].get("score") or 0.0),
            rerank_score=head["hit"].get("rerank_score"),
            truncated=truncated,
        ))
        used += tokens

    stats["adjacent_overlap_ratio"] = round(adjacent_overlap_ratio(blocks), 4)
    stats["document_fragmentation"] = round(document_fragmentation(blocks), 4)
    stats["blocks"] = len(blocks)
    stats["tokens"] = used
    return AssembledContext(blocks=blocks, tokens=used, dropped=dropped, stats=stats)
