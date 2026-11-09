"""D08 Parent Builder + Child 关联（方案 §2.3 Parent/Child 构建规则）。

Parent 的职责是**提供上下文**，不是被检索的对象；所以它的边界要服从文档结构：
先在标题/章节处断开，再在同一章节内部按 block 边界累积到目标长度。

算法三段：

1. **分组**：`heading` 块（或 `section_path` 变化）开启新组。标题块留在组内，
   这样 Parent 内容以标题开头——「不要把标题和正文分离」。
2. **切长**：组内按 block 边界累积，超过 `max_tokens` 就另起一块。
   单个 block 自己就超限时，用 D06 的递归切分在段落/句子边界劈开
   （piece 是原文的连续切片，所以偏移可以累加算出来）。
3. **并短**：短于 `min_tokens` 的块尝试并进**同一父章节**的前一块
   （§2.3 规则 4）；合并后 `section_path` 取两者共同前缀，绝不跨顶级章节（规则 5）。

Child 的 overlap 归 D07，这里只做不重叠切分 + `parent_id` 关联。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from .block import DocumentBlock
from .ids import content_hash, make_chunk_id, make_parent_id
from .overlap import (
    DEFAULT_MAX_OVERLAP_TOKENS,
    DEFAULT_MIN_OVERLAP_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    apply_overlap,
)
from .recursive_chunker import recursive_split
from .tokenizer import TokenCounter

#: 切分规则版本：改切分逻辑必须同时改这两个版本号，否则 ID 不会变（D10 负向测试会抓到）
PARENT_CHUNKER_VERSION = "recursive_parent_child_v1"

#: 子块规则版本。v2 = 重叠语义固定为 `chunk_size`（含重叠的子块不得超过 `child_max_tokens`）。
#: 改这里的代价是**所有子块 id 变化**（id 含版本号），但**父块 id 不变** —— 所以
#: `gold_chunks`（父块粒度）不会被打破，只有 `gold_points`（子块粒度）需要重算。
CHILD_CHUNKER_VERSION = "recursive_child_v2"


def ingest_chunker_version(parent_version: str | None = None) -> str:
    """入库闸门用的切分版本 = 父 + 子两段（N43）。

    为什么不能只用父块版本：这一轮改的是**子块**的重叠语义。只用父块版本当闸门，
    升级后所有 `COMPLETED` 文档会被 `[skip]`，旧语义的子块留在库里、新文档用新语义 ——
    一份库混着两套切分语义，而且看不出来。

    为什么用 `+` 拼而不是另起一个常量：两个版本号都要能在库里直接读出来，
    出事时 `--status` / `index_versions` 里一眼能看出「父是哪版、子是哪版」。
    """
    return f"{parent_version or PARENT_CHUNKER_VERSION}+{CHILD_CHUNKER_VERSION}"

#: block 之间拼接用的分隔符（Parent 内容里段落之间保留空行）
_BLOCK_JOINER = "\n\n"


@dataclass
class ParentConfig:
    """§2.3 的初始参数。"""

    target_tokens: int = 1500
    max_tokens: int = 2000
    min_tokens: int = 200
    child_target_tokens: int = 400
    child_max_tokens: int = 500
    child_overlap_tokens: int = DEFAULT_OVERLAP_TOKENS

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正数，收到 {self.max_tokens}")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens 不能为负，收到 {self.min_tokens}")
        if self.child_max_tokens <= 0:
            raise ValueError(f"child_max_tokens 必须为正数，收到 {self.child_max_tokens}")
        if self.child_overlap_tokens < 0:
            raise ValueError(f"child_overlap_tokens 不能为负，收到 {self.child_overlap_tokens}")
        if self.child_overlap_tokens >= self.child_max_tokens:
            raise ValueError(
                f"child_overlap_tokens({self.child_overlap_tokens}) 必须小于 "
                f"child_max_tokens({self.child_max_tokens})，否则切不出内容"
            )


@dataclass
class _Fragment:
    """构建过程中的中间块（还没落成 Parent）。"""

    content: str
    token_count: int
    section_path: list[str]
    block_indexes: list[int] = field(default_factory=list)
    source_start: int | None = None
    source_end: int | None = None


@dataclass
class Parent:
    parent_index: int
    content: str
    token_count: int
    section_path: list[str]
    source_start: int | None = None
    source_end: int | None = None
    block_indexes: list[int] = field(default_factory=list)
    document_id: str | None = None
    document_version: str | None = None
    title: str | None = None
    language: str | None = None
    chunker_version: str = PARENT_CHUNKER_VERSION
    parent_id: str | None = None

    def to_dict(self) -> dict:
        """完整字段快照（含内部序号），用于确定性/回归对比。"""
        return asdict(self)

    def to_record(self) -> dict:
        """按 §3.2 的字段输出（字段名与顺序固定，D12 会按它校验）。"""
        if not self.parent_id:
            raise ValueError(
                f"parent {self.parent_index} 没有 parent_id：请在 build() 时传 document_id 与 document_version"
            )
        return {
            "parent_id": self.parent_id,
            "document_id": self.document_id,
            "document_version": self.document_version,
            "language": self.language,
            "title": self.title,
            "section_path": list(self.section_path),
            "content": self.content,
            "token_count": self.token_count,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "chunker_version": self.chunker_version,
            # parent_index 必须落库：D09 的父块关联键是 (document_id, parent_index)，
            # D32 的「同文档相邻合并」也靠它。漏了它，两处都退化成不可用（实测 25,411 行全 NULL）。
            "parent_index": self.parent_index,
        }


@dataclass
class Child:
    parent_index: int
    child_index: int
    content: str
    token_count: int
    section_path: list[str]
    source_start: int | None = None
    source_end: int | None = None
    document_id: str | None = None
    document_version: str | None = None
    title: str | None = None
    language: str | None = None
    chunker_version: str = CHILD_CHUNKER_VERSION
    embedding_model: str = ""
    embedding_version: str = ""
    status: str = "active"
    chunk_id: str | None = None
    parent_id: str | None = None

    def to_dict(self) -> dict:
        """完整字段快照（含内部序号），用于确定性/回归对比。"""
        return asdict(self)

    def to_record(self) -> dict:
        """按 §3.3 的字段输出。"""
        if not self.chunk_id:
            raise ValueError(f"child {self.parent_index}/{self.child_index} 没有 chunk_id")
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "document_id": self.document_id,
            "document_version": self.document_version,
            "language": self.language,
            "title": self.title,
            "section_path": list(self.section_path),
            "content": self.content,
            "embedding_text": build_embedding_text(self.title, self.section_path, self.content),
            "token_count": self.token_count,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "chunker_version": self.chunker_version,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "content_hash": content_hash(self.content),
            "status": self.status,
        }


def build_embedding_text(title: str | None, section_path: Sequence[str], content: str) -> str:
    """§2.8 的 embedding 输入格式：把标题与章节一起送进向量模型。"""
    section = " / ".join(section_path) if section_path else ""
    lines = []
    if title:
        lines.append(f"Document: {title}")
    if section:
        lines.append(f"Section: {section}")
    lines.append(f"Content: {content}")
    return "\n".join(lines)


class ParentBuilder:
    chunker_version = PARENT_CHUNKER_VERSION

    def __init__(self, counter: TokenCounter, config: ParentConfig | None = None) -> None:
        self.counter = counter
        self.config = config or ParentConfig()

    @property
    def ingest_version(self) -> str:
        """入库闸门用的版本（父 + 子）。**不要**拿它算 id：父块 id 只认父版本。"""
        return ingest_chunker_version(self.chunker_version)

    def build(
        self,
        blocks: Iterable[DocumentBlock],
        *,
        document_id: str | None = None,
        document_version: str | None = None,
        title: str | None = None,
        language: str | None = None,
    ) -> list[Parent]:
        block_list = list(blocks)
        if not block_list:
            return []

        fragments: list[_Fragment] = []
        for group in self._group(block_list):
            fragments.extend(self._split_long(group, block_list))
        fragments = self._merge_short(fragments)

        parents: list[Parent] = []
        for index, fragment in enumerate(fragments):
            parent = Parent(
                parent_index=index,
                content=fragment.content,
                token_count=fragment.token_count,
                section_path=fragment.section_path,
                source_start=fragment.source_start,
                source_end=fragment.source_end,
                block_indexes=fragment.block_indexes,
                document_id=document_id,
                document_version=document_version,
                title=title,
                language=language,
                chunker_version=self.chunker_version,
            )
            if document_id and document_version:
                parent.parent_id = make_parent_id(
                    document_id, document_version, index, self.chunker_version
                )
            parents.append(parent)
        return parents

    # ------------------------------------------------------------------ 分组

    def _group(self, blocks: list[DocumentBlock]) -> list[list[int]]:
        """按标题/章节边界分组，返回 block 下标分组。"""
        groups: list[list[int]] = []
        current: list[int] = []
        current_path: tuple[str, ...] = ()
        for index, block in enumerate(blocks):
            path = tuple(block.section_path)
            starts_new = block.block_type == "heading" or (current and path != current_path)
            if starts_new and current:
                groups.append(current)
                current = []
            current.append(index)
            current_path = path
        if current:
            groups.append(current)
        return groups

    # ------------------------------------------------------------------ 切长

    def _split_long(self, indexes: list[int], blocks: list[DocumentBlock]) -> list[_Fragment]:
        fragments: list[_Fragment] = []
        buffer: list[int] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            fragments.append(self._fragment_from_blocks(buffer, blocks))
            buffer, buffer_tokens = [], 0

        for index in indexes:
            block = blocks[index]
            tokens = self.counter.count(block.text)
            if tokens > self.config.max_tokens:
                # 单个 block 就超限：先收掉已累积的，再把它按段落/句子切开
                flush()
                fragments.extend(self._split_oversized_block(block))
                continue
            joiner_tokens = self.counter.count(_BLOCK_JOINER) if buffer else 0
            if buffer and buffer_tokens + joiner_tokens + tokens > self.config.max_tokens:
                flush()
                joiner_tokens = 0
            buffer.append(index)
            buffer_tokens += joiner_tokens + tokens
        flush()
        return fragments

    def _fragment_from_blocks(self, indexes: list[int], blocks: list[DocumentBlock]) -> _Fragment:
        texts = [blocks[index].text for index in indexes]
        content = _BLOCK_JOINER.join(texts)
        starts = [blocks[index].source_start for index in indexes if blocks[index].source_start is not None]
        ends = [blocks[index].source_end for index in indexes if blocks[index].source_end is not None]
        return _Fragment(
            content=content,
            token_count=self.counter.count(content),
            section_path=list(blocks[indexes[0]].section_path),
            block_indexes=list(indexes),
            source_start=min(starts) if starts else None,
            source_end=max(ends) if ends else None,
        )

    def _split_oversized_block(self, block: DocumentBlock) -> list[_Fragment]:
        pieces = recursive_split(
            block.text,
            counter=self.counter,
            max_tokens=self.config.max_tokens,
            target_tokens=self.config.target_tokens,
        )
        fragments: list[_Fragment] = []
        offset = 0
        for piece in pieces:
            start = None if block.source_start is None else block.source_start + offset
            end = None if block.source_start is None else block.source_start + offset + len(piece)
            fragments.append(
                _Fragment(
                    content=piece,
                    token_count=self.counter.count(piece),
                    section_path=list(block.section_path),
                    source_start=start,
                    source_end=end,
                )
            )
            offset += len(piece)
        return fragments

    # ------------------------------------------------------------------ 并短

    def _merge_short(self, fragments: list[_Fragment]) -> list[_Fragment]:
        if self.config.min_tokens <= 0:
            return fragments

        merged: list[_Fragment] = []
        for fragment in fragments:
            if merged and fragment.token_count < self.config.min_tokens:
                previous = merged[-1]
                combined = self.counter.count(previous.content) + self.counter.count(_BLOCK_JOINER) + fragment.token_count
                if _same_parent_chapter(previous.section_path, fragment.section_path) and combined <= self.config.max_tokens:
                    previous.content = previous.content + _BLOCK_JOINER + fragment.content
                    previous.token_count = self.counter.count(previous.content)
                    previous.block_indexes = previous.block_indexes + fragment.block_indexes
                    if previous.source_start is not None and fragment.source_start is not None:
                        previous.source_start = min(previous.source_start, fragment.source_start)
                    if previous.source_end is not None and fragment.source_end is not None:
                        previous.source_end = max(previous.source_end, fragment.source_end)
                    # 合并后取两者共同前缀：内容现在属于更外层的章节
                    previous.section_path = _common_prefix(previous.section_path, fragment.section_path)
                    continue
            merged.append(fragment)
        return merged


def _common_prefix(left: Sequence[str], right: Sequence[str]) -> list[str]:
    result: list[str] = []
    for a, b in zip(left, right):
        if a != b:
            break
        result.append(a)
    return result or list(left)


def _same_parent_chapter(left: Sequence[str], right: Sequence[str]) -> bool:
    """同一父章节：顶级标题相同，且两条 section_path 有前缀关系（§2.3 规则 4/5）。"""
    if not left or not right:
        return False
    if left[0] != right[0]:
        return False
    return list(left[: len(right)]) == list(right) or list(right[: len(left)]) == list(left)


def build_children(
    parents: Sequence[Parent],
    *,
    counter: TokenCounter,
    child_target_tokens: int = ParentConfig.child_target_tokens,
    child_max_tokens: int = ParentConfig.child_max_tokens,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    embedding_model: str = "",
    embedding_version: str = "",
) -> list[Child]:
    """把每个 Parent 切成 Child 并回填 `parent_id`（overlap 由 D07 负责）。

    不变量：Child 的 content 一定是 Parent content 的连续切片——`recursive_split`
    保证 `"".join(pieces) == parent.content`，所以既能用 `in parent.content`
    校验归属，也能据此换算偏移。
    """
    if overlap_tokens >= child_max_tokens:
        raise ValueError(
            f"overlap_tokens({overlap_tokens}) 必须小于 child_max_tokens({child_max_tokens})"
        )

    # 先按「max - overlap」切，贴上重叠后仍然不超 max（否则 overlap 会把上限顶破）
    split_max = child_max_tokens - overlap_tokens
    split_target = max(1, child_target_tokens - overlap_tokens)

    # overlap 的**上限**也要收在 `overlap_tokens` 以内：`apply_overlap` 的默认
    # `max_overlap_tokens=70` 比 `DEFAULT_OVERLAP_TOKENS=60` 大，取 70 时会有几块溢出
    # `child_max_tokens`（基准库实测 5/9,795 块超 1–5 个 token）。
    # 语义选定：**含 overlap 的子块 ≤ child_max_tokens**（上游的 `max_tokens_strategy=chunk_size`）。
    overlap_cap = max(0, min(DEFAULT_MAX_OVERLAP_TOKENS, overlap_tokens))
    overlap_floor = max(0, min(DEFAULT_MIN_OVERLAP_TOKENS, overlap_cap))

    children: list[Child] = []
    for parent in parents:
        raw_pieces = recursive_split(
            parent.content,
            counter=counter,
            max_tokens=split_max,
            target_tokens=split_target,
        )
        pieces = apply_overlap(
            raw_pieces,
            counter,
            overlap_tokens=overlap_tokens,
            min_overlap_tokens=overlap_floor,
            max_overlap_tokens=overlap_cap,
        )
        for child_index, piece in enumerate(pieces):
            content = piece.content
            char_start = piece.char_start
            if not content.strip():
                continue
            # 兜底：无论 overlap 挑到多长，都不许把子块顶过 `child_max_tokens`。
            # 丢掉的是**重叠区**，不丢正文（正文仍在后一块里）。
            drop = _fit_front(content, counter, child_max_tokens)
            if drop:
                content = content[drop:]
                char_start += drop
            start = None if parent.source_start is None else parent.source_start + char_start
            end = None if parent.source_start is None else parent.source_start + char_start + len(content)
            child = Child(
                parent_index=parent.parent_index,
                child_index=child_index,
                content=content,
                token_count=counter.count(content),
                section_path=list(parent.section_path),
                source_start=start,
                source_end=end,
                document_id=parent.document_id,
                document_version=parent.document_version,
                title=parent.title,
                language=parent.language,
                chunker_version=CHILD_CHUNKER_VERSION,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
                parent_id=parent.parent_id,
            )
            if parent.parent_id:
                child.chunk_id = make_chunk_id(
                    parent.parent_id, child_index, content_hash(content), CHILD_CHUNKER_VERSION
                )
            children.append(child)
    return children


def _fit_front(content: str, counter: TokenCounter, max_tokens: int) -> int:
    """返回「从前面丢掉多少个字符」能让剩余部分 ≤ `max_tokens`（丢的是重叠区）。

    二分而不是逐字符：真实词表下「1 token ≈ 1 个字」并不成立，逐字符循环在长块上
    会变成几千次分词。
    """
    if counter.count(content) <= max_tokens:
        return 0
    low, high = 0, len(content)
    while low < high:
        middle = (low + high) // 2
        if counter.count(content[middle:]) <= max_tokens:
            high = middle
        else:
            low = middle + 1
    return low


__all__ = [
    "CHILD_CHUNKER_VERSION",
    "ingest_chunker_version",
    "PARENT_CHUNKER_VERSION",
    "Child",
    "Parent",
    "ParentBuilder",
    "ParentConfig",
    "build_children",
    "build_embedding_text",
]
