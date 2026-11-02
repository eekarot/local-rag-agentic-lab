"""D07 Overlap：在 Child 之间制造受控重叠（方案 §2.5）。

为什么不能简单「把上一块尾部贴到下一块前面」：

- **按 token 窗口硬贴**会把句子斩成两半，衔接处读起来是半个词；
  所以先在上一块里找**句子起点**，挑一个让重叠长度落进
  `[min_overlap, max_overlap]` 的最近候选。
- **句子边界可能不给面子**：最近边界可能只有 20 token（不够）或 90 token
  （太肥）。两者都要退回 token 窗口——宁可在词中间切，也不让重叠失控。
- **重叠必须能定位**：返回的 `char_start/char_end` 是相对 parent 内容的
  偏移，`parent.content[char_start:char_end] == content` 必须成立，
  这样 D09 的边界校验、D12 的 `source_start/end` 才有意义。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .tokenizer import TokenCounter

#: 重叠目标与容差（D07 验收口径：overlap token ∈ [50, 70]）
DEFAULT_OVERLAP_TOKENS = 60
DEFAULT_MIN_OVERLAP_TOKENS = 50
DEFAULT_MAX_OVERLAP_TOKENS = 70

#: 句子起点：中文标点后、英文句末标点后、换行后
SENTENCE_START_RE = re.compile(r"(?<=[。！？；!?])|(?<=\.)(?=\s)|(?<=\n)")


@dataclass
class OverlappedPiece:
    """一个带重叠的 Child 片段（偏移相对 parent 内容）。"""

    content: str
    char_start: int
    char_end: int
    overlap_chars: int = 0
    overlap_tokens: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OverlappedPiece):
            return NotImplemented
        return (
            self.content == other.content
            and self.char_start == other.char_start
            and self.char_end == other.char_end
        )


def apply_overlap(
    pieces: Sequence[str],
    counter: TokenCounter,
    *,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_overlap_tokens: int = DEFAULT_MIN_OVERLAP_TOKENS,
    max_overlap_tokens: int = DEFAULT_MAX_OVERLAP_TOKENS,
) -> list[OverlappedPiece]:
    """给相邻片段加重叠；`pieces` 必须是同一段文本的**连续切片**。

    返回值的 `content` 仍是原文的连续切片（重叠部分取自上一块尾部）。
    """
    if not pieces:
        return []
    if min_overlap_tokens < 0 or max_overlap_tokens < min_overlap_tokens:
        raise ValueError(
            f"overlap 区间不合法：min={min_overlap_tokens}, max={max_overlap_tokens}"
        )

    result: list[OverlappedPiece] = []
    position = 0
    for index, piece in enumerate(pieces):
        if index == 0 or overlap_tokens <= 0:
            result.append(
                OverlappedPiece(content=piece, char_start=position, char_end=position + len(piece))
            )
            position += len(piece)
            continue

        previous = pieces[index - 1]
        overlap_text = _pick_overlap(
            previous,
            counter,
            target=overlap_tokens,
            minimum=min_overlap_tokens,
            maximum=max_overlap_tokens,
        )
        content = overlap_text + piece
        start = position - len(overlap_text)
        result.append(
            OverlappedPiece(
                content=content,
                char_start=start,
                char_end=start + len(content),
                overlap_chars=len(overlap_text),
                overlap_tokens=counter.count(overlap_text) if overlap_text else 0,
            )
        )
        position += len(piece)
    return result


def _pick_overlap(
    previous: str,
    counter: TokenCounter,
    *,
    target: int,
    minimum: int,
    maximum: int,
) -> str:
    """在上一块里挑一段尾巴做重叠：优先句子边界，其次 token 窗口。"""
    if not previous:
        return ""

    best_text, best_distance = None, None
    for match in SENTENCE_START_RE.finditer(previous):
        start = match.end()
        if start <= 0 or start >= len(previous):
            continue
        candidate = previous[start:]
        tokens = counter.count(candidate)
        if tokens < minimum or tokens > maximum:
            continue
        distance = abs(tokens - target)
        if best_distance is None or distance < best_distance:
            best_text, best_distance = candidate, distance

    if best_text is not None:
        return best_text

    # 没有合适的句子边界：退回 token 窗口（宁可在词中间切，也不让重叠失控）
    window = counter.truncate_tail(previous, min(target, maximum))
    while window and counter.count(window) > maximum:
        window = window[1:]
    return window


__all__ = [
    "DEFAULT_MAX_OVERLAP_TOKENS",
    "DEFAULT_MIN_OVERLAP_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "OverlappedPiece",
    "apply_overlap",
]
