"""D03 TXT Parser：只做段落识别，长度交给 D06（方案 §2.2「TXT → 段落、句子识别」）。

TXT 没有可依赖的结构标记，所以这里刻意**不做**启发式标题识别：
猜错的标题会把段落错误地塞进假章节，比没有章节更糟（`section_path`
回退到文件名，至少是真话）。

段落内的软换行原样保留（用 `\\n` 连接），不改写成空格——中文正文里
插空格会污染原文，而 `\\n` 本来就在 D06 的分隔符表里。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..block import DocumentBlock
from .base import BaseParser, read_text_file

#: 段落分隔：连续两个及以上换行（允许中间夹空白）
PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
#: 句子结束标记：中文标点任意连写，英文句号要求后面是空白或行尾（避开小数、缩写）
SENTENCE_END_RE = re.compile(r"[。！？；!?]+|\.(?=\s|$)|[!?]+")


class TxtParser(BaseParser):
    parser_version = "txt_v1"
    supported_suffixes = (".txt", ".text")

    def parse(self, path: str, text: str | None = None) -> list[DocumentBlock]:
        source = text if text is not None else read_text_file(path)
        source = source.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        if not source.strip():
            return []

        stem = Path(path).stem or "document"
        blocks: list[DocumentBlock] = []
        cursor = 0
        for raw in PARAGRAPH_SPLIT_RE.split(source):
            start = source.find(raw, cursor)
            if start < 0:  # 理论上不会发生；保守处理
                start = cursor
            cursor = start + len(raw)
            body = raw.strip()
            if not body:
                continue
            # 去掉尾部空白后再算偏移，避免 source_end 落在空行上
            source_end = start + len(raw.rstrip())
            blocks.append(
                DocumentBlock(
                    block_id=f"{stem}#{len(blocks):04d}",
                    block_type="paragraph",
                    text=body,
                    source_start=start,
                    source_end=source_end,
                    section_path=[stem],
                    metadata={"n_lines": body.count("\n") + 1, "n_sentences": count_sentences(body)},
                )
            )
        return blocks


def count_sentences(text: str) -> int:
    """粗略句子数：数句子结束标记，没有标记时算 1 句。"""
    endings = SENTENCE_END_RE.findall(text)
    if not endings:
        return 1 if text.strip() else 0
    # 连续的 「！？」 只算一次结束
    collapsed = re.sub(r"([。！？；!?])\1+", r"\1", text)
    return max(1, len(SENTENCE_END_RE.findall(collapsed)))


__all__ = ["TxtParser", "count_sentences"]
