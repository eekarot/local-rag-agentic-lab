"""D10 稳定 ID 与内容哈希（方案 §3.4）。

设计取舍：

- **document_id 认来源、不认内容**。ID 用 `source_uri` 派生，内容变了只变
  `document_version`——否则「同一文档改了错别字」会被当成另一篇文档，
  旧版本的 chunk 也没法按 `document_id` 清理（§3.4 最后一条要求）。
- **parent_id / chunk_id 认内容 + 版本**。任一输入变化都必须变 ID，这样
  「chunker 规则改了但 ID 没变」这种事故在结构上不可能发生。
- 分隔符用 `\\x1f`（Unit Separator）而不是 `|` 或 `:`：ID 各段里可能出现
  任何可见字符，用可见字符拼接会撞车（`a|b` + `c` vs `a` + `b|c`）。
"""

from __future__ import annotations

import hashlib
import re

#: ID 整体格式：类型前缀 + 32 位小写十六进制（sha256 前 128 bit）
ID_PATTERN = re.compile(r"^(doc|par|chk)_[0-9a-f]{32}$")

#: 拼接 ID 各段的分隔符
_SEPARATOR = "\x1f"

#: 截取 sha256 十六进制的前多少位（128 bit 足够，ID 更短更好读）
_ID_HEX_LEN = 32


def _digest(*parts: str) -> str:
    payload = _SEPARATOR.join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_ID_HEX_LEN]


def normalize_text(text: str) -> str:
    """哈希前先归一：BOM、换行符。

    不归一的话，同一份文档在 Windows（CRLF）和仓库里（LF）会得到两个
    `document_version`，增量导入会误判成「内容变了」而全量重建。
    """
    return text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def content_hash(text: str) -> str:
    """内容哈希（§3.3 的 `content_hash` 字段）。不归一——按实际入库文本算。"""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_version(source_text: str) -> str:
    """文档版本（§3.1）：对归一化后的原文求哈希。"""
    return "sha256:" + hashlib.sha256(normalize_text(source_text).encode("utf-8")).hexdigest()


def make_document_id(source_uri: str) -> str:
    """文档 ID 由来源路径派生：同一路径永远同一个 ID，与内容无关。"""
    return "doc_" + _digest("document", normalize_text(str(source_uri)))


def make_parent_id(
    document_id: str, document_version: str, parent_index: int, chunker_version: str
) -> str:
    """§3.4：parent_id = hash(document_id + document_version + parent_index + chunker_version)"""
    return "par_" + _digest("parent", document_id, document_version, str(parent_index), chunker_version)


def make_chunk_id(parent_id: str, child_index: int, content_digest: str, chunker_version: str) -> str:
    """§3.4：chunk_id = hash(parent_id + child_index + content_hash + chunker_version)"""
    return "chk_" + _digest("child", parent_id, str(child_index), content_digest, chunker_version)


__all__ = [
    "ID_PATTERN",
    "content_hash",
    "document_version",
    "make_chunk_id",
    "make_document_id",
    "make_parent_id",
    "normalize_text",
]
