"""D02 Markdown Parser：标题层级 / 列表 / 表格 / 代码块（方案 §2.2）。

设计要点：

- **代码块优先识别**：先吃掉 fenced code，再谈标题。否则 `# 注释` 会被
  当成 Markdown 标题——这是 D02 点名的第一个陷阱。
- **表格不依赖前置空行**：GFM 表格紧跟标题是合法的，判定只看「本行有 `|`
  且下一行是分隔行」，所以紧贴标题的表格仍然是 `table` 而不是 `paragraph`。
- **标题不脱离正文**：解析结果只负责把标题和段落分开成不同 Block，
  归属关系通过 `section_path` 表达（标题块自己的 `section_path` 末尾一项
  就是它自己），D08 据此划 Parent 边界。
- **`section_path` 永不为空**：文档开头没有标题时退化成文档标题
  （首个一级标题，没有就用文件名）。D02 验收要求非空率 100%。

偏移（`source_start/source_end`）是**块级别**的，指向规范化换行
（`\r\n` → `\n`）之后的源文本；`text` 可能去掉 Markdown 标记（如标题的
`#`、引用的 `>`），所以不保证 `source[start:end] == text`。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..block import DocumentBlock
from .base import BaseParser, read_text_file

#: ATX 标题：`## 标题` / `## 标题 ##`
ATX_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*#*[ \t]*$")
#: fenced code 起始围栏：``` 或 ~~~（允许信息串，如 python）
FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^`]*)$")
#: 分割线
HR_RE = re.compile(r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
#: 列表项
LIST_ITEM_RE = re.compile(r"^ {0,3}([-*+]|\d{1,9}[.)])[ \t]+\S")
#: 引用
QUOTE_RE = re.compile(r"^ {0,3}>[ \t]?")
#: 独占一行的图片（可带 title）
IMAGE_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:[ \t]+\"[^\"]*\")?\)$")
#: 表格分隔行：|---|---| / |:--:|--:|
TABLE_DELIM_RE = re.compile(r"^ {0,3}\|?[ \t]*:?-{1,}:?[ \t]*(?:\|[ \t]*:?-{1,}:?[ \t]*)+\|?[ \t]*$")
#: setext 标题下划线
SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")


class MarkdownParser(BaseParser):
    parser_version = "md_v1"
    supported_suffixes = (".md", ".markdown")

    def parse(self, path: str, text: str | None = None) -> list[DocumentBlock]:
        source = text if text is not None else read_text_file(path)
        source = source.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        if not source.strip():
            return []

        lines = source.split("\n")
        line_offsets = _line_offsets(lines)
        stem = Path(path).stem or "document"
        # 标题回退用**文件名**而不是「首个一级标题」：文档开头若有一截
        # 无标题正文，把它算到后面才出现的章节名下是错误归属。
        fallback_title = stem

        blocks: list[DocumentBlock] = []
        sections: list[tuple[int, str]] = []  # (level, title)

        def emit(block_type, text_, start_line, end_line, **kwargs):
            """end_line 为开区间；source 偏移取 [start_line, end_line) 的行范围。"""
            block = DocumentBlock(
                block_id=f"{stem}#{len(blocks):04d}",
                block_type=block_type,
                text=text_,
                source_start=line_offsets[start_line],
                source_end=line_offsets[end_line - 1] + len(lines[end_line - 1]),
                section_path=kwargs.pop("section_path", None) or _current_path(sections) or [fallback_title],
                **kwargs,
            )
            blocks.append(block)

        def push_heading(level: int, title: str, raw: str, start_line: int, end_line: int):
            while sections and sections[-1][0] >= level:
                sections.pop()
            sections.append((level, title))
            emit(
                "heading",
                title,
                start_line,
                end_line,
                heading_level=level,
                section_path=[t for _, t in sections],
                metadata={"raw": raw.strip(), "level": level},
            )

        index = 0
        total = len(lines)
        while index < total:
            line = lines[index]
            stripped = line.strip()

            if not stripped:
                index += 1
                continue

            # ① fenced code —— 必须排在标题之前，否则 `# 注释` 会被当标题
            fence = FENCE_OPEN_RE.match(line)
            if fence:
                marker = fence.group(1)[0]
                language = fence.group(2).strip().split()[0] if fence.group(2).strip() else ""
                end = index + 1
                while end < total:
                    closing = lines[end].strip()
                    if closing.startswith(marker * 3) and set(closing) <= {marker}:
                        end += 1
                        break
                    end += 1
                emit(
                    "code",
                    "\n".join(lines[index:end]),
                    index,
                    end,
                    metadata={"language": language, "fenced": True, "unclosed": end >= total and not lines[end - 1].strip().startswith(marker * 3)},
                )
                index = end
                continue

            # ② ATX 标题
            atx = ATX_RE.match(line)
            if atx:
                title = (atx.group(2) or "").strip()
                if title:
                    push_heading(len(atx.group(1)), title, stripped, index, index + 1)
                    index += 1
                    continue

            # ③ 表格（不要求前面有空行）
            if "|" in line and index + 1 < total and TABLE_DELIM_RE.match(lines[index + 1]):
                end = index + 2
                while end < total and "|" in lines[end] and lines[end].strip():
                    end += 1
                rows = [row for row in lines[index:end] if row.strip()]
                emit(
                    "table",
                    "\n".join(lines[index:end]),
                    index,
                    end,
                    metadata={
                        "has_header": True,
                        "n_rows": len(rows) - 1,  # 去掉分隔行
                        "n_cols": len(rows[0].strip().strip("|").split("|")),
                    },
                )
                index = end
                continue

            # ④ 分割线（纯视觉分隔，不产生 block）
            if HR_RE.match(line):
                index += 1
                continue

            # ⑤ 引用
            if QUOTE_RE.match(line):
                end = index
                while end < total and QUOTE_RE.match(lines[end]):
                    end += 1
                body = "\n".join(QUOTE_RE.sub("", line_) for line_ in lines[index:end])
                emit("quote", body.strip(), index, end, metadata={"n_lines": end - index})
                index = end
                continue

            # ⑥ 列表
            if LIST_ITEM_RE.match(line):
                end = index
                while end < total and lines[end].strip():
                    if end > index and not (LIST_ITEM_RE.match(lines[end]) or lines[end].startswith((" ", "\t"))):
                        break
                    end += 1
                items = [lines[i_] for i_ in range(index, end) if LIST_ITEM_RE.match(lines[i_])]
                ordered = bool(re.match(r"^ {0,3}\d", items[0])) if items else False
                emit(
                    "list",
                    "\n".join(lines[index:end]).rstrip(),
                    index,
                    end,
                    metadata={"ordered": ordered, "n_items": len(items)},
                )
                index = end
                continue

            # ⑦ 独占一行的图片 → caption
            image = IMAGE_RE.match(stripped)
            if image:
                emit(
                    "caption",
                    stripped,
                    index,
                    index + 1,
                    metadata={"alt": image.group("alt"), "src": image.group("src")},
                )
                index += 1
                continue

            # ⑧ setext 标题（文本行 + `===`/`---`）
            if index + 1 < total and "|" not in line:
                setext = SETEXT_RE.match(lines[index + 1])
                if setext:
                    level = 1 if setext.group(1).startswith("=") else 2
                    push_heading(level, stripped, stripped, index, index + 2)
                    index += 2
                    continue

            # ⑨ 段落：连续非空、且不是任何块起始的行
            end = index + 1
            while end < total and lines[end].strip() and not _starts_block(lines, end, total):
                end += 1
            emit(
                "paragraph",
                "\n".join(lines[index:end]).strip(),
                index,
                end,
                metadata={"n_lines": end - index},
            )
            index = end

        return blocks


def _line_offsets(lines: list[str]) -> list[int]:
    offsets, position = [], 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1  # +1 为换行符
    return offsets


def _current_path(sections: list[tuple[int, str]]) -> list[str]:
    return [title for _, title in sections]


def _starts_block(lines: list[str], index: int, total: int) -> bool:
    """判断 index 行是否开启一个新块（用于段落边界与列表终止）。"""
    line = lines[index]
    if FENCE_OPEN_RE.match(line) or ATX_RE.match(line) or HR_RE.match(line):
        return True
    if QUOTE_RE.match(line) or LIST_ITEM_RE.match(line):
        return True
    if "|" in line and index + 1 < total and TABLE_DELIM_RE.match(lines[index + 1]):
        return True
    if IMAGE_RE.match(line.strip()):
        return True
    # 本行是 setext 下划线：说明上一行应该作为标题单独成块
    if SETEXT_RE.match(line) and index > 0 and lines[index - 1].strip():
        return True
    return False


__all__ = ["MarkdownParser"]
