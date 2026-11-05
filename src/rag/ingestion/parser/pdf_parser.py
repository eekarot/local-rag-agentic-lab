"""D04 PDF Parser：pymupdf 版面块解析，保留页码与字号信息（方案 §2.2）。

为什么从 pypdf 换到 pymupdf：pypdf 只给「一串文本」，拿不到版面块、字号、
坐标，于是标题层级、表格边界、页眉页脚都无从判断。pymupdf 的
``page.get_text("dict")`` 给出 block → line → span 三层结构，字号和 bbox
都在里面，标题识别与页脚剔除才有可能。

三条纪律：

1. **页码必须留**：``page_number`` 从 1 开始，每个 block 都有（D04 验收）。
2. **重复页眉页脚要剔除，但必须留痕**：数字归一化后（``第 1 页`` → ``第 # 页``）
   在 ≥60% 的页面上出现的短行判为 margin，删除并记进 ``stats["margins_removed"]``。
   静默删除等于把解析错误变成数据缺失。只对 ≥2 页的文档生效。
3. **标题靠字号相对比识别**：以正文正文字号（按字符数加权的），
   比值 ≥2.0 → h1、≥1.5 → h2、≥1.15 → h3；纯字号绝对阈值在不同
   排版（论文 / 手册 / 幻灯片）之间完全不可移植。
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..block import DocumentBlock
from ..errors import IngestionError
from .base import BaseParser

#: pymupdf 是**可选依赖**（许可证是 AGPL-3.0 / 商业双授权，不是 Apache-2.0）。
#: 所以这里做惰性导入：只解析 md/txt 的用户不该被牵连（见 NOTICE 与
#: tests/test_pdf_optional_dependency.py）。
_PYMUPDF: Any = None


def load_pymupdf() -> Any:
    """按需导入 pymupdf（1.x 叫 `fitz`，2.x 起叫 `pymupdf`），失败时报「装什么 + 什么许可证」。"""
    global _PYMUPDF
    if _PYMUPDF is not None:
        return _PYMUPDF
    try:
        import pymupdf as module
    except ImportError:
        try:
            import fitz as module  # type: ignore
        except ImportError as exc:
            raise IngestionError(
                "解析 PDF 需要 pymupdf（本仓库不默认安装）："
                "python -m pip install -r requirements-pdf.txt"
                + chr(10)
                + "注意：PyMuPDF 的许可证是 AGPL-3.0 / 商业双授权，与本仓库的 Apache-2.0 不同；"
                "商用闭源场景请自行购买授权，或只用纯文本（pypdf）模式。"
            ) from exc
    _PYMUPDF = module
    return module


#: 判定标题的最小字号比值
HEADING_RATIOS = ((2.0, 1), (1.5, 2), (1.15, 3))
#: margin 判定的页数占比阈值
MARGIN_PAGE_RATIO = 0.6
#: margin 判定的最大长度（超过这个长度不可能是页眉页脚）
MARGIN_MAX_CHARS = 80
#: 标题的最大长度（防止把长正文段落误判成标题）
HEADING_MAX_CHARS = 80

DIGITS_RE = re.compile(r"\d+")


class PdfParser(BaseParser):
    parser_version = "pdf_mupdf_v1"
    supported_suffixes = (".pdf",)

    def __init__(self) -> None:
        #: 上一次 parse 的统计信息（页数、margin 剔除清单、正文字号等）
        self.stats: dict[str, Any] = {}

    def parse(
        self,
        path: str,
        text: str | None = None,
        *,
        page_limit: int | None = None,
        drop_repeated_margins: bool = True,
    ) -> list[DocumentBlock]:
        if text is not None:
            raise ValueError(
                "PDF 没有线性文本，必须传文件路径；请用 PdfParser().parse(path) "
                "而不是传入 text（若只要纯文本，用 page.get_text() 自行提取）"
            )

        file_path = Path(path)
        if not file_path.exists():
            raise IngestionError(f"PDF 文件不存在：{file_path}")
        try:
            document = load_pymupdf().open(str(file_path))
        except Exception as exc:  # noqa: BLE001 - pymupdf 抛的异常类型不稳定
            raise IngestionError(f"无法作为 PDF 打开：{file_path}（{exc}）") from exc

        try:
            total_pages = document.page_count
            if page_limit is not None:
                total_pages = min(total_pages, max(0, page_limit))
            pages = [_collect_page(document[i]) for i in range(total_pages)]
        finally:
            document.close()

        body_size = _body_font_size(pages)
        margins = (
            _find_margins(pages, total_pages) if drop_repeated_margins and total_pages >= 2 else set()
        )

        stem = file_path.stem or "document"
        blocks: list[DocumentBlock] = []
        sections: list[tuple[int, str]] = []

        for page_number, page_lines in enumerate(pages, start=1):
            for block_lines in _group_by_block(page_lines):
                kept = [line for line in block_lines if _margin_key(line.text) not in margins]
                if not kept:
                    continue
                # pymupdf 的 block 边界不可全信：标题和正文可能被聚进同一个 block
                # （排版紧、行距一致时很常见），所以 block 内部再按字号切一次 run。
                for run in _split_runs(kept, body_size):
                    body = _join_lines([line.text for line in run]).strip()
                    if not body:
                        continue

                    heading_level = _heading_level(run[0].size, body_size, body, len(run))
                    if heading_level:
                        while sections and sections[-1][0] >= heading_level:
                            sections.pop()
                        sections.append((heading_level, body))
                        section_path = [title for _, title in sections]
                        block_type = "heading"
                    else:
                        section_path = [title for _, title in sections] or [stem]
                        block_type = "paragraph"

                    blocks.append(
                        DocumentBlock(
                            block_id=f"{stem}#p{page_number:03d}-{len(blocks):04d}",
                            block_type=block_type,
                            text=body,
                            heading_level=heading_level,
                            page_number=page_number,
                            section_path=section_path,
                            metadata={
                                "page": page_number,
                                "bbox": [round(v, 1) for v in run[0].bbox],
                                "font_size": round(run[0].size, 1),
                                "n_lines": len(run),
                                "page_blocks": len(block_lines),
                            },
                        )
                    )

        self.stats = {
            "file": str(file_path),
            "pages": total_pages,
            "blocks": len(blocks),
            "chars": sum(len(block.text) for block in blocks),
            "headings": sum(1 for block in blocks if block.block_type == "heading"),
            "body_font_size": round(body_size, 2) if body_size else None,
            "margins_removed": sorted(margins),
            "pages_missing_text": [
                index for index, lines in enumerate(pages, start=1) if not lines
            ],
        }
        return blocks


class _Line:
    __slots__ = ("text", "size", "bbox", "block_number")

    def __init__(self, text: str, size: float, bbox: tuple[float, ...], block_number: int) -> None:
        self.text = text
        self.size = size
        self.bbox = bbox
        self.block_number = block_number


def _collect_page(page) -> list[_Line]:
    """把一页拆成行，行带字号、bbox 与 pymupdf 的 block 编号。"""
    lines: list[_Line] = []
    payload = page.get_text("dict")
    for block in payload.get("blocks", []):
        if block.get("type") != 0:  # 图片块跳过；OCR 交给专门流程
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if span.get("text")]
            if not spans:
                continue
            text = "".join(span["text"] for span in spans)
            if not text.strip():
                continue
            size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
            lines.append(_Line(text, size, tuple(line.get("bbox", ())), int(block.get("number", 0))))
    return lines


def _group_by_block(lines: list[_Line]) -> Iterable[list[_Line]]:
    """按 pymupdf 的 block 编号分组：它已经按版面把行聚成了段落。"""
    current: list[_Line] = []
    current_number = None
    for line in lines:
        if current and line.block_number != current_number:
            yield current
            current = []
        current.append(line)
        current_number = line.block_number
    if current:
        yield current


def _split_runs(lines: list[_Line], body_size: float) -> list[list[_Line]]:
    """把同一个 pymupdf block 内的行按「是否标题 + 字号」切成 run。"""
    runs: list[list[_Line]] = []
    for line in lines:
        is_heading = _looks_like_heading_line(line, body_size)
        if runs:
            previous = runs[-1]
            previous_is_heading = _looks_like_heading_line(previous[0], body_size)
            same_kind = is_heading == previous_is_heading
            same_size = (not is_heading) or abs(line.size - previous[0].size) < 0.5
            if same_kind and same_size:
                previous.append(line)
                continue
        runs.append([line])
    return runs


def _looks_like_heading_line(line: _Line, body_size: float) -> bool:
    if not body_size or line.size <= 0:
        return False
    text = line.text.strip()
    return (
        line.size / body_size >= HEADING_RATIOS[-1][0]
        and len(text) <= HEADING_MAX_CHARS
        and len(text.split()) <= 12
    )


def _body_font_size(pages: list[list[_Line]]) -> float:
    """正文正文字号：按字符数加权的众数（取最常见字号，避免被标题拉高）。"""
    weights: Counter[float] = Counter()
    for lines in pages:
        for line in lines:
            weights[round(line.size, 1)] += len(line.text)
    if not weights:
        return 0.0
    return max(weights.items(), key=lambda item: (item[1], item[0]))[0]


def _heading_level(size: float, body_size: float, text: str, n_lines: int) -> int | None:
    if not body_size or size <= 0:
        return None
    if len(text) > HEADING_MAX_CHARS or n_lines > 2:
        return None
    ratio = size / body_size
    for threshold, level in HEADING_RATIOS:
        if ratio >= threshold:
            return level
    return None


def _margin_key(text: str) -> str:
    """页眉页脚归一化：数字换成 #，空格折叠。"""
    return " ".join(DIGITS_RE.sub("#", text).split())


def _find_margins(pages: list[list[_Line]], total_pages: int) -> set[str]:
    """在足够多的页面上重复出现的短行 → 页眉页脚。"""
    if total_pages < 2:
        return set()
    counter: Counter[str] = Counter()
    for lines in pages:
        keys = {_margin_key(line.text) for line in lines if len(line.text.strip()) <= MARGIN_MAX_CHARS}
        counter.update(keys)
    threshold = max(2, int(total_pages * MARGIN_PAGE_RATIO + 0.999))
    return {key for key, count in counter.items() if key and count >= threshold}


def _join_lines(parts: list[str]) -> str:
    """行内拼接：中日韩字符之间不加空格，拉丁字母之间补一个空格。"""
    if not parts:
        return ""
    result = parts[0]
    for part in parts[1:]:
        previous, following = result[-1:], part[:1]
        if _is_cjk(previous) or _is_cjk(following):
            result += part
        elif previous.isspace() or following.isspace():
            result += part
        else:
            result += " " + part
    return result


def _is_cjk(char: str) -> bool:
    return bool(char) and "\u3000" <= char <= "\u9fff" or bool(char) and "\uff00" <= char <= "\uffef"


def cli_main(argv: list[str] | None = None) -> int:
    """``report``：生成 D04 验收要的「30 页人工表」（CSV）。"""
    import argparse
    import csv
    import json

    parser = argparse.ArgumentParser(prog="python -m src.rag.ingestion.parser.pdf_parser")
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="逐页导出统计表，供人工抽查")
    report.add_argument("pdf")
    report.add_argument("--pages", default="1-30", help="页码范围，如 1-30")
    report.add_argument("--out", default=None, help="CSV 输出路径，默认打印到 stdout")
    report.add_argument("--stats", action="store_true", help="额外打印解析统计 JSON")

    dump = sub.add_parser("dump", help="把 block 以 JSONL 打到 stdout")
    dump.add_argument("pdf")
    dump.add_argument("--pages", default=None)

    args = parser.parse_args(argv)

    page_limit = None
    if getattr(args, "pages", None):
        try:
            first, _, last = args.pages.partition("-")
            start, end = int(first), int(last or first)
        except ValueError:
            print(f"页码范围格式应为 1-30，收到 {args.pages!r}")
            return 2
        if start < 1 or end < start:
            print(f"页码范围不合法：{args.pages}")
            return 2
        page_limit = end

    pdf_parser = PdfParser()
    try:
        blocks = pdf_parser.parse(args.pdf, page_limit=page_limit)
    except IngestionError as exc:
        print(str(exc))
        return 2

    if args.command == "dump":
        for block in blocks:
            print(json.dumps(block.to_dict(), ensure_ascii=False))
        return 0

    rows = []
    for page in range(1, (page_limit or pdf_parser.stats["pages"]) + 1):
        page_blocks = [b for b in blocks if b.page_number == page]
        rows.append(
            {
                "page": page,
                "chars": sum(len(b.text) for b in page_blocks),
                "blocks": len(page_blocks),
                "headings": sum(1 for b in page_blocks if b.block_type == "heading"),
                "font_size": page_blocks[0].metadata.get("font_size") if page_blocks else "",
                "sample": (page_blocks[0].text[:40].replace("\n", " ") if page_blocks else ""),
            }
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"已写入 {args.out}（{len(rows)} 页）")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.stats:
        print(json.dumps(pdf_parser.stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(cli_main())
