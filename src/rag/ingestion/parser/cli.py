"""语料级验收命令：``python -m src.rag.ingestion.parser {check,dump}``。

D02/D03/D04 的验收指标都是**比率**（section_path 非空率、空 block 数、页码
覆盖率），只能在整份语料上算，所以给一条能贴进验收单的命令：

    python -m src.rag.ingestion.parser check data/md      # 期望：非空率 100.00%、空 block 0
    python -m src.rag.ingestion.parser dump data/rag_basics.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..errors import IngestionError, UnsupportedFormatError
from . import available_suffixes, get_parser, register_builtin_parsers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.rag.ingestion.parser")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="整目录解析并打印质量比率")
    check.add_argument("directory")
    check.add_argument("--limit", type=int, default=0, help="最多解析多少个文件（0=全部）")

    dump = sub.add_parser("dump", help="把单个文件的 block 打成 JSONL")
    dump.add_argument("file")

    args = parser.parse_args(argv)
    register_builtin_parsers(replace=True)

    if args.command == "dump":
        return _dump(args.file)
    return _check(args.directory, args.limit)


def _dump(file_name: str) -> int:
    path = Path(file_name)
    if not path.exists():
        print(f"文件不存在：{path}")
        return 2
    try:
        parser = get_parser(str(path))
    except UnsupportedFormatError as exc:
        print(f"不支持的文件类型：{exc}")
        return 2
    try:
        blocks = parser.parse(str(path))
    except IngestionError as exc:
        print(f"解析失败：{exc}")
        return 2
    for block in blocks:
        print(json.dumps(block.to_dict(), ensure_ascii=False))
    return 0


def _check(directory: str, limit: int) -> int:
    root = Path(directory)
    if not root.exists():
        print(f"目录不存在：{root}")
        return 2

    suffixes = set(available_suffixes())
    files = [path for path in sorted(root.rglob("*")) if path.suffix.lower() in suffixes]
    if limit:
        files = files[:limit]
    if not files:
        print(f"没有可解析的文件（支持后缀：{', '.join(sorted(suffixes))}）")
        return 2

    type_counter: Counter[str] = Counter()
    total_blocks = 0
    empty_blocks = 0
    empty_path = 0
    missing_page = 0
    pdf_blocks = 0
    failures: list[tuple[str, str]] = []

    for path in files:
        parser = get_parser(str(path))
        try:
            blocks = parser.parse(str(path))
        except (IngestionError, ValueError) as exc:
            failures.append((str(path), str(exc)))
            continue
        total_blocks += len(blocks)
        for block in blocks:
            type_counter[block.block_type] += 1
            if not block.text.strip():
                empty_blocks += 1
            if not block.section_path:
                empty_path += 1
            if path.suffix.lower() == ".pdf":
                pdf_blocks += 1
                if block.page_number is None:
                    missing_page += 1

    if not total_blocks:
        print(f"解析了 {len(files)} 个文件，但没有任何 block（全部失败？）")
        return 2

    print(f"文件 {len(files)} 个 → block {total_blocks:,} 个")
    non_empty_rate = 1 - empty_path / total_blocks
    print(f"section_path 非空率 {non_empty_rate:.2%}（空 {empty_path}）")
    print(f"空 block {empty_blocks}")
    print("block 类型：" + "  ".join(f"{name} {count:,}" for name, count in type_counter.most_common()))
    if pdf_blocks:
        coverage = 1 - missing_page / pdf_blocks
        print(f"PDF 页码覆盖率 {coverage:.2%}（缺页 {missing_page}）")
    if failures:
        print(f"解析失败 {len(failures)} 个：")
        for name, reason in failures[:5]:
            print(f"  - {name}: {reason}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
