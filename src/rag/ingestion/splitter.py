"""D06 Recursive Splitter：按分隔符优先级递归切分，最后用 token 窗口兜底。

规格来源：方案 §2.3（切分参数）、§2.4（算法与分隔符优先级）、§17.2 D06。

三条硬不变式（下游 D07/D08 依赖）：

1. ``"".join(pieces) == text`` —— 逐字符可还原。切分**只做切、不做改写**，
   否则 ``source_start/source_end`` 会错位，也没法把 chunk 映射回原文。
2. ``token_count(piece) <= max_tokens`` —— 超限率必须为 0（D06 验收）。
3. 片段按原顺序、非空。

设计取舍：

- 合并（``merge_small_parts``）只在**最后做一次**，不在递归每层做。
  方案伪代码是每层都 merge，但那样内层已经按 target 拼好的片段到了外层
  还会被再拼一次，既费 token 计数又让「到底按哪一级目标切」变得不可预测。
- 兜底不用字符数，用 token 窗口（``split_by_token_window``），且窗口切出来的
  每一段都是原文前缀（见 ``TokenCounter.truncate``）。
- 这一层不认识 Markdown 语义：标题/表格/代码块的**结构**由 Parser 的
  ``DocumentBlock.block_type`` 表达（D02–D04），本模块只保证不破坏
  分隔符与行边界。代码块/表格之所以不会被拦腰切断，是因为 ``"\\n"``
  在分隔符优先级里排在句号之前，且切分时换行符保留在前一段末尾。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .tokenizer import TokenCounter, default_counter

#: ``check`` 子命令默认读的索引文件
STORAGE_INDEX_PATH = Path(__file__).resolve().parents[3] / "storage" / "index.json"

#: 分隔符优先级，直接照抄方案 §2.4。越靠前越"粗"，越靠后越"细"。
#: 末尾的 ``""`` 是哨兵：表示"没有分隔符可用了，改用 token 窗口兜底"。
SEPARATORS: tuple[str, ...] = (
    "\n\n",  # 段落
    "\n",  # 行（代码块、表格、列表项靠它保住结构）
    "。",
    "！",
    "？",
    ". ",  # 英文句子；小数/缩写的误切由后续 merge 吸收
    "! ",
    "? ",
    "；",
    "; ",
    "，",
    ", ",
    " ",  # 词间空格
    "",  # token 窗口兜底
)

DEFAULT_SEPARATORS = SEPARATORS


def recursive_split(
    text: str,
    *,
    counter: TokenCounter,
    max_tokens: int,
    target_tokens: int | None = None,
    separators: Sequence[str] = SEPARATORS,
) -> list[str]:
    """把 ``text`` 切成每段 ≤ ``max_tokens`` 的片段。

    :param counter: 提供 ``count`` / ``truncate`` 的 token 计数器（D05）
    :param max_tokens: 单段硬上限；超过就必须切
    :param target_tokens: 合并时的目标长度，默认等于 ``max_tokens``
    :param separators: 分隔符优先级，最后一项应为 ``""``（token 窗口兜底）
    :returns: 按原顺序排列的片段，拼接后与原文逐字符一致
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens 必须为正数，收到 {max_tokens}")
    if not text:
        return []

    target = max_tokens if target_tokens is None else target_tokens
    if target <= 0:
        raise ValueError(f"target_tokens 必须为正数，收到 {target}")

    pieces = _split_recursive(text, counter, max_tokens, tuple(separators))
    return merge_small_parts(pieces, counter, max_tokens=max_tokens, target_tokens=target)


def _split_recursive(
    text: str,
    counter: TokenCounter,
    max_tokens: int,
    separators: tuple[str, ...],
) -> list[str]:
    """递归主体：只保证「每段 ≤ max」，不做合并。"""
    if not text:
        return []
    if counter.count(text) <= max_tokens:
        return [text]
    if not separators:
        return split_by_token_window(text, counter, max_tokens)

    separator, rest = separators[0], separators[1:]
    if separator == "":
        # 哨兵：已经没有更细的分隔符了，只能按 token 窗口硬切
        return split_by_token_window(text, counter, max_tokens)

    parts = split_preserving_separator(text, separator)
    if len(parts) <= 1:
        # 这个分隔符在文本里不存在，降一级继续
        return _split_recursive(text, counter, max_tokens, rest)

    pieces: list[str] = []
    for part in parts:
        if not part:
            continue
        if counter.count(part) <= max_tokens:
            pieces.append(part)
        else:
            pieces.extend(_split_recursive(part, counter, max_tokens, rest))
    return pieces


def split_preserving_separator(text: str, separator: str) -> list[str]:
    """按 ``separator`` 切分，并把分隔符**保留在前一段末尾**。

    ``"a.\\n\\nb.\\n\\nc"`` 按 ``"\\n\\n"`` 切成 ``["a.\\n\\n", "b.\\n\\n", "c"]``：
    句号、换行符、标题符号都不会丢，拼接回去还是原文。
    """
    if not separator or separator not in text:
        return [text]
    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]]
    tail = parts[-1]
    if tail:
        pieces.append(tail)
    return pieces


def split_by_token_window(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """最后兜底：按 token 窗口硬切，每段都是原文前缀，拼接即还原。

    这是唯一会让句子被切断的地方（无分隔符、超长单词、超长代码行），
    所以只在分隔符全部用尽后才走到这里。
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens 必须为正数，收到 {max_tokens}")

    pieces: list[str] = []
    rest = text
    while rest:
        piece = counter.truncate(rest, max_tokens)
        if not piece:
            # tokenizer 认为这段连一个 token 都放不下（正常不会发生）；
            # 退化成按字符切，保证循环一定能前进
            piece = rest[:1]
        # 按 offset 回切出的前缀，重新 tokenize 时边界可能多出 1 个 token，
        # 这里做一次收缩，确保硬上限不被突破
        while piece and counter.count(piece) > max_tokens:
            piece = piece[:-1]
        if not piece:
            piece = rest[:1]
        pieces.append(piece)
        rest = rest[len(piece) :]
    return pieces


def merge_small_parts(
    parts: Iterable[str],
    counter: TokenCounter,
    *,
    max_tokens: int,
    target_tokens: int,
) -> list[str]:
    """把相邻短片段拼接，直到达到 ``target_tokens``（且绝不超过 ``max_tokens``）。

    只做字符串拼接、不插入任何字符——片段本来就是原文的连续切片，
    拼起来仍然能还原原文。
    """
    merged: list[str] = []
    buffer = ""
    buffer_tokens = 0

    for part in parts:
        if not part:
            continue
        part_tokens = counter.count(part)

        if not buffer:
            buffer, buffer_tokens = part, part_tokens
            continue

        if buffer_tokens >= target_tokens:
            merged.append(buffer)
            buffer, buffer_tokens = part, part_tokens
            continue

        combined = buffer + part
        combined_tokens = counter.count(combined)
        if combined_tokens <= max_tokens:
            buffer, buffer_tokens = combined, combined_tokens
        else:
            merged.append(buffer)
            buffer, buffer_tokens = part, part_tokens

    if buffer:
        merged.append(buffer)
    return merged


__all__ = [
    "DEFAULT_SEPARATORS",
    "SEPARATORS",
    "cli_main",
    "merge_small_parts",
    "recursive_split",
    "split_by_token_window",
    "split_preserving_separator",
]


def cli_main(argv: Sequence[str] | None = None) -> int:
    """D06 的可执行验收：``split`` 看切分结果，``check`` 算超限率。

    ``python -m src.rag.ingestion.splitter split "文本" --max-tokens 200``
    ``python -m src.rag.ingestion.splitter check --n 2000``（期望：超限率 0.00%）
    """
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="python -m src.rag.ingestion.splitter")
    sub = parser.add_subparsers(dest="command", required=True)

    split_cmd = sub.add_parser("split", help="切分一段文本并打印各段（用 | 分隔）")
    split_cmd.add_argument("text", help="要切分的文本；用 - 从 stdin 读")
    split_cmd.add_argument("--max-tokens", type=int, default=500)
    split_cmd.add_argument("--target-tokens", type=int, default=None)

    check_cmd = sub.add_parser("check", help="在真实索引上统计 token 超限率")
    check_cmd.add_argument("--index", default=None, help="index.json 路径，默认 storage/index.json")
    check_cmd.add_argument("--n", type=int, default=500, help="抽样条数，0 表示全量")
    check_cmd.add_argument("--max-tokens", type=int, default=500)
    check_cmd.add_argument("--target-tokens", type=int, default=400)

    args = parser.parse_args(argv)
    counter = default_counter()

    if args.command == "split":
        text = sys.stdin.read() if args.text == "-" else args.text
        pieces = recursive_split(
            text,
            counter=counter,
            max_tokens=args.max_tokens,
            target_tokens=args.target_tokens,
        )
        for index, piece in enumerate(pieces):
            print(f"[{index}] {counter.count(piece):>4d} tok  {piece}")
        return 0

    index_path = Path(args.index) if args.index else STORAGE_INDEX_PATH
    if not index_path.exists():
        print(f"找不到索引：{index_path}")
        return 2

    import json
    import random

    records = json.loads(index_path.read_text(encoding="utf-8"))["records"]
    texts = [record["text"] for record in records]
    if args.n and args.n < len(texts):
        texts = random.Random(20261102).sample(texts, args.n)

    total_pieces = over_limit = rejoin_failures = fragments = 0
    for text in texts:
        pieces = recursive_split(
            text,
            counter=counter,
            max_tokens=args.max_tokens,
            target_tokens=args.target_tokens,
        )
        if "".join(pieces) != text:
            rejoin_failures += 1
        for piece in pieces:
            tokens = counter.count(piece)
            total_pieces += 1
            if tokens > args.max_tokens:
                over_limit += 1
            if tokens < 80:
                fragments += 1

    rate = over_limit / total_pieces if total_pieces else 0.0
    frag_rate = fragments / total_pieces if total_pieces else 0.0
    print(f"样本 chunk {len(texts)} 条 → 片段 {total_pieces} 段")
    print(f"超限率 {rate:.2%}（{over_limit} 段 > {args.max_tokens} token）")
    print(f"还原失败 {rejoin_failures} 条")
    print(f"碎块率 {frag_rate:.2%}（{fragments} 段 < 80 token）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(cli_main())
