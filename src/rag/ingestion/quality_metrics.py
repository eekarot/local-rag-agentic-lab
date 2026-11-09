"""切片质量指标（G3 / 方案 P1.3 + P1.4）。

三个纯函数，输入全是整数坐标，**不碰 tokenizer、不碰向量库、不碰金标**：

| 函数 | 回答什么问题 | 数据来源 |
|---|---|---|
| `size_compliance` | 有多少比例的块落在 `[min_tokens, max_tokens]` 内？ | 只要每块的 token 数 |
| `block_integrity` | 有多少比例的结构块（段落/表格/代码）没被切断？ | parser 给的 block 偏移 + chunk 偏移 |
| `find_coverage_gaps` | 子块之间/末尾有没有**丢字**？ | 只要区间 |

为什么值得存在（对照上游 `ekimetrics/adaptive-chunking`，LREC 2026 的 SC / BI）：

* **分位数看不出越界**：`tokens_p50=400 / p90=430` 与「30% 的块超过上限」可以同时成立。
* **结构完整性是唯一能在「没有金标」时判断切片好坏的东西**：用户索引自己的文档时
  没有题库，`Recall@k` 无从谈起；但 parser 的块边界是现成的，切断没切断一眼可判。
* **覆盖率不是指标，是不变量**：缺一个字符就要当 bug 处理，不给分数。

上游用 `tolerance_chars=5`（块边界附近差几个字符算没切断），这里保留同口径 ——
差 1 个字符被判成「结构被破坏」会淹没真正的信号。
"""

from __future__ import annotations

from typing import Iterable, Sequence

#: 判定「块被切断」时允许的边界容差（字符）。与上游默认一致。
DEFAULT_TOLERANCE_CHARS = 5

Span = tuple[int | None, int | None]


def size_compliance(
    token_counts: Sequence[int],
    *,
    min_tokens: int,
    max_tokens: int,
) -> float | None:
    """落在 `[min_tokens, max_tokens]`（含端点）的块占比。

    空输入返回 ``None`` —— 分母为 0 时不给数（与 N36 同一条纪律：
    「没有样本」和「全都不合格」必须能区分开）。
    """
    if min_tokens > max_tokens:
        raise ValueError(f"min_tokens({min_tokens}) 不能大于 max_tokens({max_tokens})")
    if not token_counts:
        return None
    compliant = sum(1 for count in token_counts if min_tokens <= count <= max_tokens)
    return compliant / len(token_counts)


def block_integrity(
    chunk_spans: Sequence[Span],
    block_spans: Sequence[Span],
    *,
    tolerance_chars: int = DEFAULT_TOLERANCE_CHARS,
) -> float | None:
    """没有被「切分点」切开的**结构块**占比。

    * ``chunk_spans``：每块在文档里的 `(start, end)`，允许重叠、允许 ``(None, None)``（跳过）。
    * ``block_spans``：parser 给出的块边界（段落/表格/代码…），这就是这里的「金标」。

    判定：把除第一块之外的每个 chunk 起点都当成一个切分点；若某个切分点落在某个
    结构块的**内部**（距两端都超过 `tolerance_chars`），该块记为被切断。
    同一块内有多个切分点只算一次。

    返回 ``None`` 的两种情形：没有 chunk、没有 block（都算「没法判定」，不报 0）。
    """
    starts = sorted({start for start, _ in chunk_spans if start is not None})
    if not chunk_spans:
        return None
    if not block_spans:
        return None
    if len(starts) <= 1:
        # 只有一块（或都从同一点开始）：没有任何内部切分点
        return 1.0

    predicted = starts[1:]
    intact = 0
    blocks = 0
    for left, right in block_spans:
        if left is None or right is None:
            continue
        blocks += 1
        broken = any(left + tolerance_chars < point < right - tolerance_chars for point in predicted)
        if not broken:
            intact += 1
    if not blocks:
        return None
    return intact / blocks


def find_coverage_gaps(spans: Iterable[Span], *, end: int) -> list[tuple[int, int]]:
    """在 `[0, end)` 内找出没被任何区间覆盖的**连续段**。

    这是不变量检查，不是评分：返回非空列表就意味着「丢字了」，调用方应当直接报错。
    重叠区间没问题；乱序也没问题（内部会排序）。
    """
    if end < 0:
        raise ValueError(f"end({end}) 不能为负")
    ordered = sorted((left, right) for left, right in spans if left is not None and right is not None)
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for left, right in ordered:
        if right <= cursor:
            continue
        if left > cursor:
            gaps.append((cursor, min(left, end)))
        cursor = max(cursor, right)
        if cursor >= end:
            break
    if cursor < end:
        gaps.append((cursor, end))
    return [gap for gap in gaps if gap[0] < gap[1]]


__all__ = [
    "DEFAULT_TOLERANCE_CHARS",
    "Span",
    "block_integrity",
    "find_coverage_gaps",
    "size_compliance",
]
