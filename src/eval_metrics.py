"""检索层（L1）指标：纯函数，不碰模型、不碰索引，全部可手算复核。

设计约定（见 docs/eval-and-logging-design.md 第 4 章）：

- **Recall@k 必须报两种口径**：`hit`（命中任一 gold）与 `full`（覆盖全部 gold）。
  chunk 重叠会让同一句原文落在 2–3 个 chunk 里，两种口径天然不同，不能混着比。
- **不提供 Precision@k**：gold 是「够用就行」的集合而不是穷举标注，Precision 会被系统性低估。
- **不确定度必须给**：60 题的 80% 和 85% 可能只是两三个样本的抖动，任何结论都要带区间。
- **配对比较优于各自区间**：同一批题下比逐题差值（paired bootstrap / McNemar），
  否则两次独立区间重叠并不等于「没有差异」。

区间一律用 bootstrap 百分位法；配对差值用同一套重采样索引，保证 a、b 的抽样完全对齐。
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

DEFAULT_BOOT = 1000
DEFAULT_SEED = 0


def _as_ranking(ranked: Iterable[str]) -> list[str]:
    return list(ranked or [])


def first_relevant_rank(ranked: Sequence[str], gold: Iterable[str]) -> int | None:
    """第一条 gold 的名次（从 1 开始）；没有则 None。"""
    gold = set(gold or ())
    if not gold:
        return None
    for rank, doc_id in enumerate(_as_ranking(ranked), start=1):
        if doc_id in gold:
            return rank
    return None


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> tuple[bool, bool]:
    """返回 (hit, full)：hit = top-k 命中任一 gold；full = top-k 覆盖全部 gold。"""
    gold = set(gold or ())
    if not gold or k <= 0:
        return (False, False)
    top = set(_as_ranking(ranked)[:k])
    found = len(gold & top)
    return (found > 0, found == len(gold))


def recall_fraction(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """标准 Recall@k = |G ∩ R_k| / |G|（doc 4.1 的主指标）。

    与 `recall_at_k()` 的 (hit, full) 互补：hit/full 是二元的两个极端（宽松 / 严格），
    这里给连续值 —— gold 有 3 条时，捞到 1 条记 1/3 而不是 0，统计上更稳。
    """
    gold = set(gold or ())
    if not gold or k <= 0:
        return 0.0
    return len(gold & set(_as_ranking(ranked)[:k])) / len(gold)


def mrr_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """reciprocal rank，只看前 k 名。"""
    rank = first_relevant_rank(_as_ranking(ranked)[:k], gold)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """二元增益的 nDCG@k。

    现在 gold 全是 grade=2，所以等价于「带位置折扣的召回」；将来引入 grade=1 时，
    只需把 gain 换成 `grades.get(doc, 0)`，DCG/IDCG 结构不变。
    """
    gold = set(gold or ())
    if not gold or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(_as_ranking(ranked)[:k], start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(sorted_values[int(pos)])
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low))


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = DEFAULT_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict:
    """均值的 bootstrap 百分位区间（对样本重采样，独立同分布假设）。

    返回 `{n, mean, lo, hi, sd, half_width}`；`half_width` 就是回归门禁用的容差。
    """
    values = [float(v) for v in values]
    if not values:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0, "sd": 0.0, "half_width": 0.0}
    mean = _mean(values)
    if len(values) == 1 or len(set(values)) == 1:
        return {"n": len(values), "mean": mean, "lo": mean, "hi": mean, "sd": 0.0, "half_width": 0.0}

    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    lo = _percentile(means, alpha / 2)
    hi = _percentile(means, 1 - alpha / 2)
    return {
        "n": size,
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "sd": (sum((v - mean) ** 2 for v in values) / max(size - 1, 1)) ** 0.5,
        "half_width": (hi - lo) / 2,
    }


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = DEFAULT_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict:
    """配对的 bootstrap 差值：`diff = mean(a) - mean(b)`。

    同一道题的两个模式必须按同一顺序传入（靠 qid 对齐，调用方负责）。
    p 值来自「重采样后的差值分布跨过 0 的比例」，双尾。
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if len(a) != len(b):
        raise ValueError(f"配对样本长度必须相同：{len(a)} vs {len(b)}")
    if not a:
        return {"n": 0, "diff": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0}
    diff = _mean(a) - _mean(b)
    if len(a) < 2:
        return {"n": len(a), "diff": diff, "lo": diff, "hi": diff, "p_value": 1.0}

    rng = random.Random(seed)
    size = len(a)
    samples = []
    for _ in range(n_boot):
        idx = [rng.randrange(size) for _ in range(size)]
        samples.append(sum(a[i] for i in idx) / size - sum(b[i] for i in idx) / size)
    samples.sort()
    non_positive = sum(1 for value in samples if value <= 0) / n_boot
    non_negative = sum(1 for value in samples if value >= 0) / n_boot
    return {
        "n": size,
        "diff": diff,
        "lo": _percentile(samples, alpha / 2),
        "hi": _percentile(samples, 1 - alpha / 2),
        "p_value": min(1.0, 2 * min(non_positive, non_negative)),
    }


def mcnemar_exact(a_only: int, b_only: int) -> dict:
    """McNemar 精确检验：只统计「一个中一个不中」的配对。

    用于「同一批题，模式 A 与模式 B 的命中集合是否有显著差异」——它忽略了双方都中的题，
    所以比独立区间更敏感，也更贴合「换检索方式值不值」这个问题。
    """
    a_only = int(a_only)
    b_only = int(b_only)
    discordant = a_only + b_only
    if discordant == 0:
        return {"discordant": 0, "p_value": 1.0}
    tail = sum(math.comb(discordant, i) for i in range(0, min(a_only, b_only) + 1))
    p_value = min(1.0, 2 * tail / (2 ** discordant))
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_value": p_value}
