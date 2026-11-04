"""D23：要求翻译**原样保留**的 token —— 型号、缩写、数字、参数名。

为什么这一条是硬约束：翻译只要把 `DGX B300` 写成 `B300 system`、把 `8192` 写成
`8,192`，英文 lexical 路就会整条失效（词元对不上），而这恰好是 D25 测出来
英文路最能赢的那类题（number_spec / acronym_model）。所以：

* 抽取（`protect_tokens`）：翻译**前**从中文 query 里认出这些 token；
* 校验（`missing_terms`）：翻译**后**逐个检查是否原样出现，缺一个就重试，
  重试还缺就**回退原文**（宁可用中文问，也不要把术语弄丢）。
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff]")

#: 需要原样保留的 token：型号、参数名、缩写、数字
_PROTECTED = re.compile(
    r"""
    [A-Za-z]{2,}[\s-]?[A-Za-z]*\d+(?:\.\d+)?   # DGX B300 / PCIe Gen6 / RHEL8 / A100
    | \d+(?:\.\d+)?\s?(?:GB|TB|MB|ms|s|tokens?)\b   # 288 GB / 8192 tokens
    | \d{3,}                            # 8192、2024
    | [A-Z]{2,}[A-Za-z]*                # HNSW / BMC / RAG / NVLink / MIG
    | [a-z]+_[a-z_]+                    # ef_construct / top_k
    | [A-Z][a-z]+[A-Z][A-Za-z]*         # NVSwitch / TopK
    """,
    re.VERBOSE,
)


def _normalize(token: str) -> str:
    # 去空白再比：`288 GB` 与 `288GB`、`DGX B300` 与 `DGX-B300` 视为同一个术语
    return re.sub(r"\s+", "", token or "").lower()


def protect_tokens(text: str) -> set[str]:
    """抽出必须原样保留的 token。"""
    found: set[str] = set()
    for match in _PROTECTED.finditer(text or ""):
        token = match.group(0).strip()
        if not token or _CJK.search(token):
            continue
        found.add(token)
        # 「288 GB」也要能以「288」的形式带上：译文写 288GB 时不该算丢术语
        number = re.search(r"\d+(?:\.\d+)?", token)
        if number:
            found.add(number.group(0))
    return found


def missing_terms(source: str, translated: str) -> list[str]:
    """翻译后丢了哪些必须保留的 token（按首次出现位置排序，便于报错）。"""
    haystack = _normalize(translated)
    missing = [token for token in protect_tokens(source) if _normalize(token) not in haystack]
    return sorted(missing, key=lambda token: (source.find(token), token))


def retention(source: str, translated: str) -> float:
    """术语保留率（1.0 = 全部原样保留）。"""
    tokens = protect_tokens(source)
    if not tokens:
        return 1.0
    return round(1 - len(missing_terms(source, translated)) / len(tokens), 4)
