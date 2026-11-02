"""中英混合分词（BM25 / 关键词检索专用）。

设计目标
--------
1. 中文走 jieba 的 `cut_for_search`，比原先的二元字组（bigram）更贴近语义单元；
2. `cut_for_search` 会把长词同时切成子词（「检索增强生成」→ 检索增强 / 检索 / 增强），
   这样词典外的专有名词也不会整段丢失，只是退化成子词命中；
3. 英文 / 数字 / 型号 / 路径整体保留（GPT-4o、k1、bge-m3、src/retrieve.py），
   并额外拆出子片段（retrieve、py）和驼峰片段（AgenticRAG → agentic、rag）；
4. 用 NFKC 归一化全角字符，避免「ＲＡＧ」和「RAG」被当成两个词；
5. 过滤停用词与纯符号，避免「的 / 了 / 是」这类高频字主导 BM25 打分。

如果环境里没有装 jieba，会自动退回原项目的二元字组方案，
保证 `pip install -r requirements.txt` 之前也不会崩。
"""

import logging
import re
import unicodedata
from functools import lru_cache

try:  # 可选依赖：装了就用，没装就退回 bigram
    import jieba

    jieba.setLogLevel(logging.WARNING)  # 关掉「Building prefix dict」提示
    HAS_JIEBA = True
except ImportError:  # pragma: no cover - 兜底路径
    jieba = None
    HAS_JIEBA = False


# 中文停用词 + 英文停用词。只收「几乎不携带检索信息」的词，
# 像「使用 / 问题 / 检索」这类有实义的一律保留。
_STOPWORDS = """
的 地 得 了 着 过 是 在 和 与 及 或 而 并 且 但 也 就 都 还 又 再 很 更 最 太 只
把 被 让 给 对 于 以 因 为 从 到 由 向 往 跟 同 于 其 之 此 该 这 那 些 个 们 我 你
他 她 它 我们 你们 他们 她们 它们 自己 这个 那个 这些 那些 这里 那里 这样 那样
什么 怎么 怎样 如何 哪些 哪个 为什么 多少 是否 可以 可能 应该 需要 能够 会 能 要
不 没 没有 不要 不会 不是 一个 一些 一样 以及 或者 例如 比如 根据 关于 由于 不过
而且 因此 所以 但是 如果 因为 虽然 然而 于是 时候 方面 情况 起来 下来 出来 过来
进行 对于 通过 按照 除了 之外 等等 等 中 内 外 上 下 前 后 里 间 时
the a an of to in on at by for with from as is are was were be been being it its this
that these those and or not no but if then than so such can could should would will
may might must do does did has have had we you they he she i us them our your their
""".split()

STOPWORDS = frozenset(_STOPWORDS)

_CJK = "\u4e00-\u9fff"
_TOKEN_RE = re.compile(rf"[a-zA-Z0-9]+(?:[._+#/@-][a-zA-Z0-9]+)*|[{_CJK}]+")
_CJK_ONLY_RE = re.compile(rf"^[{_CJK}]+$")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_IDENT_SEP_RE = re.compile(r"[._+#/@-]")


def _ascii_tokens(piece):
    """英文/数字/型号/路径：保留整体，并补上分隔片段与驼峰片段。"""
    out = [piece.lower()]
    if _IDENT_SEP_RE.search(piece):
        # src/retrieve.py -> src, retrieve, py
        out.extend(p.lower() for p in _IDENT_SEP_RE.split(piece) if p)
    elif any(ch.isupper() for ch in piece):
        # AgenticRAG -> agentic, rag（纯大写缩写如 RAG 只会得到一个片段）
        out.extend(p.lower() for p in _CAMEL_RE.findall(piece))
    return list(dict.fromkeys(out))  # 同一 token 内去重，避免人为放大 tf


def _cjk_bigram(piece):
    """无 jieba 时的兜底：<=2 字保留整段，否则取所有二元字组。"""
    if len(piece) <= 2:
        return [piece]
    return [piece[i:i + 2] for i in range(len(piece) - 1)]


@lru_cache(maxsize=8192)
def _cjk_tokens_cached(piece):
    if HAS_JIEBA:
        words = jieba.cut_for_search(piece)
    else:
        words = _cjk_bigram(piece)

    out = []
    for word in words:
        word = word.strip()
        if not word or word in STOPWORDS:
            continue
        out.append(word)
    return tuple(out)


def tokenize(text):
    """把一段文本切成用于 BM25 的词项列表（保留重复，因为 tf 要用）。"""
    if not text:
        return []
    text = unicodedata.normalize("NFKC", str(text))
    out = []
    cjk_runs = []
    for match in _TOKEN_RE.finditer(text):
        piece = match.group()
        if _CJK_ONLY_RE.match(piece):
            cjk_runs.append(piece)
            out.extend(_cjk_tokens_cached(piece))
        else:
            out.extend(_ascii_tokens(piece))
    if out:
        return out
    # 整句被停用词吃光（如「什么是的」）：退回二元字组，至少留下字面线索
    for run in cjk_runs:
        out.extend(w for w in _cjk_bigram(run) if w not in STOPWORDS)
    return out


def tokenizer_name():
    return "jieba-cut_for_search" if HAS_JIEBA else "cjk-bigram-fallback"
