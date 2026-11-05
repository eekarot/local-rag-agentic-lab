"""给「必须有 tokenizer 才能跑、但不关心具体词表」的用例提供替身。

为什么需要它（A5 干净克隆验收逮到的真问题）：一批用例默认 `TokenCounter()`
走 `storage/tokenizer/bge-m3/tokenizer.json`（**17 MB，被 gitignore**）。
开发机上有这个文件，所以一直没暴露；`git clone` 到新目录 + 全新 venv 之后
**37 个用例直接 ERROR**（`TokenizerUnavailable`）—— 这等于「克隆下来跑不绿」。

三种处理方式里选了第三种：

1. 把这些用例挂 `@slow`：干净克隆里它们被跳过 —— 但 CI 也就永远不跑它们了，
   而它们守的正是切分/预算/管线的主干逻辑，不能瞎；
2. 把 17 MB 词表入库：违背 B13（不携语料资产）且把仓库变大 2 倍；
3. **给它们一个等价替身**（本文件）：字符级 tokenizer，`count("知识库") == 3`，
   量级与真实词表同阶（真实 bge-m3 对中文也接近 1 token/字）。
   切分、预算、截断这些**逻辑**照样被守住，只是不再依赖那份具体词表。

纪律：
* 开发机上有真实词表时**不动它**（`ensure_tokenizer()` 直接返回），
  否则「本地跑的」和「CI 跑的」用两套词表，测出来的差异查不清；
* 真实词表的**数值**对齐（与 Ollama `/api/embed` 的 token 数比对）仍然只在
  `test_token_counter_integration.py` 里做，那个模块挂 `@slow`。
"""

from __future__ import annotations

import os
import tempfile
import threading

ENV = "RAG_TOKENIZER_PATH"

_CACHE: dict[str, str] = {}
_LOCK = threading.Lock()


def real_tokenizer_path() -> str:
    """仓库默认位置的真实词表路径（可能不存在）。"""
    from src.rag.ingestion.tokenizer import DEFAULT_TOKENIZER_PATH

    return str(DEFAULT_TOKENIZER_PATH)


def build_char_tokenizer(path: str) -> str:
    """造一个「一个字符 ≈ 一个 token」的 tokenizer：量级接近真实中文词表。"""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    # 逐字符切开（未登录字符落到 [UNK]，但**照样占一个 token**）；
    # 注意 pattern 只能是 "."：Oniguruma 不接受 `(?s)` 这种内联选项
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=Regex("."), behavior="isolated")
    tokenizer.save(path)
    return path


def test_tokenizer_path() -> str:
    """进程内复用的替身词表路径（落在临时目录，随解释器退出即可丢）。"""
    with _LOCK:
        cached = _CACHE.get("path")
        if cached and os.path.exists(cached):
            return cached
        directory = tempfile.mkdtemp(prefix="rag-tiny-tokenizer-")
        path = build_char_tokenizer(os.path.join(directory, "tokenizer.json"))
        _CACHE["path"] = path
        return path


def ensure_tokenizer() -> str | None:
    """真实词表缺失时，把 `RAG_TOKENIZER_PATH` 指向替身。

    返回实际生效的路径；真实词表存在则返回 None（什么都不改）。
    ``TokenCounter.__init__`` 是在**调用时**读环境变量的，所以这里改 env 有效。
    """
    if os.path.exists(real_tokenizer_path()):
        return None
    path = test_tokenizer_path()
    os.environ[ENV] = path
    return path
