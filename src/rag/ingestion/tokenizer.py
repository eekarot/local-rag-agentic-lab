"""Token 计数：BGE-M3 tokenizer 口径（方案 §2.3 要求统一 Token 计量）。

决策 B01（走 Ollama API）带来一个约束：``/api/embed`` 只返回整个 batch 的
``prompt_eval_count``，拿不到逐条 token 数；逐条调用 ≈2.1 s/条，81k 条不可行。
因此 token 计数走**本地 tokenizer**（纯 CPU、毫秒级，不跑模型推理），
Ollama 的计数只用于抽样校验（见 ``verify_against_ollama``）。

实测对齐口径（2026-11-02，bge-m3 / XLM-R 词表 250,002，8–200 条真实 chunk）：
    单条 ollama prompt_eval_count == count(text) + 2      （BOS + EOS）
    批量 ollama prompt_eval_count == Σ count(text) + 2 × 条数
``count()`` 只算内容 token，所以切分预算（400/500/60）与 Ollama 计数差 2/条，这是预期行为。
实测逐条中位误差 0%、最大绝对差 0；换成 nomic 词表则中位 5.97%、最大差 107 token。
"""

import os
import threading
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

from tokenizers import Tokenizer

from src.rag.ingestion.errors import TokenizerUnavailable

ROOT_DIR = Path(__file__).resolve().parents[3]

#: 默认位置（storage/ 不入库，见 .gitignore）
DEFAULT_TOKENIZER_PATH = ROOT_DIR / "storage" / "tokenizer" / "bge-m3" / "tokenizer.json"

#: 环境变量覆盖
TOKENIZER_PATH_ENV = "RAG_TOKENIZER_PATH"

MODELSCOPE_URL = (
    "https://www.modelscope.cn/api/v1/models/BAAI/bge-m3/repo"
    "?Revision=master&FilePath=tokenizer.json"
)

#: 校验容差：中位逐条相对误差 ≤2%；实测 bge-m3 为 0%
DEFAULT_TOLERANCE = 0.02

#: 逐条绝对差上限（token）：短文本上 1 token 就能占 4%，只靠相对值会误报
DEFAULT_MAX_ABS_TOKEN_DIFF = 2

EmbedFn = Callable[[Sequence[str]], dict[str, Any]]

#: 进程级词表缓存：`(路径, mtime_ns, size) -> Tokenizer`（见 `TokenCounter._load`）
_TOKENIZER_CACHE: dict[tuple[str, int, int], Tokenizer] = {}
_TOKENIZER_LOCK = threading.Lock()


class TokenCounter:
    """BGE-M3 口径的 token 计数器。

    >>> counter = TokenCounter()
    >>> counter.count("HNSW uses a hierarchical graph.")
    8
    """

    def __init__(self, tokenizer_path: str | os.PathLike[str] | None = None) -> None:
        resolved = tokenizer_path or os.environ.get(TOKENIZER_PATH_ENV) or DEFAULT_TOKENIZER_PATH
        self.path = Path(resolved)
        self._tokenizer = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> Tokenizer:
        """解析词表（**带进程级缓存**：17 MB 的 `Tokenizer.from_file` 约 0.75 s）。

        词表在进程生命周期内不会变，但构造点不少（CLI、doctor、每个 IngestPipeline、
        测试里的二十几次），不加缓存时单这一项就能吃掉默认套件 20% 的时间。
        缓存键带上 `mtime_ns + size`：文件真被换掉时仍会重新解析。
        """
        if not path.is_file():
            raise TokenizerUnavailable(
                f"找不到 tokenizer.json：{path}\n"
                f"获取方式（ModelScope 镜像，约 17 MB）：\n"
                f"  curl -sL -o \"{path}\" \"{MODELSCOPE_URL}\"\n"
                f"或用环境变量 {TOKENIZER_PATH_ENV} 指向已有文件。"
            )
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        with _TOKENIZER_LOCK:
            cached = _TOKENIZER_CACHE.get(key)
            if cached is not None:
                return cached
        # 解析放锁外：0.75 s 的活不该把其它线程也一起卡住；重复解析一次比串行更划算
        tokenizer = Tokenizer.from_file(str(path))
        with _TOKENIZER_LOCK:
            _TOKENIZER_CACHE[key] = tokenizer
        return tokenizer

    def count(self, text: str) -> int:
        """内容 token 数（不含 BOS/EOS）。空白文本记 0。"""
        if not text or not text.strip():
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count_batch(self, texts: Sequence[str]) -> list[int]:
        return [self.count(text) for text in texts]

    def truncate(self, text: str, max_tokens: int) -> str:
        """把文本截到 ≤ max_tokens，且返回值必须是**原文的逐字符前缀**。

        不能用 ``decode(ids[:n])``：真实词表（XLM-R）的 decode 会重新拼空格与标点，
        得到的不是原文子串；下游要靠前缀长度算 ``source_start/source_end``（D07），
        一旦错位就会丢掉/重复字符。所以按第 n 个 token 的字符 offset 回切原文。
        """
        if max_tokens <= 0 or not text:
            return ""
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        if len(encoding.ids) <= max_tokens:
            return text
        cut = encoding.offsets[max_tokens - 1][1]
        return text[:cut]

    def truncate_tail(self, text: str, max_tokens: int) -> str:
        """取文本末尾 ≤ max_tokens 个 token，返回值必须是原文的逐字符**后缀**。

        与 ``truncate`` 相反的方向——D07 的 overlap 要从上一块的尾部取内容，
        而且取出来的必须是原文子串，否则重叠区域没法在原文里定位。
        """
        if max_tokens <= 0 or not text:
            return ""
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        total = len(encoding.ids)
        if total <= max_tokens:
            return text
        start = encoding.offsets[total - max_tokens][0]
        tail = text[start:]
        # 从 left 再收一刀：重新 tokenize 后缀时开头可能多吃出 token
        while tail and self.count(tail) > max_tokens:
            tail = tail[1:]
        return tail

    def verify_against_ollama(
        self,
        texts: Sequence[str],
        embed_fn: EmbedFn | None = None,
        tolerance: float = DEFAULT_TOLERANCE,
        max_abs_tolerance: int = DEFAULT_MAX_ABS_TOKEN_DIFF,
    ) -> dict[str, Any]:
        """抽样校验本地计数是否与 Ollama 同口径（D05 的验收方式）。

        **逐条**对比，不用聚合总量：实测 200 条 chunk 上换错模型（nomic）的聚合
        只差 0.35%，正负误差互相抵消，聚合口径分辨不出词表是否错了；
        逐条对比下 nomic 的中位相对误差 5.97%、最大绝对差 107 token。
        """
        texts = list(texts)
        if not texts:
            raise ValueError("verify_against_ollama 需要至少 1 条样本文本")

        call = embed_fn or self._ollama_embed
        per_text: list[dict[str, int]] = []
        for text in texts:
            # 单条调用下 Ollama 会加 1 个 BOS 与 1 个 EOS，扣掉后与内容 token 可比
            api_content = int(call([text])["prompt_eval_count"]) - 2
            local = self.count(text)
            per_text.append({"local": local, "ollama": api_content, "diff": api_content - local})

        relative = [abs(row["diff"]) / row["local"] for row in per_text if row["local"]]
        median_rel = statistics.median(relative) if relative else 0.0
        max_abs = max(abs(row["diff"]) for row in per_text)

        return {
            "n_texts": len(per_text),
            "local_total": sum(row["local"] for row in per_text),
            "ollama_total": sum(row["ollama"] for row in per_text),
            "median_rel_diff": round(median_rel, 6),
            "max_abs_diff": max_abs,
            "tolerance": tolerance,
            "max_abs_tolerance": max_abs_tolerance,
            "aligned": median_rel <= tolerance and max_abs <= max_abs_tolerance,
            "per_text": per_text,
        }

    @staticmethod
    def _ollama_embed(texts: Sequence[str]) -> dict[str, Any]:
        import requests

        from src.config import OLLAMA_BASE_URL

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": "bge-m3", "input": list(texts)},
            timeout=300,
        )
        if response.status_code != 200:
            raise TokenizerUnavailable(
                f"Ollama /api/embed 返回 {response.status_code}：{response.text[:200]}"
            )
        return response.json()


_default_counter: TokenCounter | None = None


def default_counter() -> TokenCounter:
    """进程内共享的默认计数器（加载 17 MB 词表约 0.3 s，不要每次新建）。"""
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter()
    return _default_counter


def cli_main(argv: Sequence[str] | None = None) -> int:
    """验收用命令行：

        python -m src.rag.ingestion.tokenizer count "HNSW 是什么"
        type data\\a.md | python -m src.rag.ingestion.tokenizer count -
        python -m src.rag.ingestion.tokenizer verify --n 8        # 需要 Ollama
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m src.rag.ingestion.tokenizer",
        description="BGE-M3 口径的 token 计数（本地词表，不调模型）",
    )
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json 路径（默认读 RAG_TOKENIZER_PATH）")
    sub = parser.add_subparsers(dest="command", required=True)

    count = sub.add_parser("count", help="输出 token 数；文本为 - 时逐行读 stdin")
    count.add_argument("text")

    verify = sub.add_parser("verify", help="抽样校验本地计数与 Ollama 是否同口径")
    verify.add_argument("--n", type=int, default=8, help="抽样条数（每条一次 Ollama 调用，约 2 s）")
    verify.add_argument("--index", default="storage/index.json", help="样本来自哪个索引文件")

    args = parser.parse_args(argv)
    try:
        counter = TokenCounter(args.tokenizer)
    except TokenizerUnavailable as exc:
        # 干净克隆里 README 第 3 步的**验证命令**最先撞上它。原始异常已经写清了
        # 「装什么 / 怎么装 / 能不能用别的路径」，唯独不该以堆栈的形式出现
        # —— 用户会以为是自己装错了（A5 真·干净环境实测）。
        print(str(exc), file=sys.stderr)
        return 1

    if args.command == "count":
        if args.text == "-":
            for line in sys.stdin:
                print(counter.count(line))
        else:
            print(counter.count(args.text))
        return 0

    records = json.loads(Path(args.index).read_text(encoding="utf-8"))["records"]
    step = max(1, len(records) // max(1, args.n))
    texts = [records[i]["text"] for i in range(0, len(records), step)][: args.n]
    report = counter.verify_against_ollama(texts)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["aligned"] else 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
