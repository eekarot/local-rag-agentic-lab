"""切片体检：**没有金标**也能判断「这份语料切得有没有结构性毛病」（G4 / 方案 P2）。

为什么需要它：题库只覆盖仓库自带的基准语料。用户索引**自己的文档**时没有题、没有
gold、没有 Recall@k —— 那时唯一能问的问题是「我的切分是不是把段落/表格/代码切断了？」
这就是本脚本回答的问题，用的是 parser 已经给出的结构边界（不是人写的答案）。

三个口径（实现见 `src/rag/ingestion/quality_metrics.py`）：

* ``sc``  尺寸合规：落在 `[--min-tokens, --max-tokens]` 的子块占比
* ``bi``  结构完整：没有被切分点切开的结构块占比（容差 5 字符）
* ``gap`` 覆盖率：父块内容有没有被子块漏掉（**这是不变量，不是评分**）

用法：

    python scripts/chunk_quality.py --dataset k8s-zh-en-v1
    python scripts/chunk_quality.py --dir data/md --threshold 0.9 --strict --json out.json

**它不预测答对率**：SC/BI 只筛「明显结构有毛病」的语料，检索质量仍要看 `eval_bench.py`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.ingest.pipeline import discover_corpus  # noqa: E402
from src.rag.ingestion.parent_builder import ParentConfig  # noqa: E402
from src.rag.ingestion.quality_inspect import format_report, inspect_paths  # noqa: E402
from src.rag.ingestion.quality_metrics import DEFAULT_TOLERANCE_CHARS  # noqa: E402
from src.rag.ingestion.tokenizer import (  # noqa: E402
    DEFAULT_TOKENIZER_PATH,
    TokenCounter,
    TokenizerUnavailable,
)

DATASETS_DIR = ROOT / "datasets"
PATTERNS = ("**/*.md",)
METRIC_NAMES = ("sc", "bi", "gap")
TOKENIZER_HINT = (
    f"[chunk_quality] 缺词表：{DEFAULT_TOKENIZER_PATH}",
    "  下载（约 17 MB，落在 storage/tokenizer/bge-m3/）：",
    "  curl -L -o storage/tokenizer/bge-m3/tokenizer.json \\",
    "    https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json",
)


def resolve_target(args: argparse.Namespace) -> Path:
    if args.dataset:
        return DATASETS_DIR / args.dataset / "docs"
    return Path(args.dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="切片体检（结构指标，不需要标注）")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dir", help="语料目录")
    target.add_argument("--dataset", help="datasets/<name>/docs（仓库自带基准数据集）")
    parser.add_argument("--metrics", default="sc,bi", help="逗号分隔：sc,bi,gap（默认 sc,bi）")
    parser.add_argument("--threshold", type=float, default=0.95, help="bi 低于此值算不合格（默认 0.95）")
    parser.add_argument("--limit", type=int, default=0, help="只体检前 N 篇（0=全部）")
    parser.add_argument("--min-tokens", type=int, default=0, help="sc 的下限（默认 0，只要不超过上限）")
    parser.add_argument("--max-tokens", type=int, default=None, help="sc 的上限（默认用 child_max_tokens）")
    parser.add_argument("--tolerance-chars", type=int, default=DEFAULT_TOLERANCE_CHARS)
    parser.add_argument("--json", dest="json_out", default=None, help="把逐篇结果写到该文件")
    parser.add_argument("--strict", action="store_true", help="有不合格文档或丢字时退出码 1")
    args = parser.parse_args(argv)

    metrics = [name.strip() for name in args.metrics.split(",") if name.strip()]
    unknown = [name for name in metrics if name not in METRIC_NAMES]
    if unknown:
        print(f"[chunk_quality] 未知指标 {unknown}，可选：{', '.join(METRIC_NAMES)}", file=sys.stderr)
        return 2

    root = resolve_target(args)
    if not root.is_dir():
        print(f"[chunk_quality] 目录不存在：{root}", file=sys.stderr)
        if args.dataset:
            print("  先拉数据集：python scripts/fetch_dataset.py --verify", file=sys.stderr)
        return 2

    paths = discover_corpus(root, patterns=PATTERNS)
    if not paths:
        print(f"[chunk_quality] {root} 里没有可体检的 Markdown（空目录不算错误，退出码 0）")
        return 0
    if args.limit:
        paths = paths[: args.limit]

    try:
        counter = TokenCounter()
    except TokenizerUnavailable:
        for line in TOKENIZER_HINT:
            print(line, file=sys.stderr)
        return 1

    summary = inspect_paths(
        paths,
        counter,
        ParentConfig(),
        threshold=args.threshold,
        tolerance_chars=args.tolerance_chars,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        display_root=ROOT,
    )
    for line in format_report(summary):
        print(line)

    if args.json_out:
        payload = summary.to_dict()
        payload["metrics"] = metrics
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[chunk_quality] 逐篇结果已写入 {out_path}")

    if args.strict and not summary.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
