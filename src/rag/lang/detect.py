"""D22：语言识别（zh / en，含「中文 + 英文术语」混排）。

为什么要自己写而不是引入现成的语言识别库：本项目的「混排」并不是均衡双语，而是
**中文问句里夹型号/缩写**（`DGX B300 装了几张 GPU？`、`HNSW 是什么？`）。
这类文本的**字母数**并不比汉字少——实测 `HNSW 是什么？` 汉字 3 / 字母 4（汉字只占字符的 42.9%）、
`NVLink 和 PCIe 有什么区别？` 6 / 10（37.5%）——所以按「字符占比」判语言的思路在这里不可靠：
占比被缩写和标识符拉低，一旦阈值给高了就会把中文问法判成 en，路由也就走错分支
（中文问法被送去英文 BM25，关键词全部丢失）。

判据因此改成**中文汉字 vs 英文单词**的个数比，不是字符占比：

    ratio = 汉字数 / (汉字数 + 英文单词数)

* 按「单词」而不是「字母」算：`DGX B300` 是两个词而不是七个字符，
  `ef_construct` 是一个词，缩写/标识符不会把比例拉偏；
* 阈值 0.20：`HNSW 是什么？` = 3/(3+1)=0.75、`DGX B300 装了几张 GPU？` = 5/(5+3)=0.63，
  都稳稳是 zh；纯英文句子汉字数为 0，直接是 en。

已知边界：日文/韩文里的汉字也会被算作 zh（本项目的语料只有中英，够用；
真要支持需要换成别的方法，届时换实现而不改接口）。

    python -m src.rag.lang.detect "HNSW 是什么？"
    python -m src.rag.lang.detect --questions experiments/questions.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

ZH = "zh"
EN = "en"
UNKNOWN = "unknown"

#: 汉字（含扩展 A）与兼容区；不含平假名/片假名/谚文
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
#: 英文单词：连续字母，允许中间出现数字/下划线/点（`ef_construct`、`qwen3.5`、`DGX`）
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.]*")

#: 汉字占比高于这个值就判 zh
ZH_THRESHOLD = 0.20


def cjk_char_count(text: str) -> int:
    """汉字个数（不含标点）。"""
    return len(_CJK.findall(text or ""))


def latin_word_count(text: str) -> int:
    """英文单词个数。按词计，不按字母计。"""
    return len(_LATIN_WORD.findall(text or ""))


def language_ratio(text: str) -> float | None:
    """汉字占「汉字 + 英文单词」的比例。没有可用字符时返回 None。"""
    cjk = cjk_char_count(text)
    words = latin_word_count(text)
    if cjk + words == 0:
        return None
    return round(cjk / (cjk + words), 4)


def detect(text: str) -> str:
    """判 `zh` / `en` / `unknown`。"""
    ratio = language_ratio(text)
    if ratio is None:
        return UNKNOWN
    return ZH if ratio >= ZH_THRESHOLD else EN


def detect_batch(texts: Iterable[str]) -> list[str]:
    return [detect(text) for text in texts]


def summarize(languages: Iterable[str]) -> dict[str, int]:
    stats = {ZH: 0, EN: 0, UNKNOWN: 0}
    for language in languages:
        stats[language if language in stats else UNKNOWN] += 1
    stats["total"] = sum(stats[key] for key in (ZH, EN, UNKNOWN))
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.rag.lang.detect", description="语言识别")
    parser.add_argument("text", nargs="*", help="要判定的文本（可给多条）")
    parser.add_argument("--questions", default=None, help="评估集路径：校验 question(zh) 与 query_en(en)")
    args = parser.parse_args(argv)

    if args.questions:
        items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        zh_pairs = [(item["qid"], item["question"], ZH) for item in items]
        en_pairs = [(item["qid"], item.get("query_en") or "", EN) for item in items if item.get("query_en")]
        pairs = zh_pairs + en_pairs
        wrong = [(qid, text[:40], want, detect(text)) for qid, text, want in pairs if detect(text) != want]
        correct = len(pairs) - len(wrong)
        print(f"样本 {len(pairs)} 条（zh {len(zh_pairs)} / en {len(en_pairs)}）："
              f"准确率 {correct / len(pairs):.2%}")
        for qid, text, want, got in wrong[:10]:
            print(f"  [判错] {qid} 期望 {want} 实际 {got}：{text}")
        return 0 if not wrong else 1

    if not args.text:
        parser.print_help()
        return 2
    for text in args.text:
        print(f"{detect(text):8s} ratio={language_ratio(text)}  {text}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
