"""D23：Query 翻译服务（Ollama + 严格 JSON + 术语保护 + 缓存 + 失败降级）。

这条服务在链路里的位置：中文 query 进来 → 翻成英文 → 喂给英文 Dense 与英文 lexical
两路（D24/D26）。它**不能**让检索因为翻译而失败，所以每个入口都设计成"永远返回
一个可用的 query"：

| 情况 | 返回 | `status` |
|---|---|---|
| 正常翻出来且术语齐全 | 英文译文 | `ok` |
| 缓存命中 | 英文译文 | `cached` |
| 输入本来就是英文 | 原文 | `skipped` |
| 模型漏了型号/数字（重试后仍漏） | **原文** | `passthrough_terms` |
| 模型超时/报错/回包解析不了 | **原文** | `passthrough_error` |

两条硬规则：

1. **绝不抛异常**。翻译层挂了最多是「少一路召回」，不能把整次检索打挂。
2. **术语优先于流畅**。`DGX B300` 被改写成 `B300 system` 会让英文 lexical 整条失效，
   所以漏术语时宁可回退中文原文（`passthrough_terms`），并在 `missing` 里留痕。

严格 JSON：让模型回 `{"translation": "..."}`，解析时容忍 ```json 围栏与裸字符串，
但**不接受**空译文或缺少 translation 键的回包（`parse_json_reply` 返回空串 → 重试/回退）。

    python -m src.rag.translate.service --questions experiments/questions.json --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..lang.detect import ZH, detect
from .cache import TranslationCache
from .terms import missing_terms, protect_tokens

STATUS_OK = "ok"
STATUS_CACHED = "cached"
STATUS_SKIPPED = "skipped"
STATUS_PASSTHROUGH_TERMS = "passthrough_terms"
STATUS_PASSTHROUGH_ERROR = "passthrough_error"

STATUSES = (STATUS_OK, STATUS_CACHED, STATUS_SKIPPED, STATUS_PASSTHROUGH_TERMS, STATUS_PASSTHROUGH_ERROR)

#: 系统提示：要求严格 JSON、术语原样、不许扩写
SYSTEM_PROMPT = (
    "You translate a Chinese RAG retrieval query into English.\n"
    "Rules:\n"
    "1. Output STRICT JSON only: {\"translation\": \"<english query>\"}\n"
    "2. Keep models, acronyms, numbers and parameter names EXACTLY as written "
    "(DGX B300, HNSW, NCU, 8192, ef_construct). Never reformat numbers (8192 stays 8192, not 8,192).\n"
    "3. Translate the question, do not answer it, do not add context, keep it one line.\n"
    "4. If the input is already English, return it unchanged."
)


@dataclass
class TranslateResult:
    source: str
    text: str
    status: str
    missing: list[str] = field(default_factory=list)
    attempts: int = 0
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "text": self.text,
            "status": self.status,
            "missing": list(self.missing),
            "attempts": self.attempts,
            "latency_ms": round(self.latency_ms, 2),
        }


def parse_json_reply(reply: str) -> str:
    """从模型回包里抠出译文。解析不出就返回空串（交给上层重试/回退）。"""
    text = (reply or "").strip()
    if not text:
        return ""
    # 去掉 ```json ... ``` 围栏
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except ValueError:
            return ""
        value = payload.get("translation") if isinstance(payload, dict) else None
        return value.strip() if isinstance(value, str) else ""
    # 裸字符串：只要没有 JSON 结构就当成译文（模型偶尔只回一行英文）
    return text.splitlines()[0].strip().strip('"')


def ollama_provider(model: str | None = None) -> Callable[..., list[str]]:
    """真 provider：走 Ollama `/api/chat`，要求 JSON 回包、关掉思考段。"""
    from ...ollama_client import chat

    def provider(texts: list[str], hints: list[str] | None = None) -> list[str]:
        out: list[str] = []
        for index, text in enumerate(texts):
            user = text
            if hints and hints[index]:
                user = f"{text}\n\n(Must keep these tokens verbatim: {', '.join(hints[index])})"
            reply = chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                temperature=0,
                think=False,
                num_predict=160,
                fmt="json",
            )
            out.append(parse_json_reply(reply))
        return out

    return provider


class TranslateService:
    def __init__(
        self,
        provider: Callable[..., list[str]] | None = None,
        *,
        cache: TranslationCache | None = None,
        max_retries: int = 1,
    ):
        self.provider = provider or ollama_provider()
        self.cache = cache
        self.max_retries = max_retries
        self._counts: dict[str, int] = {status: 0 for status in STATUSES}
        self._total = 0

    # -- 内部 ---------------------------------------------------------------
    def _call(self, texts: list[str], hints: list[list[str]] | None = None,
              temperature: float = 0.0) -> list[str]:
        """调 provider。签名不兼容就退回位置参数；任何异常都降级成空串（上层再回退原文）。"""
        if not texts:
            return []

        def attempt(call) -> list[str] | None:
            try:
                return list(call())
            except TypeError:
                return None      # provider 只接受位置参数（测试替身最常见）
            except Exception:    # noqa: BLE001  超时/连接/HTTP 错误一律降级
                return ["" for _ in texts]

        result = attempt(lambda: self.provider(texts, hints=hints, temperature=temperature))
        if result is None:
            result = attempt(lambda: self.provider(texts))
        return result if result is not None else ["" for _ in texts]

    def _one(self, text: str) -> TranslateResult:
        started = time.time()
        terms = protect_tokens(text)
        best: TranslateResult | None = None
        hints: list[str] | None = [sorted(terms)] if terms else None
        for attempt in range(1 + self.max_retries):
            # 第一次温度 0（要确定性）；重试换温度——同 prompt + 温度 0 只会拿到同一段文本
            replies = self._call([text], hints=hints, temperature=0.0 if attempt == 0 else 0.4)
            translated = (replies[0] if replies else "").strip()
            if not translated:
                best = TranslateResult(text, text, STATUS_PASSTHROUGH_ERROR, [], attempt + 1)
                continue
            missing = missing_terms(text, translated)
            result = TranslateResult(text, translated, STATUS_OK, missing, attempt + 1)
            if not missing:
                result.latency_ms = (time.time() - started) * 1000
                return result
            best = TranslateResult(text, text, STATUS_PASSTHROUGH_TERMS, missing, attempt + 1)
        assert best is not None
        best.latency_ms = (time.time() - started) * 1000
        return best

    # -- 对外 ---------------------------------------------------------------
    def translate(self, texts: Iterable[str]) -> list[TranslateResult]:
        """逐条翻译。顺序与输入一致，永不抛异常。"""
        results: list[TranslateResult] = []
        for text in texts:
            source = text or ""
            if detect(source) != ZH:
                result = TranslateResult(source, source, STATUS_SKIPPED)
            else:
                cached = self.cache.get(source) if self.cache else None
                if cached:
                    result = TranslateResult(source, cached, STATUS_CACHED)
                else:
                    result = self._one(source)
                    if result.status == STATUS_OK and self.cache:
                        self.cache.put(source, result.text)
            self._counts[result.status] = self._counts.get(result.status, 0) + 1
            self._total += 1
            results.append(result)
        return results

    def translate_text(self, text: str) -> str:
        """三路召回只需要字符串时的便捷入口（失败就是原文）。"""
        return self.translate([text])[0].text

    def stats(self) -> dict:
        success = self._counts[STATUS_OK] + self._counts[STATUS_CACHED]
        return {
            "total": self._total,
            **{status: self._counts.get(status, 0) for status in STATUSES},
            "success_rate": round(success / self._total, 4) if self._total else None,
            "cache": self.cache.stats() if self.cache else None,
        }


# ---------------------------------------------------------------------------
# CLI：D23 验收（200 条中文 query 成功率 + 术语保留率）
# ---------------------------------------------------------------------------


def build_stress_queries(questions: Sequence[dict]) -> list[str]:
    """把评估集的中文问法扩成 200+ 条的翻译压力集。

    两种问法变体：原句 + 「在 NVIDIA 文档里，<原句>」。**同一条问题只算一个唯一 query**，
    变体只是模拟真实用户的两种说法（前缀式提问很常见），报告里会把两个数都摊开，
    不拿变体充数。
    """
    base = [item["question"] for item in questions if item.get("question")]
    variants = [f"在 NVIDIA 文档里，{text}" for text in base]
    return base + variants


def main(argv: Sequence[str] | None = None) -> int:
    from ...config import OLLAMA_CHAT_MODEL, STORAGE_DIR

    parser = argparse.ArgumentParser(prog="python -m src.rag.translate.service", description="D23 翻译服务")
    parser.add_argument("--questions", default=None, help="用评估集的中文问法做压力测试")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    parser.add_argument("--text", nargs="*", default=None, help="直接翻给定文本")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    cache = None if args.no_cache else TranslationCache(
        STORAGE_DIR / "cache" / "translations.sqlite3", model=OLLAMA_CHAT_MODEL, version="v1"
    )
    service = TranslateService(ollama_provider(), cache=cache)

    if args.text:
        for result in service.translate(args.text):
            print(json.dumps(result.as_dict(), ensure_ascii=False))
        return 0

    if not args.questions:
        parser.print_help()
        return 2

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    queries = build_stress_queries(questions)
    if args.limit:
        queries = queries[: args.limit]
    unique = len({item["question"] for item in questions if item.get("question")})
    print(f"[translate] 样本 {len(queries)} 条（唯一原句 {unique} 条 + 前缀变体 {len(queries) - unique} 条）")

    started = time.time()
    results = service.translate(queries)
    elapsed = time.time() - started
    stats = service.stats()

    ok = [r for r in results if r.status in (STATUS_OK, STATUS_CACHED)]
    degraded = [r for r in results if r.status == STATUS_PASSTHROUGH_TERMS]
    errors = [r for r in results if r.status == STATUS_PASSTHROUGH_ERROR]
    print(f"[translate] 成功 {len(ok)}/{len(results)} = {len(ok) / len(results):.2%}"
          f"（验收线 98%），用时 {elapsed:.1f}s")
    print(f"[translate] 术语漏项回退 {len(degraded)} 条，调用失败回退 {len(errors)} 条")
    print(f"[translate] 状态分布：{ {k: v for k, v in stats.items() if k in STATUSES} }")
    if stats["cache"]:
        print(f"[translate] 缓存：{stats['cache']}")
    for result in (degraded + errors)[:5]:
        print(f"  [回退] {result.status} missing={result.missing} :: {result.source}")

    payload = {"results": [r.as_dict() for r in results], "stats": stats}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {args.json}")
    return 0 if len(ok) / len(results) >= 0.98 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
