"""`python -m src.doctor`：一条命令把「缺什么、怎么修」摊开。

为什么不是 `--help` 里加几行说明：用户起不来的时候，报错往往来自**下游第 N 层**
（`TokenizerUnavailable` → `tool` 层 → `ask` 层 → HTTP 500），栈里看不到「其实只要
`ollama pull bge-m3`」。doctor 把六项依赖 + 语料目录一次列全，每条失败给可复制命令。

**装配逐项惰性**：词表缺失时，Qdrant / Ollama / 父块的结论仍然必须打得出来
（`tests/test_doctor.py::test_broken_tokenizer_does_not_hide_other_checks`）。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from src.rag.ops.probes import ProbeReport, probe_data_dir, run_probes


@dataclass
class DoctorDeps:
    """真实装配的接口。字段是**工厂**而不是对象：某一项炸掉只影响它自己那一行。"""

    client_factory: Callable[[], Any]
    embed: Callable[[list[str]], Any]
    list_models: Callable[[], list[str]]
    model: str
    counter_factory: Callable[[], Any]
    reranker_factory: Callable[[], Any]
    parents_factory: Callable[[], Mapping[str, Any]]
    data_dir: Path
    embed_model: str = "bge-m3"


class _LazyClient:
    """把 `client_factory()` 的失败推迟到真正用它的那一刻（探测里会转成 `Check`）。"""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._value: Any = None
        self._error: Exception | None = None

    def _resolve(self) -> Any:
        if self._error is not None:
            raise self._error
        if self._value is None:
            try:
                self._value = self._factory()
            except Exception as exc:              # noqa: BLE001 - 交给 probe 报告
                self._error = exc
                raise
        return self._value

    def __getattr__(self, item: str) -> Any:
        return getattr(self._resolve(), item)


def _lazy_call(factory: Callable[[], Any]) -> Callable[[], Any]:
    state: dict[str, Any] = {}

    def call() -> Any:
        if "error" in state:
            raise state["error"]
        if "value" not in state:
            try:
                state["value"] = factory()
            except Exception as exc:              # noqa: BLE001
                state["error"] = exc
                raise
        return state["value"]

    return call


def run(deps: DoctorDeps) -> ProbeReport:
    """六项必查 + 单独的语料目录项（`data/` 为空不影响已在跑的检索）。"""
    report = run_probes(
        qdrant=_LazyClient(deps.client_factory),
        embed=deps.embed,
        list_models=deps.list_models,
        model=deps.model,
        counter=_LazyCounter(deps.counter_factory),
        reranker=_LazyReranker(deps.reranker_factory),
        parents=_LazyParents(deps.parents_factory),
        embed_model=deps.embed_model,
    )
    report.checks.append(probe_data_dir(deps.data_dir))
    return report


class _LazyCounter:
    """`probe_tokenizer` 只需要 `count()`，构造失败要落在这一项上。"""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._get = _lazy_call(factory)

    def count(self, text: str) -> int:
        return self._get().count(text)


class _LazyReranker:
    def __init__(self, factory: Callable[[], Any]) -> None:
        self._get = _lazy_call(factory)

    @property
    def config(self) -> Any:
        return self._get().config


class _LazyParents:
    def __init__(self, factory: Callable[[], Mapping[str, Any]]) -> None:
        self._get = _lazy_call(factory)

    def __len__(self) -> int:
        return len(self._get())


# ---------------------------------------------------------------- 真件装配


def default_deps() -> DoctorDeps:
    """真件：Qdrant 客户端、Ollama 两个模型、词表、重排配置、父块记录。

    全部包成工厂——`python -m src.doctor` 在**什么都没装好的机器上**也必须能跑出报告。
    """
    from src import ollama_client
    from src.config import OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL, STORAGE_DIR
    from src.rag.store.qdrant_store import make_client

    def parents() -> Mapping[str, Any]:
        from src.rag.retrieval.store_wiring import load_parent_records

        return {row["id"]: row for row in load_parent_records()}

    def counter() -> Any:
        from src.rag.ingestion.tokenizer import TokenCounter

        return TokenCounter()

    def reranker() -> Any:
        from src.rag.retrieval.rerank import RerankConfig
        from src.rag.retrieval.rerank import CrossEncoderReranker

        # 只读配置，不加载 449 MB 模型：doctor 不该为了"看一眼"花 2 s 和 449 MB
        class _ConfigOnly:
            config = RerankConfig()

        return _ConfigOnly()

    return DoctorDeps(
        client_factory=lambda: make_client(timeout=3),
        embed=lambda texts: ollama_client.embed_texts(texts),
        list_models=ollama_client.installed_models,
        model=OLLAMA_CHAT_MODEL,
        embed_model=OLLAMA_EMBED_MODEL,
        counter_factory=counter,
        reranker_factory=reranker,
        parents_factory=parents,
        data_dir=Path(os.getenv("RAG_DATA_DIR", "data")),
    )


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None, *, deps: DoctorDeps | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.doctor",
        description="自检 4 项外部依赖（Qdrant / Ollama / 词表 / 重排）+ 索引 + 语料目录",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON（stdout 只留 JSON）")
    args = parser.parse_args(argv)

    report = run(deps or default_deps())
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.render())
        if report.status == "degraded":
            print("\n可检索但不可生成：/api/v1/retrieve 与 `python -m src.ask --retrieve-only` 仍正常。")
    return 0 if report.status in ("ok", "degraded") else 1


if __name__ == "__main__":
    raise SystemExit(main())
