"""E07 问答编排：**产品层的端到端链路**（CLI 与评估器共用同一份代码）。

为什么单独拎出来：D35 之前把「召回 → 重排 → 组装 → prompt → 生成 → 引用体检」整段写在
`src/eval_answer.py` 里，于是「评估器」成了唯一能跑通新链的入口。产品要考虑的问题
（谁能用、怎么调、出错怎么退化）和评估要测的问题不一样，但**链路本身必须只有一份**
——否则评估到的行为和线上跑的行为会悄悄分叉。

依赖方向保持单向：这里只 import `retrieval / context / answer` 三层的公开接口，
不 import 任何 `eval_*` 或 `cli`。外部把 `retriever / parents / counter / reranker / ask`
注入进来，所以整个函数可以在没有 Qdrant、没有模型的情况下被单测覆盖。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..answer.citations import CitationReport, parse_citations, resolve_sources
from ..answer.prompt import PromptBundle, build_messages
from ..context.assembler import AssembledContext, assemble
from ..retrieval.fusion import RRF_K, rrf_fuse_routes
from ..retrieval.rerank import RerankHit, best_window, rerank_hits

#: 检索每路的深度（也当融合池大小）。三路都取 k，D28 定案的口径。
DEFAULT_POOL = 20
#: 交给组装器的块数上限
DEFAULT_K = 10
#: 上下文 token 预算
DEFAULT_BUDGET = 6000


@dataclass
class AskResult:
    """一次问答的全部可交付信息（CLI 打印、JSON 输出、评估计分都从这里取）。"""

    question: str
    question_en: str | None
    language: str
    answer: str
    abstained: bool
    abstain_reason: str
    blocks: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    report: CitationReport
    context_tokens: int
    candidates: int
    timings: dict[str, float] = field(default_factory=dict)
    route_hits: dict[str, int] = field(default_factory=dict)
    llm_called: bool = False

    def as_dict(self) -> dict[str, Any]:
        """JSON 友好（`report` 展开成标量，避免 CLI 输出不可序列化）。"""
        return {
            "question": self.question,
            "question_en": self.question_en,
            "language": self.language,
            "answer": self.answer,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "blocks": len(self.blocks),
            "context_tokens": self.context_tokens,
            "candidates": self.candidates,
            "cited": list(self.report.cited),
            "out_of_range": list(self.report.out_of_range),
            "malformed": list(self.report.malformed),
            "parse_rate": round(self.report.parse_rate, 4),
            "sources": [
                {
                    "index": block.get("index"),
                    "document_id": block.get("document_id"),
                    "parent_ids": list(block.get("parent_ids") or []),
                    "parent_indexes": list(block.get("parent_indexes") or []),
                    "section_path": list(block.get("section_path") or []),
                    "source_uri": block.get("source_uri"),
                    "title": block.get("title"),
                }
                for block in self.sources
            ],
            "timings": {key: round(value, 1) for key, value in self.timings.items()},
            "route_hits": dict(self.route_hits),
            "llm_called": self.llm_called,
        }


def retrieve_ranked(question: str, *, retriever, parents: Mapping[str, Mapping[str, Any]],
                    counter, reranker, pool: int = DEFAULT_POOL,
                    rerank: bool = True) -> tuple[list[RerankHit], int, dict[str, int]]:
    """三路召回 → RRF → 重排。返回 (候选列表, 融合候选数, 每路命中数)。

    `rerank=False` 时**直接返回 RRF 顺序**，不碰 encoder：
    HTTP 层要能按请求关重排（`{"rerank": false}`），而不去改变共享的 reranker 状态
    ——那个方向一改就会在多请求下互相踩。
    """
    result = retriever.retrieve(question, k=pool)
    routes = {name: list(route.ids)[:pool] for name, route in result.routes.items()}
    fused = rrf_fuse_routes(routes, rrf_k=RRF_K, top_k=pool)

    hits: list[RerankHit] = []
    for rank, hit in enumerate(fused, start=1):
        text = (parents.get(hit.parent_id) or {}).get("text", "")
        item = RerankHit(chunk_id=hit.parent_id, parent_id=hit.parent_id, text=text,
                         score=hit.score, rank=rank)
        hits.append(item)

    route_hits = {name: len(ids) for name, ids in routes.items()}
    if not rerank:
        return hits, len(fused), route_hits

    for item in hits:
        # 父块 p50 约 700 token，直接截 256 只看得到标题 → 先按 query 挑窗口再重排
        item.fields["window"] = best_window(item.text, question,
                                            max_tokens=reranker.config.max_length,
                                            counter=counter)
    ranked = rerank_hits(question, hits, encoder=reranker.encoder, config=reranker.config,
                         top_n=pool, stats=reranker.stats,
                         text_getter=lambda hit: hit.fields["window"])
    return ranked, len(fused), route_hits


def answer_query(
    question: str,
    *,
    retriever,
    parents: Mapping[str, Mapping[str, Any]],
    counter,
    reranker,
    ask: Callable[[list[dict[str, str]]], str] | None,
    query_en: str | None = None,
    k: int = DEFAULT_K,
    pool: int = DEFAULT_POOL,
    budget: int = DEFAULT_BUDGET,
    rerank: bool = True,
) -> AskResult:
    """跑完整条链路。`ask=None` 表示「只检索不生成」（`--retrieve-only`）。"""
    timings: dict[str, float] = {}

    started = time.perf_counter()
    ranked, candidates, route_hits = retrieve_ranked(question, retriever=retriever, parents=parents,
                                                    counter=counter, reranker=reranker, pool=pool,
                                                    rerank=rerank)
    timings["retrieve_rerank_ms"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    assembled: AssembledContext = assemble(
        [{"parent_id": hit.parent_id, "score": hit.score, "rerank_score": hit.rerank_score}
         for hit in ranked[:k]],
        parents, counter=counter, budget=budget,
    )
    timings["assemble_ms"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    bundle: PromptBundle = build_messages(question, assembled)
    timings["prompt_ms"] = (time.perf_counter() - started) * 1000

    answer, called = "", False
    if bundle.abstained:                     # 没有证据 → 不调模型（D33 定案）
        answer = bundle.messages[0]["content"]
        timings["generate_ms"] = 0.0
    elif ask is None:
        answer = ""
        timings["generate_ms"] = 0.0
    else:
        started = time.perf_counter()
        answer = ask(bundle.messages)
        timings["generate_ms"] = (time.perf_counter() - started) * 1000
        called = True

    blocks = [block.as_dict() for block in assembled.blocks]
    report = parse_citations(answer, k=len(blocks) or None)
    return AskResult(
        question=question,
        question_en=query_en,
        language=bundle.language,
        answer=answer,
        abstained=bool(bundle.abstained or report.abstained),
        abstain_reason=bundle.reason,
        blocks=blocks,
        sources=resolve_sources(report, blocks),
        report=report,
        context_tokens=assembled.tokens,
        candidates=candidates,
        timings=timings,
        route_hits=route_hits,
        llm_called=called,
    )


def render_cli(result: AskResult, *, show_sources: int = 5, max_answer_chars: int = 4000,
               list_blocks: bool = False) -> str:
    """终端输出：答案 → 引用 → 来源表 → 耗时。给人看的，不复用 `AssembledContext.render()`。"""
    lines: list[str] = []
    lines.append(f"问题：{result.question}")
    if result.question_en:
        lines.append(f"英文：{result.question_en}")
    lines.append("")

    answer = result.answer or "（未生成：--retrieve-only）"
    if len(answer) > max_answer_chars:
        answer = answer[:max_answer_chars] + f"…（截断，共 {len(result.answer)} 字符）"
    lines.append("回答：")
    lines.append(answer)
    lines.append("")

    if result.abstained:
        lines.append(f"[弃答] 原因={result.abstain_reason or 'model_says_insufficient'}")
    lines.append(f"引用：{result.report.cited or '无'}｜越界={len(result.report.out_of_range)}"
                 f"｜解析成功率={result.report.parse_rate:.2f}")

    rows = list(result.sources)
    if not rows and list_blocks:
        # `--retrieve-only`：还没有答案，所以把上下文块本身当来源列出来
        rows = [dict(block, title=block.get("title")) for block in result.blocks]
        if rows:
            lines.append("")
            lines.append(f"上下文块（{len(rows)} 块）：")

    if rows:
        if not list_blocks or result.sources:
            lines.append("")
            lines.append("来源：")
        for block in rows[:show_sources]:
            section = " > ".join(block.get("section_path") or []) or "-"
            lines.append(f"  [证据 {block.get('index')}] {block.get('title') or '-'}｜{section}")
            lines.append(f"            {block.get('source_uri') or block.get('document_id')}")
    elif result.blocks and not list_blocks:
        lines.append("")
        lines.append(f"来源：（答案没有标注引用；上下文共 {len(result.blocks)} 块，"
                     f"--sources all 可全部列出）")

    lines.append("")
    order = ("setup_ms", "retrieve_rerank_ms", "assemble_ms", "prompt_ms", "generate_ms")
    names = [name for name in order if name in result.timings] +             [name for name in result.timings if name not in order]
    timing = "｜".join(f"{name.replace('_ms', '')} {result.timings[name]:.0f} ms" for name in names)
    lines.append(f"耗时：{timing}")
    lines.append(f"上下文：{len(result.blocks)} 块 / {result.context_tokens} token"
                 f"｜候选 {result.candidates} 条｜各路命中 {result.route_hits}")
    return "\n".join(lines)
