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


def source_rows(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """上下文块 → 来源行（CLI 打印、JSON 输出、流式 `sources` 事件共用一份映射）。"""
    return [
        {
            "index": block.get("index"),
            "document_id": block.get("document_id"),
            "parent_ids": list(block.get("parent_ids") or []),
            "parent_indexes": list(block.get("parent_indexes") or []),
            "section_path": list(block.get("section_path") or []),
            "source_uri": block.get("source_uri"),
            "title": block.get("title"),
        }
        for block in blocks
    ]


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
    #: 检索降级信息（缺向量模型 / 某一路故障）。`None` = 全部正常
    degraded: Any = None

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
            "sources": source_rows(self.sources),
            "timings": {key: round(value, 1) for key, value in self.timings.items()},
            "route_hits": dict(self.route_hits),
            "llm_called": self.llm_called,
            "degraded": self.degraded.as_dict() if self.degraded else None,
        }


@dataclass
class RetrievalOutcome:
    """检索段的结果 + **降级标记**。`degraded` 非空表示这次少了几路。"""

    hits: list[RerankHit] = field(default_factory=list)
    candidates: int = 0
    route_hits: dict[str, int] = field(default_factory=dict)
    degraded: Any = None


def retrieve_ranked(question: str, *, retriever, parents: Mapping[str, Mapping[str, Any]],
                    counter, reranker, pool: int = DEFAULT_POOL,
                    rerank: bool = True) -> tuple[list[RerankHit], int, dict[str, int]]:
    """三路召回 → RRF → 重排。返回 (候选列表, 融合候选数, 每路命中数)。

    `rerank=False` 时**直接返回 RRF 顺序**，不碰 encoder：
    HTTP 层要能按请求关重排（`{"rerank": false}`），而不去改变共享的 reranker 状态
    ——那个方向一改就会在多请求下互相踩。
    """
    outcome = retrieve_with_meta(question, retriever=retriever, parents=parents, counter=counter,
                                 reranker=reranker, pool=pool, rerank=rerank)
    return outcome.hits, outcome.candidates, outcome.route_hits


def _parent_text(parent: Mapping[str, Any] | None) -> str:
    """父块正文：`text` 与 `content` 两个键都认（N45）。

    为什么不能只认 `text`：`store_wiring.load_parent_records()`（生产链路）给的是 `text`，
    而 `scripts/eval_bench.py` 的 `load_parents()` 直接塞 SQL 行、键是 `content`。
    只认 `text` 时评测里每个候选的窗口都是空串 —— cross-encoder 对 40 个空串打分全等，
    稳定排序保持原序，**重排等于没跑**，而报告的配置里写着「含重排」。
    取不到正文就退化成空串，等于悄悄关掉一条路，这种失败方式最难发现。
    """
    if not parent:
        return ""
    return parent.get("text") or parent.get("content") or ""


def _block_lang(parent: Mapping[str, Any] | None) -> str:
    """候选父块的语言。

    `parents.language` 这一列在历史库里全是 NULL（N49），所以再用正文判一次 ——
    语言判错只会让这一块的 query 选错，而「一律当中文」会让 C0d 完全失效（这一次就踩了：
    第一版 by_lang 与 question 跑出来**逐位相同**，因为每个块都拿到 None）。
    正文也判不出来时才退成中文（中文 query 是原来的行为，不会更差）。
    """
    from ..lang.detect import resolve_language

    item = parent or {}
    lang = str(resolve_language(item.get("language"), item.get("text") or item.get("content"))
               or "").lower()
    return "zh" if lang.startswith("zh") else ("en" if lang.startswith("en") else "zh")


def rerank_by_language(hits: Sequence[RerankHit], parents: Mapping[str, Mapping[str, Any]], *,
                       query_zh: str, query_en: str | None, reranker, counter,
                       merge: str = "raw") -> list[RerankHit]:
    """按**候选块自己的语言**挑重排 query（C0d）。

    C0c 的实测：中→英 的漏检在融合里排 #1/#3/#4/#8/#14，重排后掉到 #16/#17/#31/#13/#14 ——
    因为重排拿中文原句给英文窗口打分。cross-encoder 不是跨语言模型，它只看「这段像不像在回答这句」。
    但整题切换成英文又会把中文证据打下去（x-010 #13→#19），所以只能**逐候选**按块语言选。

    实现上分组调用 `rerank_hits`（每组一个 query），再把分数合起来统一排序：
    好处是不必改那个已经稳定的核心函数，重排的并列/异常/截断语义保持一模一样。
    """
    grouped: dict[str, list[RerankHit]] = {}
    for hit in hits:
        hit.fields["rerank_query"] = ("zh" if query_en is None else _block_lang(parents.get(hit.parent_id)))
        grouped.setdefault(hit.fields["rerank_query"], []).append(hit)

    for lang, group in grouped.items():
        query = query_zh if lang == "zh" or query_en is None else query_en
        for hit in group:
            hit.fields["window"] = best_window(hit.text, query,
                                              max_tokens=reranker.config.max_length,
                                              counter=counter)
        rerank_hits(query, group, encoder=reranker.encoder, config=reranker.config,
                    top_n=len(group), stats=reranker.stats,
                    text_getter=lambda hit: hit.fields["window"])
        # 两组分数来自两个不同的 query，必须先各自归一化才能跨组比大小（见 normalize_group_scores）
        normalize_group_scores(group, merge=merge)
        for hit in group:
            hit.rerank_score = float(hit.fields["norm_score"] or 0.0)

    return sorted(hits, key=lambda hit: (-(hit.rerank_score or 0.0), hit.rank, hit.chunk_id))


def normalize_group_scores(hits: Sequence[RerankHit], *, merge: str = "raw") -> list[RerankHit]:
    """把**同一组**（同一种语言、同一个 query）的分数换成跨组可比的量（C0e-2）。

    `by_lang` 给两种语种各配一个 query，于是两组的 `rerank_score` 来自两个不同的问题：
    cross-encoder 的原始分是「这段像不像在回答**这句**」，**跨 query 不可比**。
    直接混起来排序，就是让一组的 logits 尺度决定谁进 top-k —— 这也是 C0d
    「中→英 涨、中→中 掉」的直接原因，与「名额」无关（配额的实测就不赚）。

    三种口径：
      * `raw`    —— 不动（基线口径，保留作回归对照）；
      * `zscore` —— 组内标准化，保留「比组内均值高多少」的信息；
      * `rank`   —— 组内百分位，只留名次，对分数尺度完全不敏感。

    归一化后的值写进 `fields["norm_score"]`，**原始 `rerank_score` 一律不动**（诊断要用）；
    仅当 `merge != "raw"` 时改写 `rerank_score` 本身，因为下游 `rerank_hits` 与报告都读它。
    """
    if merge == "raw":
        for hit in hits:
            hit.fields["norm_score"] = hit.rerank_score
        return list(hits)
    if merge not in ("zscore", "rank"):
        raise ValueError(f"未知的合并口径：{merge}（可选 raw / zscore / rank）")

    scores = [float(hit.rerank_score or 0.0) for hit in hits]
    count = len(scores)
    if count == 0:
        return []
    if count == 1:
        hits[0].fields["norm_score"] = 0.0 if merge == "zscore" else 1.0
        return list(hits)

    if merge == "zscore":
        mean = sum(scores) / count
        variance = sum((value - mean) ** 2 for value in scores) / count
        spread = variance ** 0.5
        normalized = [0.0 if spread == 0 else (value - mean) / spread for value in scores]
    else:
        order = sorted(range(count), key=lambda i: (-scores[i], hits[i].rank))
        normalized = [0.0] * count
        for position, index in enumerate(order, start=1):
            normalized[index] = 1.0 - (position - 1) / count

    for hit, value in zip(hits, normalized):
        hit.fields["norm_score"] = value
    return list(hits)


def _question_lang(question: str | None) -> str:
    """提问语言（只用来决定「谁算对方语言」）。判不出来按中文——与 `_block_lang()` 的兜底一致。"""
    from ..lang.detect import resolve_language

    lang = str(resolve_language(None, question or "") or "").lower()
    return "zh" if lang.startswith("zh") else "en"


def select_with_language_quota(hits: Sequence[RerankHit], parents: Mapping[str, Mapping[str, Any]],
                               *, k: int, min_per_lang: int = 0,
                               max_other_lang: int = 0,
                               question: str | None = None) -> list[RerankHit]:
    """从已排序的候选里挑 `k` 条，每种语言先占住 `min_per_lang` 个名额（C0e）。

    为什么是「名额」不是「加权」：池子里中英混排，重排给谁高分谁就进 top-k。
    C0d 把英文候选抬起来，中→英 的 MRR 涨了（0.323→0.471），中→中 却掉了（0.636→0.508）——
    两种语言在**同一张名单**上抢位置，抬一边必然压另一边（N50）。

    所以这里只改「谁进名单」：每种语言按自己的分数序占住若干名额，剩下的名额仍按全局序填。
    **名单内的相对顺序一个字都不动**，`hit@k` / `MRR` / rank 分布的口径与原来完全一致。

    `max_other_lang` 是同一个想法的镜像（C0h）：限制**与提问语言不同**的候选最多占几个。
    实测（`docs/evidence/c0e-language-quota-k8s-v1.json`）最少名额在这份题集上从不触发
    —— 逐题 0/144 变化，因为 top-10 本来就是中英混排、两种语言都有位置（31 道跨语言题 100% 如此）。
    真正会被挤掉的是「同语言池子里的中文 gold 被英文块顶下去」，所以限制对方语言才有意义。

    `min_per_lang=0`（且 `max_other_lang=0`）或池子里只有一种语言时，本函数与 `hits[:k]` **逐位相同**。
    """
    if k <= 0 or not hits:
        return []
    if min_per_lang <= 0 and max_other_lang <= 0:
        return list(hits[:k])

    grouped: dict[str, list[RerankHit]] = {}
    for hit in hits:
        grouped.setdefault(_block_lang(parents.get(hit.parent_id)), []).append(hit)
    if len(grouped) <= 1:                       # 单语言池：配额无事可做
        return list(hits[:k])

    position_of = {id(hit): index for index, hit in enumerate(hits)}
    banned: set[int] = set()

    if max_other_lang > 0 and question is not None:      # C0h：对方语言超出上限的候选先剔掉
        other = "en" if _question_lang(question) == "zh" else "zh"
        seen = 0
        for hit in hits:
            if _block_lang(parents.get(hit.parent_id)) != other:
                continue
            seen += 1
            if seen > max_other_lang:
                banned.add(id(hit))

    reserved: list[RerankHit] = []
    for group in grouped.values():              # 按「该语言最好名次」的先后占位
        for hit in group[:min_per_lang]:
            if len(reserved) >= k or id(hit) in banned:
                break
            reserved.append(hit)
    chosen = {id(hit) for hit in reserved}

    result: list[RerankHit] = list(reserved)
    for hit in hits:                            # 剩下的名额按全局分数序填满
        if len(result) >= k:
            break
        if id(hit) in chosen or id(hit) in banned:
            continue
        result.append(hit)
    # 名单内顺序必须跟**输入顺序**（= 重排后的顺序），不能按 `rank`（RRF 名次）重排：
    # 重排已经把好候选挪到前面了，按 RRF 名次重排等于把重排结果丢掉（踩过一次）。
    result.sort(key=lambda hit: position_of[id(hit)])
    return result[:k]


def retrieve_with_meta(question: str, *, retriever, parents: Mapping[str, Mapping[str, Any]],
                       counter, reranker, pool: int = DEFAULT_POOL,
                       rerank: bool = True,
                       query_en: str | None = None,
                       rerank_query: str = "question",
                       rerank_merge: str = "raw",
                       k: int | None = None,
                       lang_quota: int = 0,
                       max_other_lang: int = 0) -> RetrievalOutcome:
    """`retrieve_ranked()` 的带元数据版本：多返回 `degraded`（缺能力 / 某路故障）。

    `query_en` 是翻译服务给出的英文 query，**必须转交给编排层**（N47）：
    编排层明文规定 `dense_en` / `lexical_en` 吃英文 query，缺了它会回退中文原句并标记为噪声源。
    之前这个参数根本不存在，于是翻译出来的英文只用在提示词语言和响应回显上，
    英文索引实际一直在吃中文原句。
    """
    result = retriever.retrieve(question, k=pool, query_en=query_en)
    degraded = getattr(result, "degraded", None)
    routes = {name: list(route.ids)[:pool] for name, route in result.routes.items()}
    fused = rrf_fuse_routes(routes, rrf_k=RRF_K, top_k=pool)

    hits: list[RerankHit] = []
    for rank, hit in enumerate(fused, start=1):
        text = _parent_text(parents.get(hit.parent_id))
        item = RerankHit(chunk_id=hit.parent_id, parent_id=hit.parent_id, text=text,
                         score=hit.score, rank=rank)
        hits.append(item)

    route_hits = {name: len(ids) for name, ids in routes.items()}
    if not rerank:
        if k is not None and k > 0:
            hits = select_with_language_quota(hits, parents, k=k, min_per_lang=lang_quota,
                                              max_other_lang=max_other_lang, question=question)
        return RetrievalOutcome(hits=hits, candidates=len(fused), route_hits=route_hits,
                                degraded=degraded)

    if rerank_query == "by_lang":
        ranked = rerank_by_language(hits, parents, query_zh=question, query_en=query_en,
                                    reranker=reranker, counter=counter, merge=rerank_merge)
    elif rerank_query == "question":
        for item in hits:
            # 父块 p50 约 700 token，直接截 256 只看得到标题 → 先按 query 挑窗口再重排
            item.fields["window"] = best_window(item.text, question,
                                                max_tokens=reranker.config.max_length,
                                                counter=counter)
        ranked = rerank_hits(question, hits, encoder=reranker.encoder, config=reranker.config,
                             top_n=pool, stats=reranker.stats,
                             text_getter=lambda hit: hit.fields["window"])
    else:
        raise ValueError(f"未知的 rerank_query 口径：{rerank_query}（可选 question / by_lang）")
    ranked = ranked[:pool] if pool else ranked
    if k is not None and k > 0:
        ranked = select_with_language_quota(ranked, parents, k=k, min_per_lang=lang_quota,
                                            max_other_lang=max_other_lang, question=question)
    return RetrievalOutcome(hits=ranked, candidates=len(fused),
                            route_hits=route_hits, degraded=degraded)


def answer_query(
    question: str,
    *,
    retriever,
    parents: Mapping[str, Mapping[str, Any]],
    counter,
    reranker,
    ask: Callable[[list[dict[str, str]]], str] | None,
    query_en: str | None = None,
    rerank_query: str = "question",
    rerank_merge: str = "raw",
    lang_quota: int = 0,
    max_other_lang: int = 0,
    k: int = DEFAULT_K,
    pool: int = DEFAULT_POOL,
    budget: int = DEFAULT_BUDGET,
    rerank: bool = True,
) -> AskResult:
    """跑完整条链路。`ask=None` 表示「只检索不生成」（`--retrieve-only`）。"""
    timings: dict[str, float] = {}

    started = time.perf_counter()
    outcome = retrieve_with_meta(question, retriever=retriever, parents=parents, counter=counter,
                                 reranker=reranker, pool=pool, rerank=rerank,
                                 query_en=query_en, rerank_query=rerank_query,
                                 rerank_merge=rerank_merge,
                                 k=k, lang_quota=lang_quota, max_other_lang=max_other_lang)
    ranked, candidates, route_hits = outcome.hits, outcome.candidates, outcome.route_hits
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
        degraded=outcome.degraded,
    )


def render_cli(result: AskResult, *, show_sources: int = 5, max_answer_chars: int = 4000,
               list_blocks: bool = False) -> str:
    """终端输出：答案 → 引用 → 来源表 → 耗时。给人看的，不复用 `AssembledContext.render()`。"""
    lines: list[str] = []
    if result.degraded is not None:
        lines.append(result.degraded.render().strip())
        lines.append("")
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


# -- 流式（O4a）------------------------------------------------------------

#: 流式事件名。HTTP 层（O4c）直接把它映射成 SSE 的 `event:` 名，不另起一套词表
STREAM_EVENTS = ("sources", "delta", "done", "error")


def answer_query_stream(
    question: str,
    *,
    retriever,
    parents: Mapping[str, Mapping[str, Any]],
    counter,
    reranker,
    ask: Callable[[list[dict[str, str]]], Any] | None,
    query_en: str | None = None,
    k: int = DEFAULT_K,
    pool: int = DEFAULT_POOL,
    budget: int = DEFAULT_BUDGET,
    rerank: bool = True,
):
    """流式版 `answer_query()`：先吐 `sources`，再吐 `delta`，最后 `done`。

    与同步版的关系：**同一条链路、同一套组装与引用体检**，只是把生成段拆成增量。
    没有复用 `answer_query()` 是因为它必须"等全文"才能解析引用，而流式的价值就在
    那 50 s 里先把来源给出去。

    三个不变量（`tests/test_streaming.py` 盯着）：
    1. `sources` 一定先于任何 `delta`；
    2. `done.answer == "".join(delta.text)`（**当且仅当真的调了模型**）；
    3. 调用方 `close()` 会通过 `finally` 关掉上游生成器 —— SSE 客户端断开时不能白烧算力。

    弃答（无证据）与只检索两条捷径都**不产生 delta**：前者直接给结论，后者根本没有正文。
    """
    timings: dict[str, float] = {}

    started = time.perf_counter()
    outcome = retrieve_with_meta(
        question, retriever=retriever, parents=parents, counter=counter,
        reranker=reranker, pool=pool, rerank=rerank,
    )
    ranked, candidates, route_hits = outcome.hits, outcome.candidates, outcome.route_hits
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

    blocks = [block.as_dict() for block in assembled.blocks]
    yield {
        "event": "sources",
        "data": {
            "blocks": len(blocks),
            "context_tokens": assembled.tokens,
            "candidates": candidates,
            "route_hits": dict(route_hits),
            "language": bundle.language,
            # 降级信息**必须随 sources 早发**：这是客户端决定要不要提示「结果可能不全」的唯一依据
            "degraded": outcome.degraded.as_dict() if outcome.degraded else None,
            "sources": source_rows(blocks),
        },
    }

    answer_parts: list[str] = []
    first_delta_ms: float | None = None
    called = False
    timings["generate_ms"] = 0.0

    if bundle.abstained:                      # 没有证据 → 不调模型（D33），也不发 delta
        answer_parts.append(bundle.messages[0]["content"])
    elif ask is not None:
        started = time.perf_counter()
        stream = ask(bundle.messages)
        called = True
        try:
            for piece in stream:
                if not piece:
                    continue
                if first_delta_ms is None:
                    first_delta_ms = (time.perf_counter() - started) * 1000
                answer_parts.append(piece)
                yield {"event": "delta", "data": {"text": piece}}
        except Exception as exc:              # noqa: BLE001 - 生成中途失败要保住半截
            timings["generate_ms"] = (time.perf_counter() - started) * 1000
            yield {
                "event": "error",
                "data": {
                    "detail": f"{type(exc).__name__}: {exc}",
                    "retryable": True,
                    "partial": "".join(answer_parts),
                    "timings": {key: round(value, 1) for key, value in timings.items()},
                },
            }
            return
        finally:
            # 消费者 close() 时这里会把上游生成器也关掉（Ollama 连接随之释放）
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        timings["generate_ms"] = (time.perf_counter() - started) * 1000

    answer = "".join(answer_parts)
    report = parse_citations(answer, k=len(blocks) or None)
    timings["first_delta_ms"] = first_delta_ms or 0.0
    result = AskResult(
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
        degraded=outcome.degraded,
    )
    yield {"event": "done", "data": result.as_dict()}
