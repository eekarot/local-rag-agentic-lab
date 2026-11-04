"""一条 query 的端到端逐段追踪（讲解/排查用，不是产品组件）。

    python -m experiments.trace_query "智能体化 RAG 和传统 RAG 有什么区别？"

每一段都把「输入 → 输出」打出来，便于对照文档 §4 的阶段划分：
    语言识别 → 翻译 → 三路召回 → RRF → 重排 → 上下文组装 → prompt → 生成 → 引用体检
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.answer.citations import parse_citations, resolve_sources
from src.rag.answer.prompt import build_messages
from src.rag.context.assembler import assemble
from src.rag.ingestion.tokenizer import TokenCounter
from src.rag.lang.detect import detect, language_ratio
from src.rag.retrieval.fusion import RRF_K, rrf_fuse_routes
from src.rag.retrieval.rerank import CrossEncoderReranker, RerankHit, best_window, rerank_hits
from src.rag.retrieval.store_wiring import build_default_retriever, load_parent_records
from src.config import OLLAMA_CHAT_MODEL, STORAGE_DIR
from src.rag.translate.cache import TranslationCache
from src.rag.translate.service import TranslateService


def rule(title: str) -> None:
    print("\n" + "=" * 78 + f"\n## {title}\n" + "=" * 78)


def main(argv: list[str]) -> int:
    question = argv[1] if len(argv) > 1 else "智能体化 RAG 和传统 RAG 有什么区别？"
    pool, k = 20, 10
    parents = {row["id"]: row for row in load_parent_records()}
    counter = TokenCounter()

    rule("① 用户提问")
    print("question:", question)

    rule("② 语言识别（src/rag/lang/detect.py）")
    t0 = time.perf_counter()
    lang, ratio = detect(question), language_ratio(question)
    print(f"输入: {question!r}")
    print(f"输出: language={lang}  ratio={ratio}  (汉字/(汉字+英文词)，阈值 0.20)  {1000*(time.perf_counter()-t0):.1f} ms")

    rule("③ Query 翻译（src/rag/translate/service.py，中文 → 英文）")
    t0 = time.perf_counter()
    translator = TranslateService(
        cache=TranslationCache(STORAGE_DIR / "cache" / "translations.sqlite3",
                              model=OLLAMA_CHAT_MODEL, version="v1"))
    result = translator.translate([question])[0]
    query_en = result.text if result.status in {"ok", "cached"} else None
    print(f"输入: {question!r}")
    print(f"输出: status={result.status} attempts={result.attempts} missing_terms={result.missing}")
    print(f"      query_en={query_en!r}")
    print(f"      {1000*(time.perf_counter()-t0):.0f} ms")

    rule("④ 三路召回（src/rag/retrieval/routes.py，每路深度 = pool = %d）" % pool)
    retriever = build_default_retriever(warmup_texts=[question, query_en or ""])
    t0 = time.perf_counter()
    retrieval = retriever.retrieve(question, query_en=query_en, k=pool, fuse=False)
    for name, route in retrieval.routes.items():
        print(f"\n--- route={name}  enabled={route.enabled}  query_used={route.query_used!r}")
        print(f"    latency={route.latency_ms:.0f} ms  candidates={route.candidates}")
        for hit in route.hits[:5]:
            section = (parents.get(hit.parent_id) or {}).get("section_path") or ""
            print(f"    rank {hit.rank:2d}  score {hit.score:.4f}  {hit.parent_id}  {section[:46]}")
        if not route.hits:
            print(f"    note={route.note!r}")
    print(f"\n三路总耗时 {1000*(time.perf_counter()-t0):.0f} ms")

    rule("⑤ RRF 融合（src/rag/retrieval/fusion.py，1/(60+rank) 跨路累加）")
    routes = {name: route.ids for name, route in retrieval.routes.items() if route.enabled}
    fused = rrf_fuse_routes(routes, rrf_k=RRF_K, top_k=pool)
    print("输入: 三路 parent_id 列表（各路深度 %d）" % pool)
    print("输出: 融合列表")
    for rank, hit in enumerate(fused[:10], start=1):
        parts = " + ".join(f"{src}:1/{RRF_K}+{r}" for src, r in hit.per_route)
        print(f"  {rank:2d}  {hit.parent_id}  rrf={hit.score:.6f}  路数={hit.n_routes}  {parts}")

    rule("⑥ 重排（src/rag/retrieval/rerank.py：ONNX cross-encoder，池 %d → top %d）" % (pool, k))
    reranker = CrossEncoderReranker()
    hits = []
    for rank, hit in enumerate(fused, start=1):
        text = (parents.get(hit.parent_id) or {}).get("text", "")
        item = RerankHit(chunk_id=hit.parent_id, parent_id=hit.parent_id, text=text, score=hit.score, rank=rank)
        item.fields["window"] = best_window(text, question, max_tokens=reranker.config.max_length,
                                            counter=counter)
        hits.append(item)
    print("输入: 池内 %d 条父块（每条先用 best_window 挑出 ≤%d token 的窗口）"
          % (len(hits), reranker.config.max_length))
    print("  示例窗口（池里第一条，前 80 字）:", hits[0].fields["window"][:80].replace("\n", " "))
    t0 = time.perf_counter()
    ranked = rerank_hits(question, hits, encoder=reranker.encoder, config=reranker.config,
                         top_n=k, stats=reranker.stats,
                         text_getter=lambda item: item.fields["window"])
    elapsed = 1000 * (time.perf_counter() - t0)
    print(f"\n输出: 重排后 top {len(ranked)}（耗时 {elapsed:.0f} ms，stats={reranker.stats})")
    print("  旧名次 → 新名次")
    for new_rank, hit in enumerate(ranked, start=1):
        section = ((parents.get(hit.parent_id) or {}).get("section_path") or [""])[-1]
        print(f"  {hit.rank:2d} → {new_rank:2d}  rerank={hit.rerank_score:+.4f}  rrf={hit.score:.6f}  "
              f"{hit.parent_id}  {section[:40]}")

    rule("⑦ 上下文组装（src/rag/context/assembler.py：去重 + 相邻合并 + 预算 6000）")
    assembled = assemble([{"parent_id": h.parent_id, "score": h.score, "rerank_score": h.rerank_score}
                          for h in ranked], parents, counter=counter, budget=6000)
    print(f"输入: {len(ranked)} 条重排命中")
    print(f"输出: {len(assembled.blocks)} 块 / {assembled.tokens} tokens / 丢弃 {len(assembled.dropped)} 块")
    print("stats:", assembled.stats)
    for block in assembled.blocks:
        print(f"  [证据 {block.index}] {block.tokens:4d} tok  {block.document_id[:12]}… "
              f"idx={block.parent_indexes} {(block.section_path or [''])[-1][:38]}")

    rule("⑧ Prompt 构造（src/rag/answer/prompt.py）")
    bundle = build_messages(question, assembled)
    print(f"abstained={bundle.abstained}  reason={bundle.reason!r}  messages={len(bundle.messages)}")
    print(f"--- system（{len(bundle.messages[0]['content'])} 字符）前 400 字 ---")
    print(bundle.messages[0]["content"][:400])
    print(f"--- user（{len(bundle.messages[1]['content'])} 字符）前 600 字 ---")
    print(bundle.messages[1]["content"][:600])

    rule("⑨ 生成（src/ollama_client.py → qwen3.5:9b）")
    from src.ollama_client import chat

    t0 = time.perf_counter()
    answer = chat(bundle.messages, fmt=None)
    llm_ms = 1000 * (time.perf_counter() - t0)
    print(f"model={OLLAMA_CHAT_MODEL}  延迟 {llm_ms:.0f} ms  答案 {len(answer)} 字符")
    print("--- answer ---")
    print(answer)

    rule("⑩ 引用体检（src/rag/answer/citations.py）")
    report = parse_citations(answer, k=len(assembled.blocks) or None)
    resolved = resolve_sources(report, [b.as_dict() for b in assembled.blocks])
    print(f"cited={report.cited}  out_of_range={report.out_of_range}  malformed={report.malformed}  "
          f"parse_rate={report.parse_rate:.2f}  abstained={report.abstained}")
    for block in resolved:
        print(f"  [证据 {block['index']}] → {block['document_id']} idx={block['parent_indexes']} "
              f"{block['source_uri']}  {(block['section_path'] or [''])[-1][:40]}")
    print(f"\n总耗时（不含首次模型加载）≈ {llm_ms/1000:.1f} s（生成占绝大部分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
