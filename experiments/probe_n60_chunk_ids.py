"""N60 证据探针：`ContextBlock.chunk_ids` 到底是子块 id 还是父块 id？

要回答的是「拿它去回查能不能查到东西」，所以判据必须落在**库里**：

- 子块 id 在 Document Store 的 `children` 表里，也正是 Qdrant 点的 payload `chunk_id`；
- 父块 id 只在 `parents` 表里，**Qdrant 里没有任何一个点是父块**。

于是同一个问题跑两遍口径：
  1. 现在的代码 → `chunk_ids`（应当是子块 id）；
  2. 修之前的行为 → 把 `parent_ids` 当成 `chunk_ids`；
分别去 `children` 表查「有多少条能对上」。第 2 种按构造必然是 0 —— 这不是猜，
`parents.parent_id` 与 `children.chunk_id` 是两个不相交的 id 空间。

跑法（只读，不写库；约 10 s）：
    python experiments/probe_n60_chunk_ids.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_bench import BENCH_STORE          # noqa: E402
from src.rag.ask.pipeline import answer_query     # noqa: E402
from src.rag.ingestion.tokenizer import TokenCounter  # noqa: E402
from src.rag.retrieval.store_wiring import load_parent_records  # noqa: E402
from src.rag.ask.pipeline import DEFAULT_POOL, DEFAULT_K  # noqa: E402

COLLECTION = "bench_k8s_v1"
DEST = ROOT / "docs" / "evidence" / "n60-chunk-ids-k8s-v1.json"

QUESTIONS = [
    ("zh", "topology manager 是怎么决定要不要重调度的？"),
    ("en", "How does the topology manager decide whether to reschedule a pod?"),
]


def _child_ids_in_store(store_path: Path) -> set[str]:
    from src.rag.store.document_store import DocumentStore

    store = DocumentStore(store_path)
    with store._connect() as conn:      # noqa: SLF001  同包内部
        return {row["chunk_id"] for row in conn.execute("SELECT chunk_id FROM children")}


def main() -> int:
    from src.rag.retrieval.store_wiring import build_default_retriever

    log = lambda message: print(f"  {message}", file=sys.stderr)  # noqa: E731
    parents = {row["id"]: row for row in load_parent_records(BENCH_STORE)}
    retriever = build_default_retriever(collection=COLLECTION, store_path=BENCH_STORE,
                                        warmup_texts=[q for _, q in QUESTIONS], log=log)
    counter = TokenCounter()
    children = _child_ids_in_store(BENCH_STORE)

    per_question = []
    for lang, question in QUESTIONS:
        result = answer_query(question, retriever=retriever, parents=parents, counter=counter,
                              reranker=None, ask=None, rerank=False, k=DEFAULT_K,
                              pool=DEFAULT_POOL)
        chunk_ids = [cid for block in result.blocks for cid in block["chunk_ids"]]
        parent_ids = [pid for block in result.blocks for pid in block["parent_ids"]]
        hit_children = [cid for cid in chunk_ids if cid in children]
        parent_as_child = [pid for pid in parent_ids if pid in children]
        #: `chunk_ids` 里其实还是父块 id 的那部分（只有词面路命中、没有子块 id 可退）
        fell_back = [cid for cid in chunk_ids if cid in parents]
        per_question.append({
            "lang": lang,
            "question": question,
            "blocks": len(result.blocks),
            "chunk_ids": chunk_ids,
            "parent_ids": parent_ids,
            "chunk_ids_resolvable": len(hit_children),
            "chunk_ids_fell_back_to_parent": len(fell_back),
            "chunk_ids_total": len(chunk_ids),
            "parent_ids_resolvable_as_chunk": len(parent_as_child),
            "parent_ids_total": len(parent_ids),
            "chunk_ids_from_parent_stat": None,
        })
        print(f"[{lang}] 块 {len(result.blocks)}｜chunk_ids 可回查 {len(hit_children)}/{len(chunk_ids)}"
              f"（其中 {len(fell_back)} 个是词面路独占、退回父块 id）"
              f"｜父块 id 冒充子块时可回查 {len(parent_as_child)}/{len(parent_ids)}")

    totals = {
        "chunk_ids": sum(item["chunk_ids_total"] for item in per_question),
        "chunk_ids_resolvable": sum(item["chunk_ids_resolvable"] for item in per_question),
        "chunk_ids_fell_back_to_parent": sum(item["chunk_ids_fell_back_to_parent"]
                                            for item in per_question),
        "parent_ids": sum(item["parent_ids_total"] for item in per_question),
        "parent_ids_resolvable_as_chunk": sum(item["parent_ids_resolvable_as_chunk"]
                                             for item in per_question),
    }
    payload = {
        "note": "N60：`ContextBlock.chunk_ids` 必须是子块 id（Qdrant 点的 payload.chunk_id），不是父块 id",
        "collection": COLLECTION,
        "store": str(BENCH_STORE.relative_to(ROOT)).replace("\\", "/"),
        "judgement": "拿 id 去 `children` 表查（= Qdrant 点的 id 空间）",
        "per_question": per_question,
        "totals": totals,
        "finding": (
            "修之前 `chunk_ids = parent_ids`，能回查的比例按构造是 0 —— "
            "引用只能指到「父块」这个 Store 内部 id，下游拿它去 Qdrant 取正文必然取不到。"
            "修之后 chunk_ids 是 dense 路真正命中的子块（可回查比例见 totals）；"
            "只有「词面路独占命中」的父块仍退回父块 id —— 词面索引建在父块上，本来就没有子块 id，"
            "这部分在 `assembled.stats['chunk_ids_from_parent']` 里留痕，不静默。"
        ),
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n结果：{DEST.relative_to(ROOT)}")
    print(f"合计：chunk_ids 可回查 {totals['chunk_ids_resolvable']}/{totals['chunk_ids']}；"
          f"父块 id 当子块用 {totals['parent_ids_resolvable_as_chunk']}/{totals['parent_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
