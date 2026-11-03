"""D14：`embedding_text` 前缀统计 + 「有前缀 / 无前缀」A/B 报告。

方案 §2.8 把「送进向量模型的文本」和「留给 LLM 的原文」分开：

    embedding_text = "Document: {title}\\nSection: {a / b}\\nContent: {正文}"
    content        = 正文

这样做的代价是**向量对应的是带前缀的文本**，好处是子块脱离父块也不会丢掉
「它属于哪篇文档、哪一节」这两个信息——偏偏这两条对「跨语言、问某文档某节」
的查询最有区分度。

D14 要回答三件事：

1. 两条字段是否都真的存在（`content` + `embedding_text`）；
2. 前缀缺失能不能统计（缺了会静默变差，必须可度量）；
3. **有前缀到底有没有用**——小样本 A/B，报 Recall@k 差值，而不是"我觉得有用"。

第 3 点是重点。`embedding_text` 可以完全由 `title / section_path / content`
重建（不落库也能还原），所以 A/B 的做法是：同一批子块建**两个** collection，
一个用 `embedding_text`、一个用纯 `content`，同一批 query 各查一遍，按父块算 Recall@k。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .parent_builder import build_embedding_text

#: 前缀行的标识。只要以其中之一开头就认为"带前缀"
PREFIX_LABELS: tuple[str, ...] = ("Document:", "Section:")

#: A/B 的两个变体
VARIANTS: tuple[str, ...] = ("prefixed", "plain")


# ---------------------------------------------------------------------------
# 前缀统计
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefixStats:
    total: int
    with_prefix: int
    without_prefix: int
    #: 缺失原因 → 条数。能区分原因才知道该怎么修
    reasons: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "with_prefix": self.with_prefix,
            "without_prefix": self.without_prefix,
            "prefix_ratio": round(self.with_prefix / self.total, 4) if self.total else None,
            "reasons": dict(self.reasons),
        }


def _missing_reason(record: Mapping[str, Any]) -> str:
    text = record.get("embedding_text")
    if text is None:
        return "missing_field"
    if text == record.get("content"):
        return "identical_to_content"
    if not text.lstrip().startswith(PREFIX_LABELS):
        return "content_only"
    return ""


def prefix_stats(records: Iterable[Mapping[str, Any]]) -> PrefixStats:
    """统计 `embedding_text` 里前缀的覆盖情况。"""
    total = with_prefix = 0
    reasons: dict[str, int] = {}
    for record in records:
        total += 1
        reason = _missing_reason(record)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            with_prefix += 1
    return PrefixStats(
        total=total,
        with_prefix=with_prefix,
        without_prefix=total - with_prefix,
        reasons=reasons,
    )


def reconstruction_stats(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """`embedding_text` 能否由 `title/section_path/content` 重建。

    这条决定「要不要把 `embedding_text` 落库」：能 100% 重建就不必存（多存一份
    原文会让索引体积涨 ~20%），A/B 时也能现算。
    """
    total = mismatched = 0
    examples: list[dict[str, Any]] = []
    for record in records:
        total += 1
        expected = build_embedding_text(
            record.get("title"), record.get("section_path") or [], record.get("content") or ""
        )
        if expected != record.get("embedding_text"):
            mismatched += 1
            if len(examples) < 3:
                examples.append(
                    {
                        "chunk_id": record.get("chunk_id"),
                        "expected": expected[:200],
                        "actual": str(record.get("embedding_text"))[:200],
                    }
                )
    return {"total": total, "mismatched": mismatched, "reconstructable_ratio": round(
        1 - mismatched / total, 4) if total else None, "examples": examples}


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def _dedupe(items: Iterable[Any]) -> list[Any]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def query_metrics(
    retrieved_parent_ids: Sequence[str | None],
    gold_parent_ids: Iterable[str],
    *,
    k: int | None = None,
) -> dict[str, float | int | None]:
    """按**父块**算 Recall@k / hit@k / MRR。

    分母是父块而不是子块：一个父块被切成 3 个子块，命中其中 2 个不该算 2/1 的
    超额分，也不该重复计数。`retrieved_parent_ids` 按名次排列（含重复、含 None），
    `k` 给定时只取前 k 条（与检索的 `limit=k` 对齐）。
    """
    if k is not None:
        retrieved_parent_ids = list(retrieved_parent_ids)[:k]
    gold = set(gold_parent_ids)
    ranked = _dedupe(pid for pid in retrieved_parent_ids if pid)
    matched = [pid for pid in ranked if pid in gold]
    if not gold:
        # 该题的可判定金标不在本次索引范围内 → 不参与均值，而不是算 0 分拉低结论
        return {"recall": None, "hit": None, "mrr": None, "matched": 0}
    mrr = 0.0
    for rank, pid in enumerate(ranked, start=1):
        if pid in gold:
            mrr = round(1 / rank, 4)
            break
    return {
        "recall": round(len(matched) / len(gold), 4),
        "hit": 1 if matched else 0,
        "mrr": mrr,
        "matched": len(matched),
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def ab_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """把逐题结果汇总成两种变体的均值与差值。"""
    judged = [row for row in rows if row.get("gold")]
    summary: dict[str, Any] = {
        "queries": len(rows),
        "judged": len(judged),
        "k": rows[0].get("k") if rows else None,
    }
    for variant in VARIANTS:
        metrics = [
            query_metrics(row.get(variant) or [], row.get("gold") or set(), k=row.get("k"))
            for row in judged
        ]
        summary[variant] = {
            key: _mean(m[key] for m in metrics) for key in ("recall", "hit", "mrr")
        }
    summary["delta"] = {
        key: (round(summary["prefixed"][key] - summary["plain"][key], 4)
              if summary["prefixed"][key] is not None and summary["plain"][key] is not None else None)
        for key in ("recall", "hit", "mrr")
    }
    return summary


# ---------------------------------------------------------------------------
# A/B 主体（需要 Qdrant）
# ---------------------------------------------------------------------------


def _upsert(client, collection: str, chunks: Sequence[Mapping[str, Any]], matrix, batch_size: int) -> None:
    from qdrant_client import models

    from ..store.qdrant_store import point_id_for

    total = len(chunks)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        points = [
            models.PointStruct(
                id=point_id_for(chunks[i]["chunk_id"]),
                vector=matrix[i].tolist(),
                payload={
                    "chunk_id": chunks[i]["chunk_id"],
                    "parent_id": chunks[i]["parent_id"],
                    "document_id": chunks[i].get("document_id"),
                },
            )
            for i in range(start, stop)
        ]
        client.upsert(collection, points=points, wait=(stop >= total))


def run_ab(
    chunks: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    *,
    embedder,
    client,
    collection_prefix: str | None = None,
    collection_names: list[str] | None = None,
    k: int = 10,
    batch_size: int = 256,
    log=None,
) -> dict[str, Any]:
    """建两个 collection（有前缀 / 无前缀），同一批 query 各查一遍。"""
    from ..store.qdrant_store import ensure_collection
    from ..store.search import qdrant_search

    if not chunks:
        raise ValueError("A/B 需要至少一个子块")
    prefix = collection_prefix or f"ab_embtext_{uuid.uuid4().hex[:8]}"
    names = {variant: f"{prefix}_{variant}" for variant in VARIANTS}
    # 无前缀变体就是纯正文：这正是"去掉前缀会怎样"的定义
    texts = {
        "prefixed": [c.get("embedding_text") or c["content"] for c in chunks],
        "plain": [c["content"] for c in chunks],
    }
    vectors = {variant: embedder.encode_documents(texts[variant]) for variant in VARIANTS}

    for variant in VARIANTS:
        ensure_collection(client, names[variant], recreate=True)
        if collection_names is not None:
            collection_names.append(names[variant])
        _upsert(client, names[variant], chunks, vectors[variant], batch_size)
        if log:
            log(f"[ab] {names[variant]}：{len(chunks)} 点（{variant}）")

    rows: list[dict[str, Any]] = []
    for question in questions:
        vector = embedder.encode_queries([question["question"]])[0]
        row: dict[str, Any] = {
            "qid": question["qid"],
            "question": question["question"],
            "gold": set(question.get("gold_parent_ids") or ()),
            "k": k,
        }
        for variant in VARIANTS:
            hits = qdrant_search(client, names[variant], vector, k=k)
            row[variant] = [hit.payload.get("parent_id") for hit in hits]
        metrics_prefixed = query_metrics(row["prefixed"], row["gold"], k=k)
        metrics_plain = query_metrics(row["plain"], row["gold"], k=k)
        row["prefixed_recall"] = metrics_prefixed["recall"]
        row["plain_recall"] = metrics_plain["recall"]
        rows.append(row)

    report = {
        "chunks": len(chunks),
        "queries": len(questions),
        "k": k,
        "collections": names,
        **ab_summary(rows),
    }
    report["rows"] = [
        {key: row[key] for key in ("qid", "question", "prefixed_recall", "plain_recall")}
        for row in rows
    ]
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_chunks(paths: Sequence[Any], *, root: Any, embedding_version: str = "bge-m3-v1") -> list[dict[str, Any]]:
    """把源文件解析 + 切分成子块记录（不写库，只要 `embedding_text`）。

    刻意沿用摄取链路的口径（`doc_key` = 相对 data 根的路径、`make_document_id`、
    `_version_of`），这样算出来的 `parent_id` 与 `DocumentStore` 里的完全一致，
    金标才查得到。
    """
    from ..ingest.pipeline import _version_of
    from .ids import make_document_id
    from .parent_builder import ParentBuilder, build_children
    from .parser import get_parser, register_builtin_parsers
    from .tokenizer import TokenCounter

    register_builtin_parsers(replace=True)
    counter = TokenCounter()
    builder = ParentBuilder(counter)
    out: list[dict[str, Any]] = []
    for path in paths:
        key = Path(path).relative_to(root).as_posix()
        try:
            parser = get_parser(str(path))
            blocks = parser.parse(str(path))
            parents = builder.build(blocks, document_id=make_document_id(key),
                                    document_version=_version_of(Path(path)), title=Path(path).stem)
            children = build_children(parents, counter=counter, embedding_version=embedding_version)
        except Exception:  # noqa: BLE001  解析失败的文件直接跳过，A/B 不需要全语料
            continue
        out.extend(child.to_record() for child in children)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    from ...config import DATA_DIR, QUESTIONS_PATH, STORAGE_DIR

    parser = argparse.ArgumentParser(prog="embedding_text_report", description="D14 前缀统计 + A/B 报告")
    parser.add_argument("--dir", default=str(DATA_DIR), help="源文件目录")
    parser.add_argument("--limit", type=int, default=120, help="最多读多少个文件（小样本 A/B）")
    parser.add_argument("--queries", type=int, default=20, help="最多用多少条题目")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--stats-only", action="store_true", help="只统计前缀，不建 collection")
    parser.add_argument("--from-gold", action="store_true",
                        help="按题目的 gold_spans 挑样本文件（否则按路径排序取前 N 个，有偏）")
    parser.add_argument("--distractors", type=int, default=0, help="额外加入多少个不含金标的干扰文件")
    parser.add_argument("--json", default=None, help="把完整报告写到这个路径")
    args = parser.parse_args(argv)

    root = Path(args.dir)
    suffixes = {".md", ".txt", ".pdf"}
    all_paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)

    raw = json.loads(Path(QUESTIONS_PATH).read_text(encoding="utf-8"))
    if args.from_gold:
        from ..store.document_store import DocumentStore

        probe = DocumentStore(STORAGE_DIR / "documents.sqlite3")
        try:
            uris: set[str] = set()
            for item in raw:
                uris |= probe.documents_containing(item["gold_spans"])
        finally:
            probe.close()
        gold_paths = [root / uri for uri in sorted(uris) if (root / uri).is_file()]
        rest = [p for p in all_paths if p not in set(gold_paths)]
        # 干扰文件均匀抽样，避免又变成"按字母序取前 N 个"的偏样
        step = max(1, len(rest) // args.distractors) if args.distractors else 1
        distractors = rest[::step][: args.distractors] if args.distractors else []
        paths = gold_paths + distractors
        print(f"按金标选文件：{len(gold_paths)} 个含金标 + {len(distractors)} 个干扰")
    else:
        paths = all_paths[: args.limit]
    if not paths:
        print(f"没有在 {root} 下找到可解析文件", file=sys.stderr)
        return 2

    chunks = _build_chunks(paths, root=root)
    print(f"样本：{len(paths)} 个文件 → {len(chunks)} 个子块")
    stats = prefix_stats(chunks)
    print(f"前缀覆盖：{stats.as_dict()}")
    recon = reconstruction_stats(chunks)
    print(f"可重建：{ {k: v for k, v in recon.items() if k != 'examples'} }")

    if args.stats_only:
        return 0

    # 金标：题目的 gold_spans 落在哪些父块里（只认本次索引范围内出现的父块）
    from ..store.document_store import DocumentStore

    store = DocumentStore(STORAGE_DIR / "documents.sqlite3")
    indexed_parents = {c["parent_id"] for c in chunks}
    questions = []
    for item in raw:
        gold = store.parents_containing(item["gold_spans"]) & indexed_parents
        questions.append({"qid": item["qid"], "question": item["question"],
                          "gold_parent_ids": gold})
    questions = [q for q in questions if q["gold_parent_ids"]][: args.queries]
    print(f"可用于判定的题目：{len(questions)}（金标父块在样本内）")
    if not questions:
        print("样本内没有可判定的题目，请加大 --limit", file=sys.stderr)
        store.close()
        return 2

    from ..embedding.service import EmbeddingService
    from ..store.qdrant_store import make_client

    client = make_client(timeout=300)
    names: list[str] = []
    try:
        report = run_ab(
            chunks, questions,
            embedder=EmbeddingService(), client=client,
            collection_prefix=f"ab_embtext_{uuid.uuid4().hex[:8]}",
            collection_names=names, k=args.k, log=print,
        )
    finally:
        for name in names:
            client.delete_collection(name)
        client.close()
        store.close()

    for variant in VARIANTS:
        row = report[variant]
        print(f"  {variant:8s} Recall@{args.k}={row['recall']} hit={row['hit']} MRR={row['mrr']}")
    print(f"  delta    Recall@{args.k}={report['delta']['recall']} "
          f"hit={report['delta']['hit']} MRR={report['delta']['mrr']}")
    payload = {"prefix": stats.as_dict(), "reconstruction": recon, **report}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入 {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
