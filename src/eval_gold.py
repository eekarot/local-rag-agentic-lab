"""把 `gold_spans` 映射成**新库**的 parent/child id（`gold_points`）。

背景（为什么需要这一步）：`questions.json` 里的 `gold_chunks` 全是旧 81,027-chunk
索引的 `rag_basics.md::chunk-0000` 形式；现在的库是 Qdrant `rag_chunks_v1`
（38,865 child / 25,411 parent）+ Document Store 的 parent/child 结构。两套 id
空间不相交，**不映射，新库的 Recall 恒为 0**。

匹配口径（先严后宽，两次都留下痕迹）：

1. 精确：`parents.content LIKE '%span%'` —— `gold_spans` 按规矩就是语料原文，绝大多数走这条；
2. 归一化：把两边所有空白折叠成单空格再比 —— PDF 转出来的文本常有「双空格 / 硬换行」，
   人工抄 span 时会写成单空格，这属于**同一句话**，不该判未命中。

三个必须留痕的情况（不许静默通过）：

* `unmatched`：span 在新库里找不到 → 该题的 gold 失效，必须人工修题；
* `ambiguous`：一个 span 命中超过 `max_parents` 个父块（说明这句话太泛），
  Recall@k 的分子被放大，分数不可信；
* `partial`：多 span 里只有一部分命中 → 该题的 Recall 口径不完整。

字段约定（`gold_points`）：`parent_ids` 是**评估主判据**（父块级 Recall），
`child_ids` 备用；`spans[]` 记录每个 span 落到哪些父块，便于复核。

> 与 S3「答案要点」区分：S3 的要点覆盖以后用独立字段 `answer_points`，
> 不用 `gold_points` 这个名字，避免同一个字段有两种含义。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_UNMATCHED = "unmatched"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_NEGATIVE = "negative"

STATUSES = (STATUS_OK, STATUS_PARTIAL, STATUS_UNMATCHED, STATUS_AMBIGUOUS, STATUS_NEGATIVE)

#: 一个 span 命中超过这么多父块 → 判定「太泛」
DEFAULT_MAX_PARENTS = 8

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """折叠所有空白（含换行）成单空格。"""
    return _WS.sub(" ", text or "").strip()


def _exact_parents(store, span: str) -> set[str]:
    return store.parents_containing([span])


def _all_parents(store) -> list[tuple[str, str]]:
    """(parent_id, content) 全量。25k 行的表，一次扫完 ~百毫秒，够用。"""
    with store._connect() as conn:      # noqa: SLF001  同包内部工具，复用现成连接管理
        rows = conn.execute("SELECT parent_id, content FROM parents").fetchall()
    return [(row["parent_id"], row["content"]) for row in rows]


def _child_ids(store, parent_ids: Iterable[str]) -> list[str]:
    ids = sorted(set(parent_ids))
    if not ids:
        return []
    found: list[str] = []
    with store._connect() as conn:      # noqa: SLF001
        for start in range(0, len(ids), 200):
            batch = ids[start:start + 200]
            marks = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT chunk_id FROM children WHERE parent_id IN ({marks}) ORDER BY chunk_id",
                batch,
            ).fetchall()
            found.extend(row["chunk_id"] for row in rows)
    return found


def _match_span(store, span: str, normalized_parents=None) -> tuple[set[str], str]:
    """返回 (父块集合, 匹配方式 exact|normalized|none)。"""
    exact = _exact_parents(store, span)
    if exact:
        return exact, "exact"
    needle = normalize(span)
    if not needle:
        return set(), "none"
    if normalized_parents is None:
        normalized_parents = [(pid, normalize(text)) for pid, text in _all_parents(store)]
    hits = {pid for pid, text in normalized_parents if needle in text}
    return hits, "normalized" if hits else "none"


def map_question(
    store,
    item: Mapping[str, Any],
    *,
    max_parents: int = DEFAULT_MAX_PARENTS,
    normalized_parents=None,
) -> dict[str, Any]:
    """把一道题的 gold_spans 映射成 parent/child id。"""
    existing = (item.get("gold_points") or {}).get("parent_ids") or []
    spans = [span for span in (item.get("gold_spans") or []) if span]

    if existing:
        # 人工指定优先：自动匹配不许覆盖人手填的
        return {
            "qid": item.get("qid"),
            "status": STATUS_OK,
            "parent_ids": sorted(set(existing)),
            "child_ids": _child_ids(store, existing),
            "match_mode": "explicit",
            "spans": [],
            "unmatched_spans": [],
            "ambiguous_spans": [],
        }

    if item.get("type") == "negative" or not spans:
        return {
            "qid": item.get("qid"),
            "status": STATUS_NEGATIVE,
            "parent_ids": [],
            "child_ids": [],
            "match_mode": "none",
            "spans": [],
            "unmatched_spans": [],
            "ambiguous_spans": [],
        }

    parent_ids: set[str] = set()
    detail: list[dict[str, Any]] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []
    modes: set[str] = set()

    for span in spans:
        hits, mode = _match_span(store, span, normalized_parents)
        modes.add(mode)
        if len(hits) > max_parents:
            ambiguous.append(span)
        if not hits:
            unmatched.append(span)
            detail.append({"span": span, "match_mode": mode, "parent_ids": []})
            continue
        parent_ids |= hits
        detail.append({"span": span, "match_mode": mode, "parent_ids": sorted(hits)})

    if not parent_ids:
        status = STATUS_UNMATCHED
    elif ambiguous:
        status = STATUS_AMBIGUOUS
    elif unmatched:
        status = STATUS_PARTIAL
    else:
        status = STATUS_OK

    return {
        "qid": item.get("qid"),
        "status": status,
        "parent_ids": sorted(parent_ids),
        "child_ids": _child_ids(store, parent_ids),
        "match_mode": "normalized" if "normalized" in modes else ("exact" if "exact" in modes else "none"),
        "spans": detail,
        "unmatched_spans": unmatched,
        "ambiguous_spans": ambiguous,
    }


def build_gold_points(
    questions: Sequence[Mapping[str, Any]],
    store,
    *,
    max_parents: int = DEFAULT_MAX_PARENTS,
    log=None,
) -> dict[str, dict[str, Any]]:
    """整集映射。返回 {qid: 结果}。"""
    normalized_parents = None
    mapping: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(questions, start=1):
        result = map_question(
            store, item, max_parents=max_parents, normalized_parents=normalized_parents
        )
        mapping[result["qid"]] = result
        if result["match_mode"] == "normalized" and normalized_parents is None:
            # 一旦用过归一化扫描，后面所有题复用这份缓存（25k 行 × 归一化只做一次）
            normalized_parents = [(pid, normalize(text)) for pid, text in _all_parents(store)]
        if log and index % 20 == 0:
            log(index, len(questions))
    return mapping


def apply_gold_points(
    questions: Sequence[Mapping[str, Any]], mapping: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """把映射写回题目（返回新对象，不改入参）。负样本不写 `gold_points`。"""
    out: list[dict[str, Any]] = []
    for item in questions:
        copy = dict(item)
        result = mapping.get(item.get("qid"))
        if result and result["status"] != STATUS_NEGATIVE and result["parent_ids"]:
            copy["gold_points"] = {
                "parent_ids": list(result["parent_ids"]),
                "child_ids": list(result["child_ids"]),
                "status": result["status"],
                "match_mode": result["match_mode"],
            }
            if result["unmatched_spans"]:
                copy["gold_points"]["unmatched_spans"] = list(result["unmatched_spans"])
        out.append(copy)
    return out


def summarize(mapping: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counts = {status: 0 for status in STATUSES}
    for result in mapping.values():
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    usable = sum(1 for result in mapping.values() if result["status"] in (STATUS_OK, STATUS_PARTIAL, STATUS_AMBIGUOUS))
    return {
        "n": len(mapping),
        **counts,
        "usable": usable,
        "ambiguous_qids": sorted(r["qid"] for r in mapping.values() if r["status"] == STATUS_AMBIGUOUS),
        "unmatched_qids": sorted(r["qid"] for r in mapping.values() if r["status"] == STATUS_UNMATCHED),
        "partial_qids": sorted(r["qid"] for r in mapping.values() if r["status"] == STATUS_PARTIAL),
        "n_parents_p50": _median([len(r["parent_ids"]) for r in mapping.values() if r["parent_ids"]]),
    }


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    from .config import QUESTIONS_PATH, STORAGE_DIR

    parser = argparse.ArgumentParser(
        prog="python -m src.eval_gold",
        description="gold_spans → 新库 parent/child 映射（写入 gold_points）",
    )
    parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    parser.add_argument("--store", default=str(STORAGE_DIR / "documents.sqlite3"))
    parser.add_argument("--out", default=None, help="映射明细 JSON（默认 experiments/gold_points.json）")
    parser.add_argument("--write", action="store_true", help="写回 questions.json")
    parser.add_argument("--max-parents", type=int, default=DEFAULT_MAX_PARENTS)
    args = parser.parse_args(argv)

    from .rag.store.document_store import DocumentStore

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    store = DocumentStore(args.store)
    mapping = build_gold_points(
        questions, store, max_parents=args.max_parents,
        log=lambda done, total: print(f"  已映射 {done}/{total} 题"),
    )
    stats = summarize(mapping)
    print(f"[gold] 题数 {stats['n']}：ok {stats['ok']} / partial {stats['partial']} / "
          f"ambiguous {stats['ambiguous']} / unmatched {stats['unmatched']} / negative {stats['negative']}")
    print(f"[gold] 可用于新库评估：**{stats['usable']} 题**；每题 gold 父块数 p50 = {stats['n_parents_p50']}")
    if stats["unmatched_qids"]:
        print(f"[gold] 未命中（必须人工修题）：{', '.join(stats['unmatched_qids'])}")
    if stats["ambiguous_qids"]:
        print(f"[gold] 命中过泛（分数不可信）：{', '.join(stats['ambiguous_qids'])}")
    if stats["partial_qids"]:
        print(f"[gold] 部分命中（口径不完整）：{', '.join(stats['partial_qids'])}")

    out = Path(args.out) if args.out else Path(args.questions).parent / "gold_points.json"
    out.write_text(json.dumps({"stats": stats, "mapping": mapping}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gold] 明细：{out}")

    if args.write:
        updated = apply_gold_points(questions, mapping)
        Path(args.questions).write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written = sum(1 for item in updated if item.get("gold_points"))
        print(f"[gold] 已写回 {args.questions}：{written} 题带 gold_points")

    return 0 if stats["unmatched"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
