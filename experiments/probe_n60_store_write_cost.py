"""N60 证据探针（二）：Store 双写到底贵多少？

编排放（`IngestPipeline._prepare`）与索引器（`Indexer.index_document`）各写一次
`DocumentStore.replace_document`，而后者不是「update 一行」——它是
**删掉这篇文档的全部父/子块再整篇重新插入**（一个事务内）。所以第二遍是纯浪费。

判据不用推理，直接量：拿真 Store（`storage/bench/documents.sqlite3` 的临时副本，
不动原库）读出一篇真实文档的 parents/children，连写两遍，计时。

跑法（只读原库，写临时副本；约 3 s）：
    python experiments/probe_n60_store_write_cost.py
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_bench import BENCH_STORE          # noqa: E402
from src.rag.store.document_store import DocumentStore  # noqa: E402

DEST = ROOT / "docs" / "evidence" / "n60-store-double-write-k8s-v1.json"


def _load_document(store_path: Path, document_id: str) -> tuple[dict, list[dict], list[dict]]:
    conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        document = dict(conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone())
        parents = [dict(row) for row in conn.execute(
            "SELECT * FROM parents WHERE document_id = ?", (document_id,))]
        children = [dict(row) for row in conn.execute(
            "SELECT * FROM children WHERE document_id = ?", (document_id,))]
    finally:
        conn.close()
    return document, parents, children


def main() -> int:
    conn = sqlite3.connect(f"file:{BENCH_STORE}?mode=ro", uri=True)
    try:
        documents = conn.execute("SELECT document_id FROM documents ORDER BY document_id").fetchall()
        # 父/子块数必须**分开数**：`parents JOIN children` 会按文档笛卡尔积，把 71 万放大出来
        counts = conn.execute(
            "SELECT d.document_id,"
            " (SELECT COUNT(*) FROM parents p WHERE p.document_id = d.document_id) AS parents,"
            " (SELECT COUNT(*) FROM children c WHERE c.document_id = d.document_id) AS children"
            " FROM documents d"
        ).fetchall()
    finally:
        conn.close()
    sizes = {row[0]: {"parents": row[1], "children": row[2]} for row in counts}

    with TemporaryDirectory() as tmp:
        copy = Path(tmp) / "documents.sqlite3"
        shutil.copy2(BENCH_STORE, copy)          # 原库不动
        store = DocumentStore(copy)
        samples = []
        for (document_id,) in documents[:5]:
            document, parents, children = _load_document(BENCH_STORE, document_id)
            started = time.perf_counter()
            store.replace_document(document, parents, children)
            first = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            store.replace_document(document, parents, children)   # 双写的第二遍
            second = (time.perf_counter() - started) * 1000
            samples.append({"document_id": document_id, **sizes[document_id],
                            "first_write_ms": round(first, 1),
                            "second_write_ms": round(second, 1)})

    per_doc = statistics.median(item["second_write_ms"] for item in samples)
    total_docs = len(documents)
    payload = {
        "note": "N60：`replace_document` 是「DELETE 该文档全部父/子块 + 整篇 INSERT」的事务，不是行更新",
        "store": str(BENCH_STORE.relative_to(ROOT)).replace("\\", "/"),
        "corpus": {"documents": total_docs,
                   "parents": sum(v["parents"] for v in sizes.values()),
                   "children": sum(v["children"] for v in sizes.values())},
        "samples": samples,
        "median_second_write_ms": round(per_doc, 1),
        "extrapolated_wasted_ms_full_corpus": round(per_doc * total_docs, 1),
        "finding": (
            "双写让每篇文档的 Store 写入翻倍；按本轮语料 "
            f"{total_docs} 篇 × 中位 {per_doc:.0f} ms ≈ {per_doc * total_docs / 1000:.1f} s 纯浪费，"
            "而且两次写入之间任一次失败会让 Store 与 Qdrant 版本不一致，状态表却照样是 COMPLETED。"
        ),
        "how_to_reproduce": ["python experiments/probe_n60_store_write_cost.py"],
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("corpus", "median_second_write_ms", "extrapolated_wasted_ms_full_corpus")},
                     ensure_ascii=False, indent=1))
    print(f"结果：{DEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
