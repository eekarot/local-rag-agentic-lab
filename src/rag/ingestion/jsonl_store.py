"""D12 JSONL 输出与 `data/chunks/` 目录约定（方案 §3.1/§3.2/§3.3、§6）。

为什么要**严格**字段校验（多一个字段也报错）：这些 JSONL 是 Qdrant payload、
SQLite Document Store、评估脚本三方共同的输入契约。字段悄悄多出来意味着
某处以为自己写了、下游其实没读，或者版本漂移了没人发现。

可复现性：JSONL 里不允许出现时间戳、随机数、绝对路径以外的不确定内容，
否则 D10「同一输入连跑两次 sha256 完全一致」就不成立。`generated_at`
只写进 manifest，不写进数据行。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

#: §3.2 Parent 字段（顺序即输出顺序）
PARENT_FIELDS = (
    "parent_id",
    "document_id",
    "document_version",
    "language",
    "title",
    "section_path",
    "content",
    "token_count",
    "source_start",
    "source_end",
    "chunker_version",
)

#: §3.3 Child 字段（顺序即输出顺序）
CHILD_FIELDS = (
    "chunk_id",
    "parent_id",
    "document_id",
    "document_version",
    "language",
    "title",
    "section_path",
    "content",
    "embedding_text",
    "token_count",
    "source_start",
    "source_end",
    "chunker_version",
    "embedding_model",
    "embedding_version",
    "content_hash",
    "status",
)

#: §3.1 Document 字段
DOCUMENT_FIELDS = (
    "document_id",
    "source_uri",
    "file_name",
    "file_type",
    "title",
    "language",
    "document_version",
    "parser_version",
    "created_at",
    "status",
)

DATA_FILES = {
    "documents": ("documents.jsonl", DOCUMENT_FIELDS),
    "parents": ("parents.jsonl", PARENT_FIELDS),
    "children": ("children.jsonl", CHILD_FIELDS),
}


def validate_record(record: dict, required_fields: Sequence[str], *, label: str = "record") -> dict:
    """字段集必须与方案完全一致；返回按 `required_fields` 顺序重排的副本。"""
    required = list(required_fields)
    actual = set(record)
    expected = set(required)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"缺少字段 {missing}")
        if extra:
            parts.append(f"多出字段 {extra}")
        raise ValueError(f"{label} 与方案字段不一致：{'；'.join(parts)}")
    return {key: record[key] for key in required}


def write_jsonl(path: str | Path, records: Iterable[dict], *, required_fields: Sequence[str]) -> int:
    """写 JSONL：UTF-8 无 BOM、LF 换行、一行一个 JSON、不带尾随空白。

    先校验全部记录再开文件——否则校验失败会留下一个半截文件，
    下次跑「断点续读」时读到一半数据。
    """
    validated = [validate_record(record, required_fields) for record in records]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in validated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(validated)


def write_chunks(
    root: str | Path,
    *,
    documents: Sequence[dict],
    parents: Sequence[dict],
    children: Sequence[dict],
    versions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把三类记录写到 `<root>/chunks/`，并生成 manifest。

    manifest 里记录每个文件的 sha256：D10 的验收就是拿两次运行的 manifest
    做对比，不用手工 sha256sum。
    """
    directory = Path(root) / "chunks"
    directory.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    files: dict[str, dict[str, Any]] = {}
    for key, (file_name, fields) in DATA_FILES.items():
        rows = {"documents": documents, "parents": parents, "children": children}[key]
        path = directory / file_name
        counts[key] = write_jsonl(path, rows, required_fields=fields)
        payload = path.read_bytes()
        files[file_name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "rows": counts[key],
        }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "directory": str(directory.as_posix()),
        "counts": counts,
        "files": files,
        "versions": dict(versions or {}),
        "python": os.sys.version.split()[0],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


__all__ = [
    "CHILD_FIELDS",
    "DATA_FILES",
    "DOCUMENT_FIELDS",
    "PARENT_FIELDS",
    "validate_record",
    "write_chunks",
    "write_jsonl",
]
