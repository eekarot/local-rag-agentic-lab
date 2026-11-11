"""Docling 与现有 PyMuPDF 解析器的对照探针（**只读**：不写库、不改状态表、不动基线）。

为什么要有它
------------
M1-2 想换 PDF 解析器，但「换」之前得先知道两件事：**多拿到什么**（表格结构化、
坐标、阅读顺序）与**付出什么**（耗时、内存、磁盘）。这个探针只做这两件事的记账。

三条纪律
--------
1. **不碰库**：只调 `PdfParser.parse()`（纯函数式）与 docling 的 CLI 子进程；
   不 import docling（它装在 `storage/docling-venv`，属于可选环境）。
2. **两边同口径**：都只跑前 `--max-pages` 页，指标都从「块/元素」这一层数出来，
   `summarize_blocks()` 与 `summarize_docling()` 必须能对齐比较（见用例）。
3. **产物可入库**：证据里只有匿名 id（`pdf-01`）与类别（`en-manual`），
   不含文件名与绝对路径；原始映射留在 `storage/`（已 gitignore）。

用法
----
    python experiments/probe_parser_docling.py \
        --manifest storage/docling-probe/manifest.json \
        --docling-python storage/docling-venv/Scripts/python.exe \
        --out docs/evidence/parser-docling-probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
# 直接 `python experiments/xxx.py` 时 sys.path[0] 是 experiments/，`import src...` 会炸
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DOCLING = ROOT / "storage" / "docling-venv" / "Scripts" / "python.exe"
DEFAULT_ARTIFACTS = Path.home() / ".cache" / "docling" / "models"
HOME = str(Path.home())


# --------------------------------------------------------------------------- #
# 纯函数（用例覆盖这一层）
# --------------------------------------------------------------------------- #
def _prov_with_bbox(item: dict) -> bool:
    for prov in item.get("prov") or []:
        bbox = (prov or {}).get("bbox")
        if isinstance(bbox, dict) and all(k in bbox for k in ("l", "t", "r", "b")):
            return True
    return False


def _table_cell_text(table: dict) -> str:
    data = table.get("data") or {}
    cells = data.get("table_cells") or []
    return "".join(str(cell.get("text") or "") for cell in cells)


def docling_chars(document: dict) -> int:
    """可检索字符数：正文 + 表格单元格文本（表格不算就等于把表格丢了）。"""
    texts = sum(len(str(item.get("text") or "")) for item in document.get("texts") or [])
    tables = sum(len(_table_cell_text(item)) for item in document.get("tables") or [])
    return texts + tables


def summarize_docling(document: dict) -> dict[str, Any]:
    """把 DoclingDocument 压成与 `summarize_blocks()` 同一组指标。"""
    texts = document.get("texts") or []
    tables = document.get("tables") or []
    pictures = document.get("pictures") or []
    pages = document.get("pages") or {}
    items = texts + tables + pictures

    labels: dict[str, int] = {}
    for item in texts:
        label = str(item.get("label") or "unknown")
        labels[label] = labels.get(label, 0) + 1

    positioned = [item for item in items if _prov_with_bbox(item)]
    pages_with_content = {
        prov.get("page_no")
        for item in positioned
        for prov in (item.get("prov") or [])
        if prov.get("page_no") is not None
    }
    return {
        "pages": len(pages),
        "texts": len(texts),
        "tables": len(tables),
        "pictures": len(pictures),
        "chars": docling_chars(document),
        "labels": labels,
        "positioned_items": len(positioned),
        "total_items": len(items),
        "bbox_coverage": (len(positioned) / len(items)) if items else 0.0,
        "pages_with_content": len(pages_with_content),
    }


def summarize_blocks(blocks: Iterable[Any]) -> dict[str, Any]:
    """现有解析器的 `DocumentBlock` 列表 → 同一组指标。

    只用鸭子类型（`text` / `block_type` / `page_number`），避免探针把解析器的
    具体类绑死；也让用例可以用三行假对象覆盖它。
    """
    labels: dict[str, int] = {}
    chars = 0
    tables = 0
    pages: set[int] = set()
    count = 0
    for block in blocks:
        count += 1
        chars += len(str(getattr(block, "text", "") or ""))
        block_type = str(getattr(block, "block_type", "") or "unknown")
        labels[block_type] = labels.get(block_type, 0) + 1
        if block_type == "table":
            tables += 1
        page = getattr(block, "page_number", None)
        if page is not None:
            pages.add(page)
    return {
        "blocks": count,
        "chars": chars,
        "tables": tables,
        "labels": labels,
        "pages_with_content": len(pages),
    }


def coverage_ratio(docling_value: float, baseline_value: float) -> float | None:
    """候选解析器的覆盖率；基线为 0 时返回 None（不除零，也不假装是 0%）。"""
    if not baseline_value:
        return None
    return docling_value / baseline_value


def _pct(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def sanitize(path: str | Path) -> str:
    """证据要入库，不能带本机绝对路径与用户名。"""
    text = str(path)
    if HOME and HOME in text:
        text = text.replace(HOME, "~")
    return text.replace("\\", "/")


# --------------------------------------------------------------------------- #
# 两个解析器（IO 侧）
# --------------------------------------------------------------------------- #
def run_baseline(pdf: Path, *, max_pages: int) -> tuple[dict[str, Any], float]:
    """现有 PyMuPDF 解析器：只调纯函数式 parse()，不写任何存储。"""
    from src.rag.ingestion.parser.pdf_parser import PdfParser

    parser = PdfParser()
    started = time.perf_counter()
    blocks = parser.parse(str(pdf), page_limit=max_pages)
    elapsed = time.perf_counter() - started
    stats = summarize_blocks(blocks)
    stats["declared_pages"] = int((parser.stats or {}).get("pages") or 0)
    return stats, elapsed


def docling_command(python: Path) -> list[str]:
    """venv 里的 console script 优先：`python -m docling` 会报 no module named docling.__main__。"""
    for name in ("docling.exe", "docling"):
        candidate = python.parent / name
        if candidate.is_file():
            return [str(candidate)]
    return [str(python), "-m", "docling"]


def run_docling(
    pdf: Path,
    *,
    python: Path,
    out_dir: Path,
    max_pages: int,
    artifacts: Path | None,
    timeout_s: float,
) -> tuple[dict[str, Any], float, float | None, str | None]:
    """DOCLING 走子进程：主环境里没有它，也不该有。

    返回 (指标, 墙钟秒数, 峰值 RSS MB, 错误信息)。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{pdf.stem}.json"
    if target.exists():
        target.unlink()

    command = [*docling_command(python), str(pdf), "--to", "json",
               "--output", str(out_dir), "--page-range", f"1-{max_pages}"]
    if artifacts is not None:
        command += ["--artifacts-path", str(artifacts)]

    env = {
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PATH": str(Path(sys.executable).parent) + ";" + str(Path.home() / "AppData/Local/Programs/Python/Python312/Scripts"),
    }
    import os

    run_env = {**os.environ, **env}
    started = time.perf_counter()
    peak_mb: float | None = None
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=run_env, text=True, encoding="utf-8", errors="replace")

    def _watch() -> None:
        nonlocal peak_mb
        try:
            import psutil
        except ImportError:  # pragma: no cover - psutil 在主环境里已装
            return
        proc = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    rss += child.memory_info().rss
                peak_mb = max(peak_mb or 0.0, rss / 1e6)
            except Exception:  # noqa: BLE001 - 进程已退出属正常
                return
            time.sleep(0.2)

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return {}, time.perf_counter() - started, peak_mb, f"超时 >{timeout_s:.0f}s"
    elapsed = time.perf_counter() - started

    if process.returncode != 0:
        tail = (stderr or stdout or "").strip().splitlines()[-3:]
        return {}, elapsed, peak_mb, f"退出码 {process.returncode}: {' / '.join(tail)[:300]}"
    if not target.is_file():
        return {}, elapsed, peak_mb, "docling 说成功但没有产出 json"

    document = json.loads(target.read_text(encoding="utf-8"))
    stats = summarize_docling(document)
    stats["declared_pages"] = len(document.get("pages") or {})
    return stats, elapsed, peak_mb, None


# --------------------------------------------------------------------------- #
# 汇总与 CLI
# --------------------------------------------------------------------------- #
def compare(manifest: Sequence[dict[str, Any]], *, docling_python: Path, artifacts: Path | None,
            max_pages: int, work: Path, timeout_s: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in manifest:
        pdf = Path(item["path"])
        row: dict[str, Any] = {"id": item["id"], "kind": item.get("kind", "unknown")}
        if not pdf.is_file():
            row.update(status="missing-source")
            rows.append(row)
            continue
        try:
            base_stats, base_s = run_baseline(pdf, max_pages=max_pages)
        except Exception as exc:  # noqa: BLE001 - 探针不许抛
            row.update(status="baseline-failed", error=f"{type(exc).__name__}: {exc}"[:200])
            rows.append(row)
            continue

        doc_stats, doc_s, peak_mb, error = run_docling(
            pdf, python=docling_python, out_dir=work / item["id"], max_pages=max_pages,
            artifacts=artifacts, timeout_s=timeout_s,
        )
        row.update({
            "status": "ok" if error is None else "docling-failed",
            "baseline": base_stats,
            "docling": doc_stats or None,
            "char_coverage": coverage_ratio(doc_stats.get("chars", 0), base_stats["chars"]),
            "latency_baseline_s": round(base_s, 3),
            "latency_docling_s": round(doc_s, 3),
            "peak_rss_mb": round(peak_mb, 1) if peak_mb else None,
        })
        if error:
            row["error"] = error
        rows.append(row)
        print(f"[{item['id']:8}] {row['status']:15} 基线 {base_s:6.2f}s / docling {doc_s:6.2f}s "
              f"| 字符覆盖 {row['char_coverage'] if row['char_coverage'] is None else round(row['char_coverage'], 3)}",
              flush=True)

    return {"rows": rows, **aggregate(rows)}


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    ratios = [row["char_coverage"] for row in ok if row.get("char_coverage") is not None]
    bbox = [row["docling"]["bbox_coverage"] for row in ok if row.get("docling")]
    baseline_ms = [row["latency_baseline_s"] * 1000 for row in ok]
    docling_ms = [row["latency_docling_s"] * 1000 for row in ok]
    rss = [row["peak_rss_mb"] for row in ok if row.get("peak_rss_mb")]
    return {
        "n_manifest": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "char_coverage_mean": round(statistics.fmean(ratios), 4) if ratios else None,
        "char_coverage_min": round(min(ratios), 4) if ratios else None,
        "bbox_coverage_mean": round(statistics.fmean(bbox), 4) if bbox else None,
        "bbox_coverage_min": round(min(bbox), 4) if bbox else None,
        "tables_baseline_total": sum(row["baseline"]["tables"] for row in ok),
        "tables_docling_total": sum(row["docling"]["tables"] for row in ok if row.get("docling")),
        "latency_baseline_p50_ms": round(_pct(baseline_ms, 0.5) or 0.0, 1),
        "latency_docling_p50_ms": round(_pct(docling_ms, 0.5) or 0.0, 1),
        "latency_docling_p95_ms": round(_pct(docling_ms, 0.95) or 0.0, 1),
        "peak_rss_mb_max": round(max(rss), 1) if rss else None,
        "peak_rss_mb_p50": round(_pct(rss, 0.5) or 0.0, 1) if rss else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Docling vs PyMuPDF 解析对照（只读探针）")
    parser.add_argument("--manifest", required=True, help="JSON： [{id, kind, path}]")
    parser.add_argument("--docling-python", default=str(DEFAULT_DOCLING))
    parser.add_argument("--artifacts-path", default=str(DEFAULT_ARTIFACTS),
                        help="离线模型目录；传空串则让 docling 自己解析")
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--work", default=str(ROOT / "storage" / "docling-probe"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidence" / "parser-docling-probe.json"))
    args = parser.parse_args(argv)

    docling_python = Path(args.docling_python)
    if not docling_python.is_file():
        print(f"找不到 docling 的 Python 解释器：{docling_python}", file=sys.stderr)
        print("这是可选环境，按下面两条命令之一装好即可：", file=sys.stderr)
        print("  uv venv storage/docling-venv --python 3.12", file=sys.stderr)
        print("  storage/docling-venv/Scripts/python.exe -m pip install docling", file=sys.stderr)
        return 2

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"找不到清单：{manifest_path}", file=sys.stderr)
        print("清单格式（放在 storage/ 下，不进仓库）：", file=sys.stderr)
        print('  [{"id": "pdf-01", "kind": "en-manual", "path": "<绝对路径>"}]', file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = Path(args.artifacts_path) if args.artifacts_path else None
    result = compare(manifest, docling_python=docling_python, artifacts=artifacts,
                     max_pages=args.max_pages, work=Path(args.work), timeout_s=args.timeout_s)

    payload = {
        "note": "Docling 2.150.0（独立 venv）与现有 PyMuPDF 解析器的对照；同一批 PDF、同样只跑前几页",
        "config": {"max_pages": args.max_pages, "docling_python": sanitize(docling_python),
                   "artifacts_path": sanitize(artifacts) if artifacts else None},
        **result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== 汇总 ===")
    for key in ("n_manifest", "n_ok", "n_failed", "char_coverage_mean", "char_coverage_min",
                "bbox_coverage_mean", "tables_baseline_total", "tables_docling_total",
                "latency_baseline_p50_ms", "latency_docling_p50_ms", "latency_docling_p95_ms",
                "peak_rss_mb_p50", "peak_rss_mb_max"):
        print(f"  {key:28} {result[key]}")
    print(f"\n结果 → {out}")
    return 0 if result["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
