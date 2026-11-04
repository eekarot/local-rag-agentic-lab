"""评估集的工具：gold span 自动定位 + 题目文件校验。

设计规范见 docs/eval-and-logging-design.md 第 3 章。两个用途：

1. 出题时把「语料里的原文句子」（gold_spans）自动定位成 chunk id，避免手抄 id 出错；
2. CI/提交前校验：qid 唯一、题型配额、参考答案齐全、gold 定位唯一、负样本确实查不到。

命令行：

    python -m src.eval_dataset locate "要定位的原文句子"
    python -m src.eval_dataset check                    # 校验 experiments/questions.json
    python -m src.eval_dataset find 关键词 [关键词2...]   # 找出包含关键词的 chunk（出题用）
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from .config import QUESTIONS_PATH
from .retrieve import load_index

REQUIRED_TYPES = ("fact", "multi_hop", "concept", "negative")
MIN_SPAN_CHARS = 12          # 太短的句子在全库会撞到多处，必须人工指定 id
QUOTA = {"fact": 15, "multi_hop": 15, "concept": 15, "negative": 12}
MIN_TOTAL = 57


def normalize(text):
    """统一归一化：小写 + 去掉所有空白。

    PDF 抽出来的正文有硬换行（例如「只有业务约\\n束本身」），不去空白会定位失败。
    """
    return re.sub(r"\s+", "", (text or "").lower())


def locate(span, records=None, limit=5):
    """把原文句子定位成 chunk id 列表。返回 [(id, source, chunk_id), ...]。"""
    if records is None:
        records = load_index()["records"]
    needle = normalize(span)
    if len(needle) < MIN_SPAN_CHARS:
        raise ValueError(f"gold_span 太短（{len(needle)} 字符 < {MIN_SPAN_CHARS}），会匹配到多处")
    hits = []
    for record in records:
        if needle in normalize(record.get("text")):
            hits.append((record["id"], record["source"], record["chunk_id"]))
            if len(hits) >= limit:
                break
    return hits


def find_chunks(keywords, records=None, limit=30, require_all=False):
    """出题用：找出同时/任一包含关键词的 chunk，方便挑原文句子。"""
    if records is None:
        records = load_index()["records"]
    needles = [normalize(k) for k in keywords if k]
    out = []
    for record in records:
        text = normalize(record.get("text"))
        matched = [n for n in needles if n in text]
        if (require_all and len(matched) == len(needles)) or (not require_all and matched):
            out.append({
                "id": record["id"],
                "source": record["source"],
                "chunk_id": record["chunk_id"],
                "matched": matched,
                "text": record["text"],
            })
            if len(out) >= limit:
                break
    return out


def _span_locations(item, records, cache):
    results = []
    for span in item.get("gold_spans") or []:
        key = normalize(span)
        if key not in cache:
            cache[key] = locate(span, records)
        results.append((span, cache[key]))
    return results


#: span 在旧索引里命中多少处就算「需要人工点名」。已有显式 `gold_points` 时只降级为警告。
SPAN_HIT_ERROR_THRESHOLD = 5


def validate(path=QUESTIONS_PATH, resolve_spans=False, verbose=True):
    """返回 (errors, warnings)。errors 非空表示题目文件不可用。"""
    data = json.loads(open(path, encoding="utf-8").read()) if not hasattr(path, "read_text") else json.loads(path.read_text(encoding="utf-8"))
    records = load_index()["records"]
    cache = {}
    errors, warnings = [], []

    missing_points = []
    qids = [item.get("qid") for item in data]
    if len(set(qids)) != len(qids):
        errors.append(f"qid 有重复：{sorted({q for q in qids if qids.count(q) > 1})}")
    for index, item in enumerate(data):
        label = item.get("qid") or f"#{index}"
        if not item.get("qid"):
            errors.append(f"{label}: 缺少 qid")
        question_type = item.get("type")
        if question_type not in REQUIRED_TYPES:
            errors.append(f"{label}: type 必须是 {REQUIRED_TYPES}，实际 {question_type!r}")
        if not item.get("question", "").strip():
            errors.append(f"{label}: question 为空")
        if not item.get("lang"):
            warnings.append(f"{label}: 缺少 lang")

        spans = item.get("gold_spans") or []
        declared = item.get("gold_chunks") or []
        # 新库（D20 之后）的 gold 是 parent id，写在 gold_points 里。已有显式映射时，
        # 「span 命中多处 / 旧索引没覆盖」都不再是错误——校验器的本意是逼出人工确认，
        # 而映射已经人工确认过了（详见 §17.11.3）。
        mapped_points = (item.get("gold_points") or {}).get("parent_ids") or []
        if question_type != "negative":
            if not spans and not declared:
                errors.append(f"{label}: 正样本必须有 gold_spans 或 gold_chunks")
            for span, hits in _span_locations(item, records, cache):
                if not hits:
                    if mapped_points:
                        warnings.append(
                            f"{label}: gold_span 不在旧 81k 索引里（该文档只进了新库），"
                            f"已由 gold_points 映射到 {len(mapped_points)} 个父块：{span[:30]}…"
                        )
                    else:
                        errors.append(f"{label}: gold_span 在语料里定位不到：{span[:40]}…")
                elif (len(hits) >= SPAN_HIT_ERROR_THRESHOLD
                      and not declared and not mapped_points):
                    errors.append(f"{label}: gold_span 命中 {len(hits)} 处（≥5），需要人工指定 id：{span[:30]}…")
                elif len(hits) >= SPAN_HIT_ERROR_THRESHOLD:
                    warnings.append(
                        f"{label}: gold_span 命中 {len(hits)} 处（跨文档重复），已用显式 gold_chunks "
                        f"收敛到 {len(declared)} 条"
                    )
                elif len(hits) > 1:
                    warnings.append(f"{label}: gold_span 命中 {len(hits)} 处，建议确认：{span[:30]}…")
                if len(declared) < len(spans) and resolve_spans:
                    item.setdefault("_resolved", []).extend(h["id"] for h in hits)
            if not item.get("reference_answer") and question_type in ("fact", "concept", "multi_hop"):
                errors.append(f"{label}: 正样本缺少 reference_answer")
            if not item.get("gold_points"):
                missing_points.append(label)
        else:
            if not item.get("note"):
                errors.append(f"{label}: 负样本必须在 note 里记录「已确认全库无答案」的查证方式")
            if spans:
                warnings.append(f"{label}: 负样本不该有 gold_spans")
        for chunk in declared:
            if "id" not in chunk or "grade" not in chunk:
                errors.append(f"{label}: gold_chunks 每项需要 id 与 grade")
            if chunk.get("grade") not in (1, 2):
                errors.append(f"{label}: grade 只能是 1（相关）或 2（完全支持）")

    counts = {t: sum(1 for item in data if item.get("type") == t) for t in REQUIRED_TYPES}
    for question_type, minimum in QUOTA.items():
        if counts[question_type] < minimum:
            errors.append(f"题型 {question_type} 只有 {counts[question_type]} 题，低于配额 {minimum}")
    if len(data) < MIN_TOTAL:
        errors.append(f"总题数 {len(data)} < {MIN_TOTAL}")
    if missing_points:
        warnings.append(
            f"{len(missing_points)} 题缺 gold_points（S3 补齐后即可算要点覆盖）："
            f"{', '.join(missing_points[:5])}…"
        )
    langs = {item.get("lang") for item in data}
    if verbose:
        print(f"题目数：{len(data)}；分布：{counts}；语言：{sorted(x for x in langs if x)}")
        for warning in warnings:
            print(f"  [warn] {warning}")
        for error in errors:
            print(f"  [error] {error}")
        print("校验通过。" if not errors else f"校验失败：{len(errors)} 个错误。")
    return errors, warnings


def resolve_gold_chunks(path=QUESTIONS_PATH, write=True):
    """把 gold_spans 定位结果写进 gold_chunks（保留人工已填的条目）。"""
    data = json.loads(open(path, encoding="utf-8").read())
    records = load_index()["records"]
    changed = 0
    for item in data:
        if item.get("type") == "negative":
            continue
        existing = {chunk["id"] for chunk in item.get("gold_chunks") or []}
        for span in item.get("gold_spans") or []:
            hits = locate(span, records, limit=3)
            if len(hits) != 1:
                continue
            chunk_id = hits[0][0]
            if chunk_id not in existing:
                item.setdefault("gold_chunks", []).append({"id": chunk_id, "grade": 2})
                existing.add(chunk_id)
                changed += 1
    data = [
        {k: v for k, v in item.items() if not k.startswith("_")}
        for item in data
    ]
    if write:
        target = path if hasattr(path, "write_text") else type(path)(path)
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed, data


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.eval_dataset", description="评估集工具")
    sub = parser.add_subparsers(dest="command", required=True)
    locate_cmd = sub.add_parser("locate", help="定位原文句子到 chunk id")
    locate_cmd.add_argument("span")
    find_cmd = sub.add_parser("find", help="查找包含关键词的 chunk（出题用）")
    find_cmd.add_argument("keywords", nargs="+")
    find_cmd.add_argument("--limit", type=int, default=10)
    find_cmd.add_argument("--all", action="store_true", help="要求同时包含全部关键词")
    sub.add_parser("check", help="校验题目文件")
    resolve_cmd = sub.add_parser("resolve", help="把 gold_spans 写进 gold_chunks")
    resolve_cmd.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "locate":
        hits = locate(args.span)
        if not hits:
            print("定位不到（检查是否抄自语料原文、是否有多余空白）")
            return 1
        for chunk_id, source, index in hits:
            print(f"{chunk_id}\n    source={source} chunk={index}")
        return 0
    if args.command == "find":
        for chunk in find_chunks(args.keywords, limit=args.limit, require_all=args.all):
            print(f"--- {chunk['id']} (matched={chunk['matched']})")
            print(chunk["text"][:600].replace("\n", " "))
            print()
        return 0
    if args.command == "check":
        errors, _ = validate()
        return 1 if errors else 0
    if args.command == "resolve":
        changed, _ = resolve_gold_chunks(write=not args.dry_run)
        print(f"{'（dry-run）' if args.dry_run else ''}新增 gold_chunks {changed} 条")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
