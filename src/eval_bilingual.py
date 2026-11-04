"""D27：评估集扩到 100–200 条中英成对题的骨架与校验。

方案 §11.2 的结构是 `query_zh` / `query_en` 成对 + `topic` / `difficulty` 切片。
落地口径（§17.11.3 B06）：

* `question` 字段就是 `query_zh`，不改名——已有 60 题、日志、指标、报告全都按
  `question` 走，为了一个字段名重写工具链不值得；
* 新增 `query_en`：只喂给**英文 Dense / 英文 BM25** 两路，**不参与主指标**；
* `query_en` 必须人工过一遍：机器草稿标 `draft`，人改完标 `reviewed` / `edited`。
  `check` 默认把 `draft` 当错误——评估集里混着未审的机翻，跑出来的消融结论
  就分不清是「检索不行」还是「翻译不行」。

命令行：

    python -m src.eval_bilingual status                    # 覆盖率/成对率/待审数量
    python -m src.eval_bilingual draft                     # 生成英文草稿（只补空的）
    python -m src.eval_bilingual review --qid eval-003 --en "..."   # 人工改写后标 reviewed
    python -m src.eval_bilingual check                     # CI 校验（draft 即失败）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .config import QUESTIONS_PATH

#: 方案 §11.2 要求的 8 类问题
TOPICS: tuple[str, ...] = (
    "concept_zh",       # 中文概念问题
    "zh_with_en_term",  # 中文问题包含英文术语
    "acronym_model",    # 缩写和模型名
    "number_spec",      # 数字、参数、版本号
    "paraphrase",       # 同义表达（问法与语料用词不同）
    "multi_condition",  # 多条件问题
    "cross_chunk",      # 需要跨多个 Chunk 才能回答
    "no_answer",        # 知识库没有答案
)

TOPIC_LABELS: dict[str, str] = {
    "concept_zh": "中文概念问题",
    "zh_with_en_term": "中文问题包含英文术语",
    "acronym_model": "缩写和模型名",
    "number_spec": "数字、参数、版本号",
    "paraphrase": "同义表达",
    "multi_condition": "多条件问题",
    "cross_chunk": "跨多个 Chunk",
    "no_answer": "知识库没有答案",
}

DIFFICULTIES: tuple[str, ...] = ("easy", "normal", "hard")

#: `query_en_status` 的合法值。`draft` = 机器草稿未审；`reviewed` = 人工确认；`edited` = 人工改过
PAIR_STATUSES: tuple[str, ...] = ("draft", "reviewed", "edited")
ACCEPTED_STATUSES: tuple[str, ...] = ("reviewed", "edited")

_CJK = re.compile(r"[\u4e00-\u9fff]")
#: 需要原样保留的 token：数字/版本号、全大写缩写、驼峰/下划线标识符、型号、技术名
_PROTECTED = re.compile(
    r"""
    [A-Za-z]{2,}[\s-]?[A-Za-z]*\d+(?:\.\d+)?   # DGX B300 / PCIe Gen6 / RHEL8 / A100
    | \d+(?:\.\d+)?\s?(?:GB|TB|MB|ms|s|tokens?)\b   # 288 GB / 8192 tokens
    | \d{3,}                            # 8192、2024
    | [A-Z]{2,}[A-Za-z]*                # HNSW / BMC / RAG / NVLink / MIG / DGX
    | [a-z]+_[a-z_]+                    # ef_construct / top_k
    | [A-Z][a-z]+[A-Z][A-Za-z]*         # NVSwitch / TopK / MetaFile
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# topic 推断（只用于起草，最终以 JSON 里声明的 topic 为准）
# ---------------------------------------------------------------------------


def infer_topic(item: dict) -> str:
    """按题型与问法猜一个 topic。多条件/同义表达这类语义判断猜不出来，靠人工声明。"""
    if item.get("type") == "negative":
        return "no_answer"
    if item.get("type") == "multi_hop":
        return "cross_chunk"
    question = item.get("question") or ""
    if re.search(r"\d", question):
        return "number_spec"
    if re.search(r"[A-Z]{2,}", question):
        return "acronym_model"
    if re.search(r"[A-Za-z][A-Za-z_.\-]{2,}", question):
        return "zh_with_en_term"
    return "concept_zh"


# ---------------------------------------------------------------------------
# 术语保护
# ---------------------------------------------------------------------------


def protect_tokens(text: str) -> set[str]:
    """抽出必须原样保留的 token：型号、缩写、数字、参数名。"""
    found: set[str] = set()
    for match in _PROTECTED.finditer(text or ""):
        token = match.group(0).strip()
        if not token or _CJK.search(token):
            continue
        found.add(token)
        # 「288 GB」也要能以「288」的形式带上：译文写成 288GB 时不该算丢术语
        number = re.search(r"\d+(?:\.\d+)?", token)
        if number:
            found.add(number.group(0))
    return found


def _normalize_token(token: str) -> str:
    # 去掉全部空白再比：「288 GB」与「288GB」、「DGX B300」与「DGX-B300」视为同一术语
    return re.sub(r"\s+", "", token or "").lower()


def token_retention(question_zh: str, query_en: str) -> float:
    """英文 query 里保留了多少中文 query 的术语/数字（原样，不许意译）。"""
    tokens = protect_tokens(question_zh)
    if not tokens:
        return 1.0
    haystack = _normalize_token(query_en or "")
    kept = sum(1 for token in tokens if _normalize_token(token) in haystack)
    return round(kept / len(tokens), 4)


# ---------------------------------------------------------------------------
# 成对率 / 覆盖率 / 校验
# ---------------------------------------------------------------------------


def is_paired(item: dict) -> bool:
    """既有中文问法又有独立的英文问法（抄一遍中文不算）。"""
    zh = (item.get("question") or "").strip()
    en = (item.get("query_en") or "").strip()
    return bool(zh and en and zh != en)


def pair_rate(items: Iterable[dict]) -> float:
    items = list(items)
    if not items:
        return 0.0
    return round(sum(1 for item in items if is_paired(item)) / len(items), 4)


def coverage(items: Iterable[dict]) -> dict:
    items = list(items)
    counts = {topic: 0 for topic in TOPICS}
    for item in items:
        topic = item.get("topic")
        if topic in counts:
            counts[topic] += 1
    difficulties = {level: 0 for level in DIFFICULTIES}
    for item in items:
        level = item.get("difficulty")
        if level in difficulties:
            difficulties[level] += 1
    return {
        "total": len(items),
        "counts": counts,
        "topics_covered": sum(1 for n in counts.values() if n),
        "topics_missing": [t for t, n in counts.items() if not n],
        "difficulties": difficulties,
        "pair_rate": pair_rate(items),
        "drafts": sum(1 for item in items if item.get("query_en_status") == "draft"),
    }


def check(
    items: Sequence[dict],
    *,
    allow_drafts: bool = False,
    require_topic_coverage: bool = True,
    min_total: int = 0,
) -> list[str]:
    """返回错误列表（空 = 通过）。只查 D27 新增的字段，题集内容仍由 `eval_dataset.validate` 管。"""
    errors: list[str] = []
    for index, item in enumerate(items):
        label = item.get("qid") or f"#{index}"
        if not (item.get("question") or "").strip():
            errors.append(f"{label}: question（即 query_zh）为空")
        if not is_paired(item):
            errors.append(f"{label}: query_en 缺失或与中文原文相同 → 不成对")
        status = item.get("query_en_status")
        if status not in PAIR_STATUSES:
            errors.append(f"{label}: query_en_status 必须是 {PAIR_STATUSES}，实际 {status!r}")
        elif status == "draft" and not allow_drafts:
            errors.append(f"{label}: query_en 还是 draft（机翻草稿），必须人工过审后标 reviewed/edited")
        topic = item.get("topic")
        if topic not in TOPICS:
            errors.append(f"{label}: topic 必须是 {TOPICS}，实际 {topic!r}")
        elif topic == "cross_chunk" and item.get("type") != "multi_hop":
            errors.append(f"{label}: topic=cross_chunk 的题 type 应为 multi_hop")
        elif topic == "no_answer" and item.get("type") != "negative":
            errors.append(f"{label}: topic=no_answer 的题 type 应为 negative")
        if item.get("difficulty") not in DIFFICULTIES:
            errors.append(f"{label}: difficulty 必须是 {DIFFICULTIES}，实际 {item.get('difficulty')!r}")
        retention = token_retention(item.get("question") or "", item.get("query_en") or "")
        if retention < 1.0:
            missing = sorted(
                t for t in protect_tokens(item.get("question") or "")
                if _normalize_token(t) not in _normalize_token(item.get("query_en") or "")
            )
            errors.append(f"{label}: 术语/数字没有原样保留（保留率 {retention}），缺：{missing}")

    total = len(items)
    if total < min_total:
        errors.append(f"总题数 {total} < {min_total}")
    if require_topic_coverage:
        result = coverage(items)
        if result["topics_missing"]:
            errors.append(f"topic 覆盖不全，缺：{result['topics_missing']}")
    return errors


# ---------------------------------------------------------------------------
# 起草
# ---------------------------------------------------------------------------


def draft_items(
    items: list[dict],
    translator: Callable[[list[str]], list[str]],
    *,
    force: bool = False,
) -> int:
    """给缺 `query_en` 的题生成英文草稿（标 `draft`）。返回改了几条。

    `translator` 收一批中文问法、返回等长英文列表——这样批量调用只付一次模型
    加载开销（D23 的正式服务接口也是这个形状）。
    """
    targets = [
        index for index, item in enumerate(items)
        if (item.get("question") or "").strip()
        and (force or not (item.get("query_en") or "").strip())
    ]
    if not targets:
        return 0
    drafts = translator([items[index]["question"] for index in targets])
    if len(drafts) != len(targets):
        raise ValueError(f"translator 返回 {len(drafts)} 条，期望 {len(targets)} 条")
    for index, draft in zip(targets, drafts):
        items[index]["query_en"] = (draft or "").strip()
        items[index]["query_en_status"] = "draft"
    return len(targets)


def _ollama_translator() -> Callable[[list[str]], list[str]]:
    """默认的草稿翻译器（D23 会用正式 `TranslateService` 替换这里）。

    只做草稿：输出必须人工过审，所以这里不追求完美，但要**保住术语与数字**。
    模型由 `OLLAMA_CHAT_MODEL` 环境变量决定（与主链路一致）。
    """
    from .ollama_client import chat

    system = (
        "You convert a Chinese RAG retrieval query into a short English retrieval query. "
        "Rules: keep product models, acronyms, numbers and parameter names EXACTLY as written "
        "(e.g. DGX B300, HNSW, NCU, 8192, ef_construct). Do not answer the question. "
        "Do not add context. Output the English query only, one line, no quotes."
    )

    def translate(texts: list[str]) -> list[str]:
        out = []
        for text in texts:
            reply = chat([{"role": "system", "content": system},
                          {"role": "user", "content": text}], temperature=0,
                         think=False, num_predict=96)
            out.append((reply or "").strip().strip('"').splitlines()[0] if reply else "")
        return out

    return translate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(items: Sequence[dict], path: Path = QUESTIONS_PATH) -> None:
    Path(path).write_text(json.dumps(list(items), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.eval_bilingual", description="D27 中英成对题集工具")
    parser.add_argument("--file", default=str(QUESTIONS_PATH))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="成对率 / topic 覆盖 / 待审数量")

    draft = sub.add_parser("draft", help="生成英文草稿（只补 query_en 为空的）")
    draft.add_argument("--force", action="store_true", help="连已有的也重翻")
    draft.add_argument("--model", default=None, help="（已废弃，模型由 OLLAMA_CHAT_MODEL 决定）")
    draft.add_argument("--dry-run", action="store_true")

    review = sub.add_parser("review", help="人工改写后标记过审")
    review.add_argument("--qid", required=True)
    review.add_argument("--en", default=None, help="新的英文问法（不给则仅把现有内容标为 reviewed）")
    review.add_argument("--difficulty", default=None)
    review.add_argument("--topic", default=None)
    review.add_argument("--accept-draft", action="store_true", help="草稿不改直接标 reviewed")

    check_cmd = sub.add_parser("check", help="CI 校验")
    check_cmd.add_argument("--allow-drafts", action="store_true")
    check_cmd.add_argument("--min-total", type=int, default=0)
    args = parser.parse_args(argv)

    path = Path(args.file)
    items = load(path)

    if args.command == "status":
        result = coverage(items)
        print(f"题数：{result['total']}；成对率：{result['pair_rate']:.2%}；待审草稿：{result['drafts']}")
        print(f"topic 覆盖：{result['topics_covered']}/8")
        for topic in TOPICS:
            print(f"  {topic:16s} {TOPIC_LABELS[topic]:14s} {result['counts'][topic]:3d}")
        print(f"difficulty：{result['difficulties']}")
        return 0 if not result["topics_missing"] else 1

    if args.command == "draft":
        changed = draft_items(items, _ollama_translator(), force=args.force)
        if args.dry_run:
            for item in items:
                if item.get("query_en_status") == "draft":
                    print(f"{item['qid']}: {item['question']}\n    → {item.get('query_en')}")
            print(f"（dry-run）将写入 {changed} 条草稿")
            return 0
        save(items, path)
        print(f"已写入 {changed} 条英文草稿（status=draft，需人工过审）")
        return 0

    if args.command == "review":
        target = next((item for item in items if item.get("qid") == args.qid), None)
        if target is None:
            print(f"找不到 {args.qid}", file=sys.stderr)
            return 2
        if args.en is not None:
            target["query_en"] = args.en.strip()
            target["query_en_status"] = "edited"
        elif args.accept_draft:
            target["query_en_status"] = "reviewed"
        elif target.get("query_en_status") == "draft":
            print("草稿未改，确认要标 reviewed 请加 --accept-draft", file=sys.stderr)
            return 2
        if args.difficulty:
            target["difficulty"] = args.difficulty
        if args.topic:
            target["topic"] = args.topic
        save(items, path)
        print(f"{args.qid}: {target.get('query_en')!r}（{target.get('query_en_status')}）")
        return 0

    if args.command == "check":
        errors = check(items, allow_drafts=args.allow_drafts, min_total=args.min_total)
        for error in errors:
            print(f"  [error] {error}")
        print("校验通过。" if not errors else f"校验失败：{len(errors)} 个错误。")
        return 1 if errors else 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
