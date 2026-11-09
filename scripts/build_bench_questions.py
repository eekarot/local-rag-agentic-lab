"""给基准数据集出题（`datasets/k8s-zh-en-v1/questions.jsonl`）。

沿用 D27c 的纪律，一条不改：

1. **只引用候选句子的 key**（`c0001`…），不手抄 span —— 脚本按 key 取原文，
   所以 `gold_spans` 100% 是语料原文，不会因为转写掉字而定位失败；
2. **题目必须改写**，不许把句子改成问句了事；`query_en` 人工写，不用机翻回填；
3. **命中块数 ≤ 4**，否则算 ambiguous，人工点名；
4. 一个题目一个 `expected_evidence_lang`：中文题配中文证据、英文题配英文证据，
   跨语言题（中文问 → 英文证据）单独标出来，指标按它切片。

子命令：

    python scripts/build_bench_questions.py dump --pages 30 --per-page 8
    python scripts/build_bench_questions.py append --drafts <draft.jsonl> [--dry-run]

draft 每行一个 JSON：

    {"qid": "k8s-001", "question": "……", "query_en": "……",
     "expected_evidence_lang": "zh-cn", "type": "single_hop",
     "topic": "concepts/containers", "spans": ["c0012"], "note": "……"}
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STORE_PATH = ROOT / "storage" / "bench" / "documents.sqlite3"
DATASET = ROOT / "datasets" / "k8s-zh-en-v1"
CANDIDATES_PATH = DATASET / "_candidates.json"
QUESTIONS_PATH = DATASET / "questions.jsonl"

NL = chr(10)  # 显式换行：这个文件里的字符串常量很多，不能用字面量换行

SENTENCE_LIMITS = (40, 240)        # 太短定位不到、太长跨块
MAX_HITS = 4                       # 一个 span 命中超过 4 个块 = ambiguous
SKIP_PREFIXES = ("-", "*", "|", "#", ">", "`", "www.", "http", "Note:", "NOTE:", "```")
SKIP_CONTAINS = ("http://", "https://" if False else "https://", "..........")


def connect() -> sqlite3.Connection:
    if not STORE_PATH.is_file():
        raise SystemExit(f"找不到基准库的文档表：{STORE_PATH}\n先建库：\n"
                         "  RAG_COLLECTION=bench_k8s_v1 python scripts/ingest.py "
                         "--dir datasets/k8s-zh-en-v1/docs --document-db storage/bench/documents.sqlite3 "
                         "--versions storage/bench/index_versions.sqlite3 --cache storage/bench/embeddings.sqlite3")
    conn = sqlite3.connect(str(STORE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def page_lang(source_uri: str) -> str:
    return "zh-cn" if source_uri.startswith("zh-cn/") else "en"


_FENCE = re.compile(r"^\s*```.*?^\s*```\s*$", re.S | re.M)
_TERMINAL = ("。", "！", "？", "；", ".", "!", "?", ";")


def strip_code_fences(text: str) -> str:
    """去掉 ``` 围栏代码块（整段删掉）。剩下的句子在原文里仍是连续子串。"""
    return _FENCE.sub(chr(10), text or "")


_SENTENCE_END = re.compile(r"[。！？；]|[.!?](?=\s|$)")


def split_sentences(text: str) -> list[str]:
    """按**段落**再按句末标点切，返回原文子串（不改写）。

    为什么不按换行切：Markdown 段落里有一堆硬换行，按换行切会得到
    `particularly useful for building dashboards and alerts.`、
    `setting them may require restarting processes.` 这类半句（实测 233 条候选里
    近一半是这种），问句根本无从写起。
    按段落 + 句末标点切，句子中间的换行原样保留（仍是原文子串，含换行也无妨）。
    """
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text or ""):
        para = para.strip()
        if not para:
            continue
        start = 0
        for match in _SENTENCE_END.finditer(para):
            piece = para[start:match.end()].strip()
            if piece:
                out.append(piece)
            start = match.end()
        tail = para[start:].strip()
        if tail:
            out.append(tail)
    return out


def usable(sentence: str) -> bool:
    if not (SENTENCE_LIMITS[0] <= len(sentence) <= SENTENCE_LIMITS[1]):
        return False
    # 只收完整句：截断的半句、YAML 行、表格行都过不了这一关
    if not sentence.endswith(_TERMINAL):
        return False
    if re.match(r"^[\w./-]+\s*:", sentence):        # `apiVersion: ...`
        return False
    if sentence.count("=") and sentence.startswith(("kubectl ", "$ ")):
        return False
    # 英文句子必须以大写/数字/反引号开头 —— 小写开头的都是被截断的半句
    if re.match(r"^[a-z]", sentence):
        return False
    # 图表定义、Mermaid 节点定义等
    if sentence.startswith(("classDef", "class ", "flowchart", "graph ")):
        return False
    if sentence.startswith(SKIP_PREFIXES):
        return False
    if "http://" in sentence or "https://" in sentence:
        return False
    # 链接碎片：`m/cncf/tag-security/blob/main/...` 这种从 URL 里掉出来的半句，
    # 实测会被当成「句子」进候选池（c0017），对出题没用。
    if any(mark in sentence for mark in ("github.com", "blob/", "wiki/", "](http")):
        return False
    if sentence.startswith(("m/", "www", "/zh-cn", "/docs")):
        return False
    # 表格行、纯标点、代码：
    if sentence.count("|") >= 2 or sentence.count("{") >= 2:
        return False
    if not re.search(r"[\u4e00-\u9fff]|[A-Za-z]{4}", sentence):
        return False
    return True


def load_parents(conn: sqlite3.Connection) -> dict[str, dict]:
    """parent 正文 + 所属文档（child 是 parent 的切片，句子从 parent 切更完整）。"""
    docs = {row["document_id"]: row for row in conn.execute(
        "SELECT document_id, source_uri, title FROM documents")}
    parents: dict[str, dict] = {}
    for row in conn.execute("SELECT parent_id, document_id, content, language, title, "
                            "section_path FROM parents"):
        doc = docs.get(row["document_id"])
        source_uri = (doc["source_uri"] if doc is not None else "") or ""
        parents[row["parent_id"]] = {
            "parent_id": row["parent_id"],
            "document_id": row["document_id"],
            "content": row["content"] or "",
            "source_uri": source_uri,
            "lang": page_lang(source_uri),
            "title": row["title"] or (doc["title"] if doc is not None else "") or "",
            "section_path": row["section_path"] or "",
        }
    return parents


def child_slices(conn: sqlite3.Connection, parents: dict[str, dict]) -> dict[str, list[tuple[str, str]]]:
    """parent_id → [(chunk_id, 该 child 的正文)]（正文由 offset 从 parent 切出来）。"""
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in conn.execute("SELECT chunk_id, parent_id, source_start, source_end FROM children"):
        parent = parents.get(row["parent_id"])
        if parent is None:
            continue
        start, end = row["source_start"], row["source_end"]
        text = parent["content"][start:end] if start is not None and end is not None else parent["content"]
        out[row["parent_id"]].append((row["chunk_id"], text))
    return out


def load_chunks(conn: sqlite3.Connection) -> list[dict]:
    """还原每个 child 的正文（children 表只有 offset，正文在 parent.content 里）。"""
    parents = {row["parent_id"]: row for row in conn.execute(
        "SELECT parent_id, document_id, content, language, title, section_path FROM parents")}
    docs = {row["document_id"]: row for row in conn.execute(
        "SELECT document_id, source_uri, title FROM documents")}
    chunks: list[dict] = []
    for row in conn.execute("SELECT chunk_id, parent_id, document_id, source_start, source_end "
                            "FROM children"):
        parent = parents.get(row["parent_id"])
        if parent is None:
            continue
        start, end = row["source_start"], row["source_end"]
        text = parent["content"][start:end] if start is not None and end is not None else parent["content"]
        doc = docs.get(row["document_id"])
        source_uri = (doc["source_uri"] if doc is not None else "") or ""
        chunks.append({
            "chunk_id": row["chunk_id"],
            "parent_id": row["parent_id"],
            "document_id": row["document_id"],
            "source_uri": source_uri,
            "lang": page_lang(source_uri),
            "title": parent["title"] or (doc["title"] if doc is not None else "") or "",
            "section_path": parent["section_path"] or "",
            "text": text,
        })
    return chunks


def pick_pages(by_page: dict[str, list[dict]], *, pages: int,
               only_pages: list[str] | None) -> list[str]:
    """挑要抽候选句的页。

    `only_pages` 给了就**原样照用**（去重、保留首次出现的位置）—— 扩题时要精确指定
    「这 20 页各补 2–3 题」，任何排序/抽样改写都会让清单对不上。
    没给才走原来的「按语言分层 + 等距抽样」，保证中英都有且可复现。
    """
    if only_pages:
        ordered: list[str] = []
        for page in only_pages:
            if page not in ordered:
                ordered.append(page)
        return ordered
    picked: list[str] = []
    for lang in ("zh-cn", "en"):
        candidates = sorted(p for p in by_page if p.startswith(f"{lang}/"))
        step = max(1, len(candidates) // max(1, pages // 2))
        picked.extend(candidates[::step][: pages // 2])
    return picked


def missing_pages(by_page: dict[str, list[dict]], wanted: list[str]) -> list[str]:
    """清单里在库中不存在的页 —— 必须报出来，不能静默少抽几页。"""
    return [page for page in wanted if page not in by_page]


def dump(args) -> int:
    conn = connect()
    chunks = load_chunks(conn)
    if not chunks:
        raise SystemExit("库里没有 child（还没建好？）")

    by_page: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_page[chunk["source_uri"]].append(chunk)

    only_pages = None
    if getattr(args, "only_pages", None):
        only_pages = [line.strip() for line in
                      Path(args.only_pages).read_text(encoding="utf-8").splitlines() if line.strip()]
        absent = missing_pages(by_page, only_pages)
        if absent:
            raise SystemExit("这些页在库里没有子块（路径写错？）：" + chr(10).join(absent))
    picked = pick_pages(by_page, pages=args.pages, only_pages=only_pages)

    parents = load_parents(conn)
    slices = child_slices(conn, parents)
    candidates: list[dict] = []
    seen: set[str] = set()
    for page in picked:
        taken = 0
        page_parents = sorted((p for p in parents.values() if p["source_uri"] == page),
                              key=lambda p: (p["parent_id"]))
        for parent in page_parents:
            for sentence in split_sentences(strip_code_fences(parent["content"])):
                if not usable(sentence):
                    continue
                # 整句必须落在某个 child 切片里 —— 否则 gold_chunks 无从映射
                holders = [cid for cid, text in slices.get(parent["parent_id"], [])
                           if sentence in text]
                if not holders or len(holders) > MAX_HITS:
                    continue
                key = sentence.lower()[:120]
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "key": f"c{len(candidates) + 1:04d}",
                    "text": sentence,
                    "chunk_id": holders[0],
                    "chunk_ids": holders,
                    "document_id": parent["document_id"],
                    "source_uri": parent["source_uri"],
                    "lang": parent["lang"],
                    "title": parent["title"],
                    "section_path": parent["section_path"],
                })
                taken += 1
                if taken >= args.per_page:
                    break
            if taken >= args.per_page:
                break

    CANDIDATES_PATH.write_text(
        json.dumps({"dataset": DATASET.name, "count": len(candidates), "candidates": candidates},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"候选句 {len(candidates)} 条，来自 {len(picked)} 页 → {CANDIDATES_PATH.relative_to(ROOT)}")
    for item in candidates:
        print(f"{item['key']} [{item['lang']}] {item['source_uri']}")
        print(f"    {item['text']}")
    return 0


def resolve(drafts: list[dict], candidates: dict[str, dict], chunks: list[dict]) -> tuple[list[dict], list[str]]:
    by_chunk = {chunk["chunk_id"]: chunk for chunk in chunks}
    out: list[dict] = []
    problems: list[str] = []
    seen_qid: set[str] = set()

    for draft in drafts:
        qid = draft.get("qid") or "<无 qid>"
        if qid in seen_qid:
            problems.append(f"{qid}：qid 重复")
        seen_qid.add(qid)

        # 负样本（答案不在库里）：允许没有 span，但必须显式标注 type=negative，
        # 否则就是「忘了写 gold」而不是「故意考弃答」。负样本必须留痕。
        if draft.get("type") == "negative":
            if draft.get("spans"):
                problems.append(f"{qid}：negative 题不该有 gold spans")
            out.append({
                "qid": qid,
                "question": (draft.get("question") or "").strip(),
                "query_en": (draft.get("query_en") or "").strip(),
                "expected_evidence_lang": None,
                "type": "negative",
                "topic": draft.get("topic") or "",
                "source_path": "",
                "gold_spans": [],
                "gold_chunks": [],
                "gold_points": [],
                "note": draft.get("note") or "",
            })
            continue

        spans = draft.get("spans") or []
        if not spans:
            problems.append(f"{qid}：没有 spans")
            continue

        texts: list[str] = []
        hits: set[str] = set()          # child（point）id
        parents: set[str] = set()       # parent（chunk）id —— 检索返回的就是这个粒度
        for key in spans:
            item = candidates.get(key)
            if item is None:
                problems.append(f"{qid}：候选 key 不存在 {key}")
                continue
            texts.append(item["text"])
            for chunk in chunks:
                if item["text"] in chunk["text"]:
                    hits.add(chunk["chunk_id"])
                    parents.add(chunk["parent_id"])
        if not texts:
            continue
        if len(hits) > MAX_HITS:
            problems.append(f"{qid}：span 命中 {len(hits)} 个块（>{MAX_HITS}），算 ambiguous，缩小或点名")
        if not hits:
            problems.append(f"{qid}：span 在任何块里都定位不到")

        expected = draft.get("expected_evidence_lang")
        if expected:
            langs = {by_chunk[cid]["lang"] for cid in hits if cid in by_chunk}
            if langs and expected not in langs:
                problems.append(f"{qid}：期望证据语言 {expected}，但命中块是 {sorted(langs)}")

        question = (draft.get("question") or "").strip()
        query_en = (draft.get("query_en") or "").strip()
        if not question:
            problems.append(f"{qid}：question 为空")
        if not query_en:
            problems.append(f"{qid}：query_en 必须人工写（不用机翻回填）")

        out.append({
            "qid": qid,
            "question": question,
            "query_en": query_en,
            "expected_evidence_lang": expected,
            "type": draft.get("type") or "single_hop",
            "topic": draft.get("topic") or "",
            "source_path": candidates[spans[0]]["source_uri"] if spans[0] in candidates else "",
            "gold_spans": texts,
            # 两个不相交的 id 空间，别混：
            #   gold_chunks  = parent id（检索返回的粒度，评估用这个）
            #   gold_points  = child id（向量点，建库/删库用这个）
            "gold_chunks": sorted(parents),
            "gold_points": sorted(hits),
            "note": draft.get("note") or "",
        })
    return out, problems


def append(args) -> int:
    conn = connect()
    chunks = load_chunks(conn)
    if not CANDIDATES_PATH.is_file():
        raise SystemExit(f"没有候选句子文件：{CANDIDATES_PATH}（先跑 dump）")
    payload = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    candidates = {item["key"]: item for item in payload["candidates"]}

    drafts = [json.loads(line) for line in Path(args.drafts).read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.strip().startswith("//")]
    resolved, problems = resolve(drafts, candidates, chunks)

    if problems:
        print("校验未通过：", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 1

    if getattr(args, "merge", False):
        existing = ([json.loads(line) for line in
                     QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if QUESTIONS_PATH.is_file() else [])
        resolved, problems = merge_questions(existing, resolved)
        if problems:
            print("合并未通过：", file=sys.stderr)
            for line in problems:
                print(f"  - {line}", file=sys.stderr)
            return 1

    if args.dry_run:
        print(f"[dry-run] 通过校验 {len(resolved)} 题，未写文件")
        return 0

    QUESTIONS_PATH.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in resolved) + "\n", encoding="utf-8")
    by_type = defaultdict(int)
    by_lang = defaultdict(int)
    for item in resolved:
        by_type[item["type"]] += 1
        by_lang[item["expected_evidence_lang"]] += 1
    print(f"写入 {len(resolved)} 题 → {QUESTIONS_PATH.relative_to(ROOT)}")
    print(f"  类型：{dict(by_type)}")
    print(f"  证据语言：{dict(by_lang)}")
    return 0


def merge_questions(existing: list[dict], resolved: list[dict]) -> tuple[list[dict], list[str]]:
    """老题 + 新题 = 新题集。老题**一条都不动**（id 已经是好的），只往后接。

    这里坚持三件事：qid 不许撞、新草稿内部不许重、题干不许与老题完全一致
    （完全一致基本就是复制粘贴，不是「改写」）。
    """
    problems: list[str] = []
    seen = {item["qid"] for item in existing}
    seen_text = {(item.get("question") or "").strip() for item in existing}
    added: list[dict] = []
    for item in resolved:
        qid = item["qid"]
        if any(r["qid"] == qid for r in added):
            problems.append(f"{qid}：新草稿里 qid 重复")
            continue
        if qid in seen:
            problems.append(f"{qid}：qid 已存在（老题）")
            continue
        seen.add(qid)
        text = (item.get("question") or "").strip()
        if text and text in seen_text:
            problems.append(f"{qid}：与已有题重复出题（题干完全一致）")
            continue
        seen_text.add(text)
        added.append(item)
    return list(existing) + added, problems


def remap_gold(
    question: dict, chunks: list[dict], *, same_page_only: bool = False
) -> tuple[list[str], list[str], list[str]]:
    """按 `gold_spans` 在当前库里重算 `gold_chunks`（父块）/ `gold_points`（子块）。

    **默认全库匹配，不限制页面** —— 这是出题时的原口径（`append` 也是这么做的）：
    gold 回答的问题是「这段证据文字有没有被检出来」，不是「是不是从我想的那一页检出来的」。
    改了它等于换了衡量的东西：实测限制到 `source_path` 会把这 3 道题判成
    「检索错了」，可它们检回的孪生页面上本来就写着同一段话（`k8s-e-033` 从命中变不命中）。

    想严判「必须是这一页」时传 `same_page_only=True`（命令行 `--same-page`），
    但要意识到那会改变基线口径，不能跟旧数字直接比。
    """
    if question.get("type") == "negative":
        return (list(question.get("gold_chunks") or []),
                list(question.get("gold_points") or []), [])

    spans = [text for text in (question.get("gold_spans") or []) if text]
    if not spans:
        return (list(question.get("gold_chunks") or []),
                list(question.get("gold_points") or []),
                [f"{question.get('qid')}：没有 gold_spans，无法重算（保留原值）"])

    page = question.get("source_path") or ""
    hits = {
        chunk["chunk_id"]: chunk["parent_id"]
        for chunk in chunks
        if (not same_page_only or chunk["source_uri"] == page)
        and any(span in chunk["text"] for span in spans)
    }
    if not hits:
        return (list(question.get("gold_chunks") or []),
                list(question.get("gold_points") or []),
                [f"{question.get('qid')}：span 在当前库里定位不到（保留原值）—— 题可能坏了，别当没事"])
    problems = []
    if len(hits) > MAX_HITS:
        problems.append(
            f"{question.get('qid')}：span 命中 {len(hits)} 个块（>{MAX_HITS}），出题时算 ambiguous"
        )
    return sorted(set(hits.values())), sorted(hits), problems


def refresh_ids(args) -> int:
    """重建库后重算题集里的两套 gold id（父块 + 子块）。

    为什么必须重算：`gold_chunks=par_*`、`gold_points=chk_*` 都是**内容/版本派生**的 id，
    切分或重叠语义一变就整体失效。题集里存着 `gold_spans`（语料原文），所以能重算。
    """
    conn = connect()
    chunks = load_chunks(conn)
    if not chunks:
        raise SystemExit("库里没有 child（还没建好？）")
    path = Path(args.questions) if args.questions else QUESTIONS_PATH
    if not path.is_file():
        raise SystemExit(f"找不到题集：{path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    out: list[str] = []
    problems: list[str] = []
    changed = 0
    for line in lines:
        question = json.loads(line)
        before = (question.get("gold_chunks"), question.get("gold_points"))
        parents, points, issues = remap_gold(question, chunks, same_page_only=args.same_page)
        question["gold_chunks"] = parents
        question["gold_points"] = points
        problems.extend(issues)
        if (parents, points) != (before[0], before[1]):
            changed += 1
        out.append(json.dumps(question, ensure_ascii=False))

    print(f"题集 {len(lines)} 行，gold id 变化 {changed} 题")
    for problem in problems:
        print(f"  [警告] {problem}")
    if args.dry_run:
        print("（--dry-run：没有写回）")
        return 0
    path.write_text(NL.join(out) + NL, encoding="utf-8")
    print(f"已写回 {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_bench_questions", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    dump_parser = sub.add_parser("dump", help="生成候选句子（带 key）")
    dump_parser.add_argument("--pages", type=int, default=30)
    dump_parser.add_argument("--per-page", type=int, default=8)
    dump_parser.add_argument("--only-pages", type=Path, default=None,
                             help="只抽这些页（一行一个 source_uri）；给了就忽略 --pages 的抽样逻辑")
    dump_parser.set_defaults(func=dump)

    append_parser = sub.add_parser("append", help="按 key 解析 span 并写 questions.jsonl")
    append_parser.add_argument("--drafts", type=Path, required=True)
    append_parser.add_argument("--dry-run", action="store_true")
    append_parser.add_argument("--merge", action="store_true",
                               help="与现有 questions.jsonl 合并（扩题用），而不是整份覆盖")
    append_parser.set_defaults(func=append)

    refresh_parser = sub.add_parser(
        "refresh-ids", help="重建库后按 gold_spans 重算 gold_chunks / gold_points"
    )
    refresh_parser.add_argument("--questions", type=Path, default=None)
    refresh_parser.add_argument("--dry-run", action="store_true")
    refresh_parser.add_argument("--same-page", action="store_true",
                                help="只认题目 source_path 那一页（严判；会改变基线口径）")
    refresh_parser.set_defaults(func=refresh_ids)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
