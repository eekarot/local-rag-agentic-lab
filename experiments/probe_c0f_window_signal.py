"""C0f：跨语言「挑窗口」到底还有没有信号可用（只量，不改默认）。

背景（N54 / `docs/evidence/window-strategy-k8s-v1.json`）：重排只吃 256 token，
候选父块 p50 = 322 token、63% 超过 256，所以「看哪一段」是必答题。
生产做法 `best_window()` 按**与 query 的词面重叠**挑窗口；没有重叠就退回块首。

C0f 要验证的假设是：**跨语言时重叠恒为 0，退路成了唯一策略**，于是窗口恒定是块首，
英文问中文证据时答案句常常不在里面（N54 量到「窗口含答案句」英→中只有 0.33）。

这个探针不下结论，只回答四个数（全部在**真实候选池**上算，池子口径 = pool40 / translate）：
  1. 有多少候选父块真的**有得挑**（窗口数 > 1）；
  2. 其中有多少**词面重叠为 0**（退路被触发的比例），按四格拆；
  3. 「窗口里有没有答案句」：`overlap` 挑的 / 恒定块首 / 随便哪一段都行（上限）；
  4. 上限与现状的差 = **换个挑法最多还能赚多少**（若差≈0，C0f 就该被否掉）。

不含交叉编码器，所以只跑检索，180 题约 4 分钟。
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_bench import (BENCH_STORE, QUESTIONS, _build_retriever, build_translate_fn,
                               load_parents, load_questions)
from src.rag.ask.pipeline import _parent_text
from src.rag.ingestion.tokenizer import TokenCounter
from src.rag.retrieval.rerank import best_window, window_candidates

CJK = re.compile(r"[\u4e00-\u9fff]")
POOL = 40
MAX_TOKENS = 256

questions = load_questions(QUESTIONS)
parents = load_parents(BENCH_STORE)
counter = TokenCounter()
retriever = _build_retriever("bench_k8s_v1", BENCH_STORE, [], lambda _m: None)
translate = build_translate_fn()


def cell_of(question: dict) -> str:
    zh_q = bool(CJK.search(question["question"]))
    zh_e = question.get("expected_evidence_lang") == "zh-cn"
    return "中→中" if zh_q and zh_e else "中→英" if zh_q else "英→中" if zh_e else "英→英"


def overlap_terms(query: str) -> set[str]:
    """与 `rerank._query_terms` 完全同口径：英文按词、中文按字 bigram。"""
    lowered = (query or "").lower()
    terms = set(re.findall(r"[a-z0-9_]+", lowered))
    han = "".join(CJK.findall(lowered))
    terms.update(han[i:i + 2] for i in range(max(0, len(han) - 1)))
    if len(han) == 1:
        terms.add(han)
    return {term for term in terms if len(term) >= 2}


rows = []
for question in questions:
    if question["type"] == "negative":
        continue
    qid = question["qid"]
    cell = cell_of(question)
    spans = [s for s in (question.get("gold_spans") or []) if s and len(s.strip()) >= 8]
    gold = set(question.get("gold_chunks") or [])
    english = translate(question["question"]) if cell.startswith("中") else None
    fused = retriever.retrieve(question["question"], query_en=english, k=POOL, fuse=True).fused
    terms = overlap_terms(question["question"])
    picked_anything = False

    for rank, hit in enumerate(fused, start=1):
        if hit.parent_id not in gold:
            continue                     # 只量 gold 父块：非 gold 块的窗口对错没有意义
        text = _parent_text(parents.get(hit.parent_id))
        windows = window_candidates(text, max_tokens=MAX_TOKENS, counter=counter)
        if len(windows) < 2:
            rows.append({"qid": qid, "cell": cell, "rank": rank, "windows": len(windows),
                         "multi": False})
            continue
        scores = []
        for window in windows:
            lowered = window.lower()
            scores.append(sum(1 for term in terms if term in lowered))
        overlap_idx = int(windows.index(best_window(text, question["question"],
                                                   max_tokens=MAX_TOKENS, counter=counter)))
        gold_idx = [i for i, window in enumerate(windows)
                    if any(span[:60] in window for span in spans)]
        rows.append({"qid": qid, "cell": cell, "rank": rank, "windows": len(windows), "multi": True,
                     "max_overlap": max(scores), "nonzero_windows": sum(1 for s in scores if s > 0),
                     "overlap_idx": overlap_idx, "gold_idx": gold_idx})
        picked_anything = True

    print(f"{qid:12} {cell:6} gold在池中={picked_anything}", flush=True)


# ---------------- 汇总 ----------------
def rate(values) -> float:
    values = list(values)
    return round(sum(1 for v in values if v) / len(values), 4) if values else None


summary = {}
for cell in ("中→中", "中→英", "英→中", "英→英", "整体"):
    rows_cell = [r for r in rows if cell == "整体" or r["cell"] == cell]
    multi = [r for r in rows_cell if r["multi"]]
    zero = [r for r in multi if r["max_overlap"] == 0]
    summary[cell] = {
        "gold_blocks": len(rows_cell),
        "multi_window": len(multi),
        "multi_window_share": rate(r["multi"] for r in rows_cell),
        "windows_p50": sorted(r["windows"] for r in multi)[len(multi) // 2] if multi else None,
        "zero_overlap_blocks": len(zero),
        "zero_overlap_share_of_multi": rate(r["max_overlap"] == 0 for r in multi),
        "gold_in_picked": rate(any(i == r["overlap_idx"] for i in r["gold_idx"]) for r in multi),
        "gold_in_head": rate(0 in r["gold_idx"] for r in multi),
        "gold_in_last": rate((r["windows"] - 1) in r["gold_idx"] for r in multi),
        "gold_in_any": rate(bool(r["gold_idx"]) for r in multi),
        "zero_overlap_gold_in_picked": rate(any(i == r["overlap_idx"] for i in r["gold_idx"]) for r in zero),
        "zero_overlap_gold_in_head": rate(0 in r["gold_idx"] for r in zero),
        "zero_overlap_gold_in_any": rate(bool(r["gold_idx"]) for r in zero),
    }

# 上限与现状的差：只对「有得挑」的块说话
headroom = {cell: (round(s["gold_in_any"] - s["gold_in_picked"], 4)
                   if s["gold_in_any"] is not None and s["gold_in_picked"] is not None else None)
            for cell, s in summary.items()}

payload = {
    "note": "C0f 前置测量：跨语言时窗口挑选还有没有信号（只读，不含交叉编码器）",
    "params": {"pool": POOL, "max_tokens": MAX_TOKENS, "collection": "bench_k8s_v1",
               "questions": len(questions), "gold_blocks_only": True},
    "question": "重叠为 0 时退路=块首，与「随便挑一段都对」的上限差多少",
    "summary": summary,
    "headroom_gold_in_any_minus_gold_in_picked": headroom,
    "rows": rows,
}
dest = ROOT / "docs/evidence/c0f-window-signal-k8s-v1.json"
dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print()
header = (f"{'格':<8}{'gold块':>8}{'有得挑':>8}{'窗口p50':>9}{'重叠0占比':>11}"
          f"{'现状含答案':>11}{'块首':>8}{'末窗':>8}{'上限':>8}{'可赚':>8}")
print(header)
for cell, s in summary.items():
    def fmt(v, width=8):
        return f"{v:>{width}.4f}" if isinstance(v, float) else f"{'—':>{width}}"
    print(f"{cell:<8}{s['gold_blocks']:>8}{s['multi_window']:>8}"
          f"{(s['windows_p50'] or 0):>9}{fmt(s['zero_overlap_share_of_multi'], 11)}"
          f"{fmt(s['gold_in_picked'], 11)}{fmt(s['gold_in_head'])}{fmt(s['gold_in_last'])}"
          f"{fmt(s['gold_in_any'])}{fmt(headroom[cell])}")
print("\n只看「重叠=0」的块：")
for cell, s in summary.items():
    print(f"  {cell:<8} 现状 {s['zero_overlap_gold_in_picked']} / 块首 {s['zero_overlap_gold_in_head']}"
          f" / 上限 {s['zero_overlap_gold_in_any']}（n={s['zero_overlap_blocks']}）")
print("\n证据 →", dest.relative_to(ROOT))
