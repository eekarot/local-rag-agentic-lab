"""D27c：评估集 101 → ~200 题（成对率 100%，`gold_spans` 必须是语料原文）。

与 `expand_questions.py` 的区别：**新题的 gold 走新库（parent/child）**，并新增
「句子候选」机制来彻底消除抄写错误：

1. `dump` 从 DocumentStore 的父块正文里切出候选句子，落地 `experiments/_span_candidates.json`，
   每条有稳定 key（`c0001`…）与原文字符串；
2. 写题时只引用 **key**，不手抄 span —— 脚本按 key 取原文，所以 `gold_spans`
   100% 是语料原文，不可能因为转写掉字而定位失败；
3. `append` 校验：key 存在 → span 能在父块里定位到 → 命中父块数 ≤4（否则算 ambiguous，人工点名）。

    python -m experiments.expand_questions_v2 dump --docs 26 --per-doc 8
    python -m experiments.expand_questions_v2 append --dry-run
    python -m experiments.expand_questions_v2 append
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import QUESTIONS_PATH

ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / "storage" / "documents.sqlite3"
CANDIDATES_PATH = ROOT / "experiments" / "_span_candidates.json"

ZH_MARKERS = ("rag_basics.md", "agent_basics.md", "AI-Agents-in-Depth-zh-CN.pdf")
SENTENCE_LIMITS = (30, 220)      # 太短定位不到、太长跨父块


# --------------------------------------------------------------- 候选句子


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句，保留原文（不做任何改写）。"""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；])|(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _connect():
    conn = sqlite3.connect(str(STORE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


SKIP_PREFIXES = ("-", "*", "|", "#", ">", "`", "Note:", "NOTE:", "www.", "http")
SKIP_CONTAINS = ("http://", "https://", "..........", "....")


def _usable(sentence: str) -> bool:
    """剔掉标题行、目录行、URL、表格行——它们不适合做检索题。

    最有效的一条判据：**标题不以句末标点结尾**。加上这条后候选里几乎不再出现
    `First Boot Process for DGX Servers` 这种纯标题。
    """
    if not (SENTENCE_LIMITS[0] <= len(sentence) <= SENTENCE_LIMITS[1]):
        return False
    if not sentence.endswith((".", "。", "?", "？", "!", "！", ":")) and not sentence.endswith("。"):
        return False
    if sentence.startswith(SKIP_PREFIXES):
        return False
    if any(marker in sentence for marker in SKIP_CONTAINS):
        return False
    if "|" in sentence or "](" in sentence:
        return False
    # 至少要有 3 个英文单词或 12 个汉字，否则是标签/型号行
    latin_words = len(re.findall(r"[A-Za-z]{2,}", sentence))
    chinese = len(re.findall(r"[一-鿿]", sentence))
    return latin_words >= 6 or chinese >= 15


def collect_candidates(
    docs: int, per_doc: int, *, skip_covered: bool, seed: int = 0,
    only: list[str] | None = None,
) -> list[dict]:
    """按文档取父块正文里的候选句，优先「还没有 gold 的文档」。

    父块在文档内**跨段落取样**（stride），不只看开头——开头往往是标题和目录。
    """
    import random

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    covered: set[str] = set()
    conn = _connect()
    parent_source = {
        row["parent_id"]: row["source_uri"]
        for row in conn.execute(
            "SELECT p.parent_id, d.source_uri FROM parents p JOIN documents d ON d.document_id = p.document_id"
        )
    }
    for item in questions:
        for parent_id in (item.get("gold_points") or {}).get("parent_ids") or []:
            source = parent_source.get(parent_id)
            if source:
                covered.add(source)

    rows = conn.execute(
        "SELECT p.parent_id, p.content, p.title, p.section_path, p.parent_index, d.source_uri"
        " FROM parents p JOIN documents d ON d.document_id = p.document_id"
        " ORDER BY d.source_uri, p.parent_index"
    ).fetchall()
    conn.close()

    by_doc: dict[str, list] = {}
    for row in rows:
        by_doc.setdefault(row["source_uri"], []).append(row)

    sources = sorted(s for s, items in by_doc.items() if len(items) >= per_doc)
    if only:
        sources = [s for s in sources if any(pattern in s for pattern in only)]
    if skip_covered:
        preferred = [s for s in sources if s not in covered]
        sources = preferred or sources
    rng = random.Random(seed)
    rng.shuffle(sources)
    sources = sources[:docs]

    candidates: list[dict] = []
    for source in sorted(sources):
        rows = by_doc[source]
        # 跨段落取样：从头到尾均匀取，跳过最短的前 5%（通常是标题/TOC）
        start = max(1, len(rows) // 20)
        span = rows[start:]
        step = max(1, len(span) // max(per_doc * 3, 1))
        picked = 0
        seen: set[str] = set()
        for row in span[::step]:
            for sentence in split_sentences(row["content"] or ""):
                if not _usable(sentence):
                    continue
                key = sentence.lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "key": f"c{len(candidates) + 1:04d}",
                    "source": source,
                    "title": row["title"],
                    "section_path": row["section_path"],
                    "parent_index": row["parent_index"],
                    "text": sentence,
                })
                picked += 1
                if picked >= per_doc:
                    break
            if picked >= per_doc:
                break
    return candidates


# --------------------------------------------------------------- 校验


def resolve_spans(keys: list[str], candidates: dict[str, dict]) -> tuple[list[str], list[str]]:
    spans, problems = [], []
    for key in keys:
        item = candidates.get(key)
        if item is None:
            problems.append(f"候选 key 不存在：{key}")
            continue
        spans.append(item["text"])
    return spans, problems


def locate_parents(span: str) -> list[str]:
    """在 DocumentStore 父块里定位（先严格、再折叠空白兜底）。"""
    collapsed = re.sub(r"\s+", " ", span.strip())
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT parent_id FROM parents WHERE content LIKE ?", (f"%{span}%",)
        ).fetchall()
        if not rows:
            rows = conn.execute(
                "SELECT parent_id FROM parents WHERE REPLACE(REPLACE(content, char(10), ' '), '  ', ' ') LIKE ?",
                (f"%{collapsed}%",),
            ).fetchall()
    finally:
        conn.close()
    return [row["parent_id"] for row in rows]


def corpus_language_of(parent_ids: list[str]) -> str | None:
    if not parent_ids:
        return None
    placeholders = ",".join("?" for _ in parent_ids)
    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT DISTINCT d.source_uri FROM parents p JOIN documents d ON d.document_id = p.document_id"
            f" WHERE p.parent_id IN ({placeholders})",
            parent_ids,
        ).fetchall()
    finally:
        conn.close()
    sources = [row["source_uri"] for row in rows]
    if not sources:
        return None
    return "zh" if all(any(marker in s for marker in ZH_MARKERS) for s in sources) else "en"


# --------------------------------------------------------------- 追加


def append(items: list[dict], candidates: dict[str, dict], *, dry_run: bool) -> int:
    existing = {item["question"] for item in items}
    index = max(int(item["qid"].split("-")[1]) for item in items)
    problems: list[str] = []
    added = 0
    skipped = 0

    for spec in NEW_V2:
        question = spec["question"].strip()
        if question in existing:
            skipped += 1      # 重跑本脚本时已存在 = 幂等，不算问题
            continue
        spans, span_problems = resolve_spans(spec["spans"], candidates)
        problems.extend(f"{question[:24]}… {problem}" for problem in span_problems)
        if not spans and spec["type"] != "negative":
            problems.append(f"无 span：{question[:24]}")
            continue

        parent_ids: list[str] = []
        for span in spans:
            hits = locate_parents(span)
            if not hits:
                problems.append(f"定位失败：{question[:24]}… / {span[:36]}")
            limit = int(spec.get("max_parents", 4))
            if len(hits) > limit:
                problems.append(
                    f"span 命中 {len(hits)} 个父块（超过允许的 {limit}）：{question[:24]}…"
                )
            for parent_id in hits[:4]:
                if parent_id not in parent_ids:
                    parent_ids.append(parent_id)

        index += 1
        items.append({
            "qid": f"eval-{index:03d}",
            "question": question,
            "query_en": spec["query_en"].strip(),
            "query_en_status": "edited",     # 本文件里的英文问法是逐条写的，不是机翻回填
            "type": spec["type"],
            "topic": spec["topic"],
            "difficulty": spec["difficulty"],
            "lang": "zh",
            "corpus_lang": corpus_language_of(parent_ids),
            "gold_spans": spans,
            "gold_chunks": [],
            "gold_points": {
                "parent_ids": parent_ids,
                "child_ids": [],
                "status": "ok" if parent_ids else "negative",
                "match_mode": "exact" if parent_ids else "none",
            },
            "reference_answer": spec.get("reference", ""),
            "note": spec.get("note", ""),
        })
        added += 1

    counts: dict[str, int] = {}
    for item in items:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    print(f"新增 {added} 题（跳过已存在 {skipped} 条）→ 共 {len(items)} 题：{counts}")
    for problem in problems:
        print("  [问题]", problem)
    if dry_run:
        print("（dry-run）未写入")
        return 0 if not problems else 1
    QUESTIONS_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已写入 {QUESTIONS_PATH}")
    return 0 if not problems else 1


# --------------------------------------------------------------- 题目（按批补）

# 每条：question / query_en / type / topic / difficulty / spans(候选 key) / reference / note
#
# 撰写纪律（与 §17.11.3 / B06 一致）：
# - 题目**改写**语料说法，不照抄句子；
# - `query_en` 是独立写的英文问法，不是中文的机翻；
# - 中文问法里的型号/缩写/数字必须原样出现在 `query_en`（`check` 的术语保留率要求 1.0）；
# - 不出「四选一」式候选确认题。
NEW_V2: list[dict] = [
    # ---------------------------------------------------------------- RAG 入库（英文语料）
    {"question": "这份入库指南的重点是哪类数据的批量入库？",
     "query_en": "Which data type does the ingest guide focus on for bulk ingest?",
     "type": "fact", "topic": "concept_zh", "difficulty": "normal",
     "spans": ["c0011"],
     "reference": "纯文本（text-only）的 Wiki Text 批量入库，而不是多模态数据。"},
    {"question": "把嵌入先写到 MinIO 之后，批量进 Milvus 的分段是多大？",
     "query_en": "What segment size is used to bulk ingest Milvus after embeddings are stored in MinIO?",
     "type": "fact", "topic": "number_spec", "difficulty": "normal",
     "spans": ["c0013"],
     "reference": "按 1 GB 分段（每段约 1K 文档）直接写入 Milvus。"},
    {"question": "Ingestor 服务用哪个端点接收文档入库请求？",
     "query_en": "Which endpoint does the Ingestor server use to receive document ingest requests?",
     "type": "fact", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0014"],
     "reference": "`/v1/documents`，用 HTTP GET/POST 提交，返回 200/400/500。"},
    {"question": "NV-Ingest 侧能撑多高的文档并发，取决于哪些资源？",
     "query_en": "What does the achievable document concurrency on the NV-Ingest side depend on?",
     "type": "fact", "topic": "multi_condition", "difficulty": "normal",
     "spans": ["c0016"],
     "reference": "取决于可用的 NV-Ingest server worker 数量与分配的 CPU 资源。"},
    {"question": "做 NV-Ingest 入库压测时，客户端和服务端分别要怎么扩？",
     "query_en": "When benchmarking NV-Ingest ingestion, how should the client side and the server side be scaled respectively?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0009", "c0010"],
     "reference": "客户端提高文档并发请求数（CR），服务端把 NV-Ingest 与 Embedding/Extraction NIM 一起扩。"},

    # ---------------------------------------------------------------- Mission Control
    {"question": "在 Mission Control 里，一个 Scaling Unit 对应多少台 DGX 节点？",
     "query_en": "In Mission Control, how many DGX nodes make up one Scaling Unit?",
     "type": "fact", "topic": "number_spec", "difficulty": "easy",
     "spans": ["c0042"],
     "reference": "B200/B300 架构下 1 个 Scaling Unit = 32 台 DGX 节点。"},
    {"question": "Mission Control 给 B200/B300 和 GB200/GB300 用的控制面节点，CPU 架构分别是什么？",
     "query_en": "What CPU architecture do Mission Control control plane nodes use for B200/B300 systems, and for GB200/GB300 systems?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0043", "c0048"],
     "reference": "B200/B300 全是 x86；GB200/GB300 全是 Arm。"},
    {"question": "B200/B300 的 Mission Control 把管理职责分给几组控制面节点？",
     "query_en": "Into how many groups of control-plane nodes does Mission Control for B200/B300 separate management responsibilities?",
     "type": "fact", "topic": "number_spec", "difficulty": "normal",
     "spans": ["c0044"],
     "reference": "四组 x86 控制面节点。"},
    {"question": "Slurm 作业开始和结束时自动执行的那组健康检查被称为什么？",
     "query_en": "What are the health checks called that Slurm runs automatically at the start and end of a job?",
     "type": "concept", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0065", "c0066"],
     "reference": "Prolog 与 Epilog 检查。"},
    {"question": "这些 Prolog/Epilog 脚本能不能在 AHR 界面里查看或配置？",
     "query_en": "Can these Prolog and Epilog scripts be viewed or configured in the AHR UI?",
     "type": "fact", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0067"],
     "reference": "不能；脚本自动下发到节点，界面里看不到也改不了。"},
    {"question": "周期性检查在 AHR 里被定义成什么？满足什么条件会自动启用？",
     "query_en": "What are periodic checks defined as in AHR, and under what condition are they enabled automatically?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0070", "c0071"],
     "reference": "定义成 Alarm；通过 Single Rack Testing 的机架会自动启用，也可用 ALARMS_ENABLE/ALARMS_DISABLE runbook 手动开关。"},
    {"question": "AHR 除了做健康检查，还能对机架里的固件做什么？",
     "query_en": "Besides health checks, what can AHR do to the firmware inside the racks?",
     "type": "fact", "topic": "concept_zh", "difficulty": "normal",
     "spans": ["c0072"],
     "reference": "可以升级、上电重启循环（cycling）与校验固件及对应 OS。"},
    {"question": "NVLink 分区里，默认分区占用了多少个多播组？",
     "query_en": "How many multicast teams are allocated to the default partition in NVLink partitioning?",
     "type": "fact", "topic": "number_spec", "difficulty": "normal",
     "spans": ["c0074"],
     "reference": "全部 1,024 个 multicast teams。"},
    {"question": "要新建带多播组的分区，第一步必须先做什么？",
     "query_en": "What must be done first before creating new partitions with multicast teams?",
     "type": "fact", "topic": "multi_condition", "difficulty": "normal",
     "spans": ["c0075"],
     "reference": "先删掉默认分区。"},
    {"question": "给一个分区预留多少多播组，是按什么来定的？",
     "query_en": "What determines how many multicast teams should be reserved for a partition?",
     "type": "concept", "topic": "concept_zh", "difficulty": "normal",
     "spans": ["c0077"],
     "reference": "按该分区租户的预期使用量决定。"},
    {"question": "要在 NVL72 机架上用 NVOS 改分区配置，先 SSH 到哪台机器？",
     "query_en": "To change partition settings with NVOS on an NVL72 rack, which machine do you SSH into first?",
     "type": "fact", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0080"],
     "reference": "SSH 到目标 NVL72 机架里的 leader NVSwitch。"},
    {"question": "系统里用来提交和控制作业的工作负载管理系统是哪个？",
     "query_en": "Which workload management system is used to submit and control work on the system?",
     "type": "fact", "topic": "concept_zh", "difficulty": "easy",
     "spans": ["c0089", "c0090"], "max_parents": 6,
     "note": "同一句出现在 DGX OS 快速入门/管理指南的多个版本里，命中多个父块都算正确。",
     "reference": "Slurm。"},
    {"question": "Slurm 最常用于哪类应用场景？",
     "query_en": "For which application scenario is Slurm most frequently used?",
     "type": "fact", "topic": "concept_zh", "difficulty": "easy",
     "spans": ["c0091"],
     "reference": "Linux 集群上的 HPC 应用。"},
    {"question": "Slurm 用哪三类对象来管理计算资源？",
     "query_en": "Which three object types does Slurm use to manage compute resources?",
     "type": "fact", "topic": "acronym_model", "difficulty": "normal",
     "spans": ["c0092"],
     "reference": "分区（partition/队列）、节点（node）、作业（job）。"},
    {"question": "全局在 slurm.conf 里把 DisableRootJobs 设成 Yes 会有什么后果？",
     "query_en": "What happens if DisableRootJobs is set to Yes globally in slurm.conf?",
     "type": "fact", "topic": "multi_condition", "difficulty": "hard",
     "spans": ["c0097"],
     "reference": "该配置不再生效。"},

    # ---------------------------------------------------------------- Network Operator / 驱动 / 容器
    {"question": "怎么确认 NicClusterPolicy 已经生效？该看哪个状态值？",
     "query_en": "How do you verify that the NicClusterPolicy has been applied, and which status value should you check?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0103", "c0104"],
     "reference": "查它的 status，STATUS 为 ready 表示策略已完全生效。"},
    {"question": "同一张网卡上能不能混用不同的网络类型？",
     "query_en": "Is it possible to combine different networking types on the same NICs?",
     "type": "fact", "topic": "multi_condition", "difficulty": "normal",
     "spans": ["c0136"],
     "reference": "不能，同一张网卡上无法组合不同网络类型。"},
    {"question": "Network Operator 的哪些内核变体只是 Tech Preview？",
     "query_en": "Which kernel variants for the Network Operator are supported only as a Tech Preview?",
     "type": "fact", "topic": "acronym_model", "difficulty": "hard",
     "spans": ["c0141"],
     "reference": "nvidia、aws、azure、oracle 变体（只有 generic 是 GA）。"},
    {"question": "RDMA 子系统处于 shared 模式时，设备在各命名空间里可见性如何？什么时候可以这么用？",
     "query_en": "In shared RDMA subsystem mode, how are devices exposed across namespaces, and when is shared mode acceptable?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0143", "c0144"],
     "reference": "所有网络命名空间都能访问 RDMA 设备；不需要做设备隔离时可以用，且这是默认模式。"},
    {"question": "部署 Network Operator 时官方推荐怎么传参数，为什么不用命令行参数？",
     "query_en": "How does NVIDIA recommend passing parameters when deploying the Network Operator, and why avoid CLI arguments?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "normal",
     "spans": ["c0133", "c0134"], "max_parents": 12,
     "note": "Network Operator 各版本手册重复同一段部署建议，多父块均正确。",
     "reference": "用配置文件；参数多、CLI 覆盖容易出错。"},
    {"question": "系统更新换掉了 Linux 内核版本之后，内核模块会怎样？",
     "query_en": "After a system update changes the Linux kernel version, what happens to the kernel modules?",
     "type": "fact", "topic": "concept_zh", "difficulty": "normal",
     "spans": ["c0108"],
     "reference": "除非用的是预编译模块，否则软件包会自动重建内核模块。"},
    {"question": "驱动的标准安装用哪种模块，替代方案是什么？",
     "query_en": "Which modules does the standard driver installation use, and what is the alternative?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "normal",
     "spans": ["c0111", "c0112"],
     "reference": "标准安装用 DKMS 模块；替代方案是预编译的 Open Kernel 模块。"},
    {"question": "默认情况下 Docker 去哪里找镜像？",
     "query_en": "By default, where does Docker look for images?",
     "type": "fact", "topic": "concept_zh", "difficulty": "easy",
     "spans": ["c0116", "c0117"],
     "reference": "Docker Hub（云端公共/私有镜像仓库）。"},
    {"question": "Docker Hub 对请求数量是什么态度？",
     "query_en": "What kind of limits does Docker Hub maintain on requests?",
     "type": "fact", "topic": "concept_zh", "difficulty": "normal",
     "spans": ["c0123"],
     "reference": "Docker Hub 对请求数量设有上限。"},

    # ---------------------------------------------------------------- 中文语料（AI Agents）
    {"question": "Harness 这个词原指什么？放到 Agent 语境里又指什么？",
     "query_en": "What did the word Harness originally refer to, and what does it mean in the context of an Agent?",
     "type": "concept", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0152", "c0153"],
     "reference": "原指马具（缰绳与挽具）；在 Agent 语境里是把模型能力引导成可靠任务执行的工程外壳。"},
    {"question": "Agent 的 Harness 具体包括哪些基础设施？",
     "query_en": "What infrastructure does an Agent's Harness include?",
     "type": "fact", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0154"],
     "reference": "上下文管理、工具接口、安全约束、验证与纠正等。"},
    {"question": "为什么模型越强，反而越需要更精细的约束与纠正？",
     "query_en": "Why does a stronger model need finer constraint, verification, and correction mechanisms?",
     "type": "concept", "topic": "paraphrase", "difficulty": "hard",
     "spans": ["c0155"],
     "reference": "模型自主决策空间越大，出错时影响面越大。"},
    {"question": "Harness 工程研究的到底是哪一层代码？",
     "query_en": "Which layer of code does Harness engineering actually study?",
     "type": "concept", "topic": "paraphrase", "difficulty": "hard",
     "spans": ["c0162", "c0163"],
     "reference": "Agent 边界内、模型之外的运行与治理层。"},
    {"question": "沙箱内随行动变化的文件与进程、外部数据库、网页，属于 Environment 还是 Harness？",
     "query_en": "Do sandbox files and processes, external databases, and web pages belong to the Environment or to the Harness?",
     "type": "concept", "topic": "zh_with_en_term", "difficulty": "hard",
     "spans": ["c0164"],
     "reference": "属于 Environment；即使与 Agent 同进程运行也不改变归属。"},
    {"question": "早期 Agent 框架关注什么，现在的重心又转到了哪里？",
     "query_en": "What did early Agent frameworks focus on, and where has the focus shifted now?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0167", "c0168"],
     "reference": "早期关注给模型工具与上下文（能做事）；现在转向约束、验证与纠正（不做错事）。"},

    # ---------------------------------------------------------------- 负样本（语料无答案）
    {"question": "DGX Spark 在中国大陆的官方售价是多少人民币？",
     "query_en": "What is the official price of DGX Spark in mainland China in RMB?",
     "type": "negative", "topic": "no_answer", "difficulty": "easy",
     "spans": [], "reference": "",
     "note": "价格类：语料里只有技术文档，没有任何区域定价表。"},
    {"question": "BCM 11 的商业授权一年要多少钱？",
     "query_en": "How much does a BCM 11 commercial license cost per year?",
     "type": "negative", "topic": "no_answer", "difficulty": "easy",
     "spans": [], "reference": "",
     "note": "商务报价类：文档只讲功能与许可分类，不给价格。"},
    {"question": "Mission Control 2.3.1 的默认登录密码是什么？",
     "query_en": "What is the default login password for Mission Control 2.3.1?",
     "type": "negative", "topic": "no_answer", "difficulty": "normal",
     "spans": [], "reference": "",
     "note": "凭据类：文档不会写默认口令。"},
    {"question": "NV-Ingest 在单台 ARM 节点上每小时能处理多少页文档？",
     "query_en": "How many pages per hour can NV-Ingest process on a single ARM node?",
     "type": "negative", "topic": "no_answer", "difficulty": "hard",
     "spans": [], "reference": "",
     "note": "缺失基准：文档给的吞吐基准不含 ARM 单机口径。"},
    {"question": "DGX B300 在中国的首次交付日期是哪一天？",
     "query_en": "On what date was DGX B300 first delivered in China?",
     "type": "negative", "topic": "no_answer", "difficulty": "normal",
     "spans": [], "reference": "",
     "note": "日期/地区类：没有任何按国家划分的交付时间表。"},

    # ---------------------------------------------------------------- 第二批：BCM / SuperPOD / DGX Spark
    {"question": "BCM headnode 的高可用靠什么共享服务，哪些目录必须在两台 head 之间共享？",
     "query_en": "What shared service does BCM headnode HA rely on, and which directories must be shared between head nodes?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0178", "c0179"],
     "reference": "NFS；/home 与 /cm_shared（含 DGX OS 镜像）必须共享。"},
    {"question": "给集群用的 NFS 导出必须满足哪两个条件？",
     "query_en": "Which two requirements must an NFS export satisfy to be usable by the cluster?",
     "type": "fact", "topic": "multi_condition", "difficulty": "hard",
     "spans": ["c0183"],
     "reference": "可读写挂载，且允许文件属于 UID 0（root）。"},
    {"question": "什么情况下需要重启 CMDaemon？",
     "query_en": "Under what circumstance do you need to restart CMDaemon?",
     "type": "concept", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0186"],
     "reference": "修改了 CMDaemon 配置文件、需要激活变更时。"},
    {"question": "BCM 里的 software image 实际是什么东西？",
     "query_en": "In BCM, what is a software image in practice?",
     "type": "concept", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0192", "c0193"],
     "reference": "head 节点上的一个目录，内含完整 Linux 文件系统，是计算节点本地文件系统的蓝图。"},
    {"question": "给 DGX 节点装系统时用的镜像，名字上有什么标识？",
     "query_en": "What prefix identifies the image used to provision DGX nodes?",
     "type": "fact", "topic": "acronym_model", "difficulty": "normal",
     "spans": ["c0196"],
     "reference": "镜像名带 dgx- 前缀，是 DGX OS 镜像的副本。"},
    {"question": "BCM 的 trigger 由哪两部分组成？文档里用什么例子说明？",
     "query_en": "In BCM, what two parts make up a trigger, and what example does the document use?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0200", "c0201", "c0202"],
     "reference": "一个样本状态条件 + 一个动作；例：CPUUser 超过 50 时执行 killallyesaction。"},
    {"question": "机柜的最小尺寸和 NVIDIA 推荐的尺寸分别是多少？",
     "query_en": "What is the minimum cabinet size, and what size does NVIDIA recommend?",
     "type": "fact", "topic": "number_spec", "difficulty": "normal",
     "spans": ["c0214", "c0215"],
     "reference": "至少 24“ x 48”（600 mm x 1,200 mm）、48U 高；NVIDIA 推荐 32“ x 48”（800 mm x 1,200 mm）。"},
    {"question": "每个机架要接几个温度传感器，装在哪两个位置？",
     "query_en": "How many temperature sensors per rack are connected, and at which two positions are they mounted?",
     "type": "fact", "topic": "number_spec", "difficulty": "normal",
     "spans": ["c0217", "c0218"],
     "reference": "两个，装在机架前侧 4U 与 42U 位置。"},
    {"question": "DGX SuperPOD 的节点数跨度是多少？",
     "query_en": "Across what node count does DGX SuperPOD scale?",
     "type": "fact", "topic": "number_spec", "difficulty": "normal",
     "spans": ["c0226"],
     "reference": "从 31 到 1,023 个 DGX 节点。"},
    {"question": "DGX B300 单机是几个 GPU 的配置？",
     "query_en": "How many GPUs does a DGX B300 system have?",
     "type": "fact", "topic": "number_spec", "difficulty": "easy",
     "spans": ["c0228"],
     "reference": "八 GPU 配置。"},
    {"question": "InfiniBand XDR 的单向峰值速率是多少？",
     "query_en": "What is the peak per-direction speed of InfiniBand XDR?",
     "type": "fact", "topic": "number_spec", "difficulty": "normal",
     "spans": ["c0233"],
     "reference": "每向 800 Gbps。"},
    {"question": "Kubernetes 升级里的 N-2 规则是什么意思？能不能跳着升？",
     "query_en": "What does the N-2 rule mean for Kubernetes upgrades, and can you skip versions?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0253", "c0254"],
     "reference": "每次只能跳两个小版本，例如从 1.28 升 1.31 要 1.28→1.29→1.30→1.31。"},
    {"question": "Kubernetes 升级先升哪类节点？worker 节点怎么利用 N-2 兼容性省事？",
     "query_en": "Which Kubernetes nodes are upgraded first, and how do worker nodes use N-2 compatibility?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0255", "c0256"],
     "reference": "先升 control plane，再升 worker；worker 可以每两轮 control plane 升级才做一次。"},
    {"question": "启动 DCGM 时报版本不匹配，在哪些系统上可以忽略这个提示？",
     "query_en": "When DCGM reports a version mismatch at startup, on which systems can the message be ignored?",
     "type": "fact", "topic": "acronym_model", "difficulty": "hard",
     "spans": ["c0262"], "max_parents": 30,
     "note": "DCGM 版本不匹配段落在 DGX OS 各版本手册里重复，命中多个父块都算正确。",
     "reference": "非 NVSwitch 系统（如 DGX Station / DGX-1）可以忽略。"},
    {"question": "node category 是用来做什么的？里面节点会继承什么？",
     "query_en": "What are node categories for, and what do nodes inside a category inherit?",
     "type": "concept", "topic": "concept_zh", "difficulty": "normal",
     "spans": ["c0274", "c0275"],
     "reference": "把硬件规格/角色/配置需求相同的节点分组；节点继承该 category 的默认属性与配置。"},
    {"question": "node group 和 node category 的区别是什么？",
     "query_en": "What is the difference between a node group and a node category?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "normal",
     "spans": ["c0274", "c0280"],
     "reference": "category 是共享属性的分组且子节点继承默认值；node group 只是为了批量操作而把节点拼在一起。"},
    {"question": "换了内核之后，为什么还得在 BCM 里重新登记？",
     "query_en": "After changing the kernel, why does it still need to be registered with BCM?",
     "type": "concept", "topic": "zh_with_en_term", "difficulty": "hard",
     "spans": ["c0281", "c0282"],
     "reference": "因为计算节点是从 head 节点（PXE）引导内核、不用本地盘，且还要生成新的 initramfs。"},
    {"question": "升级 DGX OS 镜像前为什么建议先克隆一份？",
     "query_en": "Why is it recommended to clone the DGX OS image before upgrading?",
     "type": "concept", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0247", "c0248"],
     "reference": "方便回滚到之前的版本/状态，可用 cmsh 的 clone image 功能。"},
    {"question": "在 chroot 里刷新包索引之后，还要给镜像装什么？",
     "query_en": "After refreshing package metadata inside chroot, what else should be installed into the image?",
     "type": "fact", "topic": "concept_zh", "difficulty": "normal",
     "spans": ["c0249", "c0250"],
     "reference": "装对应 DGX OS 版本支持的最新 CUDA toolkit。"},
    {"question": "在集群环境里对 DNS 和 NTP 有什么基本要求？",
     "query_en": "What basic requirement applies to DNS and NTP servers in the cluster environment?",
     "type": "fact", "topic": "concept_zh", "difficulty": "easy",
     "spans": ["c0184"],
     "reference": "必须能从集群环境内部访问到。"},
    {"question": "DGX Spark 官方推荐的管理平台是哪个？许可怎么算？",
     "query_en": "Which management platform is recommended for DGX Spark, and how is its license provided?",
     "type": "fact", "topic": "acronym_model", "difficulty": "normal",
     "spans": ["c0297"],
     "reference": "Canonical Landscape；许可包含在 Ubuntu Pro 授权里。"},
    {"question": "除了 Landscape，还能用哪些工具管 DGX Spark 机群？这些工具由谁支持？",
     "query_en": "Besides Landscape, which tools can manage a DGX Spark fleet, and who supports them?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0300", "c0301"],
     "reference": "Ansible、Puppet、Tanium 等有 ARM64 agent 的生态工具；由各自厂商支持。"},
    {"question": "Pre-Release 版本的软件在使用和支持上各有什么限制？",
     "query_en": "What limitations apply to Pre-Release software in terms of use and support?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "normal",
     "spans": ["c0303", "c0306"],
     "reference": "可能丢失使用/内容，不适合关键业务；按 AS-IS 提供且不在 Enterprise Support 范围内。"},
    {"question": "NGC 对 DGX Spark 用户的价值体现在哪？",
     "query_en": "Where does the value of NGC for DGX Spark users come from?",
     "type": "concept", "topic": "acronym_model", "difficulty": "normal",
     "spans": ["c0310", "c0312"],
     "reference": "提供针对 Grace Blackwell 架构优化的最新框架、工具与优化环境。"},
    {"question": "在 NGC 上跑 NIM 容器需要什么认证？创建 API Key 时要勾选什么？",
     "query_en": "What authentication is required to run NIM containers on NGC, and what must be selected when creating the API Key?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0313", "c0314"],
     "reference": "需要已认证的 NGC 访问；创建 Personal API Key 时 Services Included 里至少勾 NGC Catalog。"},
    {"question": "NIM for LLMs 能用旧版 API key 吗？",
     "query_en": "Can NIM for LLMs use legacy API keys?",
     "type": "fact", "topic": "acronym_model", "difficulty": "normal",
     "spans": ["c0315"],
     "reference": "不支持旧版 API key。"},
    {"question": "为什么更新过程中绝对不能关机或重启？",
     "query_en": "Why must you never power down or reboot during the update process?",
     "type": "concept", "topic": "paraphrase", "difficulty": "normal",
     "spans": ["c0317", "c0318"],
     "reference": "更新一旦开始不能中断，down 机可能造成系统损坏。"},
    {"question": "DGX Spark 连不上热点时，看到热点可用和看不到热点分别说明什么？",
     "query_en": "When DGX Spark fails to join a hotspot, what does it mean if the hotspot is visible versus not visible?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0320", "c0322"],
     "reference": "看到热点→确实没连上网络；看不到热点→其实已连上网络，只是笔记本和它不通。"},
    {"question": "交换机端口速率组合为什么必须考虑向后兼容？有哪些手段？",
     "query_en": "Why must switch port speed mixes consider backwards compatibility, and what techniques are used?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0206", "c0211", "c0212"],
     "reference": "通过每端口速率与端口拆分兼容旧代以太网；可降 lane 速率（如 100 GbE 端口跑 40 GbE）或用 breakout 线拆 lane。"},
    {"question": "怎么验证 helm values 真的被应用到发布里了？",
     "query_en": "How do you verify that Helm values were actually applied to a release?",
     "type": "fact", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0100", "c0101", "c0102"],
     "reference": "对已装 release 跑 helm get values 看用户传入值，再和自己那份 values.yaml 对比。"},
    {"question": "验证 SR-IOV 和 IB 资源配置时，从状态里看什么？",
     "query_en": "When validating SR-IOV and IB resource configuration, what should you look for in the status output?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0105", "c0106", "c0107"],
     "reference": "看节点是 ready 还是仍在 draining；整套配置生效需要时间。"},
    {"question": "怎么查哪些驱动版本可用于数据中心？",
     "query_en": "How do you find out which driver versions are certified for Datacenter use?",
     "type": "fact", "topic": "concept_zh", "difficulty": "easy",
     "spans": ["c0109", "c0110"],
     "reference": "查 Datacenter 驱动文档里的认证版本列表及各自 release notes。"},
    {"question": "Docker 客户端拉镜像前后分别会做什么事？",
     "query_en": "What does the Docker client do before and after pulling an image?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "normal",
     "spans": ["c0119", "c0120", "c0121"],
     "reference": "先在仓库里搜镜像、拉起时先确认本地是否已有最新版本，没有才下载并本地保存。"},
    {"question": "外部存储和高性能 NAS 的连接应该用哪类端口？",
     "query_en": "Which port type should be used to connect external storage and high-speed NAS?",
     "type": "fact", "topic": "multi_condition", "difficulty": "normal",
     "spans": ["c0125"],
     "reference": "存储端口（storage ports）接高性能 NAS，集群端口用于节点间通信。"},
    {"question": "Network Operator 在哪些场景下被验证过？",
     "query_en": "In which scenarios has the Network Operator been validated?",
     "type": "fact", "topic": "acronym_model", "difficulty": "normal",
     "spans": ["c0132"], "max_parents": 5,
     "note": "验证场景清单在 Network Operator 各版本文档里重复。",
     "reference": "文档列出的验证场景（单机与组合部署）。"},
    {"question": "怎么启用 systemd 模式的 SR-IOV 配置？",
     "query_en": "How do you enable systemd-based SR-IOV configuration?",
     "type": "fact", "topic": "acronym_model", "difficulty": "hard",
     "spans": ["c0142"],
     "reference": "在 SriovOperatorConfig CRD 里指定 configurationMode。"},
    {"question": "装 Network Operator 的第一步是什么？",
     "query_en": "What is the first step when installing the Network Operator?",
     "type": "fact", "topic": "concept_zh", "difficulty": "easy",
     "spans": ["c0146", "c0147"],
     "reference": "先给它建一个 namespace，再在里面创建对象完成安装。"},
    {"question": "Slurm 相关配置失效时，文档给的临时办法是什么？",
     "query_en": "What workaround does the documentation give when the Slurm-related configuration no longer works?",
     "type": "fact", "topic": "zh_with_en_term", "difficulty": "hard",
     "spans": ["c0170"],
     "reference": "在 head 节点和 Slurm 用的软件镜像/计算节点里安装 cm-hwloc2-local。"},
    {"question": "BCM 的 ISO 有哪几种安装方式？",
     "query_en": "Which installation methods are available for the BCM ISO?",
     "type": "fact", "topic": "acronym_model", "difficulty": "normal",
     "spans": ["c0172", "c0173"],
     "reference": "烧到 DVD / 可引导 U 盘，或用 BMC 虚拟控制台挂成虚拟介质。"},
    {"question": "装 BCM headnode 之前，存储介质要处于什么状态？",
     "query_en": "What state must the headnode storage media be in before installing BCM?",
     "type": "fact", "topic": "multi_condition", "difficulty": "normal",
     "spans": ["c0174", "c0175"],
     "reference": "全盘擦除干净，并把 BIOS 恢复默认（会重置引导设备顺序）。"},
    {"question": "DGX BasePOD 对 NFS 存储有什么要求？",
     "query_en": "What does DGX BasePOD require regarding NFS storage?",
     "type": "fact", "topic": "multi_condition", "difficulty": "hard",
     "spans": ["c0180", "c0181"],
     "reference": "不限定 NFS 存储形态，但要用站点勘察里给的导出路径。"},
    {"question": "怎么只重启一段计算节点上的 CMDaemon？",
     "query_en": "How do you restart CMDaemon on only a range of compute nodes?",
     "type": "fact", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0188", "c0190"],
     "reference": "用 service cmd 的 init 脚本参数，按节点范围批量重启（如 dgx001 到 dgx040）。"},
    {"question": "DGX SuperPOD 里所有节点是不是共用同一个基础 OS？DGX 节点的镜像怎么识别？",
     "query_en": "Do all nodes in a DGX SuperPOD share the same base OS, and how is a DGX node image identified?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0194", "c0196"],
     "reference": "是，管理节点与 DGX 节点共用同一基础 OS；DGX 节点镜像名带 dgx- 前缀。"},
    {"question": "节点完全启动后能对本地文件系统做什么？镜像平时怎么改？",
     "query_en": "What can you do with the local filesystem after a node is fully booted, and how are software images normally modified?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "normal",
     "spans": ["c0197", "c0198"],
     "reference": "可让节点与软件镜像重新同步；镜像用 apt、chroot 等常规 Linux 工具修改。"},
    {"question": "在 BCM 里怎么克隆一个已有的监控 trigger？",
     "query_en": "How do you clone an existing monitoring trigger in BCM?",
     "type": "fact", "topic": "zh_with_en_term", "difficulty": "normal",
     "spans": ["c0199"],
     "reference": "Monitoring > Triggers > Failing health checks > Clone。"},
    {"question": "BCM 提供的两个建盘镜像命令分别适合什么场景？",
     "query_en": "When are BCM's two disk-image creation commands appropriate?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0267", "c0268", "c0270"],
     "reference": "一个适合已有镜像目录/本地或其他集群的副本；另一个适合头节点之外的发行版或不同 CPU 架构。"},
    {"question": "在 cmsh 里怎么把镜像的内核版本换成新的？",
     "query_en": "How do you set a new kernel version for an image in cmsh?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0285", "c0286", "c0287"],
     "reference": "进入 softwareimage 选镜像→setkernelversion（Tab 补全选版本）→提交修改。"},
    {"question": "同一个软件镜像能不能被多个 category 共用？",
     "query_en": "Can several categories share the same software image?",
     "type": "fact", "topic": "concept_zh", "difficulty": "normal",
     "spans": ["c0279"],
     "reference": "可以，且磁盘布局、内核参数等可按 category 覆盖。"},
    {"question": "软件订阅和永久许可在支持服务上有什么差别？",
     "query_en": "How does support differ between software subscriptions and perpetual licenses?",
     "type": "multi_hop", "topic": "cross_chunk", "difficulty": "hard",
     "spans": ["c0307", "c0308"],
     "reference": "订阅默认含 Enterprise Support/维护/更新；永久许可的支持与更新可单独购买。"},

    # 第二批负样本
    {"question": "DGX SuperPOD 在中国区的起订价格是多少？",
     "query_en": "What is the starting price of DGX SuperPOD in China?",
     "type": "negative", "topic": "no_answer", "difficulty": "easy",
     "spans": [], "reference": "",
     "note": "价格类：语料只有产品与部署技术文档。"},
    {"question": "NGC 企业版一年的订阅费是多少？",
     "query_en": "What is the annual subscription fee for NGC Enterprise?",
     "type": "negative", "topic": "no_answer", "difficulty": "easy",
     "spans": [], "reference": "",
     "note": "商务报价类：文档不给价格。"},
    {"question": "Mission Control 支持中文界面吗？",
     "query_en": "Does Mission Control support a Chinese-language interface?",
     "type": "negative", "topic": "no_answer", "difficulty": "normal",
     "spans": [], "reference": "",
     "note": "本地化类：文档没有任何多语言界面说明。"},
    {"question": "DGX B300 机架在中国的交付周期是几周？",
     "query_en": "What is the rack delivery lead time for DGX B300 in China, in weeks?",
     "type": "negative", "topic": "no_answer", "difficulty": "normal",
     "spans": [], "reference": "",
     "note": "交付周期类：没有任何地区性交付时间表。"},
    # ---------------------------------------------------------------- 第三批：补 paraphrase / multi_condition
    {"question": "模型越强大，为什么围绕它构建的 Harness 反而更关键？",
     "query_en": "Why does the Harness built around a model matter more as the model gets stronger?",
     "type": "concept", "topic": "paraphrase", "difficulty": "hard",
     "spans": ["c0151"],
     "reference": "模型越强，越需要工程外壳把能力导向可靠的任务执行，而不是限制能力本身。"},
    {"question": "模型厂商真正的竞争优势是什么？",
     "query_en": "What is the real competitive advantage of model vendors?",
     "type": "concept", "topic": "paraphrase", "difficulty": "hard",
     "spans": ["c0156"],
     "reference": "不是把框架做薄，而是能对模型与外围 Harness 协同优化并持续迭代。"},
    {"question": "Harness 里的约束、验证、纠正分别负责什么？",
     "query_en": "What do constraint, verification, and correction each handle inside a Harness?",
     "type": "fact", "topic": "concept_zh", "difficulty": "normal",
     "spans": ["c0161"],
     "reference": "约束规定能做什么不能做什么，验证检查做得对不对，纠正处理做错了怎么补救。"},
    {"question": "CPU 时间片指的是什么？",
     "query_en": "What does a CPU time slice mean?",
     "type": "concept", "topic": "concept_zh", "difficulty": "normal",
     "spans": ["c0203"],
     "reference": "进程在被内核切换出去之前能占用 CPU 的最短时间。"},
    {"question": "机架的侧壁有什么强制要求？",
     "query_en": "What is mandatory regarding rack side walls?",
     "type": "fact", "topic": "multi_condition", "difficulty": "easy",
     "spans": ["c0216"],
     "reference": "所有机架都必须安装侧壁。"},
    {"question": "DGX OS 版本早于哪个版本的系统，升级前必须先更新密钥？",
     "query_en": "If a system runs a DGX OS version earlier than which release must the keys be updated before upgrading?",
     "type": "fact", "topic": "number_spec", "difficulty": "hard",
     "spans": ["c0264"],
     "reference": "早于 DGX OS 4.12。"},
    {"question": "DGX Spark 的可管理性基线划清了什么边界？",
     "query_en": "What boundary does the DGX Spark manageability baseline draw?",
     "type": "concept", "topic": "paraphrase", "difficulty": "normal",
     "spans": ["c0295"],
     "reference": "划清「NVIDIA 提供并支持的部分」与「企业 IT 环境自己负责的部分」。"},

    {"question": "DGX OS 7.5 的默认 root 口令是什么？",
     "query_en": "What is the default root password for DGX OS 7.5?",
     "type": "negative", "topic": "no_answer", "difficulty": "normal",
     "spans": [], "reference": "",
     "note": "凭据类：文档从不写默认口令。"},
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.expand_questions_v2")
    sub = parser.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump", help="生成句子候选")
    dump.add_argument("--docs", type=int, default=26)
    dump.add_argument("--per-doc", type=int, default=8)
    dump.add_argument("--all-docs", action="store_true", help="包括已有 gold 的文档")
    dump.add_argument("--seed", type=int, default=0)
    dump.add_argument("--only", default="", help="只用路径含这些子串的文档，逗号分隔")
    dump.add_argument("--merge", action="store_true", help="追加到已有候选文件（key 顺延）")

    app = sub.add_parser("append", help="把 NEW_V2 追加进 questions.json")
    app.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "dump":
        only = [part.strip() for part in args.only.split(",") if part.strip()]
        candidates = collect_candidates(
            args.docs, args.per_doc, skip_covered=not args.all_docs, seed=args.seed,
            only=only or None,
        )
        if args.merge and CANDIDATES_PATH.exists():
            old = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))["candidates"]
            start = len(old)
            for offset, item in enumerate(candidates, start=1):
                item["key"] = f"c{start + offset:04d}"
            candidates = old + candidates
            print(f"并入已有 {start} 条 → 共 {len(candidates)} 条")
        CANDIDATES_PATH.write_text(
            json.dumps({"candidates": candidates}, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"候选 {len(candidates)} 条 → {CANDIDATES_PATH}")
        current = None
        for item in (candidates if not args.merge else candidates[-(len(candidates)):]):
            if item["source"] != current:
                current = item["source"]
                print(f"\n## {current}")
            print(f"{item['key']} [{item['parent_index']}] {item['text']}")
        return 0

    payload = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    candidates = {item["key"]: item for item in payload["candidates"]}
    items = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return append(items, candidates, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
