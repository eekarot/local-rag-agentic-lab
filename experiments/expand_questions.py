"""D27：把评估集扩到 100 条中英成对题，并给原有 60 题补 `topic`/`difficulty`。

为什么要单独一个脚本而不是改 `build_questions.py`：本文件只**追加**新题、只**补字段**，
不重算已有题的 gold——`build_questions.py` 是全量重建（会覆盖人工过审的 `query_en`）。

两条硬规则（与方案 §17.11.3 / B06 一致）：

1. `gold_spans` 必须是语料**原文**，由 `locate()` 自动定位成 chunk id，定位不到立刻报错；
2. `query_en` 是新写的英文问法，**不是**中文问法的机翻（本脚本里的新题由人写，
   老题用 `python -m src.eval_bilingual draft` 出草稿后再人工过审）。

    python -m experiments.expand_questions --dry-run
    python -m experiments.expand_questions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import QUESTIONS_PATH
from src.eval_dataset import locate, normalize  # noqa: F401
from src.retrieve import load_index

# ---------------------------------------------------------------- 原有 60 题切片字段
# 键是 qid；值是 (topic, difficulty)
ANNOTATE = {
    "eval-001": ("concept_zh", "easy"),
    "eval-002": ("zh_with_en_term", "normal"),
    "eval-003": ("zh_with_en_term", "easy"),
    "eval-004": ("zh_with_en_term", "normal"),
    "eval-005": ("paraphrase", "normal"),
    "eval-006": ("acronym_model", "normal"),
    "eval-007": ("zh_with_en_term", "normal"),
    "eval-008": ("zh_with_en_term", "normal"),
    "eval-009": ("zh_with_en_term", "normal"),
    "eval-010": ("zh_with_en_term", "easy"),
    "eval-011": ("paraphrase", "normal"),
    "eval-012": ("concept_zh", "easy"),
    "eval-013": ("concept_zh", "normal"),
    "eval-014": ("zh_with_en_term", "normal"),
    "eval-015": ("zh_with_en_term", "normal"),
    "eval-016": ("concept_zh", "hard"),
    "eval-017": ("number_spec", "easy"),
    "eval-018": ("number_spec", "easy"),
    "eval-019": ("acronym_model", "normal"),
    "eval-020": ("number_spec", "easy"),
    "eval-021": ("number_spec", "easy"),
    "eval-022": ("number_spec", "normal"),
    "eval-023": ("number_spec", "normal"),
    "eval-024": ("acronym_model", "normal"),
    "eval-025": ("number_spec", "normal"),
    "eval-026": ("acronym_model", "normal"),
    "eval-027": ("multi_condition", "hard"),
    "eval-028": ("number_spec", "normal"),
    "eval-029": ("number_spec", "normal"),
    "eval-030": ("acronym_model", "normal"),
    "eval-031": ("number_spec", "normal"),
    "eval-032": ("multi_condition", "normal"),
    "eval-033": ("cross_chunk", "hard"),
    "eval-034": ("cross_chunk", "normal"),
    "eval-035": ("cross_chunk", "easy"),
    "eval-036": ("cross_chunk", "normal"),
    "eval-037": ("cross_chunk", "normal"),
    "eval-038": ("cross_chunk", "hard"),
    "eval-039": ("cross_chunk", "easy"),
    "eval-040": ("cross_chunk", "hard"),
    "eval-041": ("cross_chunk", "hard"),
    "eval-042": ("cross_chunk", "hard"),
    "eval-043": ("cross_chunk", "normal"),
    "eval-044": ("cross_chunk", "normal"),
    "eval-045": ("cross_chunk", "hard"),
    "eval-046": ("cross_chunk", "hard"),
    "eval-047": ("cross_chunk", "hard"),
    "eval-048": ("cross_chunk", "hard"),
    "eval-049": ("no_answer", "easy"),
    "eval-050": ("no_answer", "easy"),
    "eval-051": ("no_answer", "easy"),
    "eval-052": ("no_answer", "easy"),
    "eval-053": ("no_answer", "easy"),
    "eval-054": ("no_answer", "easy"),
    "eval-055": ("no_answer", "easy"),
    "eval-056": ("no_answer", "normal"),
    "eval-057": ("no_answer", "easy"),
    "eval-058": ("no_answer", "easy"),
    "eval-059": ("no_answer", "normal"),
    "eval-060": ("no_answer", "easy"),
}

# ---------------------------------------------------------------- 新增题目
# (question, query_en, type, topic, difficulty, spans, reference, note)
NEW = [
    # --- fact：DGX SuperPOD / 网络 / Mission Control（英文语料、中文问）
    ("DGX SuperPOD 里 NDR InfiniBand 的单向峰值速率是多少？",
     "What is the peak speed of NDR InfiniBand per direction in a DGX SuperPOD?",
     "fact", "number_spec", "normal",
     ["The latest generation InfiniBand, NDR, has a peak speed of 400 Gbps per direction with an extremely low port-to-port latency."],
     "400 Gbps（单向），端口到端口延迟极低。", ""),
    ("HDR100 是用几条通道、每条多少速率凑出 100 Gbps 的？",
     "How does HDR100 reach 100 Gbps in terms of lanes and per-lane speed?",
     "fact", "number_spec", "normal",
     ["Recall that 100 Gbps EDR and 100 Gbps HDR100 are not the same: EDR is four lanes of 25 Gbps and HDR100 is two lanes of 50 Gbps."],
     "HDR100 = 2 条 50 Gbps 通道；EDR 是 4 条 25 Gbps，两者不是一回事。", ""),
    ("两层 fabric 用 16 台 spine 交换机能支持多大的 DGX SuperPOD？",
     "How large a DGX SuperPOD can a two-layer fabric with 16 spine switches support?",
     "fact", "number_spec", "normal",
     ["The two-layer fabric with 16 spine switches supports a DGX SuperPOD with up to 16 SUs (9,000 GPUs)."],
     "最多 16 个 SU，约 9,000 张 GPU。", ""),
    ("DGX GB200 一个机架里放多少张 GPU？",
     "How many GPUs are in one DGX GB200 rack?",
     "fact", "number_spec", "easy",
     ["Each DGX GB200 delivers breakthrough AI performance in a rack-scale, 72 GPU configuration."],
     "72 张 GPU（rack-scale 配置）。", ""),
    ("DGX GB200 的 72 张 GPU 是怎么组成单个 NVLink 域的？",
     "How do the 72 GPUs in a DGX GB200 form a single NVLink domain?",
     "fact", "acronym_model", "normal",
     ["The compute nodes for DGX GB200 rack system are using the 72x1 NVLink topology, which contains 72 GPUs in a single NVLink domain."],
     "用 72x1 NVLink 拓扑，72 张 GPU 落在同一个 NVLink 域。", ""),
    ("一个 GB200 Superchip 里包含哪些芯片、用什么互连？",
     "What dies does one GB200 Superchip combine and what interconnect does it use?",
     "fact", "acronym_model", "normal",
     ["GB200 Superchips, each chip combines two NVIDIA B200 Tensor Core GPUs and one NVIDIA Grace CPU, connected over a 900GB/s ultra-low-power NVLink-C2C interconnect."],
     "两颗 B200 GPU + 一颗 Grace CPU，用 900 GB/s 的 NVLink-C2C 互连。", ""),
    ("DGX SuperPOD 机房的噪声暴露上限是多少？",
     "What is the maximum acceptable noise exposure limit in a DGX SuperPOD data center?",
     "fact", "number_spec", "normal",
     ["The maximum acceptable unprotected exposure limit for a standard 8-hour duty cycle is a time weighted average of 85 dBA."],
     "8 小时工时下 85 dBA（时间加权平均）。", ""),
    ("Mission Control 里 NetQ 的评估许可能用多久？",
     "How long is the NetQ evaluation license valid for in Mission Control?",
     "fact", "number_spec", "easy",
     ["When installing NetQ, the system receives an evaluation license valid for 60 days."],
     "60 天。", ""),
    ("Mission Control 离线安装大致要下载多少数据？",
     "Roughly how much data does the Mission Control air-gapped installation download?",
     "fact", "number_spec", "normal",
     ["The download script fetches approximately 130 GB of data (PID files alone account for roughly 9 GB)."],
     "约 130 GB，其中 PID 文件约占 9 GB。", ""),
    ("NVL72 机架默认的 NVLink 分区包含多少张 GPU？",
     "How many GPUs does an NVL72 rack's default NVLink partition contain?",
     "fact", "number_spec", "easy",
     ["For instance, each NVL72 rack is by default an NVLink partition that includes 72 GPUs connected through NVLink."],
     "72 张 GPU，通过 NVLink 连接。", ""),
    ("按照 DGX 文档，GPU driver 至少要到哪个版本？",
     "According to the DGX documentation, what is the minimum installed GPU driver release?",
     "fact", "number_spec", "normal",
     ["For DGX systems, the installed GPU driver release must be 580 or greater."],
     "580 或更高。", ""),
    ("DGX-H200 系统要求的 GPU driver 最低版本是多少？",
     "What is the minimum GPU driver release for DGX H200 systems?",
     "fact", "number_spec", "normal",
     ["For the DGX H200 systems, the installed GPU driver release must be 550 or greater."],
     "550 或更高。", ""),
    ("MLNX_OFED 最后一个独立发布版本的支持周期有多长？",
     "What is the support term of the last standalone MLNX_OFED release?",
     "fact", "number_spec", "normal",
     ["MLNX_OFED last standalone release is October 2024 Long Term Support (3 years)."],
     "2024 年 10 月的长期支持版，支持 3 年。", ""),
    ("DGX SuperPOD 建议的用户存储最低带宽是多少？",
     "What minimum bandwidth should user storage provide for best performance in a DGX SuperPOD?",
     "fact", "number_spec", "normal",
     ["For best performance, user storage should provide a minimum bandwidth of 100 Gb/s."],
     "至少 100 Gb/s。", ""),
    ("一个 SU 里所有 DGX 系统的聚合非阻塞带宽是多少？",
     "What is the total aggregated non-blocking bandwidth of all DGX systems in a single SU?",
     "fact", "number_spec", "hard",
     ["The total aggregated non-blocking bandwidth for all DGX systems within a single SU is 96 × 400 Gb/s."],
     "96 × 400 Gb/s。", ""),
    ("NDR200 适配器端用的是什么样的分光器和光模块？",
     "What splitters and transceivers are used at the adapter end for NDR200?",
     "fact", "acronym_model", "hard",
     ["For NDR200 adapters, twin two-way single-mode splitters are used with 200 Gbps OSFP flat single-mode transceivers at the adapter."],
     "twin two-way 单模分光器 + 200 Gbps OSFP flat 单模光模块。", ""),
    ("DGX GB200 机架的每个计算节点有几条 GPU rail、要配几个 IP 池？",
     "How many GPU rails does each compute tray in a DGX GB200/GB300 rack have, and how many IP pools do you define?",
     "fact", "number_spec", "normal",
     ["Each compute node (AKA compute tray) in a DGX GB200/GB300 rack has 4 GPU rails, so you would define 4 IP pools."],
     "4 条 GPU rail → 4 个 IP 池。", ""),

    # --- concept：中文语料，靠改写覆盖 paraphrase / multi_condition
    ("把用户问题也变成向量、再按相似度找文本块，这属于哪种检索？",
     "Which retrieval method converts the user query into a vector and looks for chunks by similarity?",
     "concept", "paraphrase", "easy",
     ["向量检索把用户问题也转换成向量，再根据向量相似度或距离寻找最相关的文本块。"],
     "稠密向量检索。", ""),
    ("生产系统为什么不直接用简单词项重叠，而是用 BM25？",
     "Why do production systems use BM25 instead of simple term overlap for keyword retrieval?",
     "concept", "paraphrase", "normal",
     ["生产系统常使用 BM25 等成熟算法，而不是简单的词项重叠。"],
     "BM25 是更成熟的词项匹配算法，比简单重叠更可靠。", ""),
    ("Agent Loop 每一轮具体做哪些动作？",
     "What does each iteration of the Agent Loop do?",
     "concept", "concept_zh", "normal",
     ["Agent Loop 是一个反复执行的控制循环：向模型提供目标和当前状态，解析模型的下一步决策，执行允许的工具，把工具结果加入状态，再决定继续还是结束。"],
     "给目标与状态 → 解析决策 → 执行工具 → 结果入状态 → 决定继续或结束。", ""),
    ("模型会不会自己直接执行代码？工具到底是谁执行的？",
     "Does the model execute code directly, and who actually runs the tool?",
     "concept", "concept_zh", "normal",
     ["工具调用不是模型直接执行代码。模型输出一个结构化动作，宿主程序验证动作名称和参数，再执行对应的工具，并把结果返回模型。"],
     "模型只输出结构化动作；由宿主程序校验参数后执行并回传结果。", ""),
    ("固定 RAG 的「固定」体现在哪一步，Agentic 又改动了哪一步？",
     "Which step is fixed in fixed RAG, and which step does Agentic RAG change?",
     "concept", "paraphrase", "normal",
     ["固定 RAG 的顺序一般是：检索一次、组装上下文、生成答案。Agentic RAG 可以在证据不足时改写查询或追加检索。"],
     "固定版固定为「检索一次→组装→生成」；Agentic 允许在证据不足时改写查询或追加检索。", ""),
    ("什么情况下该用 RAG，什么情况下微调更合适？",
     "When is RAG the right choice and when is fine-tuning more appropriate?",
     "concept", "multi_condition", "normal",
     ["RAG 更适合需要更新、引用或隔离的知识，微调更常用于行为、格式或任务模式调整。两者可以结合使用。"],
     "要更新/引用/隔离的知识用 RAG；改行为、格式或任务模式用微调；两者可结合。", ""),
    ("除了限制最大步数，还能靠什么让 Agent Loop 停下来？",
     "Besides a maximum step count, what else keeps the Agent Loop from running forever?",
     "concept", "paraphrase", "normal",
     ["最大步骤数、重复查询检测、工具错误处理与最终回退能让系统在可控范围内终止。"],
     "重复查询检测、工具错误处理与最终回退。", ""),

    # --- multi_hop：两个不同片段拼起来
    ("DGX SuperPOD 的 NDR fabric 单向能跑多快，用户存储又要求多少带宽？",
     "How fast is the NDR fabric per direction in a DGX SuperPOD, and what storage bandwidth is required?",
     "multi_hop", "cross_chunk", "hard",
     ["The latest generation InfiniBand, NDR, has a peak speed of 400 Gbps per direction with an extremely low port-to-port latency.",
      "For best performance, user storage should provide a minimum bandwidth of 100 Gb/s."],
     "NDR 单向 400 Gbps；用户存储至少 100 Gb/s。", ""),
    ("一个 DGX GB200 机架的 GPU 数量和 NVLink 拓扑规模分别是多少？",
     "How many GPUs are in one DGX GB200 rack, and what is the scale of its NVLink topology?",
     "multi_hop", "cross_chunk", "normal",
     ["Each DGX GB200 delivers breakthrough AI performance in a rack-scale, 72 GPU configuration.",
      "The compute nodes for DGX GB200 rack system are using the 72x1 NVLink topology, which contains 72 GPUs in a single NVLink domain."],
     "一个机架 72 张 GPU，拓扑是 72x1，72 张同处一个 NVLink 域。", ""),
    ("GB200 Superchip 由哪几颗芯片组成，整个机架的 NVLink 分区又是多大？",
     "What dies make up a GB200 Superchip, and how large is the NVLink partition of the rack?",
     "multi_hop", "cross_chunk", "hard",
     ["GB200 Superchips, each chip combines two NVIDIA B200 Tensor Core GPUs and one NVIDIA Grace CPU, connected over a 900GB/s ultra-low-power NVLink-C2C interconnect.",
      "For instance, each NVL72 rack is by default an NVLink partition that includes 72 GPUs connected through NVLink."],
     "每颗 Superchip = 2×B200 + 1×Grace（NVLink-C2C 900 GB/s）；NVL72 机架默认一个 72 GPU 的 NVLink 分区。", ""),
    ("两层 fabric 用 16 台 spine 能撑多少个 SU，B300 那套参考架构又写到多少个节点？",
     "How many SUs can 16 spine switches support, and how many nodes does the B300 reference architecture fabric cover?",
     "multi_hop", "cross_chunk", "hard",
     ["The two-layer fabric with 16 spine switches supports a DGX SuperPOD with up to 16 SUs (9,000 GPUs).",
      "Figure 6 shows the compute fabric layout for the full 512-node DGX SuperPOD."],
     "16 台 spine 最多 16 个 SU（约 9,000 GPU）；B300 参考架构画的是 512 节点满配。", ""),
    ("HDR100 和 EDR 都是 100 Gbps，通道配置差在哪，NDR200 又用什么分光器？",
     "HDR100 and EDR both reach 100 Gbps — how do their lane configs differ, and what splitters does NDR200 use?",
     "multi_hop", "cross_chunk", "hard",
     ["Recall that 100 Gbps EDR and 100 Gbps HDR100 are not the same: EDR is four lanes of 25 Gbps and HDR100 is two lanes of 50 Gbps.",
      "For NDR200 adapters, twin two-way single-mode splitters are used with 200 Gbps OSFP flat single-mode transceivers at the adapter."],
     "EDR = 4×25 Gbps，HDR100 = 2×50 Gbps；NDR200 用 twin two-way 单模分光器 + 200 Gbps OSFP flat 光模块。", ""),
    ("DGX GB200 机架的每个计算节点怎么配 IP 池，整套机架又是多大规模？",
     "How are IP pools sized per compute tray in a DGX GB200 rack, and how large is the rack itself?",
     "multi_hop", "cross_chunk", "normal",
     ["Each compute node (AKA compute tray) in a DGX GB200/GB300 rack has 4 GPU rails, so you would define 4 IP pools.",
      "Each DGX GB200 delivers breakthrough AI performance in a rack-scale, 72 GPU configuration."],
     "每个计算托盘 4 条 GPU rail → 4 个 IP 池；整套机架 72 张 GPU。", ""),
    ("NetQ 的评估许可能用多久，Mission Control 离线装一次要拉多少数据？",
     "How long does the NetQ evaluation license last, and how much data does an air-gapped Mission Control install download?",
     "multi_hop", "cross_chunk", "normal",
     ["When installing NetQ, the system receives an evaluation license valid for 60 days.",
      "The download script fetches approximately 130 GB of data (PID files alone account for roughly 9 GB)."],
     "NetQ 评估许可 60 天；离线安装约 130 GB（PID 文件约 9 GB）。", ""),
    ("DGX 系统要求 GPU driver 至少什么版本，DGX OS 8 又是为哪些驱动版本推荐的操作系统？",
     "What minimum GPU driver release do DGX systems require, and for which driver releases is DGX OS 8 recommended?",
     "multi_hop", "cross_chunk", "hard",
     ["For DGX systems, the installed GPU driver release must be 580 or greater.",
      "For driver release 595 and later, upgrading to DGX OS 8 is recommended because DGX OS 8 is the validated software stack for these driver releases."],
     "DGX 要求 580+；595 及以后建议升到 DGX OS 8（该驱动的验证栈）。", ""),
    ("MLNX_OFED 最后一个独立版的支持周期有多长，ConnectX-7 SuperNIC 双口能到多少速率？",
     "How long is the support term for the last standalone MLNX_OFED release, and what speed does a dual-port ConnectX-7 SuperNIC reach?",
     "multi_hop", "cross_chunk", "hard",
     ["MLNX_OFED last standalone release is October 2024 Long Term Support (3 years).",
      "The **MCX715105AS-WEAT **ConnectX-7 SuperNIC, featuring a dual-port QSFP112 module, supports up to 200Gb/s."],
     "MLNX_OFED 最后独立版支持 3 年；双口 QSFP112 的 ConnectX-7 SuperNIC 最高 200 Gb/s。", ""),
    ("一个 SU 的聚合非阻塞带宽是多少，整个 SuperPOD 的 fabric 又按多少节点设计？",
     "What is the aggregated non-blocking bandwidth per SU, and how many nodes does the full SuperPOD fabric serve?",
     "multi_hop", "cross_chunk", "hard",
     ["The total aggregated non-blocking bandwidth for all DGX systems within a single SU is 96 × 400 Gb/s.",
      "Figure 5 shows the compute fabric layout for the full 576-node DGX SuperPOD."],
     "单 SU 聚合 96 × 400 Gb/s；满配 fabric 按 576 节点设计。", ""),
    ("DGX SuperPOD 机房对噪声的上限是多少，DGX-1 搬运又需要什么设备？",
     "What is the noise exposure limit in a DGX SuperPOD data center, and what equipment is needed to lift a DGX-1?",
     "multi_hop", "cross_chunk", "hard",
     ["The maximum acceptable unprotected exposure limit for a standard 8-hour duty cycle is a time weighted average of 85 dBA.",
      "The DGX-1 weighs approximately 134 lbs, so an equipment lift is required to safely lift the unit and then accurately align the chassis rails with the rack rails"],
     "噪声上限 85 dBA（8 小时加权）；DGX-1 约 134 lbs，需要吊装设备。", ""),

    # --- negative：库里没有答案
    ("DGX GB200 一个机架的报价是多少？", "What is the price of one DGX GB200 rack?",
     "negative", "no_answer", "easy", [], "", "价格类：全库只有 GB200 的规格、拓扑与部署说明，无任何报价。"),
    ("NetQ 的永久授权怎么买？", "How do you buy a perpetual NetQ license?",
     "negative", "no_answer", "easy", [], "", "商务类：文档只提 60 天评估许可，无购买方式与永久授权。"),
    ("DGX SuperPOD 的保修期是几年？", "How many years of warranty does a DGX SuperPOD come with?",
     "negative", "no_answer", "easy", [], "", "商务类：文档讲设计与部署，无保修年限。"),
    ("ConnectX-7 SuperNIC 的零售价是多少美元？", "What is the retail price of a ConnectX-7 SuperNIC in USD?",
     "negative", "no_answer", "easy", [], "", "价格类：网卡文档只有端口拆分与速率，无价格。"),
    ("MLNX_OFED 的付费支持热线是多少？", "What is the paid support hotline for MLNX_OFED?",
     "negative", "no_answer", "easy", [], "", "联系方式类：文档只有 LTS 周期说明，无电话。"),
    ("DGX Spark 在大陆的上市时间是什么时候？", "When will DGX Spark be released in mainland China?",
     "negative", "no_answer", "normal", [], "", "日期/地区类：文档无任何区域上市时间表。"),
]


def corpus_language(gold):
    chinese = ("rag_basics.md", "agent_basics.md", "AI-Agents-in-Depth-zh-CN.pdf")
    if not gold:
        return None
    sources = {item["id"].split("::")[0] for item in gold}
    return "zh" if all(any(marker in s for marker in chinese) for s in sources) else "en"


def main():
    parser = argparse.ArgumentParser(prog="experiments.expand_questions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_index()["records"]
    items = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    # 1) 老题补切片字段（不改 gold、不动 question）
    missing = 0
    for item in items:
        topic, difficulty = ANNOTATE.get(item["qid"], (None, None))
        if topic is None:
            missing += 1
            continue
        item["topic"] = topic
        item["difficulty"] = difficulty
        item.setdefault("query_en", "")
        item.setdefault("query_en_status", "")
    if missing:
        print(f"[warn] {missing} 条老题没有切片标注（保持原样）")

    # 2) 追加新题
    existing = {item["question"] for item in items}
    index = max(int(item["qid"].split("-")[1]) for item in items)
    problems, added = [], 0
    for question, query_en, qtype, topic, difficulty, spans, reference, note in NEW:
        if question in existing:
            problems.append(f"重复题目：{question}")
            continue
        index += 1
        gold, whole_hits = [], []
        for span in spans:
            hits = locate(span, records)
            whole_hits.append(len(hits))
            if not hits:
                problems.append(f"定位失败（{question[:20]}）：{span[:40]}")
            for chunk_id, _source, _chunk_index in hits[:3]:
                if chunk_id not in [g["id"] for g in gold]:
                    gold.append({"id": chunk_id, "grade": 2})
        if qtype != "negative" and not gold:
            problems.append(f"无 gold：{question[:30]}")
        if any(n >= 5 for n in whole_hits):
            problems.append(f"span 命中 ≥5 处（需人工收敛 gold）：{question[:30]}")
        items.append({
            "qid": f"eval-{index:03d}",
            "question": question,
            "query_en": query_en,
            "query_en_status": "edited",   # 本文件里的英文问法是人写的
            "type": qtype,
            "topic": topic,
            "difficulty": difficulty,
            "lang": "zh",
            "corpus_lang": corpus_language(gold),
            "gold_spans": spans,
            "gold_chunks": gold,
            "reference_answer": reference,
            "note": note,
        })
        added += 1

    counts = {}
    for item in items:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    print(f"新增 {added} 题 → 共 {len(items)} 题：{counts}")
    for problem in problems:
        print("  [问题]", problem)
    if args.dry_run:
        print("（dry-run）未写入")
        return 0 if not problems else 1
    QUESTIONS_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {QUESTIONS_PATH}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
