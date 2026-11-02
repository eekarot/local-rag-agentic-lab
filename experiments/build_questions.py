"""生成/更新 experiments/questions.json（评估集 v1）。

为什么不手写 JSON：gold 必须与语料一字不差。这里只写「原文句子」（gold_spans），
由 `src.eval_dataset.locate` 自动定位出 chunk id；定位失败会立刻暴露，不会静默错配。

    python -m experiments.build_questions            # 生成并校验
    python -m experiments.build_questions --dry-run  # 只看定位结果
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import QUESTIONS_PATH
from src.eval_dataset import locate, normalize
from src.retrieve import load_index
from src.tokenizer import tokenize

# ---------------------------------------------------------------- concept：来自 rag_basics.md / agent_basics.md
CONCEPT = [
    ("RAG 是什么？", "系统先从外部知识源中检索与问题相关的材料，再把这些材料连同问题交给语言模型生成答案。",
     "检索增强生成（Retrieval-Augmented Generation）：先检索外部材料，再连同问题交给语言模型。"),
    ("为什么文档要切分成 chunk？", "若整份长文档只生成一个向量，细粒度问题可能难以匹配到正确段落",
     "一个向量压不住长文档的多个主题，细粒度问题匹配不到；切太短又会丢上下文。"),
    ("Embedding 模型负责生成最终答案吗？", "Embedding 模型不是负责撰写答案的聊天模型。",
     "不负责。Embedding 只把文本映射成向量，写答案是聊天模型的职责。"),
    ("Top-K 太大有什么风险？", "Top-K 太小可能漏掉证据，太大则可能带入无关内容。",
     "太小漏证据，太大带入无关内容并抬高延迟；需要实验选择。"),
    ("关键词检索在哪些问题上可能更有优势？", "关键词检索通过词项匹配寻找文档，适合专有名词、编号、错误代码、产品型号等精确文本。",
     "专有名词、编号、错误代码、产品型号等需要精确字面匹配的场景。"),
    ("RRF 是用来做什么的？", "由于不同检索方法的原始分数尺度可能不同，可以使用 Reciprocal Rank Fusion（RRF）融合排名。",
     "融合多路检索的排名：只用名次不用原始分数，避开量纲不一致的问题。"),
    ("Reranker 能否找回第一阶段完全没召回的证据？", "它不能找回第一阶段检索完全没有召回的证据，所以仍需要关注初始召回率。",
     "不能。Reranker 只重排已召回的候选，因此初始召回率仍是上限。"),
    ("固定 RAG 与 Agentic RAG 有什么区别？", "Agentic RAG 让模型根据问题和已有证据决定下一步，例如搜索、改写查询、再次检索或回答。",
     "固定 RAG 按预设流程走；Agentic RAG 由模型决定是否再检索、改写查询或直接回答。"),
    ("为什么 Agent Loop 需要最大步数？", "如果模型反复认为证据不足并不断搜索，循环可能永远无法结束。",
     "防止无终止循环；配合重复查询检测、错误处理与最终回退把系统收敛在可控范围。"),
    ("更换 Embedding 模型后需要做什么？", "更换 Embedding 模型后，通常需要重新计算知识库中所有文本块的向量。",
     "重建索引：旧向量与新模型的语义空间不兼容。"),
    ("RAG 和模型微调有什么区别？", "RAG 在推理时向模型提供外部检索内容；微调则通过训练过程改变模型参数。",
     "RAG 改的是推理时的上下文，微调改的是模型参数；前者适合易变/需引用的知识。"),
    ("工具调用时权限应遵循什么原则？", "工具权限应遵循最小权限原则。",
     "最小权限原则：只授予完成任务必需的权限。"),
    ("RAG 常见失败类型有哪些？", "常见失败包括：语料中没有答案、文档解析错误、切分边界不合理、检索没有召回正确证据、证据彼此冲突、Prompt 没有约束生成",
     "语料缺答案、解析错误、切分不当、召回失败、证据冲突、Prompt 缺约束、模型不忠于证据。"),
    ("Agentic RAG 相比固定 RAG 要付出什么代价？", "Agentic RAG 更灵活，但会增加延迟、成本、失败模式与控制复杂度。",
     "更灵活，但增加延迟、成本、失败模式与控制复杂度。"),
    ("chunk 切得过短会有什么问题？", "若切分过短，上下文又可能丢失。",
     "上下文被切断，单块信息不足以回答；Chunk Size 与 Overlap 需实验选择。"),
    ("为什么 RAG 不能自动保证答案正确？", "它可以为模型提供更新或私有的知识，也便于答案引用来源，但不能自动保证答案正确。",
     "RAG 只提供外部知识，是否忠实使用、是否相关仍取决于检索质量与生成约束。"),
]

# ---------------------------------------------------------------- fact：来自 NVIDIA 文档（跨语言：中文问、英文语料）
FACT = [
    ("DGX B300 单机的 GPU 显存总量是多少？", "8 x 288 GB = 2.3 TB total",
     "8 × 288 GB = 2.3 TB（八张 B300，每张 288 GB）。"),
    ("DGX B300 默认配置的系统内存是多少？", "2 TB default (up to 4 TB)",
     "默认 2 TB，可扩到 4 TB。"),
    ("DGX B300 的 GPU 通过什么方式连网络？", "ConnectX-8 networking connects directly to GPUs via PCIe Gen6.",
     "通过 ConnectX-8 网络，经 PCIe Gen6 直连 GPU。"),
    ("DGX B300 的系统内存最多能扩到多少？", "Supports up to 4 TB of RAM, configurable on-site with 128 GB DIMMs purchased from NVIDIA.",
     "最多 4 TB，现场用 NVIDIA 的 128 GB DIMM 配置。"),
    ("DGX B300 装了几张 GPU？", "The system is built on eight NVIDIA B300 Tensor Core GPUs.",
     "8 张 NVIDIA B300 Tensor Core GPU。"),
    ("NVIDIA RAG 参考架构里，文本通常切分成多少 token 的 chunk？",
     "The tokenized data is typically chunked or split into smaller chunks (256 or 512 tokens) before it is sent to Embedding NIM.",
     "256 或 512 token，然后送入 Embedding NIM。"),
    ("RAG 流水线里 chunk 大小不能超过 embedding 模型的什么限制？",
     "the chunk size should not exceed the maximum context length (8192 tokens) supported.",
     "不能超过 embedding 模型的最大上下文长度（8192 tokens）。"),
    ("MIG 关闭后要恢复 GPU 的 NVLink 点对点能力，需要什么在运行？",
     "the FM service must be running",
     "需要 Fabric Manager（FM）服务在运行。"),
    ("Base Command Manager 免费授权每台机器最多支持多少加速器？",
     "the free license is available for up to eight accelerators",
     "每台服务器/节点/系统最多 8 个加速器。"),
    ("哪种 Base Command Manager 授权套餐包含 GPU Operator、Network Operator 和 vGPU？",
     "Base Command Manager license, Enterprise Support, GPU Operator, Network Operator, vGPU, and AI and data science software",
     "NVIDIA AI Enterprise 套餐包含 BCM 授权、企业支持、GPU/Network Operator、vGPU 与 AI 软件。"),
    ("NVIDIA 认为哪些参数会显著影响 RAG 的准确率？",
     "**Accuracy:** LLM Model, Chunk Size, Reranker ON/OFF, TopK.",
     "LLM 模型、Chunk Size、Reranker 开关、TopK。"),
    ("Enterprise RAG 的 Chat 场景首字响应目标是多少？",
     "Required first response in under 3s-5s",
     "首字响应 3–5 秒以内（TTFT p90 ≤ 3–5 s）。"),
    ("Enterprise RAG 的 Summarization 场景典型输入/输出长度是多少？",
     "Long ISL (1024 tokens); Medium OSL (512 tokens).",
     "输入 1024 token（含多轮历史）、输出 512 token。"),
    ("DGX RHEL8 安装指南建议如何处理 BMC 的默认用户名？",
     "NVIDIA recommends disabling the default username and creating a unique username and password as soon as possible. [SCOPE:dgx-rhel8-install-guide]",
     "尽快禁用默认用户名，创建唯一的用户名与密码。"),
    ("NVIDIA RAG 基准里的 Wikipedia 数据集规模有多大？",
     "5.9 M documents, 38M vector embeddings (chunk size 256/50, 2048dim X FP32)",
     "590 万文档、3800 万向量，chunk 256/50，2048 维 FP32。"),
    ("RAG 里增大 chunk size 或送入更多 chunk 会带来什么开销？",
     "The larger the chunk size selected and the more the number of VDB chunks sent to the LLM the larger the latency overhead",
     "chunk 越大、送给 LLM 的 chunk 越多，延迟开销越大。"),
]

# ---------------------------------------------------------------- multi_hop：每题两个不同来源的片段
MULTI_HOP = [
    ("MIG 模式下 DGX A100 的 NVLink 会被关闭，那关闭 MIG 之后要靠什么才能恢复点对点？",
     ["On DGX A100 and NVIDIA HGX A100 systems, the corresponding GPU NVLinks and NVSwitch side NVLinks are trained off when MIG mode is enabled",
      "the FM service must be running"],
     "A100 上 MIG 开启会把 GPU 与 NVSwitch 侧 NVLink 训练关闭；关闭 MIG 后需要 Fabric Manager 服务运行才能重新训练恢复 P2P。"),
    ("DGX B300 一张卡 288 GB，整机显存多少？系统内存最大能扩到多少？",
     ["8 x 288 GB = 2.3 TB total",
      "Supports up to 4 TB of RAM, configurable on-site with 128 GB DIMMs purchased from NVIDIA."],
     "整机 2.3 TB（8×288 GB）；系统内存默认 2 TB，最大 4 TB。"),
    ("DGX B300 有几张 GPU，默认系统内存多少？",
     ["The system is built on eight NVIDIA B300 Tensor Core GPUs.",
      "2 TB default (up to 4 TB)"],
     "8 张 B300 Tensor Core GPU，默认 2 TB 系统内存（可扩到 4 TB）。"),
    ("RAG 参考架构推荐 chunk 用多少 token，同时这个大小还受什么上限约束？",
     ["The tokenized data is typically chunked or split into smaller chunks (256 or 512 tokens) before it is sent to Embedding NIM.",
      "the chunk size should not exceed the maximum context length (8192 tokens) supported."],
     "常用 256 或 512 token，且不得超过 embedding 模型最大上下文 8192 token。"),
    ("RAG 里 chunk 大小和 TopK 同时影响什么？",
     ["The larger the chunk size selected and the more the number of VDB chunks sent to the LLM the larger the latency overhead",
      "**Accuracy:** LLM Model, Chunk Size, Reranker ON/OFF, TopK."],
     "两者同时影响延迟与准确率：越大/越多越慢，同时也会影响准确率，需要权衡。"),
    ("Enterprise RAG 的 Chat 与 Summarization 两种场景，首字响应时间目标差多少？",
     ["Required first response in under 3s-5s",
      "Long ISL (1024 tokens); Medium OSL (512 tokens)."],
     "Chat 要求首字 3–5 秒内；Summarization 输入 1024/输出 512 token，首字目标 10–15 秒。"),
    ("DGX B300 单机能否落在 Base Command Manager 免费授权范围内？",
     ["the free license is available for up to eight accelerators",
      "The system is built on eight NVIDIA B300 Tensor Core GPUs."],
     "能：免费授权每台最多 8 个加速器，B300 正好 8 张 GPU。"),
    ("想要 GPU Operator、vGPU 这些能力，需要选哪种 BCM 授权？免费版的上限是多少？",
     ["Base Command Manager license, Enterprise Support, GPU Operator, Network Operator, vGPU, and AI and data science software",
      "the free license is available for up to eight accelerators"],
     "要选 NVIDIA AI Enterprise 套餐；免费版每台最多 8 个加速器。"),
    ("NVIDIA 的 Wikipedia RAG 基准用了多大的数据量，chunk 是怎么设的？",
     ["5.9 M documents, 38M vector embeddings (chunk size 256/50, 2048dim X FP32)",
      "The tokenized data is typically chunked or split into smaller chunks (256 or 512 tokens) before it is sent to Embedding NIM."],
     "590 万文档 / 3800 万向量，chunk 256/50、2048 维 FP32；参考架构一般推荐 256 或 512 token。"),
    ("Reranker 找不回的漏召回，Agentic RAG 能补吗？",
     ["它不能找回第一阶段检索完全没有召回的证据，所以仍需要关注初始召回率。",
      "Agentic RAG 让模型根据问题和已有证据决定下一步，例如搜索、改写查询、再次检索或回答。"],
     "Reranker 本身不能；但 Agentic RAG 可以通过改写查询、再检索一轮来补召回。"),
    ("为什么要有最大步数上限，它防的是哪类失败？",
     ["如果模型反复认为证据不足并不断搜索，循环可能永远无法结束。",
      "常见失败包括：语料中没有答案、文档解析错误、切分边界不合理、检索没有召回正确证据、证据彼此冲突、Prompt 没有约束生成"],
     "防的是「证据不足→反复搜索」的循环；失败类型里语料缺答案与召回失败最容易触发这种循环。"),
    ("决定 RAG 准确率的参数里包含 TopK，那 TopK 调太大有什么副作用？",
     ["**Accuracy:** LLM Model, Chunk Size, Reranker ON/OFF, TopK.",
      "Top-K 太小可能漏掉证据，太大则可能带入无关内容。"],
     "TopK 是准确率的关键参数；太大带入无关内容，太小漏证据。"),
    ("更换 embedding 模型要重建索引，那新模型的 chunk 还受什么长度限制？",
     ["更换 Embedding 模型后，通常需要重新计算知识库中所有文本块的向量。",
      "the chunk size should not exceed the maximum context length (8192 tokens) supported."],
     "要重算全部向量；且 chunk 不能超过新模型最大上下文（如 8192 token）。"),
    ("固定 RAG 延迟更低，那什么时候才值得上 Agentic RAG？",
     ["不是所有问题都需要 Agentic RAG；固定 RAG 通常更简单、更可预测、延迟更低。",
      "Agentic RAG 更灵活，但会增加延迟、成本、失败模式与控制复杂度。"],
     "只有在需要改写查询/多轮检索时才值得；代价是延迟、成本、失败模式与复杂度上升。"),
    ("RAG 有的问题一次检索就答不好，这属于哪类失败，能不能用 Agentic 方式补？",
     ["常见失败包括：语料中没有答案、文档解析错误、切分边界不合理、检索没有召回正确证据、证据彼此冲突、Prompt 没有约束生成",
      "Agentic RAG 让模型根据问题和已有证据决定下一步，例如搜索、改写查询、再次检索或回答。"],
     "「检索没有召回正确证据」是典型失败；Agentic RAG 可以改写查询再检索一轮来补。"),
    ("RAG 的准确率和性能是一回事吗？为什么 chunk/topk 的选择要权衡？",
     ["Both Accuracy and Performance are critical for RAG Inference",
      "The larger the chunk size selected and the more the number of VDB chunks sent to the LLM the larger the latency overhead"],
     "两者都关键且互相牵制：chunk 越大、送入越多，延迟越高。"),
]

# ---------------------------------------------------------------- negative：库里确实没有答案（note 记录查证方式）
NEGATIVE = [
    ("DGX B300 在中国的官方售价是多少？", "价格类问题：全库无 B300 报价，price 与「价格」检索只命中其它产品文档的许可/成本论述。"),
    ("NVIDIA 企业级支持的客服电话是多少？", "联系方式类：全库无电话号码或热线，只有「联系支持」的流程说明。"),
    ("Slurm 的许可证费用是多少？", "价格类：Slurm 相关文档只讲配置与使用，无定价信息。"),
    ("Java 版 Base Command Manager SDK 的下载地址是什么？", "版本/语言类：BCM 文档无 Java SDK 相关内容。"),
    ("Mission Control 的月度订阅费用是多少？", "价格类：Mission Control 文档无订阅价格。"),
    ("DGX B300 的上市日期是哪一天？", "日期类：文档只有硬件规格与配置，无发布/上市日期。"),
    ("H100 的官方零售价是多少美元？", "价格类：全库无 H100 零售价。"),
    ("DGX SuperPOD 的交付周期是多久？", "交付/商务类：文档讲部署架构，不讲交付周期。"),
    ("InfiniBand 线缆的采购折扣是多少？", "商务类：无采购与折扣信息。"),
    ("DGX B300 的机箱有几种颜色可选？", "外观类：文档只讲规格与接口，无机箱配色。"),
    ("本项目使用的 qwen3.5:9b 模型训练数据截止到什么时候？", "模型元信息：语料是 NVIDIA 产品文档与一本书，不含该本地模型的训练数据说明。"),
    ("我们公司内网知识库的管理员密码是多少？", "私有信息：属于用户内部环境，绝不可能出现在公开语料里（安全性正样本）。"),
]


# 证据语言：从 gold 的源路径判定（中文语料只有这三份）
CHINESE_SOURCES = ("rag_basics.md", "agent_basics.md", "AI-Agents-in-Depth-zh-CN.pdf")


def corpus_language(gold):
    if not gold:
        return None
    sources = {item["id"].split("::")[0] for item in gold}
    return "zh" if all(any(marker in source for marker in CHINESE_SOURCES) for source in sources) else "en"


def slug(text, index):
    return f"eval-{index:03d}"


def build(dry_run=False):
    records = load_index()["records"]
    items = []
    problems = []
    index = 0

    def add(question, question_type, spans, reference, note=""):
        nonlocal index
        index += 1
        gold = []
        for span in spans:
            scope = None
            if "[SCOPE:" in span:
                span, scope = span.split("[SCOPE:", 1)
                scope = scope.rstrip("]")
            hits = locate(span.strip(), records)
            if scope:
                hits = [h for h in hits if scope in h[1]]
            if not hits:
                problems.append(f"{question_type} 定位失败：{span[:40]}")
            for chunk_id, source, chunk_index in hits[:3]:
                if chunk_id not in [g["id"] for g in gold]:
                    gold.append({"id": chunk_id, "grade": 2})
        if not gold and question_type != "negative":
            problems.append(f"{question_type} 无 gold：{question}")
        items.append({
            "qid": slug(question, index),
            "question": question,
            "type": question_type,
            "lang": "zh",
            "corpus_lang": corpus_language(gold),
            "gold_spans": [s.split("[SCOPE:")[0].strip() for s in spans],
            "gold_chunks": gold,
            "reference_answer": reference,
            "note": note,
        })

    for question, span, reference in CONCEPT:
        add(question, "concept", [span], reference)
    for question, span, reference in FACT:
        add(question, "fact", [span], reference)
    for question, spans, reference in MULTI_HOP:
        add(question, "multi_hop", spans, reference)
    for question, note in NEGATIVE:
        # 负样本查证：整句在库里应当 0 命中（关键词级查证见 note 里记录的语义 top1）
        whole = sum(1 for r in records if normalize(question) in normalize(r["text"]))
        add(question, "negative", [], "",
            note=f"{note}｜查证：整句在全库命中 {whole} 条 chunk")

    if not dry_run:
        QUESTIONS_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return items, problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    items, problems = build(dry_run=args.dry_run)
    counts = {}
    for item in items:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    print(f"生成 {len(items)} 题：{counts}")
    unresolved = [i for i in items if i["type"] != "negative" and not i["gold_chunks"]]
    print(f"无 gold 的题：{len(unresolved)}")
    for problem in problems:
        print("  [问题]", problem)
    if not args.dry_run:
        print(f"已写入 {QUESTIONS_PATH}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
