# Local RAG Lab — Windows + Ollama

一套从零学习 RAG 到 Agentic RAG 的本地实验项目。默认不使用 LangChain，直接调用 Ollama HTTP API，便于观察每个步骤的数据流。

## 学习目标

1. 理解文档切分、Embedding、向量检索、Prompt 组装与生成。
2. 通过实验比较 chunk size、Top-K、向量检索与混合检索。
3. 了解 RAG 的常见失败类型，并建立小型评估集。
4. 实现一个受控的 Agentic RAG：模型可以选择检索、改写查询、基于证据回答，但有最大步骤数。

## 先决条件

- Windows 10/11
- Python 3.10+
- 已安装并运行 Ollama
- 至少一个可用的 Ollama 聊天模型

打开 PowerShell，先检查：

```powershell
ollama list
ollama --version
```

下载一个 Embedding 模型（聊天模型不能代替 Embedding 模型）：

```powershell
ollama pull bge-m3
```

> 索引是按某个 Embedding 模型建的。`storage/index.json` 里的 `embedding_model` / `embedding_dim` 记录了建库用的模型（当前为 `bge-m3` / 1024 维），换成别的模型必须重建索引。

如果电脑内存较小，聊天模型优先选你已经能流畅运行的小模型。项目默认 `qwen3.5:9b`；如果你的机器跑不动它，换成本地已有的小模型即可（不改代码，只设环境变量）。

```powershell
$env:OLLAMA_CHAT_MODEL = "你在 ollama list 中看到的模型名"
$env:OLLAMA_EMBED_MODEL = "bge-m3"
```

> 如果报 `Ollama 上找不到模型 'xxx'`，说明拉下来的模型名与配置不一致：
> 跑 `ollama list` 看真实名字（要带 tag，如 `qwen3.5:9b`），再改环境变量或 `ollama pull`。

## 安装

在项目根目录执行：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，可以不激活虚拟环境，直接使用：
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
然后把下文的 `python` 替换为 `.\.venv\Scripts\python.exe`。

### 额外准备：BGE-M3 tokenizer 词表（约 17 MB，不入库）

新版切分按 token 计量（决策 B02），需要 BGE-M3 的词表做本地计数。
HuggingFace 在国内常被墙，用 ModelScope 镜像下载：

```powershell
New-Item -ItemType Directory -Force storage/tokenizer/bge-m3 | Out-Null
curl.exe -sL -o storage/tokenizer/bge-m3/tokenizer.json "https://www.modelscope.cn/api/v1/models/BAAI/bge-m3/repo?Revision=master&FilePath=tokenizer.json"
python -m src.rag.ingestion.tokenizer count "HNSW 是什么"
```

也可以用环境变量 `RAG_TOKENIZER_PATH` 指向已有的 `tokenizer.json`。
这一步只加载词表、不跑模型推理，向量仍由 Ollama 计算。

## 第一次运行

```powershell
python -m src.build_index
python -m src.rag
```

第一次构建索引会调用 Embedding 模型，耗时取决于文档数量和电脑性能。索引保存在 `storage/` 下：文本块与 BM25 倒排表在 `index.json`，向量在 `embeddings.npy`（float32 二进制，按行序与 `records` 一一对应）。

大语料建库支持断点续建：中途中断后重新运行 `build_index` 会从上次的进度继续，不会白跑。

示例问题：
- RAG 和普通 LLM 问答有什么区别？
- 为什么需要文本切分？
- 什么情况下向量检索可能找不到准确答案？
- Agentic RAG 与固定 RAG 流程有什么区别？

## 项目结构

```text
local-rag-agentic-lab/
├── data/                       # 自己的 Markdown/TXT/PDF 语料
│   ├── rag_basics.md
│   └── agent_basics.md
├── storage/                    # 自动生成的索引（不入库）
│   ├── index.json              # 元数据 + 文本块 + BM25 倒排表（不含向量）
│   ├── embeddings.npy          # 向量矩阵 float32 (N × dim)，行序对应 records
│   └── tokenizer/bge-m3/tokenizer.json   # BGE-M3 词表，token 计量口径
├── src/rag/                    # 目标架构（方案 §6 目录；旧扁平模块在 E06 迁移时并入）
│   └── ingestion/
│       ├── block.py            # DocumentBlock：统一中间结构（D01）
│       ├── errors.py           # IngestionError / UnsupportedFormatError / TokenizerUnavailable
│       ├── tokenizer.py        # TokenCounter：token 计数 + Ollama 抽样校验（D05）
│       ├── recursive_chunker.py  # Recursive Splitter：分隔符优先级 + token 窗口兜底（D06）
│       └── parser/
│           ├── base.py            # BaseParser 抽象基类 + read_text_file（D01）
│           ├── markdown_parser.py # 标题/列表/表格/代码块/引用/图注（D02）
│           ├── txt_parser.py      # 空行分段 + 句子数统计（D03）
│           ├── pdf_parser.py      # pymupdf 版面块：页码/字号/页眉页脚（D04）
│           ├── cli.py             # python -m src.rag.ingestion.parser check|dump
│           └── __init__.py        # 按后缀分派的解析器注册表
├── src/
│   ├── __init__.py
│   ├── config.py               # 模型、切分与检索参数
│   ├── ollama_client.py        # 直接调用 Ollama API
│   ├── text_utils.py           # 文档加载与文本切分
│   ├── tokenizer.py            # 中英混合分词（jieba + 停用词 + 标识符保留）
│   ├── bm25.py                 # Okapi BM25 倒排索引（建索引 + 查询）
│   ├── build_index.py          # 建立 Embedding 向量 + BM25 倒排表
│   ├── retrieve.py             # 向量、BM25、两种混合检索
│   ├── rag.py                  # 固定 RAG Pipeline
│   ├── evaluate.py             # L1 检索评估 CLI（run / measure / compare / list）
│   ├── eval_metrics.py         # Recall@k / MRR / nDCG / bootstrap / McNemar（纯函数）
│   ├── eval_logs.py            # 日志 → 指标（只读日志，不调模型）
│   ├── eval_compare.py         # 基线对比（回归门禁）与模式配对比较
│   ├── eval_dataset.py         # 题目集校验、gold_spans → chunk id 定位
│   ├── raglog.py               # trace / span / 事件落 JSONL
│   ├── raglog_report.py        # review 报告（md / html / failures / stats / diff）
│   └── agentic_rag.py          # 受控 Agentic RAG
├── experiments/
│   ├── questions.json          # 评估集（60 题：16/16/16/12）
│   ├── build_questions.py      # 由原文句子生成题目集（gold 自动定位）
│   ├── baseline.json           # L1 基线（含 CI），回归门禁用
│   ├── probe_embed_model.py     # 探针：两个 Embedding 模型的跨语言区分度（不重建索引）
│   ├── probe_full_rank.py       # 探针：全量 81k 语料上 gold 排名（精确 / 邻近族两个口径）
│   ├── probes/                  # 探针结果 JSON
│   └── reports/                # 每次 measure 生成的报告
├── tests/                      # 单元测试（stdlib unittest）：指标、日志解析、建库重试、Block/token 计数
├── logs/                       # 自动生成的运行日志（JSONL）+ review 报告
├── docs/
│   └── eval-and-logging-design.md   # 评估方案 + 日志模块设计标准（开发前必读）
├── requirements.txt
└── README.md
```

## 评估与日志的设计标准

改评估集、加指标、调整检索日志埋点之前，先看 [`docs/eval-and-logging-design.md`](docs/eval-and-logging-design.md)。它规定：

- 哪些指标可用（Recall@k / MRR / nDCG + bootstrap CI），哪些**不可用**（关键词覆盖率已废弃）；
- 评估集的字段格式、题型配额与样本量要求（10 道题的置信区间宽达 40+ 个百分点）；
- 日志事件字段（`run` / `retrieve` / `generate` / `error` / `label`）、文件布局与 review CLI；
- 评估与日志的分工：指标只从日志算，不重复调模型。

## 运行日志与 review

`python -m src.rag` 和 `python -m src.agentic_rag` 每次提问都会自动落一份 JSONL 日志（默认 `logs/`，等级 `detail`），记录候选规模、分路排名、三段耗时、引用与弃答标记。

```powershell
python -m src.raglog_report latest --html      # 单文件 HTML，[证据 n] 可点击跳转
python -m src.raglog_report latest --md        # 打印 Markdown review 卡片
python -m src.raglog_report latest --failures  # 只看被标记的题（空检索/分数平坦/来源坍缩/零引用/引用越界/弃答）
python -m src.raglog_report latest --stats     # 延迟 p50/p95、分模式告警统计、来源 top-N
python -m src.raglog_report list               # 列出历史 run
```

## 评估集

`experiments/questions.json` 共 60 题（concept 16 / fact 16 / multi_hop 16 / negative 12），gold 由原文句子自动定位到 chunk：

```powershell
python -m experiments.build_questions          # 重新生成题目（gold_spans → gold_chunks）
python -m src.eval_dataset check               # 校验 qid 唯一、题型配额、gold 定位、负样本留痕
python -m src.eval_dataset locate "一句原文"    # 定位某句话落在哪个 chunk
python -m src.eval_dataset find 关键词         # 出题时找候选段落
```

现状基线（仅检索，60 题）：`hybrid` 的中文语料 Recall@4 70.0%、英文语料（中文提问）15.9%；`hybrid_score` 分别是 75.0% 与 23.0%。跨语言检索仍是最大短板——把 Embedding 从 `nomic-embed-text` 换成 `bge-m3` 后，英文语料的 Recall@10 从 15.3% 提到 31.9%（≈2 倍），但 k=4 上仍只有 15.9%，见实验 7 与设计文档 3.3.1。

环境变量：`RAG_LOG_LEVEL=off|summary|detail|debug`、`RAG_LOG_DIR`、`RAG_LOG_TEXT=off`（只记 hash 不记正文）、`RAG_LOG_SAMPLE`。

## 学习路线与验收

### 实验 0：确认本地模型

```powershell
ollama list
```

`bge-m3` 是 Embedding 模型，通常不支持普通聊天交互。更推荐直接运行项目的索引构建，它会通过 API 调用模型。

**验收：** `ollama list` 中能看到聊天模型和 `bge-m3`。

### 实验 1：理解最小 RAG 数据流

```powershell
python -m src.build_index
python -m src.rag
```

观察每个问题的检索片段、来源、向量相似度和最终答案。

**验收：** 能解释“问题也要向量化”以及“模型生成答案前，检索结果如何进入 Prompt”。

### 实验 2：切分策略

打开 `src/config.py`，修改 `CHUNK_SIZE` 为 `250`、`500`、`900`，每次重建索引并测试同一批问题：

```powershell
python -m src.build_index
python -m src.rag
```

注意：这里 chunk size 以字符数近似，不是 Token 数。真实项目还需要考虑标题、表格、代码块和语义边界。

**记录：** 文本块数量、相关证据是否完整、回答是否遗漏上下文。

### 实验 3：Top-K 与检索策略

在 `src/config.py` 中修改 `TOP_K` 为 `1`、`3`、`5`。运行时可在程序里输入 `/mode vector`、`/mode bm25`、`/mode hybrid`、`/mode hybrid_score` 切换检索方式（`keyword` 是 `bm25` 的旧别名，仍可用）。

- `vector`：向量余弦相似度，brute-force 全表扫（N 只有一千多，够用）。
- `bm25`：Okapi BM25 稀疏检索。索引在 `build_index` 阶段建好（`storage/index.json` 的 `bm25` 字段），查询时只扫命中词项的 posting list。
- `hybrid`：向量排名与 BM25 排名做 Reciprocal Rank Fusion（RRF，常数 60）。只看排名，不看分数，稳健。
- `hybrid_score`：两路分数各自 min-max 归一化后按 `HYBRID_ALPHA`（默认 0.5）加权。会保留「命中得多好」的信息，但容易被异常分值放大。

BM25 用的是标准公式，参数在 `src/config.py`（也可用环境变量覆盖）：

```text
idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))     # 恒为正
score(q) = Σ qtf(t) · idf(t) · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))
```

其中 `k1=1.5` 控制词频饱和（同一个词出现 3 次和 20 次差别不该太大），`b=0.75` 控制长度归一化（避免长 chunk 天然占优）。

**分词方案**（`src/tokenizer.py`）：中文走 jieba `cut_for_search`，长词同时产出子词，词典外的专有名词也能靠子词命中；英文/数字/型号/路径整体保留并额外拆片段（`GPT-4o` → `gpt-4o`、`gpt`、`4o`；`src/retrieve.py` → `src`、`retrieve`、`py`）；NFKC 归一化全角字符；过滤停用词。没装 jieba 时会自动退回二元字组（bigram）方案。

**验收：** 至少记录 5 个问题在四种模式下的差异，并解释为什么关键词检索对精确术语、缩写或标识符可能更有帮助。

### 实验 4：L1 检索评估（带置信区间）

```powershell
python -m src.evaluate run                                  # 60 题 × 4 模式，落 raglog
python -m src.evaluate measure latest --write-baseline      # 从日志算指标 + 出报告 + 存基线
python -m src.evaluate compare latest                       # 与基线对比，跌出 CI 半宽 → exit 1
python -m src.evaluate compare latest --baseline <旧基线>    # 换 embedding 模型后对比旧基线（会报 not_comparable）
python -m src.evaluate list                                 # 历史 run
```

指标定义与门禁规则见 [`docs/eval-and-logging-design.md`](docs/eval-and-logging-design.md) 第 4–6 章：

- `Recall@k = |G∩R_k|/|G|` 是主指标，另报 `hit`（命中任一 gold）与 `full`（覆盖全部 gold）；
- 必报 `MRR@10`、`nDCG@10`，全部带 bootstrap 95% CI；
- 按 `type`（fact / multi_hop / concept）与**证据语言**（中/英）切片；
- 模式之间的差异用 McNemar 配对检验，不看「区间是否重叠」；
- 负样本报**相对全库的 z 分数**（`top1_z_p50/max`、`share_z_ge_alert`）和 `separation`，**不用绝对余弦阈值**——余弦尺度随 embedding 模型变，`≥0.75` 这种阈值换模型后会得出反向结论；
- 旧的关键词覆盖率指标已删除（对排名不敏感、对 k 单调，且 `keyword` 只是 `bm25` 的别名）。

`measure` 完全不调用模型：它只读日志。换 gold、改公式、加切片都不用重跑 embedding。

**验收：** 报告含 CI 与三个维度的切片；`compare` 在人为劣化时 exit 1；`experiments/baseline.json` 可复现。

### 实验 5：观察 RAG 失败类型

尝试这些问题：
- 语料里没有答案的问题。
- 需要跨两个文档综合的问题。
- 使用与原文不同说法的问题。
- 包含产品名、缩写或数字的问题。

**验收：** 将失败归类为：语料缺失、切分问题、检索问题、证据冲突、生成幻觉或 Prompt 约束不足。

### 实验 6：运行 Agentic RAG

```powershell
python -m src.agentic_rag
```

该 Agent 可以选择：
- `search`：检索知识库；
- `answer`：依据已获得证据回答；
- `rewrite_search`：改写查询后再检索。

每一步都打印模型的动作和检索证据，并限制最大步骤数。它是教学用的受控 Agent Loop，不是生产级安全执行环境。

**验收：**
1. 对语料中有明确答案的问题，Agent 至少检索一次再回答。
2. 对模糊问题，观察它是否改写查询。
3. 对语料中没有答案的问题，检查它是否承认证据不足。
4. 确认达到最大步数后程序会停止。

### 实验 7：换 Embedding 模型解决跨语言检索

语料里 97.8% 的 chunk 是英文，而提问是中文。用 `nomic-embed-text` 时，中文提问会**优先命中中文 chunk**（`top-4` 里 96% 是中文），英文文档里的金证据根本进不来。

两个探针可以在**不重建索引**的前提下先判断值不值得换模型：

```powershell
python -m experiments.probe_embed_model            # 48 题 + gold + 3500 条随机候选，看区分度与无关问题对照
python -m experiments.probe_full_rank              # 全量 81,027 条上 gold 的排名（精确 / 邻近族）
python -m experiments.probe_full_rank --matrix <另一个模型的 embeddings.npy>
```

实测（60 题 × 4 模式，全量重建后）：

| 切片 | 指标 | nomic-embed-text | bge-m3 |
|---|---|---|---|
| 英文语料（中文提问，28 题） | Recall@4 / @10 | 11.6% / 15.3% | **15.9% / 31.9%** |
| 中文语料（20 题） | Recall@4 / @10 | 77.5% / 92.5% | 70.0% / 82.5% |
| fact 题型（16 题，`hybrid_score`） | Recall@4 / @10 | 12.5% / 15.6% | **28.1% / 40.6%** |
| 全体（`hybrid_score`） | Recall@4 / @10 | 44.3% / 48.4% | 44.7% / **58.5%** |
| 负样本 top1 相似度 ≥0.75 占比 | — | 16.7% | **0%** |

结论：`bge-m3` 把「英文语料捞不回来」修好了一半（`Recall@10` 翻倍），代价是中文语料在 `k=4` 上退步——因为跨语言能力拆掉了原来那层「语言防火墙」，英文大池（79,244 条）开始和中文小池（1,783 条）抢位置。

**验收：** 能解释为什么「换更强的多语言模型」不等于「指标一定上升」，并能用探针数据说明收益落在哪个切片上。

### 实验 8：把切分从「字符」换成「token」（方案 Phase 1 的起点）

现在的切分是 `CHUNK_SIZE=500` **字符**，实测平均只有 **134 token**（BGE-M3 词表口径），而方案要求 Child 400–500 token——差 3 倍。换口径之前必须先有 token 计数器，并且要和 Ollama 的计数对齐，否则「超限率 0」无从验收。

```powershell
# ① 计数（本地词表，毫秒级，不调模型）
python -m src.rag.ingestion.tokenizer count "HNSW 是用多层图做近似最近邻搜索的算法。"
Get-Content data/rag_basics.md | python -m src.rag.ingestion.tokenizer count -

# ② 与 Ollama 逐条对齐（每条一次调用，约 2 s）
python -m src.rag.ingestion.tokenizer verify --n 8
```

实测口径（2026-11-02，bge-m3 / XLM-R 词表 250,002）：

| 项 | 结果 |
|---|---|
| 单条 Ollama `prompt_eval_count` | `count(text) + 2`（BOS + EOS） |
| 批量 Ollama `prompt_eval_count` | `Σcount(text) + 2 × 条数` |
| 逐条对齐（8–200 条真实 chunk） | 中位相对误差 **0%**，最大绝对差 **0 token** |
| 负向对照（拿 nomic 冒充） | 中位 **5.97%**、最大差 **107 token** → 判为不一致 |
| 聚合总量对比（错误做法） | 200 条时 nomic 只差 **0.35%**，正负抵消 → 测不出词表错 |

所以：**校验必须逐条做**，聚合口径分辨率不够；切分预算用 `count()` 衡量，与 Ollama 计数天然差 2/条，这是定义不是误差。

**验收：** 能说明为什么「聚合校验」分辨不出错模型，并能用 `verify` 输出证明当前计数器与 Ollama 同口径。

### 实验 9：递归切分——先切到 token 预算里，再谈检索质量

有了 token 计数器就能按方案 §2.3 的口径切：Child 目标 400 token、硬上限 500、Parent 1500/2000。

```powershell
python -m src.rag.ingestion.recursive_chunker check --n 2000          # 期望：超限率 0.00%、还原失败 0
python -m src.rag.ingestion.recursive_chunker split "长文本..." --max-tokens 500 --target-tokens 400
Get-Content data/rag_basics.md -Raw | python -m src.rag.ingestion.recursive_chunker split - --max-tokens 200
```

切分有三条必须成立的不变式（`tests/test_recursive_splitter.py` 逐条断言）：

1. **`"".join(pieces) == 原文`** —— 只切不改写，否则 `source_start/source_end` 一定错位；
2. **每段 ≤ max_tokens** —— 超限率必须为 0，这是 D06 的验收线；
3. **分隔符不丢** —— 句号、换行符留在前一段末尾，所以代码块和表格不会被拦腰切断。

实测（2,000 条真实 chunk，target 400 / max 500）：

| 指标 | 结果 |
|---|---|
| 超限率 | **0.00%** |
| 还原失败 | **0** |
| 碎块率（< 80 token） | 3.95% |
| 长文档（拼 8 条 chunk）699 段 | token p50 318 / p95 479 / max 499 |
| 吞吐 | ≈ 3,000 chunk/s（81k 全量约 27 s） |

顺便得到一个关键事实：现有索引（500 **字符**切的）chunk token 数是 p50 128、max 434 —— **全部低于 500 token 上限**。所以「按 token 重切分」不是把块变碎，而是让每块真正装满设计的上下文预算。

**验收：** 能解释「超限率」和「碎块率」分别在防什么，并说明为什么切分必须保证逐字符可还原。

### 实验 10：解析成统一 Block——结构丢了，后面再好的切分也救不回来

三个格式各一个解析器，统一输出 `DocumentBlock`（`heading` / `paragraph` / `list` / `table` / `code` / `quote` / `caption`）。

```powershell
# 整份语料的质量比率（D02/D03 验收）
python -m src.rag.ingestion.parser check data/md
python -m src.rag.ingestion.parser check data          # 含 PDF 页码覆盖率

# 看单个文件解析成什么样
python -m src.rag.ingestion.parser dump data/rag_basics.md

# D04 验收要的「30 页人工抽查表」
python -m src.rag.ingestion.parser.pdf_parser report data/AI-Agents-in-Depth-zh-CN.pdf --pages 1-30 --out report.csv
```

实测（全语料 1,467 个文件）：

| 指标 | 结果 |
|---|---|
| block 总数 | **217,950** |
| `section_path` 非空率 | **100.00%** |
| 空 block | **0** |
| PDF 页码覆盖率 | **100.00%** |
| md 语料 block 构成 | paragraph 37,557 / list 36,754 / heading 13,430 / code 8,783 / table 2,442 / caption 513 |

三个必须牢记的坑：

1. **代码块要先于标题识别**。真实语料里有 4,295 行 `# 注释` 出现在 fenced code 里，先切 code 才不会被当成标题。
2. **pymupdf 会把标题和正文聚进同一个 block**。只按 block 首行字号判标题，实测会整段漏掉标题（`1.1 向量检索` 就这样消失过）。要在 block 内部再按字号切 run。
3. **不要用绝对字号阈值判标题**。不同排版的正文基准差一倍，用「相对正文正文字号的比值」（≥2.0 → h1、≥1.5 → h2、≥1.15 → h3）才可移植。

PDF 字符数对比（30 页）：pypdf 62,779 vs pymupdf 59,874 —— 差的 4.6% 全是空白，**去掉空白后 47,687 vs 47,464（差 0.47%）**。pypdf 会给每个 span 补空格，所以比字符数一定要先去空白。

**验收：** 能解释「为什么标题必须和正文分块但又要能关联」，并能用 `check` 输出证明 `section_path` 非空率 100%。


## 重要说明

- 本项目默认每次 `build_index` 都重建索引，方便实验；生产系统应使用增量更新、文档哈希、版本管理和删除策略。
- `bge-m3` 是独立的 Embedding 模型（1024 维、8192 上下文、多语言）。聊天模型与 Embedding 模型职责不同。
- 向量距离的数值取决于距离度量与向量库实现；不要跨实现直接比较绝对值。
- 索引分两个文件：`storage/index.json` 存元数据、`records`（id/source/chunk_id/text）和 `bm25` 倒排表；`storage/embeddings.npy` 存 float32 向量矩阵，第 i 行对应 `records[i]`。JSON 里存浮点文本会膨胀 3.3 倍，且必须整文件解析，所以向量单独用二进制。
- 换 Embedding 模型、chunk 参数或分词方案后必须重新 `build_index`；`index.json` 里的 `embedding_model`、`embedding_dim`、`tokenizer`、`n_records` 就是用来事后核对是否配套的。老格式索引（向量存在 `records[].embedding`、没有 `bm25` 字段）仍能读取，BM25 会现场建一次并缓存。
- 向量矩阵在进程内只加载一次（`retrieve._embedding_matrix` 按 `records` 对象身份缓存），`rag`/`agentic_rag` 启动时用 `warmup()` 提前加载，避免第一个问题出现冷启动尖峰。
- 建库参数可用环境变量调整：`EMBED_BATCH_SIZE`（默认 256）、`EMBED_400_RETRIES`（默认 3，Ollama 偶发 400 时的原地重试次数）、`CHUNK_SIZE`、`CHUNK_OVERLAP`；`RAG_DATA_DIR` / `RAG_STORAGE_DIR` 可把语料与索引指到别处，方便小批量试跑。实测 `bge-m3` 在 224–320 条/批都能跑，但约 30% 的请求会间歇性返回 400（runner 重启后 tokenize 代理连不上），所以 `build_index` 的策略是「400 先原地重试，仍失败再二分」。
- 检索结果对象不再携带 `embedding`，避免每条命中都复制上千维浮点数组。
- 不要把不可信文档中的指令当作 Agent 系统指令。当前实现只允许检索工具，不执行任意 shell、Python 或网络操作。
