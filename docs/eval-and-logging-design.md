# RAG 评估方案 + 日志模块 设计标准

| 项 | 内容 |
|---|---|
| 文档版本 | **v1.6**（当前生效版本；变更见第 11 节变更记录） |
| 适用范围 | 本仓库 `local-rag-agentic-lab` 的检索评估、答案评估、运行日志与 review 工具 |
| 约束对象 | 后续任何改动 `src/evaluate.py`、新增评估集、新增指标、调整 `retrieve`/`answer` 埋点的人 |
| 观测基线 | 现行基线 run `20261102T143004-2188`（`content_hash 3e08ef4c1c4f058e`）；历史数字见附录 A，重切分（决策 B02）后按 E05 重立 |

> 本文档是**开发依赖的标准**：指标定义、评估集格式、日志字段名以本文为准。实现与文档不一致时，要么改实现，要么先改本文档并记录到第 11 节变更记录，不允许"实现悄悄漂移"。

---

## 目录

1. [现状诊断：现有评估能测什么、不能测什么](#1-现状诊断现有评估能测什么不能测什么)
2. [术语与分层](#2-术语与分层)
3. [评估集规范](#3-评估集规范)
4. [指标定义](#4-指标定义)
5. [实验设计规范](#5-实验设计规范)
6. [基线文件与回归门禁](#6-基线文件与回归门禁)
7. [日志模块设计](#7-日志模块设计)
8. [评估与日志的接口契约](#8-评估与日志的接口契约)
9. [实施路线图与验收标准](#9-实施路线图与验收标准)
10. [待确认决策](#10-待确认决策)
11. [变更记录](#11-变更记录)

---

## 1. 现状诊断：现有评估能测什么、不能测什么

现有 `src/evaluate.py` + `experiments/questions.json` 的机制：把检索到的 4 条证据拼成一个大字符串，判断 `expected_keywords` 里的词是否出现，输出「至少命中一个关键词的问题比例」与「关键词覆盖率」。

这套机制的实测体检结果（复现命令见附录 A.5）：

| 诊断项 | 实测结果 | 结论 |
|---|---|---|
| 对排名是否敏感 | top-4 **倒序后分数完全相同**（80% / 80%） | ❌ 排名信息被丢弃，测不出「第一条对不对」 |
| 对 Top-K 是否敏感 | bm25 下 k=1: 60% → k=4: 80% → k=10: 90% → k=40: 90%，**单调不减** | ❌ 测不出「K 太大引入噪声」（而这是评估集第 4 题本身） |
| 五个模式是否独立 | `_resolve_mode` 把 `keyword` 归一为 `bm25`，两行数字恒相同 | ❌ 报告里并列 5 行是重复计数，实际只有 4 个 |
| 对「完全无关证据」是否有区分力 | 随机取 4 条 chunk → 覆盖率 **0.0%**（关键词 df 仅 0.0%–0.5%） | ✅ 这一优点真实存在，可作为「冒烟级」检查 |
| 金证据分布 | 命中集中：`AI-Agents-in-Depth-zh-CN.pdf` 1,888 条，第二名 `rag_basics.md` 17 条 | ❌ 10 题全部只能靠一本书回答，实为「大海捞针」，不测 NVIDIA 文档域的领域区分度 |
| 统计可靠性 | 8/10 → 95% CI **[49%, 94%]**；9/10 → **[60%, 98%]** | ❌ 「80% vs 90%」只差 1 题，统计上无意义 |
| 是否判定答案对错 | 不涉及答案，也无 gold 答案 | ❌ 无法回答「有没有编」 |

**结论**：现有脚本保留为「冒烟检查」（能发现检索彻底崩坏），**不得再作为效果指标**出现在报告、结论或 README 的成效描述里。

---

## 2. 术语与分层

| 层 | 定义 | 主要归因的失败 |
|---|---|---|
| **L1 检索层** | query → top-k chunk（`vector`/`bm25`/`hybrid`/`hybrid_score`） | 检索漏召回、排序差、来源坍缩 |
| **L2 答案层** | top-k chunk + prompt → 模型答案 | 幻觉、漏要点、引用错、该弃答不弃答 |
| **L3 系统层** | 端到端耗时、失败率、资源占用 | 性能与稳定性，不是"效果" |

单层数字不能互相替代：检索 Recall 高不代表答案对（模型可能编），答案对也不代表检索好（可能靠先验答对）。

---

## 3. 评估集规范

### 3.1 文件与格式

主文件：`experiments/questions.json`（保持向后兼容，现有字段保留）。新增字段如下：

```json
[
  {
    "qid": "eval-007",
    "question": "Reranker 能否找回第一阶段完全没召回的证据？",
    "query_en": "Can a reranker recover evidence the first stage never retrieved?",  // B06：跨语言消融的英文对照 query
    "type": "concept",                    // fact | multi_hop | concept | negative
    "lang": "zh",
    "expected_keywords": ["不能", "召回"],  // 仅保留作冒烟检查，不用于指标
    "gold_chunks": [                       // 主指标依据，chunk 级
      {"id": "AI-Agents-in-Depth-zh-CN.pdf::chunk-0512", "grade": 2}
    ],
    "gold_spans": ["Reranker 只能对第一阶段返回的候选重排"],  // 定位用原文片段
    "reference_answer": "不能。Reranker 只对已召回的候选重排……",
    "gold_points": ["只能重排已召回候选", "无法找回未召回的文档"],
    "note": "出自书 4.3 节"
  }
]
```

字段规则：

- `qid`：**唯一且永不复用**，日志、标注、基线全部靠它 join。格式 `eval-NNN`。
- `type`：四类固定枚举，不允许新增自由文本（否则切片统计失效）。
- `gold_chunks[].grade`：`2` = 完全支持答案；`1` = 相关但不足以回答；`0` = 不相关（一般不显式写）。二元标注时全部写 `2`。
- `gold_spans`：从语料复制原文句子，用于**自动定位 gold chunk**（见 3.3）。
- `query_en`（v1.6 新增，决策 B06）：与 `question` 同义的英文检索 query，只用于「中文 Dense vs 英文翻译 Dense vs 英文 BM25」三路消融（方案 §11.4 的 A–D 组）。它**不参与**主指标计算，缺省为 `null` 并在报告里标「英文对照缺失 n 题」。
- `reference_answer` 只对 `fact` / `concept` / `multi_hop` 必填；`negative` 不需要。

### 3.2 题型切片与配额

`type` | 定义 | 最少题数 | 语料来源
---|---|---|---
`fact` | 单跳事实，答案是一句话/一个数字 | 15 | NVIDIA 文档（`md/`、`pdf_md/`）
`multi_hop` | 需 2 个及以上 chunk 或跨文档 | 15 | NVIDIA 文档 + 书
`concept` | 概念解释（现有 10 题属此类） | 15 | `AI-Agents-in-Depth-zh-CN.pdf`、`rag_basics.md`
`negative` | 库里确实没有答案，正确行为是弃答 | 12 | 任意

**总量目标 60–80 题**。理由（样本量数学）：p≈0.8 时，95% CI 半宽 ≤10 pt 需 **n ≈ 61**；半宽 ≤7 pt 需 **n ≈ 120**。当前 10 题的 CI 宽度约 40–45 pt，**任何模式间比较都不成立**。

### 3.3 gold 标注方式

**A. 便宜档（推荐先用）**：写 `gold_spans`，由脚本自动定位 chunk：

1. 读取 `records[].text`，对**双方**做归一化：`lower()` + 去掉所有空白（含换行，PDF 抽文本有硬换行，例如 `只有业务约\n束本身`）；
2. 用归一化后的 `gold_span` 做子串搜索（长度 ≥ 12 字符以避免歧义）；
3. 命中 0 处 → 报错（标注有误或语料版本不符）；命中 >3 处 → 要求人工指定 `id`；
4. 生成的 `gold_chunks` 写回题目文件，之后不再自动覆盖（人工优先）。

**B. 严格档**：人工直接点 `id`，双人独立标 30 题，算 Cohen's κ，**κ ≥ 0.6** 才认为标注规范可用；否则先修规范（例如"相关"的定义）。

**实际落地方式（v1 数据集）**：题目由 `experiments/build_questions.py` 生成，作者只写 `gold_spans`，chunk id 由 `src.eval_dataset.locate()` 自动定位写入 `gold_chunks`。三条配套规则：

- **chunk 重叠导致的重复命中是正常的**：CHUNK_OVERLAP=80 会让同一句原文落在相邻 2–3 个 chunk 里，全部记为 `grade=2`（`gold` 有多个 id）。因此指标会同时报「命中任一 gold」与「覆盖全部 gold」，后者在 gold 多 id 时天然偏低，**不要**把两者混着比。
- **同一句在多份文档重复**（如 DGX 各代用户指南都写了同一句安全建议）时，在 span 后加 `[SCOPE:源路径片段]` 把 gold 收敛到指定文档，并把题目措辞限定到该文档，避免「四选一抽奖」。
- **PDF 抽取的硬换行不影响定位**：`locate()` 双方都做「小写 + 去空白」归一化，所以 `gold_span` 可以按自然句子书写。

**负样本的查证方式（必须留痕）**：① 整句在库中命中条数必须为 0（脚本自动写入 note）；② 用 embedding 取该问题的 top-1 相似 chunk，人工确认其中**不含答案**（只有主题相关不算可答），把 cos 值与来源写进 note。

### 3.3.1 v1 数据集实测构成（2026-10-30）

| 项 | 值 |
|---|---|
| 题数 | 60（concept 16 / fact 16 / multi_hop 16 / negative 12） |
| gold | 85 条（唯一 31 条），涉及 10 份文档 |
| 每题 gold 数 | 1 条 24 题、2 条 15 题、3 条 6 题、4 条 2 题、5 条 1 题 |
| 语料分布 | 中文 3 份（`rag_basics.md`、`agent_basics.md`、`AI-Agents-in-Depth-zh-CN.pdf`）+ 英文 7 份（DGX B300 用户指南、RAG 参考架构 3 份、BCM 授权、Fabric Manager、DGX RHEL8 安装指南） |
| 校验 | `python -m src.eval_dataset check` 通过（0 error） |
| 负样本 | 12 题，整句在库中全部 0 命中；语义 top-1 cos 0.645–0.778（最高者经人工核对仍不含答案） |

**用这套题跑出来的第一条基线（hybrid，k=10，仅检索）**：

| 题型 | n | hit@4 | hit@10 | 语料语言 |
|---|---|---|---|---|
| concept | 16 | 81.2% | 93.8% | 中文 |
| fact | 16 | 12.5% | 18.8% | 英文（中文提问） |
| multi_hop | 16 | 43.8% | 56.2% | 英文（中文提问） |

**这不是数据集缺陷，是真实的系统短板**：把同样三条 fact 问题的 query 换成英文后，其中 2 条的 gold 立刻从「未命中」变成「命中」。即 `nomic-embed-text` + BM25 在当前配置下**跨语言检索能力很弱**，中文提问基本捞不到英文文档。后续要么做查询翻译/多语言 embedding，要么在报告里如实标注「英文语料只能用英文提问」。这条结论以前的评估完全看不到，因为它只有中文题 + 中文语料。

**2026-11-02 补充：换成 `bge-m3` 之后（v1.4）**

全量重建索引（81,027 × 1024）后重跑同一套题，机制也测清楚了：

| | `nomic-embed-text` | `bge-m3` |
|---|---|---|
| 中文提问 top-4 里中文 chunk 占比 | 96% | 68% |
| 英文语料题 top-4 里中文 chunk 占比 | 52% | 1% |
| 中文提问 → 英文 gold 的全库排名 p50 / p90 | 158 / 15,394 | **22 / 145** |

`nomic` 存在一层「语言防火墙」：中文 query 几乎只匹配中文 chunk（96%），而 28 道题的金证据在英文文档里（79,244 条 chunk 中的少数）——所以必然漏。`bge-m3` 拆掉了这层墙（英文题 top-4 里中文 chunk 只剩 1%），跨语言的尾部改善约 100 倍（p90 15,394 → 145）。

**但拆墙有代价**：中文语料题也因此被英文大池挤占（top-4 里中文 chunk 96% → 68%），20 道中文题里有 6 道从 top-4 掉出、其中 4 道直接跌出 top-10。所以指标是分化的，不是单纯变好：

| 切片（`hybrid` / `hybrid_score`） | nomic | bge-m3 |
|---|---|---|
| 英文语料 Recall@4 / @10 | 11.6% / 15.3% | 15.9% / **31.9%** |
| 中文语料 Recall@4 / @10 | 77.5% / 92.5% | 70.0% / 82.5% |
| 全体 Recall@4 / @10（`hybrid_score`） | 44.3% / 48.4% | 44.7% / **58.5%** |

结论：多语言 embedding **不是银弹**，它把「语言不匹配」换成了「语料规模不均衡」。下一步应该在检索层做语言路由或来源配额（见第 10 章待确认项），而不是继续换更大的模型。

### 3.4 出题禁忌

- ❌ 不要从被检索的同一段落里逐字抄题目（BM25 白送分，会系统性高估）；
- ❌ 不要出「语料里没有金证据」的正样本（无法判定对错）；
- ❌ `negative` 题必须事先确认全库无答案（对全库做关键词 + 语义双查，并把查询记录写进 `note`）；
- ❌ 不要把同一事实换 5 种说法当 5 题（会虚增 n，压窄 CI 但没增加信息量）。

---

## 4. 指标定义

### 4.1 L1 检索层

记号：`G` = 该题 `gold_chunks` 集合；`R_k` = 返回的前 k 条 chunk id 列表；`rel(i)` = 第 i 位的相关等级（0/1/2）。

| 指标 | 公式 | 地位 | 备注 |
|---|---|---|---|
| **Recall@k** | `|G ∩ R_k| / |G|` | **主指标**，k=4 与 k=10 都报 | 检索天花板；k 必须与生产配置一致 |
| **MRR@10** | `mean(1 / rank_first_relevant)` | 单 gold 题必报 | 对"第一条就对"敏感 |
| **nDCG@10** | `DCG@10 / IDCG@10`，`DCG = Σ (2^{rel_i} − 1) / log2(i+1)` | 有分级标注时用 | 二元标注时退化为位置折扣的 Recall |
| **Precision@k** | `|G ∩ R_k| / k` | ⚠️ 仅参考 | `|G|=1..3`、k=4 时上限只有 25%–75%，易被误读 |
| Hit@k | `1[|G ∩ R_k| > 0]` | ⚠️ 仅参考 | 小样本下 CI 极宽 |
| **证据充分率** | 人工判「返回的 chunk 中是否存在足以回答的原文」 | ✅ 人工抽查 20 题 | 不依赖 gold 标注质量 |
| 来源多样性 | `n_unique_sources / k` | ✅ 诊断 | 低值 = 来源坍缩（4 条来自同一文档相邻 chunk） |

**实现口径（S2 落地，`src/eval_metrics.py`）**：`Recall@k` 报连续值 `|G∩R_k|/|G|`；另外同时报 `hit@k`（命中任一 gold）与 `full@k`（覆盖全部 gold）——前者宽松、后者严格，`|G|>1` 时三个数字必然不同，**禁止只挑好看的报**。`Precision@k` 未实现（理由见上表：`|G|=1..3`、k=4 时上限只有 25%–75%）。

**禁止**：用检索分数绝对值跨模式比较。BM25 分与余弦相似度量纲不同；RRF 融合后 `score` 是排名分（例如 `0.0164`），无相似度语义。要比就比**排名类指标**。

**负样本的口径（v1.5 修正）**：不能用「top1 余弦 ≥ 0.75 」这种**绝对阈值**，因为余弦尺度随 embedding 模型变。实测换 `bge-m3` 后负样本 top1 绝对余弦 0.682 → 0.619（看上去像变好），但相对全库的 z 分数是 +5.74 → +5.08（实际略变差）——正样本 gold 的绝对余弦也从 0.682 降到 0.590，整个尺度在下移。

所以负样本段改报三个相对量（由 `retrieve.vector_search` 在日志里带上 `cos_mean` / `cos_std`，`measure` 不重调模型）：

| 字段 | 含义 |
|---|---|
| `top1_z_p50` / `top1_z_max` | `(top1_cos − 全库均值) / 全库 std` 的中位数与最大值 |
| `share_z_ge_alert` | z ≥ `NEG_Z_ALERT`（3.0）的比例 = 「检索器很自信地凑了个答案」 |
| `separation` | `正样本 top1 z 中位数 − 负样本 top1 z 中位数` |

`separation` 是决定「能不能用单一向量阈值做弃答」的根据：实测 `bge-m3` 下正样本 +5.69、负样本 +5.28，**separation 只有 +0.42** → 负样本在向量空间里和正样本一样「自信」，**单靠余弦阈值弃答不可行**，必须靠别的信号（证据一致性、覆盖度、LLM 自评）。旧日志没有 `cos_mean` / `cos_std` 时，`measure` 会明确提示「请重跑 run」而不是默默报一个不可比的数。

### 4.2 L2 答案层

| 指标 | 定义 | 需要什么 | 备注 |
|---|---|---|---|
| **答案正确率** | 与 `reference_answer` 比对，人工 0 / 0.5 / 1 | 参考答案 | 参考答案需第二人复核 |
| **Faithfulness（忠实度）** | 答案拆成原子陈述，`被证据支持的陈述数 / 总陈述数` | 仅需证据 + 答案 | **不需要参考答案**，优先做 |
| **Citation precision** | `答案中 [证据 n] 且该证据确实支持该句 / 总引用数` | 同上 | 引用越界（n > k）单独计为错误 |
| **要点覆盖（completeness）** | `命中 gold_points / |gold_points|` | `gold_points` 清单 | 必须语义判定，不做字面匹配 |
| **弃答正确率** | `negative` 题中输出"证据不足/无法回答"的比例 | negative 题集 | 越高越好 |
| **错误自信率** | `negative` 题中给出确定性答案的比例 | 同上 | 越低越好；比弃答正确率更值得报 |
| **过度弃答率** | 正样本中被弃答的比例 | 正样本 | 与上面两项构成三角，需同时报 |

`abstained` 的自动检测只能做**启发式**（匹配「证据不足 / 无法回答 / 未提供」等短语），日志与报告中必须标注 `heuristic: true`，不得当作判定结论。

### 4.3 L3 系统层

- **延迟分解 p50/p95**：`embed.query` / `retrieve` / `generate` 三段独立统计（禁止只报端到端平均值）；
- **失败率**：空检索、HTTP 错误、答案截断、引用越界、超时；
- **规模**：prompt 字符数、证据字符占比、`prompt_eval_count` / `eval_count`（Ollama 原生返回）；
- **资源**：进程峰值内存、索引体积。

### 4.4 LLM-as-judge 使用规范

只有在满足以下全部条件时，judge 分数才可以出现在结论里：

1. 先在 **30–50 条人工校准集**上算 judge 与人的一致率，**Cohen's κ ≥ 0.6**；
2. judge 的每条判定必须输出**依据的原文片段**，无法定位到证据的判定作废；
3. 同一份输入用两种顺序各判一次，自洽率必须 ≥ 90%；
4. judge 与被测模型**不得同源**（当前两者都是 `qwen3.5:9b` 时，只能用于"回归排序"，不得报绝对准确率）；
5. 报告中必须写明 judge 模型名、prompt 版本、κ 值。

---

## 5. 实验设计规范

### 5.1 配对设计

所有模式/参数必须在**同一批题**上跑，只比较**逐题差值**，不允许跨题集比较平均值。

### 5.2 不确定度

- 指标一律给 **bootstrap 95% CI**（对题目重采样 1,000 次，固定 seed）；
- 两个模式的命中差异用 **McNemar 检验**（配对二元）或**配对 bootstrap**（连续指标）；
- 报告格式：`Recall@4 = 0.72 [0.61, 0.82]`，禁止只写 `0.72`。

### 5.3 一次只动一个变量（实验矩阵）

变量 | 取值 | 观察指标
---|---|---
`chunk_size` / `chunk_overlap` | 200/500/1000，0/80/200 | Recall@4、证据充分率
`k` | 1 / 4 / 10 / 20 | Recall@k 曲线、答案忠实度随 k 的变化
检索模式 | `vector` / `bm25` / `hybrid` / `hybrid_score`（**4 个，不含 `keyword` 别名**） | nDCG@10、MRR@10
`embedding` 模型 | `nomic-embed-text` vs 备选 | 重建索引后整体重跑
`rrf_k` | 30 / 60 / 100 | nDCG@10

改动 `chunk_size` / `embedding` 属于**索引语义变更**：必须重建索引、重新生成 `content_hash`，旧基线不可复用。

### 5.4 可复现要求

每次 run 必须固化：题目集文件 sha256、索引指纹（`n_records` / `dim` / `embedding_model` / `tokenizer` / chunk 参数 / `bm25_terms`）、`chat` 与 `embed` 模型 tag、`temperature`（评估固定为 0）、`seed`、Python 版本。全部写入日志的 `run` 事件（见 7.3）。

### 5.5 切片报告（强制）

必须分别给出：`type`（4 类）、**证据语言**、gold 来源类型（书 vs NVIDIA 文档）。**禁止只给一个总分**。

**S2 实测修正**：原来写的是按 `lang`（题面语言）切片，落地时发现 60 题全是中文提问，这个轴的信息量为零。真正有区分度的是 **`corpus_lang`（gold 所在文档的语言）**：中文语料 `Recall@4 = 77.5%`，英文语料（中文提问）`11.6%`，差 6.6 倍。所以题目集新增 `corpus_lang` 字段（由 gold 源路径推导），切片一律用它；题面语言只作为元信息保留。

---

## 6. 基线文件与回归门禁

### 6.1 基线文件格式

`experiments/baseline.json`：

```json
{
  "content_hash": "<run 事件里的指纹 hash>",
  "created_at": "2026-10-30T17:20:00+08:00",
  "dataset": {"file": "experiments/questions.json", "sha256": "..."},
  "index": {"n_records": 81027, "dim": 1024, "embedding_model": "bge-m3",
            "tokenizer": "jieba-cut_for_search", "chunk_size": 500, "chunk_overlap": 80},
  "models": {"chat": "qwen3.5:9b", "embed": "bge-m3"},
  "metrics": {
    "all": {"recall@4": [0.72, 0.61, 0.82], "mrr@10": [0.66, 0.55, 0.76], "nDCG@10": [0.68, 0.57, 0.78]},
    "by_type": {"fact": {"recall@4": [0.80, 0.64, 0.93]}, "negative": {"over_abstain": [0.10, 0.03, 0.25]}},
    "by_mode": {"vector": {}, "bm25": {}, "hybrid": {}, "hybrid_score": {}}
  },
  "run_id": "20261030T164012-3f9a"
}
```

### 6.2 门禁规则

- `content_hash` 不同 → **先警告，不做回归判定**（不是干净对比）；
- 任一主指标下降幅度 **超过其 95% CI 半宽** → 判定为回归，需要说明或回滚；
- 只允许「指标持平或上升」才可合并到主分支（本仓库无 git，等价于"才可作为新的默认配置"）。

**实现细节（`src/eval_compare.py`，踩过的坑）**：

- 容差取**基线**那一侧的 `half_width`，不用当前 run 的 —— 否则样本越多容差越紧，历史基线会越来越容易被判回归；
- `baseline.json` 经过 JSON 往返后，整数 `k` 键会变成字符串 `"4"`，比较前必须两边都归一成 `str`，否则每一项都会被误报成「缺失」（实测第一版报了 464 项缺失）；
- `not_comparable`（指纹不同 / 题目集不同 / 有一侧缺项）**不返回非零退出码**，只打印警告：它不是回归，是「没得比」；此时逐行判定一律显示「不可比」而不是「噪声内」（否则 +14 pt 的差值会被标成噪声，误导阅读）；
- 题目集用 `sha256` 单独比对，因为换题会直接让指标不可比，而这个变化不会体现在 `content_hash` 里（`content_hash` 只覆盖索引与模型）；
- 换 Embedding 模型后用 `compare latest --baseline <旧基线路径>`：默认只比 `experiments/baseline.json`，而那时它已被新模型的新基线覆盖，不改用 `--baseline` 就永远只会和「自己」比。

---

## 7. 日志模块设计

### 7.1 目标与非目标

**必须能回答 review 时的 8 个问题**：

| # | review 问题 | 对应字段 |
|---|---|---|
| 1 | 答错了，是检索没捞到还是模型编的？ | `hits[].text` + `answer` + `cited` |
| 2 | 关键词题为什么 BM25 没命中？ | `tokens`、`branch.bm25.terms_hit`、`postings_touched` |
| 3 | 这条命中是向量给的还是 BM25 给的？ | `hits[].rank_vector` / `rank_bm25` / `cos` |
| 4 | 正确的 chunk 为什么排第 7？ | 完整 top-k 排名 + `scores.gap` |
| 5 | 慢在哪？ | 三段 `latency_ms` |
| 6 | 和上次比退化了没？ | `run` 事件指纹 + `--diff` |
| 7 | 没证据时是老实说还是硬编？ | `abstained`、`evidence_count`、`cited_valid` |
| 8 | 日志本身可信吗？ | `run.index` / `run.models` / `schema` |

**非目标**：分布式追踪、Prometheus 指标、SQL 数据库。本地实验台：**JSONL + 一个 CLI**。

### 7.2 事件模型

```
trace  = 一次提问（交互式一问，或评估集里 一题×一个模式）
 └ span = 阶段：embed.query / retrieve.vector / retrieve.bm25 / retrieve.fuse
          / prompt.build / generate / verify.citations
 └ event = 告警或错误（空检索、分数平坦、引用越界、HTTP 404…）
```

上下文用 `contextvars` 传递，**不改 `retrieve()` / `answer()` 的签名**。

### 7.3 事件字段规范

公共字段（每行必有）：`ts`、`schema`（`"raglog/1"`）、`run_id`、`trace_id`、`span_id`、`parent_id`、`event`、`level`、`qid`、`mode`、`k`。

#### (a) `run` 事件（每个 run 首行）

```json
{"event":"run","schema":"raglog/1","run_id":"20261030T164012-3f9a",
 "started_at":"2026-10-30T16:40:12+08:00","python":"3.12.7","platform":"Windows-11",
 "index":{"n_records":81027,"dim":768,"embedding_model":"nomic-embed-text",
          "tokenizer":"jieba-cut_for_search","chunk_size":500,"chunk_overlap":80,
          "bm25_terms":124950,"index_mtime":"2026-10-30T16:38:11","embeddings_mtime":"2026-10-30T16:00:08"},
 "models":{"chat":"qwen3.5:9b","embed":"nomic-embed-text"},
 "params":{"modes":["vector","bm25","hybrid","hybrid_score"],"k":4,"temperature":0,"seed":42},
 "dataset":{"file":"experiments/questions.json","sha256":"..."},
 "content_hash":"..."}
```

#### (b) `retrieve` 事件（review 主力）

```json
{"event":"retrieve","level":"info","query":"Reranker 能否找回第一阶段没召回的证据","query_chars":20,
 "tokens":["reranker","能否","找回","第一","阶段","没","召回","的","证据"],
 "branch":{"bm25":{"terms_hit":6,"postings_touched":1842,"candidates":521},
           "vector":{"candidates":81027,"cos_mean":0.2453,"cos_std":0.0447,
                     "top1_cos":0.7061,"topk_cos_floor":0.5488}},
 "fuse":{"rule":"rrf","rrf_k":60,"candidates_in":587,"candidates_out":4},
 "scores":{"top1":0.0164,"topk":0.0148,"gap":0.0016,"gap_ratio":0.0998},
 "empty":false,"n_unique_sources":2,"latency_ms":4.2,
 "hits":[{"rank":1,"id":"AI-Agents-in-Depth-zh-CN.pdf::chunk-0154","chunk_id":154,
          "source":"AI-Agents-in-Depth-zh-CN.pdf","len":498,"score":0.0164,
          "rank_vector":3,"rank_bm25":1,"cos":0.5488,"text":"..."}]}
```

字段语义要点：

- `rank_vector` / `rank_bm25` / `cos`：hybrid 调参的**唯一证据来源**，必须记；
- `scores.gap` = `top1 - topk`：用于标记**分数平坦**（排序不可信）；
- `n_unique_sources`：用于标记**来源坍缩**（4 条来自同一文档相邻 chunk，实际只覆盖一段）；
- `text` 记全文（4×~500 字符 ≈ 2 KB）；超上限时截断并置 `text_truncated: true` + 外置 artifacts；
- `branch.vector.cos_mean` / `cos_std`：**全库分数分布**。负样本要靠 `(top1_cos − cos_mean) / cos_std` 判断离群程度，因为绝对余弦随 embedding 模型变（v1.5）。没有这两个数，`measure` 会明确提示旧日志不可比；
- 单路模式也要写 `branch`：纯 `vector` 模式没有 `bm25` 统计、纯 `bm25` 模式没有向量统计，对应子项写 `null` 而不是整个 `branch` 缺失（实测曾因 `if bm25:` 漏记 60 条 vector 事件的分数分布）。

#### (c) `generate` 事件

```json
{"event":"generate","level":"info","model":"qwen3.5:9b",
 "params":{"temperature":0,"seed":42,"stream":false},
 "context_chars":1987,"evidence_count":4,"evidence_truncated":false,
 "latency_ms":6120,"prompt_eval_count":742,"eval_count":186,
 "answer":"...","answer_chars":312,
 "cited":[1,2],"cited_valid":true,"citations_missing":true,
 "abstained":false,"abstained_heuristic":true,"finish_reason":"stop"}
```

- `prompt_eval_count` / `eval_count` 来自 Ollama 原生返回，不额外计算；
- `cited` 解析答案中的证据编号，**不要求方括号**（`[证据 1]`、`（证据 1）`、`证据 1` 都算）：实测 agentic 的回答写成「（证据 1）」，只认方括号会漏计；`cited_valid` = 全部落在 `1..k`；`citations_missing` = 有结论但零引用；
- `abstained` 为启发式，**必须**伴随 `abstained_heuristic: true`；
- 完整 `prompt.user` 不落主日志：记 `prompt_sha256`，超长时外置 `artifacts/<trace_id>.txt`。

#### (d) `error` 事件

```json
{"event":"error","level":"error","where":"generate","model":"qwen3:4b","http_status":404,
 "body":"{\"error\":\"model 'qwen3:4b' not found\"}",
 "hint":"模型未安装，见 ollama list","attempt":1,"recoverable":false}
```

#### (e) `label` 事件（人工 review 回流，独立文件）

```json
{"trace_id":"...","qid":"eval-007","label":"retrieval_miss",
 "flags":["gold_not_in_topk","source_collapse"],
 "note":"gold 在 chunk-512，被同一文档 4 条挤掉","by":"zcl","at":"2026-10-30T17:02:11"}
```

`label` 固定枚举：`ok` / `retrieval_miss` / `ranking_bad` / `answer_wrong` / `hallucinated_citation` / `over_abstain` / `under_abstain`。

### 7.4 文件布局

```
logs/
├── runs.jsonl                            # 每个 run 一行清单（含 content_hash）
└── 2026-10-30/
    ├── 20261030T164012-3f9a.jsonl        # 事件流，单行 JSON
    ├── 20261030T164012-3f9a.labels.jsonl # 人工标注回流
    └── artifacts/                        # 仅超长时外置 prompt / 答案
```

规则：单文件 > 20 MB 滚动为 `....part2.jsonl`；UTF-8、单行 JSON、不 pretty print（保证 `jq` / `rg` / `wc -l` 可用）；保留策略 `python -m src.raglog_report prune --days 14`。

### 7.5 review CLI

```
python -m src.raglog_report <run_id|latest> [--md|--html|--failures|--stats|--diff OTHER]
```

| 子命令 | 输出 | 用途 |
|---|---|---|
| `--md`（默认） | 每题一张 review card：问题 / 三段耗时 / 证据表（rank·分路来源·score·cos·首行·长度）/ 答案 / 引用校验 / 告警标签 | 逐题精读 |
| `--html` | 单文件、内嵌 CSS、无外部依赖；`[证据 2]` 可点击跳到对应证据块；分数条形图；证据可折叠 | 人工 review 首选 |
| `--failures` | 仅列被标记的题：空检索 / 分数平坦 / 来源坍缩 / 零引用 / 引用越界 / 弃答 / 错误 | review 从这十几条开始 |
| `--stats` | 延迟 p50/p95、耗时分解、`k` 分布、来源 top-N、模式间逐题胜负面 | 快速体检 |
| `--diff A B` | 逐题对比证据集合新增/消失、答案变化、指标变化；`content_hash` 不同则顶部警告 | 回归审查 |

`--failures` 的标记规则（阈值可配置）：

| 标记 | 判定条件 |
|---|---|
| `empty` | `empty == true` |
| `flat_scores` | `gap_ratio = (top1 − topk) / |top1| < 0.02`；**RRF 融合只在 `gap == 0`（完全同分）时标记**（原因见 7.8） |
| `source_collapse` | `n_unique_sources / k < 0.5` |
| `no_citation` | `citations_missing == true` |
| `citation_out_of_range` | `cited_valid == false` |
| `abstained` | `abstained == true`（启发式，见 7.8） |
| `error` | 该 trace 内存在 `error` 事件 |

### 7.6 性能、开关、隐私

| 项 | 规范 |
|---|---|
| 开销目标 | 每 trace < 0.3 ms（内存攒批，trace 结束一次写入） |
| 级别 | `RAG_LOG_LEVEL=off\|summary\|detail\|debug`，默认 `detail`；`debug` 才记候选分布统计 |
| 快速路径 | 埋点处 `if not raglog.enabled("debug"): return`，`off` 时零成本 |
| 采样 | 交互式与评估集全量；压测支持 `RAG_LOG_SAMPLE=0.05` |
| 大字段 | `text` / `answer` > 8 KB 外置 `artifacts/<trace_id>.txt`，主日志留路径 + sha256 |
| 脱敏 | 提供 `redact_patterns`（`sk-`、`Bearer` 等）与 `RAG_LOG_TEXT=off`（只记 hash） |
| 并发 | 文件写入加锁 + 每行原子写，支持并行评估 |
| 版本 | `schema` 字段 + report 对缺失字段降级显示 |

### 7.7 模块拆分与 API 草案

```
src/raglog.py          # 公共 API：configure / trace / span / emit / hits / answer
src/raglog_schema.py   # 事件名与字段常量 + SCHEMA_VERSION（report 与埋点共用）
src/raglog_report.py   # CLI：--md / --html / --failures / --stats / --diff / prune
```

```python
# 用法（埋点侧）
with raglog.trace(qid="eval-007", mode="hybrid", k=4) as t:
    with raglog.span("retrieve"):
        hits = retrieve(q, mode="hybrid", k=4)
    t.retrieve(hits, tokens=..., branch=..., fuse=..., scores=...)
    with raglog.span("generate"):
        ans = chat(messages)
    t.generate(ans, model=..., tokens_prompt=..., tokens_completion=...)
```

埋点位置（**只新增，不改返回值**）：

| 文件 | 位置 | 记录内容 |
|---|---|---|
| `src/retrieve.py` | `bm25_search` | `tokens`、`terms_hit`、`postings_touched` |
| `src/retrieve.py` | `vector_search` | `candidates`、`top1_cos`、`topk_cos_floor` |
| `src/retrieve.py` | `hybrid_search` / `hybrid_score_search` | 每 hit 的 `rank_vector` / `rank_bm25` / `rrf` 贡献 |
| `src/rag.py` | `answer()` | `prompt` 统计、`cited` 解析、`abstained` 启发式 |
| `src/ollama_client.py` | `_post` | 失败时 `error` 事件（含 HTTP 状态与响应体） |
| `src/agentic_rag.py` | 每步 `plan/act/observe` | 每步一个 span，`step_index` |

### 7.8 两条判定规则（S0 实测后修订）

这两条规则初版写错，是**跑真实数据后暴露的**，记录在此避免以后重犯。

**(1) 「分数平坦」不能用绝对差值**

初版用 `top1 − topk < 0.02`。但 RRF 的分数就是 `Σ 1/(k+rank)`，k=60 时相邻名次天然只差约 `2.7e-4`，于是**每一条 RRF 结果都被判成「平坦」**（实测 10/10 条 hybrid 全中）。

修订：改用相对值 `gap_ratio = gap / |top1| < 0.02`；并且 RRF 只在**完全同分**（`gap == 0`，即融合后仍打平）时标记。

**(2) 「弃答」要分强弱信号，且弱信号要求零引用**

初版只要答案里出现「证据不足」就判弃答。实测有 4/10 条正常答案被误判，因为它们在**正文里讨论这个概念**（例：`s0-09 的答案写"如果模型反复认为证据不足并不断搜索…"`）。

修订：
- 强信号（`无法回答 / 无法确定 / 不能回答 / 没有足够证据 / cannot answer`）出现即判弃答；
- 弱信号（`证据不足 / 没有找到 / 未提供 / 找不到`）只有在**答案零引用**时才判弃答。

修订后同一批数据从 10/40 告警降到 1/40（剩下那条是真实问题：`RAG 是什么？` 的 BM25 只分出 `["rag"]` 一个词项，4 条证据分数几乎并列）。

---

## 8. 评估与日志的接口契约

```
评估脚本 → 只负责「跑」：读题目集、按模式调用 retrieve/answer（埋点自动生效）
日志     → 只负责「存」：原始证据、分数、答案、耗时、错误
report   → 负责「看」：人读卡片、失败清单、run 间 diff
metrics  → 负责「算」：从日志 + gold + labels 计算指标
```

强制约束：

1. **metrics 只读日志，不自己调模型**。改指标公式不需要重跑 LLM 调用；同一份日志可用不同 gold 反复计算。
2. **日志不含 gold，logs 与 `experiments/questions.json` 通过 `qid` join**。这样换标注不需要重跑。
3. 任何指标若无法从日志字段推出，说明日志缺字段 → **补日志字段**，而不是在评估脚本里另起一套数据采集。
4. 人工标注写 `*.labels.jsonl`，不写进主日志（保持主日志只追加、不可变）。

### 8.1 实现对照（S1 + S2 已落地）

| 环节 | 模块 | 命令 |
|---|---|---|
| 出题（gold_spans → chunk id） | `experiments/build_questions.py` | `python -m experiments.build_questions` |
| 题目集校验 | `src/eval_dataset.py` | `python -m src.eval_dataset check` |
| 跑（落日志） | `src/evaluate.py`（`cmd_run`） | `python -m src.evaluate run --modes ... --k 10` |
| 存 | `src/raglog.py` | 自动 |
| 算（只读日志） | `src/eval_logs.py` + `src/eval_metrics.py` | `python -m src.evaluate measure latest` |
| 看 | `src/evaluate.py`（Markdown 报告）、`src/raglog_report.py`（逐题 review） | `... measure` / `... --html` |
| 门禁 | `src/eval_compare.py` | `python -m src.evaluate compare latest` |

`run` 会把全部 query 的向量一次性批量预计算（1 次 `/api/embed`），并对 jieba 先做一次 warmup，否则首题分词的 800 ms 会污染 p95。

---

## 9. 实施路线图与验收标准

| 阶段 | 内容 | 验收标准（DoD） |
|---|---|---|
| **S0** | `raglog` 三件套 + `rag.py` / `agentic_rag.py` 埋点 | 跑现有 10 题，生成一份可 `--md` / `--html` review 的报告；`off` 级别下 `answer()` 端到端耗时差异 < 2% |
| **S1** | 题目集扩到 60 题 + 12 负样本，`gold_spans` → 自动定位 `gold_chunks` | ✅ 已完成：60 题（16/16/16/12）、gold 定位失败 0、`qid` 唯一、`check` 0 error；并实测出跨语言检索短板（见 3.3.1） |
| **S2** | 检索指标脚本（Recall@k / MRR / nDCG + bootstrap CI），产出 `baseline.json` | ✅ 已完成：`run/measure/compare/list` 四子命令；31 个单元测试；报告含 `type`/`corpus_lang`/`mode` 切片 + CI；keyword coverage 已删除；`baseline.json` 可生成，人为劣化时 `compare` 返回 exit 1 |
| **S3** | 答案层：40 题参考答案 + 人工打分表；faithfulness / citation precision 抽检 | 产出答案正确率、忠实度、引用准确率；弃答三角（正确/错误自信/过度弃答） |
| **S4** | judge 校准（30–50 条人工 + κ）后再自动化 | κ ≥ 0.6 才启用 judge；报告写明 κ 与 judge 模型 |
| **S5** | 回归门禁 + `--diff` 接入日常流程 | 任一主指标跌幅 > CI 半宽即报警；`content_hash` 变化时自动提示 |

---

## 10. 待确认决策

| # | 决策 | 建议默认 | 状态 |
|---|---|---|---|
| 1 | 存储格式：JSONL 单文件 vs SQLite | JSONL（`jq` 友好、零依赖） | **已确认（S0 已实现）** |
| 2 | 是否记录 `hits[].text` 全文 | 记录全文（仅超 8 KB 外置） | **已确认（S0 已实现）** |
| 3 | 默认日志级别 | `detail` | **已确认（S0 已实现）** |
| 4 | 是否实现 `--html` 单文件 viewer | 实现（可点击证据跳转） | **已确认（S0 已实现）** |
| 5 | Embedding 模型：`nomic-embed-text` vs `bge-m3` | **已换成 `bge-m3`**：英文语料 Recall@10 15.3%→31.9%、负样本 top1≥0.75 占比 16.7%→0%；代价是中文语料 k=4 上 77.5%→70.0% | **已实施（2026-11-02）** |
| 6 | 是否在检索层做语言路由 / 来源配额（修复「英文大池挤占中文小池」） | 建议先试语言加权：同语言 chunk 乘 1.1–1.2 再排序；只改 `retrieve`，不动索引 | 待确认 |

---

## 11. 变更记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v1.0 | 2026-10-30 | 初稿：现状诊断、评估集与指标规范、实验设计、日志模块设计、路线图 | zcl |
| v1.1 | 2026-10-30 | S0 落地：4 项决策确认；新增 7.8 两条判定规则（RRF 平坦判定、弃答强弱信号）；补 `gap_ratio` 字段 | zcl |
| v1.2 | 2026-10-30 | S1 落地：数据集 v1（60 题 / 85 gold / 10 文档）；新增 3.3 落地细则与 3.3.1 实测构成；记录跨语言检索基线 | zcl |
| v1.3 | 2026-10-30 | S2 落地：`run/measure/compare`；指标三口径（Recall / hit / full）+ bootstrap CI + McNemar；切片轴由「题面语言」改为「证据语言」（5.5）；补 6.2 门禁实现细节与 8.1 模块对照 | zcl |
| v1.4 | 2026-11-02 | Embedding 换 `bge-m3`：全量重建 81,027×1024，新基线 `3e08ef4c1c4f058e`；3.3.1 补「语言防火墙」实测（中文题 top-4 中文 chunk 96%→68%、英文题 52%→1%）；真实 400 重试（旧二分分支因匹配字串不对而从未生效）；新增决策 5/6 | zcl |
| v1.6 | 2026-11-02 | 题集新增 `query_en` 字段（决策 B06，仅用于三路消融，不进主指标）；记录重切分（B02）对题集的三个影响：`gold_chunks` 全部重定位、多 gold 题收敛导致 `hit`/`full` 差异消失、旧基线作废需按 E05 重立 | zcl |
| v1.5 | 2026-11-02 | 负样本口径从「绝对余弦 ≥0.75」改为「相对全库 z 分数 + separation」：`vector_search` 记 `cos_mean`/`cos_std`，`measure` 报 `top1_z_p50/max`、`share_z_ge_alert`、`separation`；旧日志自动降级并提示重跑；新增 9 个单元测试（共 45）；顺带修掉「纯 vector 模式不记 branch」「branch.vector=null 时 measure 崩溃」两个 bug | zcl |

---

## 附录 A：现状诊断的实测证据

### A.1 指标对排名不敏感

bm25 模式、top-4 正序与倒序，代理指标完全相同：

```
k= 4 正序: 命中问题比例=80%  关键词覆盖率=80%
    倒序: 命中问题比例=80%  关键词覆盖率=80%
```

原因：`evaluate.py` 中 `evidence = "\n".join(...)` 后做 `in` 判断，排名信息在拼接时被丢弃。

### A.2 指标对 Top-K 单调不减

```
k= 1: 命中=60%  覆盖=60%
k= 4: 命中=80%  覆盖=80%
k=10: 命中=90%  覆盖=85%
k=40: 命中=90%  覆盖=90%
```

因为它是「并集是否包含关键词」，只会随 k 增大而增大 → 无法反映「K 变大引入噪声」。

### A.3 `keyword` 与 `bm25` 是同一实现

`src/retrieve.py` 的 `_resolve_mode` 把 `keyword` 归一为 `bm25`；`evaluate.MODES` 并列二者导致重复计数。

### A.4 关键词是否稀有 / 金证据是否集中

各题关键词在全库的出现比例仅 0.0%–0.5%；随机取 4 条 chunk 的覆盖率 0.0%（说明该指标能识破完全无关证据）。10 题的金证据命中集中在 `AI-Agents-in-Depth-zh-CN.pdf`（1,888 条），第二名 `rag_basics.md` 仅 17 条 → 评估实为「大海捞针」。

### A.5 复现命令

```bash
# 指标对排名 / k 的敏感性
python - <<'PY'
import json
from src.config import QUESTIONS_PATH
from src.retrieve import load_index, retrieve
recs = load_index()["records"]; qs = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
def score(pairs):
    hit=checks=hits=0
    for item, res in pairs:
        ev="\n".join(r["text"].lower() for r in res); kw=[k.lower() for k in item["expected_keywords"]]
        hit+=int(any(k in ev for k in kw)); checks+=len(kw); hits+=sum(k in ev for k in kw)
    return hit/len(pairs), hits/checks
for k in (1,4,10,40):
    print(k, score([(q, retrieve(q["question"], mode="bm25", k=k, records=recs)) for q in qs]))
print("倒序", score([(q, list(reversed(retrieve(q["question"], mode="bm25", k=4, records=recs)))) for q in qs]))
PY

# Wilson 95% CI（10 题时的统计噪声）
python - <<'PY'
def wilson(k, n, z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    m=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/d
    return max(0,c-m), min(1,c+m)
for name,k in (("vector 8/10",8),("hybrid 9/10",9)): print(name, ["%.0f%%"%(x*100) for x in wilson(k,10)])
PY
```

## 附录 B：`jq` 常用查询

```bash
# 所有出现空检索的题
jq -c 'select(.event=="retrieve" and .empty==true) | {qid, query}' logs/*/*.jsonl

# 引用越界的答案
jq -c 'select(.event=="generate" and .cited_valid==false) | {qid, cited, k}' logs/*/*.jsonl

# 各模式的检索延迟分布
jq -r 'select(.event=="retrieve") | "\(.mode) \(.latency_ms)"' logs/*/*.jsonl | sort -k1,1

# 答案里零引用的题（幻觉信号）
jq -c 'select(.event=="generate" and .citations_missing==true) | {qid, answer}' logs/*/*.jsonl
```

## 附录 C：字段命名约定

- 时间：`ts` / `latency_ms` / `started_at`（ISO 8601 带时区）；
- 标识：`run_id` / `trace_id` / `span_id` / `parent_id` / `qid`；
- 布尔：`is_`/`has_` 前缀非必需，但语义必须能在字段名里看出（`empty`、`citations_missing`、`text_truncated`）；
- 金额/比例不用百分数字符串，统一存 0–1 的浮点（渲染时再乘 100）；
- 任何启发式判定必须带 `*_heuristic: true` 标记。
