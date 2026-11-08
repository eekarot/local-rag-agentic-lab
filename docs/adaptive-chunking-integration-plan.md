# 切片体检与结构保护：整合 `ekimetrics/adaptive-chunking` 的方案

> 版本：v1 · 2026-11-06 · 配套设计文档 v2.37
>
> 上游：<https://github.com/ekimetrics/adaptive-chunking>（LREC 2026，MIT，401 stars）
> 本地阅读副本：`$TEMP/adaptive-chunking`（只读，不入库）
>
> **本方案只管两件事**：① 给现有切片加护栏（不改默认参数）② 让「切片质量」在没有金标时也能被体检。
> 不包含、也不建议做：逐文档自适应选择、语义切分、coref 指标。
>
> 验收一律写成「命令 + 期望值」。任何一项做不到期望值，结论就是**不采纳**，不是「再调调」。

## 1. 结论先行

| 判断 | 内容 |
|---|---|
| **能不能整合** | 能，但**借思想不借代码**。上游核心依赖（`sentence-transformers`/torch、`maverick-coref`）与本仓库 B08（只用 onnxruntime）、D39（16 GB 预算）冲突 |
| **最值钱的三样** | ① **无缝隙覆盖**不变量（我们真有洞）② **无金标体检**（SC + BI）③ 一条**负面证据**（语义切分在最严的 SC 上崩到 48.1） |
| **立刻做（1.5 天，不碰默认参数）** | P0 冻结现状 → P1 三条护栏用例 + 两个指标 → P2 体检命令 |
| **要做先扩题** | P3 结构桶对照实验（现在 2.21 题/页，按文档选方案选的是噪声） |
| **不做** | 逐文档自适应选择 / 语义切分 / RC 指标（CC BY-NC-SA）/ LLM regex 切分 / 整包引入 |
| **意外收获** | 上游 parsing 层的 **Docling（MIT）** 可能一次解掉 **D7**（PyMuPDF AGPL）与 **N9**（PDF 偏移全 NULL） |

## 2. 上游仓库速览（可核对的事实）

| 项 | 事实 | 出处 |
|---|---|---|
| 它是什么 | **选切分方法的框架**，不是一种切分算法：N 种方法 × 5 个无金标指标 → 逐文档取加权均值最大者 | `README.md`、`paper/analysis.py:294 find_best_method` |
| 选择器 | 权重**全 0.2**、加权均值、`idxmax`、NaN 指标自动跳过 | `paper/analysis.py:294`；`paper/replicate.py:92 WEIGHTS` |
| 五个指标 | SC 尺寸合规 `metrics.py:32` / ICC 块内聚 `:69` / DCC 上下文一致性 `:166` / BI 结构完整 `:280` / RC 引用完整 `:534` | `metrics.py` |
| 结构不变量 | `check_chunk_gaps`（`postprocessing.py:66`）、`repair_gaps_between_chunks`（`:128`）——**保证不丢字** | `postprocessing.py` |
| 切分参数面 | `merging ∈ {to_chunk_size, small_only}`、`merging_order ∈ {forward, backward}`、`max_tokens_strategy ∈ {chunk_size, chunk_size_plus_overlap}`、`attach_separator_to` | `splitters.py:21` |
| 实验规模 | **33 篇文档 × 3 领域 ≈ 1.18M token，全英文 PDF**；99 条查询 | `README.md` |
| 许可证 | 包 MIT；`[coref]` extra = **CC BY-NC-SA 4.0（非商业）**；`[parsing]` 里 `pymupdf4llm` = AGPL-3.0，`docling` = **MIT** | `NOTICE`（实查）、PyPI `docling.license_expression=MIT` |
| 用例数 | 43（含 `test_covers_original_text` / `test_no_gaps` / `test_repair_covers_full_text` / `test_chunk_size_respected`） | `tests/` |

论文数字（**全英文语料 + 另一套检索链路，只能当方法参考**）：

| 方法 | SC | ICC | DCC | BI | RC | 均值 |
|---|---:|---:|---:|---:|---:|---:|
| 自适应 | 99.0 | 68.2 | 88.8 | 99.4 | 99.9 | **91.07** |
| LLM regex | 98.0 | **70.9** | 82.4 | 98.1 | 99.6 | 89.80 |
| LangChain recursive | 96.1 | 65.6 | 88.8 | 95.0 | 97.7 | 88.62 |
| **Semantic** | **48.1** | 69.3 | **76.3** | 91.3 | **48.1** | 76.49 |

> **两条反常识，值得记进决策**：
> ① **语义切分在 SC/RC 上崩到 48.1** —— 它不是「更高级」，它是最差的一档。本仓库 `quality_report.py` 把 E 方案标 `not_implemented` 是对的，现在有外部证据。
> ② 自适应相对最好单方法的固有分只高 **1.27**（91.07 vs 89.80），但检索端高 **9.6** 分（67.7 vs 58.1）。**收益主要来自「避免最差」，不是「每次都选最对」。**

## 3. 与本地系统的逐项对照

本地现状：Parent(1500/2000, min 200) / Child(400/500, overlap 50) 两层；block 偏移齐全；
`quality_report.py` 已有 7 项自动指标（sentence_cut / table_intact / code_intact / heading_link /
overlap_in_range / duplicate / parent_section）。

| 上游能力 | 本地现状 | 判断 |
|---|---|---|
| **无缝隙覆盖**（不丢字） | `tests/test_parent_builder.py:289 test_children_cover_every_parent` **只断言「parent 有 child」，不断言内容连续** | **有真缺口 → P1.1** |
| **overlap 后是否超上限** 显式二选一 | 只有 `child_overlap_tokens < child_max_tokens` 校验，「含 overlap 的 child ≤ child_max」**没有用例也没有文档** | **语义未定义 → P1.2** |
| **SC 尺寸合规比例** | 只有 `tokens_p50/p90/max`，**没有「落在区间内的比例」** | 缺一个数 → P1.3 |
| **BI 结构完整** | **已有更细的版本**（sentence_cut / table_intact / code_intact），但没有「单文档单一分 + 阈值」 | 复用，不新造 → P1.4 + P2 |
| **ICC / DCC** | 无 | 需换实现（HTTP embedder），且**跨模型绝对值不可比** → 暂缓（见 §4-B3） |
| **RC 引用完整** | 无 | **不做**：NC-SA 许可证 + 只支持英文代词 |
| **`titles_context`** | **已有**：`build_embedding_text(title, section_path, content)` | 打平，不抄 |
| **`chunk_pages`** | N9：PDF `source_start/end` 全 NULL | 更大缺口 → P5（Docling） |
| **逐文档自适应选择** | 无 | **不做**（金标太薄，见 §4-C4） |
| **体检脚本给用户用** | `quality_report.py` 是开发脚本（跑 A–D 五变体做对照） | 提成命令 → P2 |

本地实测数据（这次为写方案实拉的，作为选型依据）：

| 事实 | 数字 | 来源 |
|---|---|---|
| 基准库文档 / 父块 / 子块 | 883 / 7,108 / 9,795 | `storage/bench/documents.sqlite3` |
| **有代码块的文档** | **590 篇（66.8%）**、含表格的 87 篇（9.9%）、纯散文 206 篇（23.3%） | 同上（统计围栏代码块标记与表格行正则） |
| 正样本 / 覆盖页面 / 每题每页 | 95 / **43** / **2.21** | `datasets/k8s-zh-en-v1/questions.jsonl` |
| 单题页面数 | **14 页只有 1 题** | 同上 |

> 最后两行是「不做逐文档自适应」的直接依据：2.21 题/页，选出来的是噪声。

## 4. 采纳清单

### A. 立刻做（护栏，不改默认参数，合计约 1.5 天）

| 编号 | 内容 | 为什么 |
|---|---|---|
| **A1** | **无缝隙覆盖不变量**：每个 parent 的 `[source_start, source_end)` 必须被 children 的区间并集连续覆盖，缺口 0 | 现状用例只查「有没有 child」。child 之间少一个字符，答案就会在切断处说不完整，而现有用例抓不到 |
| **A2** | **overlap 上限语义钉死**：二选一并写进 docstring + 用例（推荐 `chunk_size_plus_overlap`，即「含 overlap ≤ 550」），并在 `docs/` 写明 | 改一个数就会静默超限，然后检索质量漂移没人知道 |
| **A3** | **`size_compliance`**：落在 `[min,max]` 的 chunk 占比（约 10 行） | 现在只有分位数，看不出「有多少块越界」 |
| **A4** | **`block_integrity`**：结构块未被切断的比例（约 20 行，只需 block 偏移） | 已有三个细分指标，缺一个能对单篇文档下判断的总分 |

### B. 要做先做实验（有前置条件）

| 编号 | 内容 | 前置条件 | 判定 |
|---|---|---|---|
| **B1** | **切片体检命令**（`scripts/chunk_quality.py`）：逐文档 SC + BI、失败清单、阈值闸门、非零退出码 | A3 + A4 | 面向「用户索引自己的文档」——那时没有任何金标可测，这是唯一的体检手段（对应 ROADMAP F 节的门槛问题） |
| **B2** | **结构桶对照实验**：代码密集 / 表格密集 / 纯散文三桶，各测「结构保护」开关 | **题集扩到 ≥250 题、覆盖 ≥120 页**，让每个桶有 20+ 题 | 三口径 + 更好/更差/持平三计数；**只升一桶不算赢** |
| **B3** | **ICC / DCC**（换实现，走 HTTP embedder） | B2 结论显示「SC/BI 看不出差异但检索结果有差异」 | 只做同模型内相对比较；**禁止与论文绝对值比** |

### C. 不做（有硬理由）

| 编号 | 内容 | 理由 |
|---|---|---|
| **C1** | 整包当依赖 | `sentence-transformers`+torch 1–2 GB RSS，冲 B08（只用 onnxruntime）与 D39（16 GB 预算） |
| **C2** | RC（引用完整）指标 | `maverick-coref` = **CC BY-NC-SA 4.0 非商业**（与排除 MLQA 同一条纪律）；且依赖英文代词表 + `en_core_web_sm`，45 道中文证据题不可用 |
| **C3** | LLM Regex splitter | 需 OpenAI；本地语料是 Markdown，`section_path` 已是真结构，让 LLM 生成正则是降级重来 |
| **C4** | 逐文档自适应选择 | 2.21 题/页（14 页只有 1 题），选的是噪声；论文自身也只有 33 文档/99 查询 |
| **C5** | 抄它的 1100 / 600 / 100 | 上游用 `tiktoken` gpt-4o 口径；本地是 bge-m3 tokenizer，**同一个数字换口径是另一回事**，阈值必须重标 |
| **C6** | 搬它的数字做基线 | 33 篇全英文 PDF + Qwen3-Embedding + snowflake 重排 + haystack —— 与本地 bge-m3/qwen3.5:9b/自研 RRF **不同链路**；按既有纪律「评估基线只在同一检索器 + 同一题集下可比」 |

## 5. 分期计划

### P0 冻结现状（0.5 天，只读）

**目标**：先把「现在的切片长什么样」变成可核对的数字，否则改完无法判断是改好了还是改坏了。

**交付物**
- 本文档 §3 的对照表（已完成）
- `docs/evidence/chunk-quality-k8s-v1.json`：883 篇 × `{num_children, tokens_p50/p90/max, sc, bi}`

**影响面**：无（不碰代码、不碰库）

**验收**
```bash
python scripts/chunk_quality.py --dataset k8s-zh-en-v1 --metrics sc,bi --out docs/evidence/chunk-quality-k8s-v1.json
python -c "import json;d=json.load(open('docs/evidence/chunk-quality-k8s-v1.json'));print(d['summary'])"
```
→ 期望：`{'docs': 883, 'sc_mean': 0.9x, 'bi_mean': 0.9x, 'bi_below_threshold': N}`，且 N 与人工抽 5 篇核对一致（≥4 篇确认为真切断，否则指标是假的）

### P1 三条护栏 + 两个指标（1 天，纯加测试，不动默认参数）

**交付物**
- `src/rag/ingestion/quality_metrics.py`：`size_compliance()` / `block_integrity()`（纯函数，输入 chunk 列表 + block 偏移，输出 [0,1]）
- `tests/test_ingest_invariants.py`：3 个用例
  - `NoTextLossTest`：parent 区间被 children 连续覆盖，缺口 0（**新**）
  - `OverlapWithinLimitTest`：含 overlap 的 child ≤ 声明上限（**新**）
  - `ConfigSemanticsTest`：`max_tokens_strategy` 语义与文档一致
- `tests/test_quality_metrics.py`：4 个用例（区间内外、空输入、边界、单块）
- `src/rag/ingestion/parent_builder.py`：只在 docstring 里把 overlap 语义写清（**不改逻辑**）

**影响面**：不改 ID、不改默认参数、不重建库

**验收**
```bash
python -m unittest tests.test_ingest_invariants tests.test_quality_metrics
```
→ 期望：`OK`（≥7 用例）；两处实现故意改坏（少切一个字符叠加、overlap 加到 600）时用例必红

```bash
python -c "
from src.rag.ingestion.quality_metrics import block_integrity
print(block_integrity(chunks=['ab','cd'], blocks=[(0,4)], text='abcd'))"
```
→ 期望：`0.0`（块被从中间切开）

### P2 体检命令（1 天，面向用户）

**目标**：用户索引自己的文档时，**没有任何金标**也能知道「我的切片有没有结构性毛病」。这是 ROADMAP F 节「门槛」问题的正解之一。

**交付物**
- `scripts/chunk_quality.py`：`--dataset <name>` / `--dir <path>` / `--metrics sc,bi` / `--threshold 0.95` / `--json out.json` / `--limit N`
- 接进 `ingest --dry-run`：建库前打印「体检摘要 + 低于阈值的文档清单」，`--strict` 时非零退出
- `docs/troubleshooting.md`（或 README 一节）：症状「检索总是答一半」→ 先跑体检

**影响面**：只读；不建库、不改库

**验收**
```bash
python scripts/chunk_quality.py --dataset k8s-zh-en-v1 --metrics sc,bi --threshold 0.95
```
→ 期望：打印 `883 篇，SC 均值 ≥0.90，BI 均值 ≥0.90，BI<0.95 的 N 篇`，并对每篇列出「被切断的 block 行号」；`echo $?` 在 `--strict` 且超阈值时为 1

### P3 结构桶对照实验（2 天 + 扩题 1 天）

**前置**：题集扩到 **≥250 题、结论覆盖 ≥120 页**，三个结构桶各有 ≥20 题。

**交付物**
- `questions.jsonl` 扩题（走既有 `build_bench_questions.py append` 校验链，不手抄 span）
- 4 个变体 × SC/BI 的**只算不建库**对照表（先砍一半）
- 1 个变体的独立集合全量重建 + `eval_bench` 对照

**影响面**：**用带随机后缀的临时集合**（如 `bench_k8s_v1_<rand>`），**不动 `bench_k8s_v1` 与正式库**

**验收**
```bash
# 第 1 步：只算不建库，砍掉结构上明显更差的变体
python scripts/chunk_quality.py --dataset k8s-zh-en-v1 --metrics sc,bi --variants current,min400,code_guard,table_guard
```
→ 期望：输出 4 行对照，至少 2 个变体 BI 显著更差（可被直接淘汰）

```bash
# 第 2 步：只重建存活的 1 个变体，独立集合
RAG_COLLECTION=bench_k8s_v1_<rand> python scripts/ingest.py --dir datasets/k8s-zh-en-v1/docs --only md
python scripts/eval_bench.py --k 10 --collection bench_k8s_v1_<rand> --compare docs/baseline-k8s-v1.json
```
→ 期望：`hit@10 提升 ≥2 题` **且** `MRR 下降 <0.02` **且** 三计数里「更好 > 更差」；任一不满足 → 结论写「无差别」，保留现状

### P4 与既有消融合并（0.5 天，可选）

**目标**：不重复付「全量重建 2,000 s」的代价。

- `min_tokens 200→400`（ROADMAP D1/B07）与 P3 的变体重建**合并成一次**，共用同一批临时集合
- 顺手做 `merging_order=backward` 的纯消融（不设默认）

**验收**
```bash
python scripts/eval_bench.py --k 10 --collection <合并用的临时集合> --compare docs/baseline-k8s-v1.json
```
→ 期望：一张表同时回答「min_tokens 的影响」与「结构保护的影响」，且两者效应可分离（各自单变量）

### P5 支线：Docling 评估（1 天，独立价值最高）

**目标**：上游 parsing 层用 Docling（**MIT**，实查 PyPI `license_expression=MIT`）。本地 PDF 版面解析依赖 PyMuPDF（**AGPL-3.0**，D7 卡着），且 PDF 的 `source_start/end` 全 NULL（N9）。

**交付物**
- 只读试验脚本：docling 解 20 篇 PDF → 记录 `page / bbox / 表格结构 / 耗时 / 体积`
- 结论写回 `docs/RAG数据切分与跨语言检索技术方案.md`（采纳或记录不采纳理由）

**影响面**：不替换现有 parser，只做旁路评估

**验收**
```bash
python scripts/probe_docling.py --pdf-dir data/pdf_md --limit 20
```
→ 期望：每篇输出 `{pages, blocks, has_bbox: true, seconds}`；**能取到页码与坐标**（这是 N9 的前置），且不引入 AGPL 依赖（`pip list | grep -i pymupdf` 为空）

## 6. 判定口径（不许临时改）

1. 指标同时报 `Recall@k` + `hit@k` + `full@k` + `MRR`；**只报均值不算结论**，必须带「更好/更差/持平」三计数。
2. 只在「同一检索器 + 同一题集 + 同一数据集版本」下可比；换任何一个 → 标 `not_comparable`，不给 delta。
3. **不能凭切片指标改默认参数**（本仓库既有纪律）：SC/BI 只用来**筛掉明显更差的候选**，最终采纳要看检索级三口径。
4. 压测/实验一律用带随机后缀的 collection 名；`_candidates.json` / `_drafts.jsonl` 不进 git。
5. 扩题必须走 `build_bench_questions.py append`（span 由脚本取原文），不许手抄。
6. 任何「更好」的结论都要能复现：命令、集合名、基线文件三样齐全。

## 7. 风险与回滚

| 风险 | 影响 | 预防 / 回滚 |
|---|---|---|
| 体检命令被当成「质量评分」对外宣传 | 误导用户（SC/BI 只是结构合理性，不预测答对率） | 命令输出里固定带一行说明：「结构指标不预测答对率，只筛明显结构毛病」 |
| 覆盖率不变量实现错（把 overlap 当成缺口） | 用例假红，浪费半天 | 用例同时给两种输入：无 overlap（缺口 0）与有 overlap（区间必须**精确包含**，不是「不等」） |
| P3 用错集合撞正式库 | 污染 38,865 点正式库（N27 教训） | 只用 `RAG_COLLECTION` + 随机后缀；`ingest --status` 先确认集合名 |
| 扩题后基线不可比 | 新基线替代旧基线却没标注 | 新题集 → **新基线文件**（`baseline-k8s-v2.json`），旧文件永久保留 |
| Docling 体积/耗时不可接受 | 白花一天 | P5 限 20 篇只读试验，先量耗时再决定 |
| 采纳了 BI 更差但 hit 更高的变体 | 「指标打架」时容易选错 | §6.3 已定：**检索级优先**，但必须把冲突写进文档，不许只挑好看的报 |

## 8. 与既有 ROADMAP 的关系（不重复记账）

| 既有项 | 关系 |
|---|---|
| **D1 / B07**（`min_tokens` 200→400 消融） | **合并**进 P4，共用一次全量重建 |
| **C0b**（中文证据排序偏弱） | 无关（那是检索侧，本节是切片侧）；两条线各自独立 |
| **N9**（PDF 偏移 NULL） | P5 是解它的候选路径之一 |
| **D7**（PyMuPDF 商业授权） | P5 若采纳 Docling，可能直接消解 |
| **F 节**（工程化与上手门槛） | P2 的体检命令服务于「用户索引自己文档」这个场景，属 F 节的具体落地 |
| **D5**（`citation_precision`） | 切片护栏（A1）会先降低「引用指向被切断处」的失败率，但 D5 仍需单独测 |

## 9. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1 | 2026-11-06 | 首版：对照上游 `adaptive-chunking`（LREC 2026）后定 P0–P5 分期；A 类 4 项立刻做、B 类 3 项待条件、C 类 6 项不做（含许可证实查记录） |
