# 架构说明：从一份文档到一条带引用的答案

> 版本：2026-11-06 · 与 `docs/RAG数据切分与跨语言检索技术方案.md`（v2.53）配套。
> 设计文档讲「为什么这么定」，`docs/ROADMAP.md` 讲「还剩什么」，**本文件讲「由哪些部件组成、每一步吃什么吐什么、坏了在哪一步能被看见」**。
>
> 文中所有数字都是实测，出处标在句子后面（基线 JSON / 证据文件 / SQLite 现查）。

---

## 0. 全景

```
                          ┌──────────────────── 离线：入库 ────────────────────┐
 data/ 或 datasets/*/docs │                                                      │
      │                  │  ① 前置闸门   ② 发现+跳过判定   ③ 解析   ④ 切父块      │
      ▼                  │  ⑤ 切子块+overlap  ⑥ 窗口攒批→embedding  ⑦ 索引  ⑧ 收尾 │
  ┌────────┐             └───────────┬──────────────────────────┬─────────────────┘
  │ 原始文件│                         │                          │
  └────────┘                         ▼                          ▼
                          ┌────────────────────┐      ┌──────────────────────┐
                          │ SQLite Document    │      │ Qdrant collection    │
                          │ Store（正文）       │      │（子块向量 + payload）│
                          │ documents/parents/  │      │ bge-m3 1024/Cosine   │
                          │ children + 状态表   │      └──────────┬───────────┘
                          └─────────┬──────────┘                 │
                                    │                            │
                          ┌─────────▼──────────┐      ┌──────────▼───────────┐
                          │ BM25（父块正文）    │      │ ANN 检索（子块→父块）│
                          └─────────┬──────────┘      └──────────┬───────────┘
                                    └────────┬───────────────────┘
                                             ▼
                   ┌──────────────────── 在线：问答 ─────────────────────────┐
                   │ ⓪ 翻译 query_en  ① 三路召回   ② RRF 融合   ③ 取父块正文    │
  question ───────▶│ ④ 按 query 挑窗口  ⑤ 交叉编码器重排  ⑥ 语言配额   ⑦ 预算内   │
                   │ 组装父块  ⑧ 证据编号 + 弃答闸门  ⑨ 生成  ⑩ 引用体检 → 响应 │
                   └──────────────────────────┬───────────────────────────────┘
                                              ▼
                    CLI / HTTP 同步 / HTTP SSE / 浏览器聊天页 / 检索日志 JSONL
```

一句话：**离线把「文档」拆成「能检索的小块 + 能阅读的大块」，在线用「三路召回 → 融合 → 挑窗口 → 重排 → 预算内组装 → 带证据编号生成」把小块换回大块再答**。

---

## 1. 整体架构

### 1.1 为什么是这样（每条都对应一个被否掉的替代方案）

| # | 决策 | 采用 | 否掉的替代 | 原因（有证据的） |
|---|---|---|---|---|
| 1 | **父块 + 子块两层** | 子块（≤500 token，带 50–60 重叠）进向量库负责「被搜到」，父块（1500–2000 token）存正文负责「被读到」 | 单层 chunk 同时负责检索与阅读 | 单层只能二选一：切小了上下文断、切大了向量语义糊。实测父块 p50 263 token、每父块子块数 p50 1 / 均值 1.38（`storage/bench/documents.sqlite3` 现查），说明这套参数下多数父块只需 1 个子块就能覆盖，代价可控 |
| 2 | **向量库与关系库分离** | Qdrant 只放子块向量 + 15 个检索/展示必需字段；正文在 SQLite | 正文也塞进 payload | payload p50 **1,620 字节**（Qdrant 侧抽样），塞正文会到 KB 级并拖慢 ANN；而「整篇父块存进每条子块」会从 7.04 M 字符涨到 13.40 M 字符（**1.90×**） |
| 3 | **三路召回 + RRF** | `dense_zh` + `dense_en` + `lexical_en`，各取 top-k(=pool)，RRF(k=60) 融合 | 单向量路；加权分数求和 | 单向量路对「中文问、英文文档」这类跨语言格子失效；加权求和要先做分数归一化（不同模型余弦不可比）。RRF 只用名次，天然免归一化 |
| 4 | **重排前先挑窗口，不整块截断** | 父块超 256 token 时，按 query 词面挑一段 ≤256 token 的窗口送交叉编码器（`best_window()`） | 直接截前 256 token | 父块 p50 **263 token**、63% 超过 256 ⇒ 截断等于只看标题。实测「窗口里有 gold」的比例：英→英 0.8302、中→中 0.8000（`docs/evidence/retrieval-stage-ceilings-k8s-v1.json`），窗口挑错就直接丢分 |
| 5 | **语言在建库时判一次** | `apply_document_language()` 写进 `documents.language` 与 `parents.language`，并抄进每个 `Child` | 查询时按正文现判 | 现判 = 每次查询重复算；更重要的是会漏写 payload 字段（见 §2.6 盲区 1）。读取侧仍保留 `COALESCE(p.language, d.language)` + 正文兜底，这样旧库不重建也能用 |
| 6 | **两套 id 空间** | `parent_id`（`par_…`，不含内容哈希）与 `chunk_id`（`chk_…`，含 `content_hash`） | 一套 id 走到底 | 父块 id 稳定 ⇒ 卷级缓冲/去重/评估金标不会因为子块切法微调就全废；子块 id 带内容哈希 ⇒ 内容变了必须换 id，否则向量与正文对不上 |
| 7 | **引用编号由组装器分配** | `[1]…[k]` 对应本次进去的上下文块，答案里只允许引这些号 | 让模型写文档名/文件名 | 模型编不出合法编号，`parse_citations()` 能机械校验；换语料/换标题不会让引用失效 |
| 8 | **没有证据就不调模型** | 组装后为空或证据不足 → 直接回固定弃答文案（`ABSTAIN_TEXT_ZH/EN`），`timings["generate_ms"]=0` | 让模型「尽量答」 | 生成 p50 **53.5 s**（`qwen3.5:9b` CPU + 3k 上下文，N8），而没证据时答案必然是编的；先弃答省 53 s/次 |
| 9 | **有界队列 + 单飞** | 生成队列 inflight 1 / 排队 2 / 超时 120 s；检索队列 inflight 4 / 排队 2 / 超时 5 s（两条队列分开） | 无限并发；只用一个队列 | 生成 p50 53.5 s 而原 `QUEUE_TIMEOUT=20 s` ⇒ 排队者 100% 在 20.2 s 收 429（N19）。检索 63.7 ms 级（关重排）与生成差三个数量级，混在一个队列里队列参数必然对一边错一边 |
| 10 | **入库状态表按三个版本断点续建** | `is_done()` 要求 stage=`COMPLETED` 且 `document_version` / `chunker_version`（父+子）/ `embedding_version` 三者都没变 | 只看 mtime；或强制全量重建 | 换切分参数、换 embedding 模型、改正文，三种变化必须分别能触发重建；只看 mtime 会把「改了切分代码」当成没事 |
| 11 | **按窗口攒批 embedding** | 攒够 256 子块或 64 篇再调一次 `/api/embed` | 一篇一调 | 单次调用固定开销远大于批量摊薄；并且窗口 = 事务边界，中断只丢一个窗口，重跑靠 embedding 缓存（命中 0.019 ms/条）近乎免费 |
| 12 | **只追加的检索日志 + 独立标注文件** | 主日志 `logs/YYYY-MM-DD/*.jsonl` 只追加；人工标注写独立 `*.labels.jsonl` | 主日志原地更新 | 日志被改写就没法回溯「当时到底检索到什么」；标注另存才能重放与多人并行标注 |

### 1.2 离线段：每一步的输入 / 输出

以 `scripts/ingest.py` → `IngestPipeline.run()` 为主线（`src/rag/ingest/pipeline.py`）。**8 步**，每步都标了「输入 → 输出 → 落在哪 → 怎么观测」。

| 步 | 名称 | 输入 | 输出 | 落在哪 | 观测点 |
|---|---|---|---|---|---|
| ① | **前置闸门**（可选 `--precheck`） | 语料目录 + 切分参数 | 体检结论（`sc` 尺寸合规率 / `bi` block 完整性 / `gap` 覆盖空洞） | 只读，**不写库、不动状态表** | `scripts/chunk_quality.py --strict` 退出码；`--precheck-only` 保证完全不碰 Qdrant |
| ② | **发现 + 跳过判定** | `--dir` / `--patterns` / 状态表 | 待处理文件清单（相对路径排序） | `ingest_state` 表 | `--status` 看各 stage 计数；`--failures` 看失败清单；源文件消失时 `prune_missing()` 先清幽灵记录 |
| ③ | **解析** | 单个文件路径 | `list[DocumentBlock]`（9 字段：`block_type`/`text`/`heading_level`/`source_start`/`source_end`/…） | 不落库，只在内存 | 失败进 `Stage.PARSING` + `ingest_failures`；日志 `[fail] key @PARSING`。实测 3 页 PDF → **137 块**（`{heading 2, paragraph 135}`），PDF 的 block 粒度是**行** |
| ④ | **切父块** | blocks + `document_id` + `document_version` + title | `list[Parent]`（正文 ≤2000 token，句子/标题边界优先，短块合并） | `parents` 表（仅在索引步写入） | 切完为空 → `IngestionError` 进 `Stage.CHUNKING`；父块 p50 263 token |
| ⑤ | **切子块 + overlap** | parents | `list[Child]`（≤500 token，相邻重叠 50–60 token） | `children` 表 | 子块数 p50 1 / 均值 1.38；相邻重叠 p50 **59**，99.8% 落在 [50,60] |
| ⑥ | **窗口攒批 + embedding** | 子块的 `embedding_text`（父块没有该字段，子块用 `content`） | `(n, 1024)` 归一化向量矩阵 | 向量缓存 `storage/cache/embeddings.sqlite3` + 内存 | `summary.embedding_calls`；缓存命中；embedding 失败整窗标 `Stage.EMBEDDING` 失败（不停整轮） |
| ⑦ | **索引**（6 子步） | document + parents + children + 向量 | 三方一致：SQLite 事务 → Qdrant upsert → 删陈旧点 → 状态 `COMPLETED` | SQLite + Qdrant + `ingest_state` | 写入前版本对账（同维不同模型混写会静默出错，所以闸在 upsert 之前，失败时一个点都不写）；`summary.indexed` |
| ⑧ | **收尾** | — | `IngestSummary`（discovered/skipped/indexed/failed/parents/children/elapsed_s） | 返回 dict，`--json` 可直接消费 | 每篇一行 `[ok] key：8 child / 2 parent`；失败带可复制 `fix` |

**索引步的 6 个子步**（顺序不能换）：

1. **版本对账**（`_assert_versions`）：向量维度 / `embedding_model` / `embedding_version` 与 collection 不一致 → 直接失败，一个点都不写。
2. **写 Document Store**（单个事务）：这是**唯一写入点**（曾一度在编排层和索引器各写一遍，每篇白写 6.0 ms，883 篇 ≈ 5.3 s，见 §2.6 盲区 3）。
3. **算向量**（一般已在 ⑥ 算好，这里只做补齐）。
4. **分批 upsert**：point id 由 `chunk_id` 决定 ⇒ 重复导入是覆盖，不是新增。
5. **删本次不再存在的点**：文档变短/版本变化时清陈旧点。
6. **状态置 `COMPLETED`**：三个版本一起记，供下次 `is_done()` 判定。

**几个容易踩的点，已写成不变量**：

- **overlap 语义 = `chunk_size`**：`apply_overlap` 先找句子/行边界，找不到才退回 token 窗口；`_fit_front()` 兜底保证贴上重叠后仍不超过 `child_max_tokens`（丢掉的是重叠区，不是正文）。
- **PDF 的 `source_start/end` 是 `NULL`**（N9）：PDF 解析器不提供字符偏移，所以「按偏移切文本」的断言对 PDF 必须跳过。
- **切片不变量在「父块内容坐标」里断言**：`"".join(children) == parent.content`，而不是在全文坐标里，避免把父子坐标混用。

### 1.3 在线段：每一步的输入 / 输出

以 `answer_query()` / `retrieve_with_meta()` 为主线（`src/rag/ask/pipeline.py`）。**10 步**，默认参数是评估口径 `k=10, pool=40, rerank=True, query_en_mode=translate, rerank_query=question, rerank_merge=raw, lang_quota=0, max_other_lang=0, window=overlap`。

| 步 | 名称 | 输入 | 输出 | 关键参数 | 失败时 |
|---|---|---|---|---|---|
| ⓪ | **翻译** | 中文原句 | `query_en`（**只有 `ok`/`cached` 状态才算真译文**） | `TranslateService`（provider 只能由工厂造） | 失败 → `query_en=None`，英文两路回退中文原句并在 `notes` 里标「噪声源」 |
| ① | **三路召回** | `query_zh` + `query_en`，`k=pool=40` | `{路由名: RouteResult(hits, latency_ms, candidates)}` | 每路深度 = k | 单路异常只关这一路（`enabled=False` + `error`），其它路继续 |
| ② | **RRF 融合** | 三路父块 id 有序列表 | `list[FusedHit]`（带 `per_route` 名次） | `RRF_K=60` | 全空 → 下游直接弃答 |
| ③ | **取父块正文 + 找回子块 id** | 融合列表 | `list[RerankHit(parent_id, chunk_id, text)]` | `chunk_of[(route, parent_id)] → child_id`，取「贡献最好名次的那一路」 | 词面路本来没有子块 id ⇒ 退回父块 id 并计数（`stats["chunk_ids_from_parent"]`） |
| ④ | **按 query 挑窗口** | 父块正文 + query | ≤256 token 的窗口 | `window=overlap` 走 `best_window()`；`window=tiles` 走 `tile_windows()` 全格重排 | 挑不到 → 退回块首（会掉分但不出错） |
| ⑤ | **交叉编码器重排** | 窗口文本 + query | `RerankHit.score/rerank_score/rank`，截到 `pool` | ONNX `mmarco-mMiniLMv2-L12-H384`，`max_length=256`、`batch=8`、并发 1 | `fallback_on_error=True`：重排挂了就用 RRF 原序（而不是报错） |
| ⑥ | **语言配额**（默认关） | 重排结果 + 父块语言 | 按 `lang_quota` / `max_other_lang` 重选 top-k | 默认 0/0 | 配额不足时用剩余位补，不返回空 |
| ⑦ | **组装上下文** | top-k 命中 + 父块正文 | `AssembledContext`（blocks / tokens / 去重与邻近合并结果） | 预算 **6000 token**；近重复 Jaccard 去重；相邻块拼接缝 ≤400 字符合并 | 全被去重掉 → `abstained` |
| ⑧ | **提示词 + 弃答闸门** | 组装结果 + 问题 | `PromptBundle(messages, language, abstained, reason)` | `should_abstain()` 在**调模型之前**判定 | 弃答 → 直接回固定文案，`llm_called=False` |
| ⑨ | **生成** | messages | 答案文本 | `qwen3.5:9b`，`temperature=0.0`、`think=False`；生成队列 inflight 1 + 排队 2 + 超时 120 s | 熔断打开 → `CircuitOpenError`（不再等 53 s）；降级信息进响应 |
| ⑩ | **引用体检** | 答案 + blocks | `CitationReport`（引用号 / malformed / 越界 / 是否弃答） | 引用号必须 ≤ 本次块数 | 越界算 `invalid`，`invalid_citation_rate` 的分母含 malformed |

**RRF 融合公式**（`src/rag/retrieval/fusion.py`）：

```
score(parent) = Σ_routes 1 / (60 + rank_route(parent))
```

只用名次，所以**不需要跨模型归一化分数** —— 这是「禁止绝对余弦跨模型比较」在架构上的落地。

**语言配额的实测定位**（`docs/evidence/c0e-language-quota-k8s-v1.json`）：给它加名额反而掉分，所以默认 0/0；跨语言的差距不在「名额」而在召回与排序（§2.5）。

### 1.4 三份落库、两套 id、一次删除

| 落库 | 内容 | 关键字段 | 谁写 |
|---|---|---|---|
| **SQLite Document Store** | 正文与元数据 | `documents`(11) / `parents`(12, 含 `content`) / `children`(10, **不存 content**) | 只有 `Indexer.index_document()` 一处 |
| **Qdrant** | 子块向量 + 检索必需 payload | 15 字段（`chunk_id`/`parent_id`/`document_id`/`language`/`title`/`section_path`/`source_start`/`source_end`/`content_hash`/…）+ 6 个 payload 索引 | 索引器分批 upsert |
| **状态表 `ingest_state`** | 断点续建与失败清单 | `doc_key`/`stage`/三个版本/失败原因 | 编排器逐步 `advance()` |

- **`gold_chunks` 与 `gold_points` 是两个不相交的 id 空间**：前者是 `chunk_id`（子块），后者是 Qdrant point id（`UUID5(chunk_id)`）。评估代码不许混用。
- **删除文档 = 三方一致**：Qdrant 点 + Document Store 行（`ON DELETE CASCADE`）+ 状态表记录，缺一个就会「库里没有、状态说 COMPLETED、重跑被跳过」。

### 1.5 服务层

| 入口 | 形态 | 检索/生成 | 备注 |
|---|---|---|---|
| `python -m src.ask` | CLI | 同一条链（`answer_query`） | `--retrieve-only` 只检索不生成 |
| `POST /api/v1/ask` | 同步 JSON | 走生成队列 | 429 带 `Retry-After` |
| `GET /api/v1/ask/stream` | SSE | 同一条链，分阶段推事件 | 心跳 15 s，请求超时 300 s |
| `GET /api/v1/retrieve` | 同步 JSON | **不占生成名额**（独立队列 inflight 4） | 返回三路明细 + 融合结果 |
| `GET /health` | **永远 200** | 不检索、不生成 | 六项必查探针 + `data/` 信息项；`status` ∈ `ok`/`degraded`/`down` |
| 浏览器聊天页 | `/` | 同上 | 无前端构建步骤 |

**降级与熔断**：`degrade.py` 的 `Breaker`（失败即打开、TTL 30 s）套在 embedding 调用上；打开期间 `CircuitOpenError` 立即返回，不再等模型超时。降级信息字段刻意与 `probes.Check` 对齐（`capability` + `fix`），所以 `/health` 和响应里的「怎么修」是同一份文案。

**日志**：一次请求一条 trace，内部按阶段记 `retrieve` / `prompt` / `generate` / `trace` 事件；级别 `off|summary|detail|debug`，默认 `detail`（**记 `hits[].text` 全文是已定案决策**）；超 8000 字符的文本外置到 `artifacts/`，主日志只留引用与 sha256。

---

## 2. 测试、基准、以及每一步怎么被监测

### 2.1 测试分层

| 层 | 规模 | 门槛 | 谁能跑 |
|---|---|---|---|
| **默认套件** | **1,246 用例，`skipped=50`**，本机 65–77 s | **不联网**：`QDRANT_URL`/`OLLAMA_BASE_URL` 指向死端口仍全绿 | `python -m unittest discover -s tests` |
| **慢用例** | 需要 Qdrant + Ollama + 17 MB 词表 + 449 MB ONNX | `@slow` 标记，`RAG_RUN_SLOW=1` 才跑 | 本机 |
| **CI** | 同一份默认套件 × 3 平台 | ubuntu / windows / macos，py3.12，**期望 <60 s** | push 到 main 或 PR（不跑慢用例、不装 PDF 依赖） |

三个「防止门槛被无声破坏」的守卫：

- **`tests/test_suite_layering.py`**：用 AST 查**导入期**有没有在探真实服务（`make_client` / `embed_texts` / `Tokenizer` …）。函数体内的调用不管 —— 那些才是假件该覆盖的。同时维护 `SLOW_MODULES` 名单，防止有人悄悄删 `@slow`。
- **`tests/test_dev_env_manifest.py`**：`.env.example` 与代码里的环境变量必须**集合相等**、默认值不漂移（只认 `getenv(` / `environ.get(` 字面量 ⇒ 读环境变量必须写字面量形式）。
- **`tests/test_legal_and_ci.py`**：扫**被 git 跟踪**的文件，禁止本机绝对路径 / 用户名 / 个人邮箱 ⇒ 新文件必须先 `git add` 再跑套件。

### 2.2 两套基准（一个自建、一个对外可比）

| 基准 | 规模 | 为什么需要它 | 当前数字 |
|---|---|---|---|
| **k8s-zh-en-v1**（自建） | 883 篇 / 51 页 / **180 题**（175 正 + 5 负），中英对照 | 跨语言是主战场，公开基准没有「中文问、英文答」的格子 | hit@10 **0.8571**、full@10 **0.8514**、recall@10 **0.8533**、MRR **0.5316**，p50 1668.2 ms / p90 1935.8 ms（`docs/baseline-k8s-v1.json`） |
| **BEIR–SciFact**（导入型） | 5,183 摘要 / 1,109 claim，评测 300 题 | 对外可比：能不靠自建题集证明「不是自己给自己出题」 | nDCG@10 **0.6843**、R@10 0.7906、hit@10 0.8067、MRR@10 0.6563，p50 1658.1 ms（`docs/evidence/beir-scifact-v1-baseline.json`）；BEIR 论文 BM25 参考 **0.665** |

**SciFact 的口径特别之处**（N55）：必须先「父块 → 文档」去重收敛再算 nDCG，否则同一篇文档的多个父块会重复计分，数字虚高。

**指标口径（不允许四选一抽奖）**：同时报 `Recall@k` + `hit@k` + `full@k`；`MRR` 的分母是**全部正样本**（未命中记 0，且只算 top-k 内命中）；负样本不进 recall 分母，只判「是否弃答」。

### 2.3 门禁：怎么判定「变好了」而不是「抖了一下」

| 机制 | 规则 | 实现 |
|---|---|---|
| **可比性指纹** | `k` / `pool` / `rerank` / `query_en_mode` / `rerank_query_mode` / `lang_quota` / `rerank_merge` / `max_other_lang` / `window` + `questions_sha256` 任一不同 → `not_comparable` | `scripts/eval_bench.COMPARABILITY_KEYS` |
| **`not_comparable` 不返回非零退出** | 不可比是「结论无效」，不是「构建失败」，不能卡住流水线 | 退出码 0 + 明确原因 |
| **容差** | 基线一侧 95% bootstrap CI 半宽（当前 hit/full ±0.0514、recall ±0.0505、MRR ±0.0588） | `bootstrap_ci()` |
| **逐题计数** | 结论必须同时报「均值 delta」与「更好 / 更差 / 持平」计数 | `--compare` 输出 + 逐题 rows JSONL |
| **A/A 噪声底** | 同一份代码重跑可差 **3 好 / 5 差 / 167 平 ≈ ±0.6 点 hit** ⇒ 小于这个幅度的「提升」不算提升 | 每轮新实验先做 A/A |
| **题集变了必须重立基线** | 扩题 144→180 后 `questions_sha256` 变 ⇒ 老基线一律作废 | 指纹里的 `questions_sha256` |
| **README 数字必须与基线一致** | 6 条守卫逐位比对 | `tests/test_bench_questions.py` |

### 2.4 每一步的监测点（这一节是「怎么知道它是好的」）

| 阶段 | 监测手段 | 看什么 | 报警/门禁形式 |
|---|---|---|---|
| **解析** | `scripts/chunk_quality.py --metrics sc,bi,gap` | 尺寸合规率、block 完整性（丢字）、覆盖空洞 | `--strict` 退出码 1 |
| **切分** | 同一脚本 + 不变量用例 | `"".join(children)==parent.content`、overlap 落在 [50,60]、子块不超 500 token | 单测（`test_ingest_invariants.py`） |
| **embedding** | `summary.embedding_calls` / 缓存命中 / 范数 | 向量 L2 范数 = 1.0000、维度 1024 | `_assert_versions` 写入前对账 |
| **入库** | `ingest_state` 状态机 | 每个 `doc_key` 的 stage、三版本、失败原因 | `--status` / `--failures` / `--retry-failed` |
| **入库正确性** | Document Store 与 Qdrant 抽样对账 | 行数、payload 字段完整性（曾发现 `language` 恒空） | `scripts/backfill_payload_language.py --dry-run`、探针 |
| **检索** | `raglog` 的 `retrieve` 事件 | 每路 `latency_ms` / `candidates` / `query_used` / `notes`、融合后 `n_fused`、分数 gap | `logs/*.jsonl` → `raglog_report.py` |
| **重排** | `reranker.stats`（调用次数、批数、失败回退次数） | 回退到 RRF 原序的次数必须为 0 | 单元用例 + 探针 `probe_reranker` |
| **组装** | `AssembledContext`（tokens / 去重数 / 邻近合并数） | token 用量是否接近预算、近重复率 | `context_tokens` 进 AskResult |
| **生成** | `generate` 事件 | `latency_ms` / `cited` / `cited_valid` / `abstained`（启发式）/ `finish_reason` | 日志审查 |
| **服务** | `/health` 六探针 + `/healthz` | `status` ∈ ok/degraded/down、`blocking_fixes`、队列快照 | 探针自身异常也降级成一条失败项 |
| **端到端质量** | `eval_bench` / `eval_beir` / `eval_answer` | hit/full/recall/MRR/nDCG + 延迟 p50/p90（**必须先 warmup**） | `--compare` 门禁 |
| **答案层** | `src/eval_answer.py` + 引用体检 | `citation_precision` / `invalid_citation_rate` / 弃答率 | 人工标注（`faithfulness` **不能自动**） |

### 2.5 现在量到的阶段天花板（说明「下一步该动哪一段」）

`docs/evidence/retrieval-stage-ceilings-k8s-v1.json`（k_pool=40、k_deep=200）：

| 格子 | n | 最终 hit@10 | 某一路 pool@40 里含 gold | 深挖到 200 含 gold | 融合后 @10 | 窗口里有 gold |
|---|---|---|---|---|---|---|
| 英→英 | 54 | 0.9630 | 0.9815 | 1.0 | 0.9259 | 0.8302 |
| 中→中 | 54 | 0.7963 | 0.9259 | 0.9815 | 0.7407 | 0.8000 |
| 英→中 | 9 | 0.5556 | 0.8889 | 1.0 | 0.7778 | 0.4286 |
| 中→英 | 22 | 0.7273 | 0.9545 | 0.9545 | 0.7727 | 0.9048 |

读法：**召回上限（0.93–0.98）远高于最终 hit@10**，说明丢分主要发生在「融合 → 挑窗口 → 重排 → 截断」这几段，而不是「没搜到」。关掉重排的对照也印证了这点：

| 配置 | hit@10 | MRR | p50 |
|---|---|---|---|
| 关重排（RRF 原序） | 0.7789 | 0.5456 | **63.7 ms** |
| 重排 pool=20 | 0.8000 | 0.5770 | 604.3 ms |
| **重排 pool=40（当前默认）** | **0.8632** | **0.6075** | 1192.1 ms |
| 重排 pool=60 | 0.8737 | 0.6116 | 1775.2 ms |

（`docs/evidence/c0b-rerank-pool-ablation.json`，95 正样本那一轮）⇒ **重排是最大单项收益（+8.4 点 hit），也是最贵的一段**；pool 40→60 只再涨 1 个点却多 49% 延迟，所以默认停在 40。

**各段延迟占比**（`docs/evidence/query-en-modes-k8s-v1.json` 与基线）：

| 段 | p50 | 说明 |
|---|---|---|
| 三路召回 + RRF（关重排） | ~64 ms | 检索本身很快 |
| 翻译（`--query-en translate`） | 1200.0 → 1564.7 ms | 翻译占掉约 365 ms，同时换来跨语言格子得分（所以默认开） |
| 重排（pool=40） | ~1192 ms | **最贵的一段** |
| 端到端检索+重排 | 1668.2 ms | 基线口径 |
| 生成 | **53.5 s** | `qwen3.5:9b` CPU + 3k 上下文（N8），所以「弃答省 53 s」是产品级收益 |

### 2.6 已知盲区（这几条都是「监测没覆盖到」才被漏掉的）

| # | 漏在哪 | 后果 | 现在怎么防 |
|---|---|---|---|
| 1 | **payload 字段从没被读过**：`language` 的 payload 索引建了，但 `Child.language` 因赋值时机（先建子块后判语言）恒为 `None`，`payload_from_child()` 又把 `None` 整个剔除 | Qdrant 里从来没有 `language`，按语言过滤恒为 0 命中且不报错 | 行为用例 + **行序断言**（`assertIn` 索引位置比较）；回填脚本 `backfill_payload_language.py`（只补缺失、`--dry-run` 零写入、幂等） |
| 2 | **`chunk_ids` 装的是 `parent_id`**：融合后只保留父块 id，三处组装入口直接把 `parent_id` 填进 `Context.chunk_ids` | 日志/评估里「命中了哪个子块」全是错的（可回查 0/20） | `chunk_of[(route, parent_id)] → child_id` + `stats["chunk_ids_from_parent"]` 计数；`tests/test_chunk_ids_plumbing.py`(8 例) |
| 3 | **同一个文档写两遍 Store**：编排层与索引器各写一次 | 每篇白写 6.0 ms（883 篇 ≈ 5.3 s），且两次写入之间失败会出现 Store 与 Qdrant 版本不一致而状态照样 `COMPLETED` | 唯一写入点留在索引器 + `test_store_written_exactly_once_per_document` |
| 4 | **「字段存在 ≠ 被写过」** | 建库时把 `"language": None` 写死，列存在但 100% NULL | 体检类脚本必须抽样**读值**，不能只查列是否存在 |
| 5 | **引用精度没人盯** | `citation_precision` 仅 **38.46%**（D5） | 引用体检进 `AskResult.report`；这一项仍未解决，见 §3 |

---

## 3. 未来优化方向

按「已有证据 + 收益/成本」排序。**每一档都写明为什么不是现在做**。

### 第一档：证据已足，可以直接动手

| # | 方向 | 现状证据 | 预期收益 | 为什么能做 |
|---|---|---|---|---|
| 1 | **引用精度**（D5） | `citation_precision` **38.46%** —— 引用错比不给引用更糟 | 直接决定可用性，比任何新功能都值钱 | 已有 `parse_citations()` + 引用体检 + 组装器的块编号，缺的是「生成后就地校验并在提示词里回喂」 |
| 2 | **PDF 标题兜底**（ROADMAP S7） | 3 页 PDF 实测：137 块首块是 h2，`_guess_title()` 只认 h1 ⇒ title 变成文件名 | 引用展示、`payload.title` 立刻正确 | 不动 id、不动向量（子块向量只认 `content`）⇒ 半天内可验证 |
| 3 | **bench 库语言列**（ROADMAP S6） | `documents.language` 883 篇 / `parents.language` 7,108 行**全 NULL**，全靠查询时现判 | 让 `_block_lang()` 从「每次现判」变成「读库」，也让 `backfill_payload_language.py` 不再全跳过 | 不改 `chunk_id` ⇒ `gold_points` 不失效；但会改变「重排选哪段 query」⇒ **必须先 `--compare`**（N59 教训） |
| 4 | **C0f2：短名单换窗口口径** | 全量 tiles 重排 hit 0.8571→0.8629 但延迟 ×1.84（3070 ms）；收益落在 A/A 噪声底内 | 只对「overlap 前 10」再 tiles 重排，尝试「整体 ≥0.8629 且 p50 ≤2100 ms」 | 已有 `tile_windows()` / `rerank_tiles()` 全套实现，只改调用范围 |

### 第二档：需要新证据才能定

| # | 方向 | 缺什么证据 | 怎么取 |
|---|---|---|---|
| 5 | **C0g：补齐 en→zh 翻译** | 现在 `dense_zh`/`dense_en` 在「英文问」时可能吃同一份列表（三路退化成两路） | 先数「两种 query 在每路的 `query_used` 是否逐字相同」；再 A/B |
| 6 | **B07 `min_tokens` 200→400** | 只有 chunking 指标，没有检索级证据 | 随机后缀 collection + 全量 ingest + 复跑消融（会改 `parent_index` ⇒ gold 全废，必须整轮重建） |
| 7 | **D36 ANN Recall@K / D37 HNSW 参数** | 只有单点 0.98 观测，没有 `ef_search` 扫描表 | 扫表脚本：精度 / 内存 / 延迟三角 |
| 8 | **D38 百万级压测** | 当前最大 9,795 点，规模效应未知 | 生成 100 万点 → ingest 耗时、RSS、p50/p95，再反推 D39 的 `--workers` 上限 |
| 9 | **SciFact 上的窗口/切分口径**（S3） | 摘要 p50 1,331 字符本来贴着 256 token 窗口，可能更敏感 | 同一套 `--window` A/B，注意 nDCG 要先去重收敛 |

### 第三档：结构性方向（不是调参，是改形状）

| # | 方向 | 为什么它是结构性的 |
|---|---|---|
| 10 | **重排这一段太贵**（1192 ms / 占比 71%） | 单个收益最大（+8.4 点 hit）也是最贵的一段。可走的路线：交叉编码器蒸馏小模型、两阶段（小模型粗排 40→10 再大模型精排）、把窗口选择从「词面挑」换成「用重排分数挑」 |
| 11 | **生成 53.5 s 是产品天花板** | CPU 上 9B + 3k 上下文。要么减上下文（预算 6000 → 按需）、要么换更小模型、要么把「没证据先弃答」的覆盖再往前推（现在是组装为空才弃） |
| 12 | **PDF 走旁路解析器（Docling）只补缺口** | 实测 5.3 s/页、峰值 RSS p50 1.5 GB ⇒ **不许全量迁移**。门禁要按文档类型分档：扫描件/复杂表格才走，文本 PDF 继续用 MuPDF（S4/S5） |
| 13 | **切片优化必须走同一受保护入口** | 优化路径绕过 `/ask` 主入口，就等于用另一条链自证有效；所有实验脚本必须调 `retrieve_with_meta()` |
| 14 | **数据集只增不改 / 金标是可失效资产** | 任何「按偏移切文本」的改动都会让 `gold_points` 失效 ⇒ 每次先想「这条改动要不要重建库、要不要重立基线」 |

### 明确不做（已经量过、别再试一遍）

| 已否掉的 | 数字 | 结论 |
|---|---|---|
| 语言配额调节 | 加名额反而掉分（`c0e-language-quota-k8s-v1.json`） | 跨语言差距不在名额 |
| 让交叉编码器自己挑窗口（全量 tiles） | 0.8571→0.8629，落在 A/A 噪声底内，延迟 ×1.84 | 未达标，不翻默认 |
| 窗口预算精确卡到 256 token | hit 0.8571→**0.8286（−2.85 点）** | 「看起来更正确的收紧」必须先 `--compare` |
| 重排 pool 60 | +1.0 点 hit，延迟 +49% | 默认留 40 |
| SciFact 上关 BM25 路 | 300/300 题检索列表**逐位相同** | 词面路在这一数据集上零贡献（但没证据说它没用，所以保留） |

---

## 附：代码地图

| 文件 | 职责 |
|---|---|
| `src/rag/ingestion/parser/` | 解析器与注册表（`.md`/`.txt`/`.pdf` → `DocumentBlock`） |
| `src/rag/ingestion/parent_builder.py` | 父块/子块切分、overlap、`build_embedding_text()` |
| `src/rag/ingestion/splitter.py` `overlap.py` | 递归切分与重叠（句子/行边界优先） |
| `src/rag/ingest/pipeline.py` | 入库编排（8 步 + 窗口攒批 + 状态机） |
| `src/rag/ingest/state.py` | `ingest_state` 状态机、三版本 `is_done()`、失败清单 |
| `src/rag/store/document_store.py` | SQLite 三表 + `load_parent_records()` |
| `src/rag/store/qdrant_store.py` | collection 配置、payload 字段、payload 索引 |
| `src/rag/store/indexer.py` | 索引 6 子步、版本对账、陈旧点清理 |
| `src/rag/store/lexical.py` | 英文 BM25（父块正文） |
| `src/rag/retrieval/routes.py` `orchestrator.py` `fusion.py` | 三路召回 + RRF + 单路失败隔离 |
| `src/rag/retrieval/rerank.py` | ONNX 交叉编码器、窗口挑选、tiles 口径 |
| `src/rag/retrieval/degrade.py` | 熔断、能力缺失、降级文案 |
| `src/rag/ask/pipeline.py` | 在线主链（`retrieve_with_meta` / `answer_query` / 流式） |
| `src/rag/context/assembler.py` | 预算内组装、去重、邻近合并、块编号 |
| `src/rag/answer/prompt.py` `citations.py` | 提示词与弃答闸门、引用解析 |
| `src/rag/api/app.py` `queue.py` `streaming.py` | HTTP/SSE、有界队列、`/health` |
| `src/rag/ops/probes.py` `doctor.py` | 六探针 + 可复制 `fix` |
| `src/raglog.py` `raglog_report.py` | 检索日志（JSONL 单文件、只追加）与报告 |
| `scripts/ingest.py` `chunk_quality.py` | 入库 CLI（含前置闸门）与切分体检 |
| `scripts/eval_bench.py` `eval_beir.py` | 两套基准跑批 + `--compare` 门禁 |
| `docs/baseline-k8s-v1.json` `docs/evidence/` | 唯一有效的数字，以及每次实验的原始证据 |
