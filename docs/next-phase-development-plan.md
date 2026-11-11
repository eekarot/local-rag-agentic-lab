# 下一阶段开发计划：从本地 RAG 实验链到可管理知识库平台

> 状态：Draft 1  
> 制定日期：2026-11-11  
> 计划周期：约 6–8 周（单人全职估算）  
> 适用基线：`v0.1.0` 之后  
> 上位原则：每个任务必须有可执行验收命令和明确期望值；算法改动必须与当前基线做配对比较。

## 1. 阶段目标

下一阶段不以继续微调单个检索参数为主，而是把现有“解析 → 切片 → 检索 → 回答”的实验链升级为可持续运行的知识库平台：

1. PDF、Word、Excel、Markdown、TXT 等文档可通过统一入口解析；
2. 文本、表格、图片、页码、坐标等信息进入统一中间结构；
3. 文档具备版本历史、删除同步、精确去重和近重复识别能力；
4. 切片策略可配置、可预览、可评估、可回滚；
5. 评估覆盖解析、切片、检索、Context、答案和系统性能；
6. 对外提供文档管理、异步摄取、检索和问答接口；
7. Prompt 与 Context 组装具备版本记录、注入防护和配对评估能力。

## 2. 非目标

本阶段暂不做：

- 不把“逐文档自动选择切片算法”设为默认能力；金标覆盖不足时容易过拟合；
- 不在解析第一版同时引入视觉向量库；图片先走资产存储、OCR、caption 和文本检索；
- 不直接替换当前生产解析器；新解析器先旁路评估，再通过门禁切换；
- 不在文档生命周期未完成前开放多租户公网部署；
- 不因 SC/BI 等结构指标变好就修改默认切片参数；最终采纳仍以检索和答案层评估为准；
- 不优先投入 HNSW、量化、百万级压测，除非现有规模已经出现明确性能瓶颈。

## 3. 总体里程碑

| 里程碑 | 目标 | 预计工期 | 退出条件 |
|---|---|---:|---|
| M0 | 修复存储与在线视图一致性 | 3–4 天 | 删除、更新、失败恢复后四本账一致 |
| M1 | Document IR v2 + 多格式解析旁路 | 8–10 天 | MD/TXT/PDF/DOCX/XLSX 均输出统一 IR，图片有资产引用 |
| M2 | 文档版本、去重与生命周期管理 | 6–8 天 | 可查历史、回滚、去重、删除、对账 |
| M3 | 切片策略插件化 + L0/L2/L3 评估 | 7–9 天 | 可选择策略并生成配对评估报告 |
| M4 | 文档管理 API + 异步任务 | 6–8 天 | 上传到可检索全程可查询、可重试、幂等 |
| M5 | Prompt/Context 优化与发布门禁 | 5–7 天 | 引用和答案指标不回归，Prompt 可复现 |

依赖顺序：

```text
M0 一致性
 └─ M1 统一解析 IR
     └─ M2 版本与去重
         ├─ M3 切片与评估
         └─ M4 管理 API
             └─ M5 Prompt/Context 与发布门禁
```

## 4. M0：一致性与可恢复性

### M0-1 源文件删除同步

现状：源文件消失时只清理 `ingest_state`，不会自动清理 Document Store 和 Qdrant。

工作项：

- 扫描阶段生成 `missing_source` 清单，不直接静默删除；
- 新增 `--sync-delete`，显式启用后同步删除 Qdrant、Document Store、状态表；
- 默认先输出 dry-run 清单；
- 删除动作写审计记录，保留 document/version/source 信息；
- 单篇删除失败时不得把状态标记为完全删除。

验收：

```powershell
python scripts/ingest.py --dir tests/fixtures/sync_delete --sync-delete --dry-run
# 期望：列出 missing_source，但三处数据不变

python scripts/ingest.py --dir tests/fixtures/sync_delete --sync-delete
# 期望：Qdrant points、documents.sqlite3、ingest state 均不再包含目标 document_id
```

### M0-2 收敛 Document Store 写入点

工作项：

- `IngestPipeline` 只负责准备数据，不直接提交 Store；
- `Indexer` 或新的 `IngestUnitOfWork` 作为唯一写入口；
- 明确提交顺序和失败状态；
- 为“Store 新、Qdrant 旧”和“Qdrant 新、Store 旧”分别增加恢复测试；
- 每次 ingest 生成 `run_id`，写入状态与审计记录。

验收：

```powershell
python -m unittest tests.test_ingest_pipeline tests.test_indexer tests.test_ingest_invariants
# 期望：全部通过；测试断言一次文档更新只调用一次 replace_document
```

### M0-3 对账与修复工具

新增：

```powershell
python scripts/reconcile.py --check
python scripts/reconcile.py --repair --dry-run
python scripts/reconcile.py --repair
```

至少检查：

- Qdrant child 找不到 Document Store parent；
- Store child 在 Qdrant 不存在；
- 状态为 `COMPLETED` 但文档或向量缺失；
- source 已删除但索引仍 active；
- 同一 document 当前存在多个 active version；
- Context 来源字段中的 chunk/parent 标识是否一致。

退出条件：人工构造六类不一致后，`--check` 全部发现，`--repair` 后再次检查返回 `ok=true`。

### M0-4 在线快照刷新

工作项：

- 不再永久缓存启动时的父块字典；
- 第一版可采用版本号检测 + 原子替换内存快照；
- ingest 完成后发布 `knowledge_base_revision`；
- API 响应和 `/health` 返回当前 revision；
- 正在执行的请求继续使用旧快照，新请求切到新快照。

验收：服务不重启，新增一篇唯一关键词文档后，下一次检索能够命中；删除后不能再命中。

## 5. M1：统一解析与多模态 Document IR

### M1-1 定义 Document IR v2

保留现有 `DocumentBlock` 的简单使用方式，同时增加以下结构：

```text
ParsedDocument
  document_metadata
  elements[]
  assets[]
  parse_diagnostics

DocumentElement
  element_id
  element_type       heading/paragraph/list/table/code/image/formula/caption
  text
  section_path
  locator            page+bbox / paragraph / sheet+cell_range
  asset_ids[]
  relations[]        caption_of / follows / child_of
  extraction_method  native/ocr/caption
  confidence
  metadata
```

设计要求：

- IR 必须可 JSON 序列化并带 `schema_version`；
- 原始定位与规范化文本分离；
- 表格同时保留结构化 cells 和可检索文本；
- 图片二进制不直接塞进 block，使用 `asset_id`；
- 旧 `DocumentBlock` 可以通过 adapter 转换为 v2 element；
- parser 版本和配置哈希进入解析结果。

验收：IR schema 测试覆盖字段缺失、坏 locator、悬空 asset、非法 relation 和 v1 adapter。

### M1-2 Docling 旁路评估

先做 20 篇 PDF 的旁路 probe，不直接替换当前 PDF Parser。

记录：

- 页数、元素数、表格数、图片数；
- 页码/bbox 覆盖率；
- 文本覆盖率；
- 解析耗时 p50/p95；
- 峰值 RSS；
- 扫描 PDF 的 OCR 成功率；
- 与 PyMuPDF 结果的标题、表格、正文差异。

采纳门禁：

- `page_number` 覆盖率 = 100%；
- 可定位元素的 bbox 覆盖率 ≥ 95%；
- 正常文本 PDF 的字符覆盖率 ≥ 当前解析器的 98%；
- 20 篇无崩溃；
- 许可证与闭源使用目标兼容；
- 性能不满足时允许作为 `high_quality` profile，而不是默认 profile。

### M1-3 Word 与 Excel Parser

Word 第一版支持：

- 标题层级、段落、列表、表格；
- 页眉页脚或重复区域诊断；
- 内嵌图片导出为 asset；
- 图片 alt/caption/OCR 文本；
- 段落级 locator。

Excel 第一版支持：

- sheet、表格区域、合并单元格；
- 公式文本与显示值分离；
- 空白行列裁剪；
- `sheet + cell_range` locator；
- 大表分区，但表头必须随分区保留；
- 隐藏 sheet、图表只记录诊断，不在第一版推断语义。

验收：每种格式至少 10 个 fixtures，覆盖中文、英文、表格、图片、空文档和损坏文件。

### M1-4 Parse Preview

新增 CLI：

```powershell
python scripts/parse_preview.py sample.docx --json out.json --html out.html
```

HTML 至少展示：元素顺序、类型、章节路径、页码/单元格范围、图片缩略图、解析告警和被忽略内容。

## 6. M2：文档版本、去重与生命周期

### M2-1 数据模型迁移

目标表：

- `documents`：逻辑文档；
- `document_sources`：路径、URL、上传来源；
- `document_versions`：不可变版本；
- `assets`：图片和附件；
- `parse_runs`：解析器、配置、结果与诊断；
- `index_runs`：切片、embedding、collection 与状态；
- `duplicate_groups`：精确重复和近重复关系；
- `audit_events`：创建、更新、删除、回滚记录。

要求：旧库可迁移；迁移前自动备份；迁移可重复执行；失败不破坏旧库。

### M2-2 精确去重

分别计算：

- `blob_hash`：原始字节；
- `normalized_content_hash`：规范化正文；
- `structure_hash`：元素类型、顺序和内容；
- `asset_hash`：图片等资产内容。

默认策略：

- 同来源、同内容：跳过；
- 同来源、新内容：创建新 version；
- 不同来源、同内容：保留 source alias，不重复 embedding；
- 同正文、附件不同：创建新 version，但复用未变资产和向量。

### M2-3 近重复识别

- 使用 MinHash/SimHash 生成候选，不在全库做 O(n²) 比较；
- 第一版只标记 duplicate group，不自动删除；
- 提供相似度、差异摘要和人工确认入口；
- Context 组装仍保留近重复去重作为最后一道保险。

验收数据集至少覆盖：复制文件、改名、换行差异、只改标题、只改一段、同正文不同附件。

### M2-4 版本查询与回滚

新增 CLI：

```powershell
python scripts/documents.py list
python scripts/documents.py versions <document_id>
python scripts/documents.py diff <version_a> <version_b>
python scripts/documents.py activate <version_id>
python scripts/documents.py delete <document_id> --dry-run
```

回滚采用切换 active version + 索引协调，不修改历史版本内容。

## 7. M3：切片策略与分层评估

### M3-1 ChunkStrategy 接口

统一接口至少包含：

```text
name
version
supported_element_types
validate_config()
chunk(parsed_document, config)
describe()
```

首批策略：

- `recursive_parent_child`：现有默认策略；
- `structure_aware`：按标题、列表、代码和段落边界；
- `fixed_window`：作为基线；
- `table_aware`：表头重复、行分区；
- `page_aware`：适合页级引用要求；
- `semantic`：实验策略，默认关闭。

每个 chunk 必须记录 `strategy_name/version/config_hash/parser_version`。

### M3-2 Chunk Preview 与 Profile

新增：

```powershell
python scripts/chunk_preview.py sample.pdf --profile structure --html chunks.html
python scripts/chunk_quality.py --dir data --profile table --compare-profile recursive
```

内置 profile：

- `default`：现有行为；
- `structure`：Markdown/Word/技术 PDF；
- `table`：Excel/财务报表；
- `page-citation`：强调页码引用；
- `high-quality`：允许更慢解析/OCR。

### M3-3 评估分层

#### L0 解析层

- text coverage；
- heading path accuracy；
- table cell coverage / table intact；
- image extraction coverage；
- locator coverage；
- OCR CER/WER；
- parser failure rate。

#### L1 检索层

沿用 Recall@k、hit@k、full@k、MRR、nDCG、bootstrap CI 和配对比较。

#### L2 Context 层

- context recall；
- context precision；
- gold token coverage；
- duplicate token ratio；
- truncation loss；
- document fragmentation；
- token utilization。

#### L3 答案层

- answer correctness；
- completeness；
- faithfulness；
- citation precision；
- citation recall；
- invalid citation rate；
- wrong confidence；
- over-abstain。

#### L4 系统层

- ingest documents/s、pages/s；
- parse/embed/index p50/p95；
- ask/retrieve p50/p95；
- peak RSS；
- cache hit rate；
- failure/retry rate；
- 每题 context token 与生成 token。

### M3-4 多格式评估集

建立小而稳定的 `mixed-docs-v1`：

- PDF、扫描 PDF、DOCX、XLSX、Markdown 各不少于 10 篇；
- 至少覆盖表格问答、图片文字、跨页内容、版本更新和重复文档；
- 题型包括 single-hop、multi-hop、table、image-text、negative；
- 语料和题目带固定 hash，数据只增不改。

## 8. M4：文档管理与异步摄取 API

### M4-1 API 契约

建议端点：

```text
POST   /api/v1/documents
GET    /api/v1/documents
GET    /api/v1/documents/{id}
DELETE /api/v1/documents/{id}
GET    /api/v1/documents/{id}/versions
POST   /api/v1/documents/{id}/activate
POST   /api/v1/documents/{id}/reindex
POST   /api/v1/parse-preview
POST   /api/v1/chunk-preview
POST   /api/v1/ingest-jobs
GET    /api/v1/ingest-jobs/{job_id}
POST   /api/v1/ingest-jobs/{job_id}/retry
```

保留现有 `/ask`、`/retrieve`、SSE 和 `/health`。

### M4-2 异步任务

- 上传请求只落文件和 job，不在 HTTP 请求线程完成 embedding；
- job 状态：queued/parsing/chunking/embedding/indexing/completed/failed/cancelled；
- 支持阶段进度、失败原因、重试和取消；
- `Idempotency-Key` 防止重复提交；
- 同一逻辑文档串行更新，不同文档允许有限并发；
- API 重启后任务状态可恢复。

### M4-3 安全边界

- API Key；
- 上传大小、文件数、解压大小和页数限制；
- 文件类型不只看后缀，检查 MIME/magic bytes；
- ZIP bomb、路径穿越和宏文件防护；
- tenant_id/knowledge_base_id 进入过滤条件；
- 日志默认不记录完整文档正文和用户问题；
- SSE 最终提供 POST streaming 入口，避免敏感问题长期出现在 URL 和代理日志。

验收：OpenAPI 契约测试、鉴权测试、并发测试、幂等测试和重启恢复测试全部通过。

## 9. M5：Prompt 与 Context 组装优化

### M5-1 Prompt Registry

每个模板记录：

- `prompt_id`、`version`；
- system/user 模板；
- 支持的任务类型和语言；
- 变更说明；
- 输入变量 schema；
- 评估基线引用。

每次回答日志记录 prompt version、context strategy、模型、采样参数和知识库 revision。

### M5-2 Context 安全与结构化

- 用明确边界包裹证据，声明证据内容不是系统指令；
- 对文档中的 prompt injection 模式打标并进入安全评估集；
- metadata 与正文分离，不把内部相似度分数暴露给模型；
- 证据编号与来源对象一一对应；
- 引用先生成结构化对象，再渲染为 `[证据 n]`；
- 修复并测试 chunk_id、parent_id、source locator 的全链路追踪。

### M5-3 Query-aware Context Selection

从当前“按排名装入、最后一块截尾”升级为：

1. 相关父块内选择 query-aware window；
2. 在预算内做 MMR/覆盖式选择，降低重复；
3. 多跳问题保留不同文档的互补证据；
4. 表格证据保留表头；
5. 截断时按元素边界，不在句子或表格行中间切断；
6. 为回答预留固定 token，不让 Context 吃满模型窗口。

### M5-4 回答模板

首批模板：

- factual：直接结论 + 引用；
- comparison：按对比维度组织；
- multi-hop：只输出结论和可核对依据，不暴露模型思维过程；
- table：保留单位、列名和行条件；
- summary：覆盖关键主题并限制重复；
- negative：说明缺少什么证据，不猜测。

采纳门禁：

- citation precision 不低于当前基线，并达到预设阶段目标；
- citation recall 不因“少引”策略显著下降；
- wrong confidence 不上升；
- 正样本 correctness/faithfulness 配对比较不回归；
- p95 延迟和 context token 增幅在预算内；
- 所有结论同时报告整体和题型切片，不只报综合平均。

## 10. 横向工程任务

### 测试与 CI

- 外部命令测试先检查 executable availability；Windows 上不可执行时给明确 skip 原因；
- 默认套件继续保持离线；
- 多格式 fixture 不使用有版权争议的真实业务文档；
- 慢解析、OCR、真模型评估进入 nightly；
- schema migration、回滚和 reconcile 必须进默认套件；
- 引入 coverage、lint、type check 前先冻结当前基线，门槛逐步提高。

### 可观测性

新增指标：

- `ingest_jobs_total{status,stage}`；
- `parse_duration_seconds{parser,file_type}`；
- `elements_total{type}`；
- `dedupe_total{kind}`；
- `index_revision`；
- `reconcile_mismatch_total{kind}`；
- `context_tokens`、`context_gold_coverage`；
- `citation_invalid_total`；
- 队列深度和各阶段 p95。

### 迁移与兼容

- `DocumentBlock` v1 至少保留一个版本周期；
- 旧 SQLite 自动备份后迁移；
- 旧 CLI 参数保持可用并打印迁移提示；
- 新字段必须有安全默认值；
- 新 collection 先 shadow build，评估通过后再切 active alias；
- 回滚只切 alias/revision，不覆盖唯一一份旧数据。

## 11. 建议首个迭代

第一周只做以下内容，避免一开始同时铺开所有能力：

1. M0-1 源文件删除同步；
2. M0-2 去掉 Store 双写；
3. M0-3 reconcile `--check`；
4. 修复 Context 的 `chunk_ids`；
5. 在线父块快照增加 revision 检测；
6. 写 Document IR v2 ADR 和 schema，不接新解析器；
7. 建立 5 篇 PDF + 2 篇 DOCX + 2 篇 XLSX 的无版权 fixture 骨架。

首周退出标准：

```powershell
python -m unittest discover -s tests
# 期望：无业务失败；外部 jq 不可执行时明确 skip

python scripts/reconcile.py --check --json
# 期望：{"ok": true, "mismatches": 0, ...}

python scripts/ingest.py --dir <fixture-dir> --sync-delete
# 期望：新增、更新、删除三种操作后 Store/Qdrant/state/revision 一致
```

## 12. Definition of Done

一个任务只有同时满足以下条件才算完成：

- 代码与 schema 已实现；
- 有正向、失败、恢复和幂等测试；
- 有可复制的验收命令与期望值；
- API/CLI 文档已更新；
- 日志中能定位 parser/chunker/prompt/model/index revision；
- 数据迁移和回滚路径明确；
- 性能变化有 p50/p95 与资源数据；
- 影响检索或回答行为时，已完成同题集配对评估；
- 没有用新综合平均掩盖具体题型退化；
- 工作区内没有无法解释的孤儿数据或未登记 collection。

## 13. 决策检查点

以下决策不能在编码过程中临时拍板，应在对应里程碑开始前确认：

1. Docling 是否满足默认解析器的性能与部署体积要求；
2. PDF 商业使用的许可证路线；
3. 原始文档与图片资产是否允许持久化，以及保留周期；
4. 精确重复文档是“合并逻辑文档”还是“保留多个 source alias”；
5. API 第一版是否需要 tenant/knowledge-base 隔离；
6. OCR 和图片 caption 使用本地模型还是可选外部服务；
7. LLM-as-judge 使用本地模型还是只在离线评估环境启用；
8. Prompt Registry 存代码、SQLite，还是独立配置目录。

