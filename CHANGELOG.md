# 变更记录

本文件只记**对外可见**的变化（能用起来的、会踩到的、行为变了的）。
内部的决策与踩坑细节在 [`docs/RAG数据切分与跨语言检索技术方案.md`](docs/RAG数据切分与跨语言检索技术方案.md)（含 30 条 N 编号的坑）
与 [`docs/ROADMAP.md`](docs/ROADMAP.md)（还剩什么、怎么算做完）。

## 未发布

### 新增

- **`scripts/chunk_quality.py`：切片结构体检**（不需要标注）。输出 `sc` 尺寸合规、`bi` 结构完整
  （段落/表格/代码有没有被切断）、`gap` 丢字检查。支持 `--dir` / `--dataset` / `--threshold` / `--strict` / `--json`。
- **`ingest --precheck` / `--precheck-only` / `--precheck-strict`**：建库前先体检。`--precheck-only`
  **不连 Qdrant、不写库**（把 `QDRANT_URL` 指到死端口也能跑），适合还没起服务的机器。
- **`eval_bench --collection / --store`**：可以把评估指到另一个集合，做参数 A/B 时不必覆盖唯一一份基线数据。

### 行为变化（会影响已有库）

- **子块尺寸现在严格不超过 `child_max_tokens`**。旧实现里 overlap 上限（70）大于目标重叠（60），
  会把子块顶过上限——基准库实测有 5 个子块到 501–505 token，现在为 0。
- **这条改动会让 10.9% 的子块换 `chunk_id`**（id 是内容哈希，重叠区一改就重新哈希）。
  升级后请**整库重建**；`gold_points`（子块级标注）需要重算。
- **基准基线已重立**：`recall@10 0.7737` / `hit@10 0.7789` / `full@10 0.7684` / **`MRR 0.5456`**
  （原 0.5529）。命中率一格没变，只有 6 道中文题在 top-10 内部换了名次。旧基线归档在
  `docs/evidence/baseline-k8s-v1-before-overlap-fix.json`。

## v0.1.0 — 2026-11-05

第一个可用版本。**能做什么、不能做什么，写在这里，免得你试了才发现。**

### 包含

- **完整检索链路**：`dense_zh`（bge-m3，中文问中文）+ `dense_en`（翻译后问英文索引）+ `lexical_en`
  （BM25），RRF 融合 + cross-encoder 重排；每路检索深度 = k，E 由三路原始列表自算。
- **三种入口**
  - CLI：`python -m src.ask "问题"`（`--k/--pool/--budget/--no-rerank/--no-translate/--retrieve-only/--sources/--json`）
  - 同步 HTTP：`POST /api/v1/ask`、`GET /api/v1/retrieve`
  - 流式 HTTP：`GET /api/v1/ask/stream`（SSE：`queued → sources → delta* → done`），
    以及 `GET /` 的浏览器聊天页（零依赖、不引 CDN，内网断网可开）
- **引用体检**：越界引用、解析成功率、引用精度都随答案返回，不靠人肉看。
- **可自助排障**：`python -m src.doctor` 与 `GET /health` 共用一份六项探针
  （qdrant / embed_model / chat_model / tokenizer / reranker / parents），每条失败都给可复制命令；
  `/health` **永远 200**，状态看 `status` 字段（`ok` / `degraded` / `down`）。
- **降级而不是全挂**：Ollama 挂了 → 熔断 + 明确告知「能检索不能生成」；`retrieve_only` 不占生成名额。
- **有界队列 + 单飞**：`MAX_INFLIGHT=1`、`MAX_QUEUE=2`，满了返 429 + `Retry-After`。
- **断点续建**：`scripts/ingest.py` 幂等，Ctrl+C 后重跑从断点继续；空 `data/` 给可复制指引并 exit 0。
- **测试分层**：默认套件 **941 用例、约 35 s、不联网**；碰真实 Qdrant/Ollama/大字表的用例挂 `@slow`，
  `RAG_RUN_SLOW=1` 才跑。分层本身有守卫（AST 检查导入期有没有探真实服务）。

### 不包含（v0.1.0 明确不做）

| 没有 | 原因 / 替代做法 |
|---|---|
| 语料与评估集 | 版权 + 体积 + 与你的业务无关。`data/` 只放 `README.md` 说明规则 |
| 一键安装 | 要起 4 个外部进程（Qdrant/Ollama/模型/词表），一条命令只会把失败藏得更深 |
| `--workers > 1` | 重排模型常驻约 945 MB，多 worker 各加载一份；单飞 + 有界队列更省内存 |
| 鉴权与配额 | 定位是内网/本机工具；要公网必须自己加网关（见 ROADMAP O13） |
| Qdrant local 模式 | 走 HTTP 服务端是刻意选择（多进程共享、可观测）；local 模式见 ROADMAP O10 |
| `pip install` 后的 console script | 现在用 `python -m src.xxx` / `python scripts/xxx.py`，见 ROADMAP O11 |
| 词表自动下载 | 需你显式执行一条 `curl`（README 第 3 步），避免首次运行偷偷下 17 MB |

### 已知限制（不是 bug，但会踩到）

- **首次问答会慢**：要下载 449 MB 重排模型（`storage/models`），之后才是 28–53 ms/对。
- **PDF 解析需要额外装**：PyMuPDF 是 **AGPL-3.0/商业双授权**，`requirements-pdf.txt` 单独一份，
  不装也能用 md/txt（见 [`NOTICE`](NOTICE) 第三节）。
- **生成 p50 约 53 s**（9B 模型在 CPU 上）：所以 `QUEUE_TIMEOUT=120 s`、`Retry-After=60 s`。
- **PDF 的 `source_start/end` 为 NULL**（页码映射尚未做，N9）。
- **`embedding_model` payload 是空串**（历史建库遗留，N10）。

### 升级/兼容

第一个版本，无兼容问题。`RAG_OLLAMA_CHECK` 作为 `RAG_RUN_SLOW` 的历史别名继续认。
