# Local RAG Lab —— 一个带评估的本地 RAG（Ollama + Qdrant）

用 Ollama 做 embedding 与生成，用 Qdrant 存向量：文档 → 父子块 → 三路召回（中文 dense / 英文 dense / 英文词法）→ RRF 融合 → 交叉编码器重排 → 父块组装 → 带引用回答。

这个仓库的重点不是"能答"，而是**每个"更好"都有数字、每条数字都能重跑**：202 题中英成对评估集、正负样本分开计分、bootstrap 置信区间、逐题配对显著性检验、以及一份把所有失败留痕的导入状态机。

## 它和常见 RAG demo 的差别

| 常见 demo | 这里 |
|---|---|
| 一个 `chain.invoke()` 看不到中间态 | 三路候选、RRF 分数、重排前后排名、组装后 token 数，每步都能单独打印或关掉对比 |
| "感觉效果变好了" | 每个改动都有 A/B：同批题、逐题配对、报"更好/更差/持平计数"+ 显著性 |
| 检索错了无从下手 | 导入状态机记录每篇文档走到哪个阶段；失败清单可单条重试；索引带版本闸门 |
| 引用是装饰 | 引用可解析、可校验（越界/畸形会单独计数），并做过引用纪律 A/B |

## 快速开始

按顺序做，每一步都给了**验证方法**。全部做完大约 15 分钟（不含建库；建库按你的语料量算）。

### 第 0 步：先决条件

| 依赖 | 验证命令 | 期望输出 |
|---|---|---|
| Python 3.10+ | `python -V` | `Python 3.10.x` 或更高 |
| Ollama 已运行 | `ollama list` | 打印模型表；连不上会直接报连接错误 |
| Embedding 模型 | `ollama pull bge-m3` | 拉完后 `ollama list` 里出现 `bge-m3`（1024 维，建库口径） |
| 聊天模型 | `ollama pull qwen3.5:9b` | 拉完后 `ollama list` 里出现该名字。**机器带不动 9B 就换小的**，只是改环境变量 |
| Docker | `docker compose version` | 打印版本号。Qdrant 跑在容器里 |

> `ollama list` 里的名字**必须带 tag**（如 `qwen3.5:9b`）。只写 `qwen3.5` 一般会 404。

### 第 1 步：安装依赖

**Windows（PowerShell）**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> PowerShell 若禁止执行脚本，不要激活，直接用 `.\.venv\Scripts\python.exe` 代替下文的 `python`。

只想**跑起来**用上面这两行就够。想跑测试套件再加一行（`httpx` 是 TestClient / SSE 用例的依赖，
它现在是被 `qdrant-client` 顺手装进来的，所以这里显式声明，避免哪天它换 HTTP 栈就 ImportError）：

```bash
pip install -r requirements-dev.txt
```

**验证**（必须打印 `deps ok`）：

```bash
python -c "import qdrant_client, onnxruntime, tokenizers, pypdf; print('deps ok')"
```

装完依赖先跑一次 `python -m src.doctor`：缺什么、该跑哪条命令，它会直接列出来（下面第 2–4 步要装的东西都在里面，详见文末「自检与排障」）。

### 第 2 步：启动 Qdrant

```bash
docker compose up -d qdrant
```

**验证**（期望 `version` 里是 `1.16.2`，与本仓库 `qdrant-client` 的 minor 对齐）：

```bash
curl -s localhost:6333
```

> 为什么钉 1.16.2：1.15.0 在大集合优化期会 panic（`OutputTooSmall`），且 ANN 召回骤降到 0.5。版本纪律见 `docker-compose.yml` 注释。

### 第 3 步：准备 BGE-M3 词表（约 17 MB，不入库）

切分按 token 计量，需要 BGE-M3 的词表做本地计数（只加载词表，不跑推理；向量仍由 Ollama 算）。

```powershell
New-Item -ItemType Directory -Force storage/tokenizer/bge-m3 | Out-Null
curl.exe -sL -o storage/tokenizer/bge-m3/tokenizer.json "https://www.modelscope.cn/api/v1/models/BAAI/bge-m3/repo?Revision=master&FilePath=tokenizer.json"
```

**验证**（期望打印一个 token 数，不是报错）：

```bash
python -m src.rag.ingestion.tokenizer count "HNSW 是什么"
```

> 已有词表的话，用 `RAG_TOKENIZER_PATH=/path/to/tokenizer.json` 指过去即可，不必重复下载。

### 第 4 步：放你自己的文档

仓库**不携带任何语料**（版权 + 体积 + 与你无关）。把 Markdown / TXT / PDF 放进 `data/`：

```text
data/
├── md/产品手册/安装.md
└── pdf/白皮书.pdf
```

规则（子目录递归、命名如何影响 `--only`、PDF 页码限制）都写在 [`data/README.md`](data/README.md)。

### 第 5 步：建库

先试跑，**不落库**，看清楚会切出什么：

```bash
python scripts/ingest.py --dir data --dry-run
```

正式导入（可反复执行，已完成的不重算）：

```bash
python scripts/ingest.py --dir data
```

**期望输出**（末尾那行是判据）：

```text
[ingest] 发现 12 个文件，本次处理 12 个
[ingest] 完成：skipped=0 indexed=12 failed=0 children=214 用时 96.4s
```

- `failed=0` 才算成功；非 0 时看 `python scripts/ingest.py --failures`；
- 中途 Ctrl+C 随时安全，重跑会从断点继续（状态表 + embedding 缓存 + 点 id 幂等）；
- `data/` 是空的话会直接打印"该放哪 / 支持什么格式 / 放完跑哪条命令"，退出码仍是 0。

### 第 6 步：问答

```bash
python -m src.ask "智能体化 RAG 和传统 RAG 有什么区别？"
```

**期望输出**：一段带 `[证据 n]` 标注的答案 + 来源表 + 耗时行：

```text
引用：[2, 7, 5]｜越界=0｜解析成功率=1.00

来源：
  [证据 2] 深入理解AI Agent > 3.2.1 文档分块（Chunking）   doc_9626d3e8…
  ...

耗时：setup 12.3 s｜retrieve_rerank 5.5 s｜assemble 53 ms｜generate 16.9 s
上下文：9 块 / 6000 token｜候选 20 条｜各路命中 {'dense_zh': 20, 'dense_en': 20, 'lexical_en': 20}
```

常用开关：

```bash
python -m src.ask "问题" --retrieve-only        # 只检索，不跑 9B（秒级，调库用）
python -m src.ask "问题" --no-rerank            # 关重排（快 ~0.7 s/次，召回略降）
python -m src.ask "问题" --k 5 --sources all    # 少给几块证据 / 列全部来源
python -m src.ask "问题" --json                 # 机器可读输出
```

### 流式问答（SSE）

生成一次要十几到几十秒。流式的价值不是「显得快」，而是**证据先到**：检索 + 重排只要 1–5 s，
那时就能告诉用户「我会从这 3 块里答」，然后正文再一段段出来（实测 `sources` 4.99 s、首字 9.66 s、结束 17.67 s）。

```bash
curl -N "localhost:8000/api/v1/ask/stream?q=chunk%20size&k=3"
```

```
event: queued
data: {"inflight":0,"waiting":0,"max_inflight":1,"max_queue":2,"will_wait":false,"retry_after_s":60}

event: sources
data: {"blocks":3,"context_tokens":1674,"degraded":null,"sources":[…]}

event: delta
data: {"text":"切"}

…（每个 token 一条 delta）

event: done
data: {"answer":"…","cited":[2,1],"timings":{…}}
```

浏览器直接用（`EventSource` 只能发 GET，所以这个端点就是 GET）：

```js
const es = new EventSource('/api/v1/ask/stream?q=' + encodeURIComponent('chunk size'));
es.addEventListener('sources', e => showSources(JSON.parse(e.data)));
es.addEventListener('delta', e => append(JSON.parse(e.data).text));
es.addEventListener('done', e => { es.close(); renderCitations(JSON.parse(e.data)); });
```

三条只有长连接才会踩到的坑，都已经处理：

| 情况 | 行为 |
|---|---|
| 用户关页面 / 断网 | **立刻取消生成**（`chat_stream` 的连接被关）。否则生成继续烧 50 s，队列名额很快被僵尸占满 |
| 反代缓存 / 长时间无数据 | 每 **15 s** 发一行 `: ping`（SSE 注释，不产生事件）；响应头带 `X-Accel-Buffering: no` |
| 生成超时 | 发 `error` 且带 `status: 504` 与 `partial`（**已生成的部分不丢**），`sources` 早在前面发过了 |

流里失败不会再改 HTTP 状态码（一开始就是 200），所以失败一律用 `error` 事件表达，
队列满时是 `error` 里带 `status: 429` + `retry_after_s`。

### 能调什么（全部开关在一张表里）

所有可调项都是环境变量，**默认值摊开在仓库根的 `.env.example`**（36 个，分 9 组：

```
cp .env.example .env      # 只是给你看和抄，本仓库不自动读 .env
```

分组：路径 / Ollama / 向量库 / 切分与词表 / 向量调用 / 检索与融合 / 降级熔断 /
队列（生成 + 检索）/ 流式 / 日志 / 测试开关。

最常改的四个：

| 变量 | 默认 | 什么时候改 |
|---|---|---|
| `OLLAMA_CHAT_MODEL` | `qwen3.5:9b` | 机器带不动 9B，换小的 |
| `RAG_EMBED_MAX_CHARS` | `8192` | 单块很长被截断时 |
| `RAG_LOG_LEVEL` | `detail` | 只想留摘要用 `summary`，压测时 `off` |
| `RAG_OLLAMA_CHECK` | `0` | 想跑连真实 Ollama 的用例时设 `1` |

> `.env.example` 与代码的一致性由 `tests/test_dev_env_manifest.py` 保证：
> 代码读了却没写进文件的变量、文件里写了但没人读的变量、默认值和代码不一致的变量，
> 三种都会让用例失败。

### 并发与限流（HTTP 服务的默认值）

生成很慢（实测 p50 **53.5 s**、单次 12–25 s），且 CPU 上跑 9B 是算力独占：并发不会让 10 个人各快 10 倍，
只会让 10 个人一起等更久。所以服务端**一次只生成一个，其余排队，排不下就明确拒绝**。

| 参数 | 默认 | 含义 |
|---|---|---|
| `RAG_MAX_INFLIGHT` | 1 | 同时生成的上限 |
| `RAG_MAX_QUEUE` | 2 | 排队位（1 跑 + 2 排 = 最多 3 个进来） |
| `RAG_QUEUE_TIMEOUT` | 120 s | 排队多久没轮到就放弃（>2× 生成 p50） |
| `RAG_RETRY_AFTER` | 60 s | 429 里的 `Retry-After`（≈生成 p50，回来大概率能直接进） |
| `RAG_RETRIEVE_MAX_INFLIGHT` | 4 | `/retrieve` 独立并发（检索是纯 CPU 短任务，不能被生成饿死） |

拒绝时是这样（不是 500，也不是无限等）：

```
HTTP/1.1 429 Too Many Requests
Retry-After: 60

{"detail": "生成队列已满（1 个在跑 + 2 个排队），请 60 s 后重试（Retry-After: 60）。"}
```

实测（4 个并发 `/ask`）：**第 4 个 0.22 s 拿到 429**；同一时刻 `/retrieve` 仍 200（2.5 s）；
`/health` 的 `queue` 字段能直接看到 `inflight / waiting / served / rejected_full / rejected_timeout`。

`retrieve_only=true` 不调模型，**不占生成名额** —— 批量检索不会把问答堵死。

### 没有 Ollama 时会发生什么（降级）

向量模型（`bge-m3`）不在时，服务**不会整个挂掉**：三路检索里 `dense_zh` / `dense_en` 自动停用，
`lexical_en`（英文关键词）继续工作，答案旁边明确告诉你降级了。

```bash
python -m src.ask "Ollama 停掉时还能不能检索" --retrieve-only
```

```
[降级] embed_unavailable：CapabilityError: 调用 Ollama embedding 失败（model=bge-m3）…
  停用的路：dense_en, dense_zh
  修复：ollama serve  # 先确认 Ollama 在跑（桌面版/服务均可）
  修复：ollama pull bge-m3  # 向量模型必须与建库时一致：bge-m3（换模型需整库重建）
```

三条要知道的：

1. **失败只烧一次**：第一次失败后熔断 30 秒（`RAG_BREAKER_TTL`），期间不再发请求 ——
   实测连问三次是 `2029 ms → 0 ms → 0 ms`，而不是每次都白等 3 秒的重试；
2. **会自动恢复**：`ollama serve` 起好后不用重启服务，熔断到期自动半开重试（实测 2 s 后恢复正常、`degraded=None`）；
3. **中文问题降级后基本搜不到**：`lexical_en` 用的是英文分词，中文 query 几乎匹配不上；
   英文关键词仍可用。降级只是「不至于 500」，不是「等价能力」。

HTTP 侧同一个信息在 `degraded` 字段里（`/api/v1/retrieve` 与 `/api/v1/ask` 都有，正常时为 `null`）。

### 第 7 步（可选）：HTTP 服务

命令行之外，同一套链路也跑在 HTTP 上（FastAPI，自动生成 OpenAPI 文档）：

```bash
python -m uvicorn src.rag.api.app:app --host 127.0.0.1 --port 8000
```

**打开 <http://127.0.0.1:8000/> 就是聊天页**：输入问题 → 流式出正文 → 下面列出引用了哪几块证据
（含路径、来源文件、检索/首字/生成耗时；没有依据时会明确弃答）。页面上「停止」会同时取消服务端生成。

也可以打开 <http://127.0.0.1:8000/docs> 用 Swagger 调，或命令行：


```bash
curl -s localhost:8000/health
curl -s "localhost:8000/api/v1/retrieve?q=chunk%20size&k=3&debug=true"
curl -s -X POST localhost:8000/api/v1/ask   -H "Content-Type: application/json"   -d '{"question": "what is chunk size?", "retrieve_only": true}'
```

| 端点 | 用途 | 会调模型吗 |
|---|---|---|
| `POST /api/v1/ask` | 完整问答：检索 → 重排 → 组装 → 生成 → 引用体检 | 会（除非 `retrieve_only=true`） |
| `GET /api/v1/retrieve` | 只检索，返回融合+重排后的候选（含 `rrf_rank` 与 `rerank_score`） | 不会 |
| `GET /health` | 存活探针：父块数、路由、重排是否启用 | 不会 |

几个实测注意事项：

- **首次请求慢 15–18 s**：依赖（Qdrant 连接、25k 父块、ONNX 重排模型）在第一次请求时才装配，
  之后 `/retrieve` 约 1–5 s，`/ask` 主要是生成 10–50 s；
- **Windows 上用 curl 发中文会乱码**（终端编码问题，不是服务端）：改用 `/docs` 页面，
  或把 body 写进文件后 `--data-binary @body.json`；
- **只能单 worker**：`--workers 1`（默认）。多 worker 会各自加载一份 945 MB 的 reranker；
- 流式（SSE）与并发队列限制在下一批（O4），当前的 `/ask` 是一次性返回。

### 服务自检：`GET /health`

`/health` 与 `doctor` 跑**同一套探针**（一份实现，两处复用），区别只是它给监控用：

```bash
curl -s localhost:8000/health
```

```json
{
  "status": "down",
  "checks": [
    {"name": "qdrant", "ok": false, "detail": "连不上 Qdrant：…目标计算机积极拒绝…",
     "fix": "docker compose up -d qdrant  # 起库",
     "optional": false, "info": false}
  ],
  "blocking_fixes": ["docker compose up -d qdrant  # 起库", "python scripts/ingest.py --dir data  # 建库"],
  "degraded_fixes": [],
  "parents": 25411,
  "queue": {"inflight": 0, "waiting": 0, "served": 0}
}
```

三条与监控有关的约定：

| 约定 | 为什么 |
|---|---|
| **HTTP 永远 200**，状态看 `status` 字段 | 返 5xx 会被负载均衡直接摘掉，你就再也看不到「哪一项坏了」 |
| `status` 取最差项：`ok` / `degraded`（能检索不能生成）/ `down` | 「聊天模型没拉」不该让检索端点一起被摘 |
| 每条失败都带 `fix` | 与 `doctor` 共用探针，所以提示永远一致 |

实测：全绿时热请求 **0.65–0.81 s**（词表、父块这类昂贵构造会缓存）；`docker stop rag-qdrant` 后
`status=down` 且 HTTP 仍 200；`docker start` 后自动回 `ok`。

### 自检与排障：`python -m src.doctor`

一条命令查七项：Qdrant / 向量模型 / 生成模型 / 词表 / 重排 / 父块 / 语料目录。

```bash
python -m src.doctor          # 人类可读
python -m src.doctor --json   # 给脚本用（stdout 只留 JSON）
```

全绿时：

```
[ok] 全部就绪
  ✓ qdrant       rag_chunks_v1：38865 点
  ✓ embed_model  bge-m3 可用，维度 1024
  ✓ chat_model   qwen3.5:9b 已安装
  ✓ tokenizer    分词器可用（「健康检查」→ 3 tokens）
  ✓ reranker     已启用（mmarco-mMiniLMv2，首次调用自动下载 449 MB；离线需 HF_HUB_OFFLINE）
  ✓ parents      父块 25411 个（启动时加载，重建库后需重启）
  ✓ data_dir     1468 个文件（.md/.txt/.pdf）
```

坏掉时（这里把 Ollama 指到一个没在跑的端口）：

```
[down] 检索不可用
  ✓ qdrant       rag_chunks_v1：38865 点
  ✗ embed_model  调用失败：RuntimeError: 无法调用 Ollama API：http://127.0.0.1:11999/api/embed
      修复：ollama serve  # 先确认 Ollama 在跑（桌面版/服务均可）
      修复：ollama pull bge-m3  # 向量模型必须与建库时一致：bge-m3（换模型需整库重建）
  · chat_model   未安装 qwen3.5:9b（已安装：无）
      修复：ollama pull qwen3.5:9b
  ✓ tokenizer    分词器可用（「健康检查」→ 3 tokens）
```

三种状态与退出码：

| 状态 | 含义 | 退出码 |
|---|---|---|
| `ok` | 全部就绪 | 0 |
| `degraded` | 能检索、不能生成（只缺生成模型） | 0 |
| `down` | 检索都跑不起来（缺向量库/向量模型/词表/父块） | 1 |

符号：`✓` 正常，`✗` 硬失败，`·` 可降级或缺省。

> **为什么向量模型不能随便换**：索引用 `bge-m3`（1024 维）建。换成 768 维模型会直接报错，
> 但换成**另一个 1024 维模型不会报错**，只会让检索结果悄悄变差。`doctor` 就是为了替你抓这一类。

### 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `ConnectionError` / `localhost:11434` 拒绝 | Ollama 没启动 | `ollama serve` 或打开 Ollama 应用 |
| `Ollama 上找不到模型 'x'` | 名字少 tag 或没 pull | `ollama list` 看真实名字，改 `OLLAMA_CHAT_MODEL` |
| `404 Client Error` 调 embedding | 用了聊天模型当 embedding | `ollama pull bge-m3`，并确认 `OLLAMA_EMBED_MODEL=bge-m3` |
| `找不到 tokenizer.json` | 第 3 步没做 | 按第 3 步下载，或设 `RAG_TOKENIZER_PATH` |
| `client version ... incompatible` | Qdrant 镜像与客户端 minor 不一致 | 用仓库自带 `docker-compose.yml`（1.16.2） |
| `发现 0 个文件` | `data/` 里没有文档 | 见 [`data/README.md`](data/README.md) |
| 每次调用 Ollama 都慢 2 s | Windows 上 `localhost` 先解析 `::1`，Ollama 只听 IPv4。`OLLAMA_BASE_URL` 用 `http://127.0.0.1:11434`（本仓库默认已是），并走复用连接的 Session |
| 首次运行卡住 1 分钟 | 首次要从 HuggingFace 拉 449 MB 重排模型 | 等它下完（缓存进 `storage/models`）；或 `--no-rerank` |

## 项目结构

```text
local-rag-agentic-lab/
├── data/                     # 你的语料（.gitignore 排除，只保留 README 说明放置规则）
├── docker-compose.yml        # Qdrant v1.16.2（版本必须与 qdrant-client 同 minor）
├── src/
│   ├── ask.py                # 产品入口：python -m src.ask "问题"
│   ├── rag/
│   │   ├── ingestion/        # 解析 → Block → 递归切分 → 父子块 → overlap → 校验 → 质量报告
│   │   ├── embedding/        # bge-m3 批量编码 + 校验 + SQLite 内容 hash 缓存
│   │   ├── ingest/           # 导入状态机（十阶段）+ 失败清单 + 断点恢复
│   │   ├── store/            # Qdrant（collection/payload index）+ SQLite Document Store + 版本闸门
│   │   ├── retrieval/        # 三路召回 / RRF 融合 / 窗口重排 / 依赖装配
│   │   ├── context/          # 父块组装：去重、邻接合并、预算截尾
│   │   ├── answer/           # 提示词 + 引用解析 + 弃答判定
│   │   ├── translate/        # 中→英 query 翻译（术语表 + 缓存）
│   │   └── lang/             # 语言识别（按「词」不按「字符」计比例）
│   ├── eval_*.py             # 评估：数据集校验 / 日志算指标 / 门禁对比 / 消融 / L2 答案层
│   └── raglog*.py            # JSONL trace 落盘与 review 报告
├── scripts/
│   ├── ingest.py             # 一条命令：文档 → Qdrant + Document Store
│   └── backfill_parent_index.py
├── experiments/              # 探针、题库、消融（每个实验在 README 下半部分有命令 + 期望值）
├── legacy/                   # 旧链（单文件索引 index.json + embeddings.npy），仅存档
├── tests/                    # stdlib unittest，754 用例
└── docs/                     # 设计文档（切分与检索方案 / 评估与日志设计）
```

## Legacy：旧链（仅存档，不要用于新工程）

仓库早期版本用单文件索引（`storage/index.json` + `embeddings.npy` + 内存 BM25）跑通了 RAG 与 Agentic RAG 的全部教学实验。新链改用 Qdrant + SQLite 之后，旧链被移到 `legacy/` 存档：

| 文件 | 说明 |
|---|---|
| `legacy/legacy_rag_cli.py` | 原来的 `src/rag.py`。**它被 `src/rag/` 包遮蔽，已无法作为 `src.rag` 导入**，所以移到这里并改绝对导入 |
| `legacy/legacy_agentic_rag.py` | 原来的 `src/agentic_rag.py`（受控 Agentic：模型可选检索/改写，有最大步数） |
| `src/build_index.py` `src/retrieve.py` `src/bm25.py` | 旧链的建库/检索/BM25，仍在 `src/`（README 下半部分的教学实验要跑它），文件头标了 deprecated |

旧链需要先建索引，且指令与新链**不通用**：

```bash
python -m src.build_index        # 旧链建库（写 storage/index.json，慢）
python legacy/legacy_rag_cli.py  # 旧链问答
```

新链只需要记两条：`python scripts/ingest.py --dir data` 与 `python -m src.ask "问题"`。

## 评估与日志的设计标准

改评估集、加指标、调整检索日志埋点之前，先看 [`docs/eval-and-logging-design.md`](docs/eval-and-logging-design.md)。它规定：

- 哪些指标可用（Recall@k / MRR / nDCG + bootstrap CI），哪些**不可用**（关键词覆盖率已废弃）；
- 评估集的字段格式、题型配额与样本量要求（10 道题的置信区间宽达 40+ 个百分点）；
- 日志事件字段（`run` / `retrieve` / `generate` / `error` / `label`）、文件布局与 review CLI；
- 评估与日志的分工：指标只从日志算，不重复调模型。

## 运行日志与 review

旧链的 `legacy/legacy_rag_cli.py` 和 `legacy/legacy_agentic_rag.py` 每次提问都会自动落一份 JSONL 日志（默认 `logs/`，等级 `detail`），记录候选规模、分路排名、三段耗时、引用与弃答标记。

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
python legacy/legacy_rag_cli.py
```

观察每个问题的检索片段、来源、向量相似度和最终答案。

**验收：** 能解释“问题也要向量化”以及“模型生成答案前，检索结果如何进入 Prompt”。

### 实验 2：切分策略

打开 `src/config.py`，修改 `CHUNK_SIZE` 为 `250`、`500`、`900`，每次重建索引并测试同一批问题：

```powershell
python -m src.build_index
python legacy/legacy_rag_cli.py
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
python legacy/legacy_agentic_rag.py
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

### 实验 11：Parent / Child 两层 + 稳定 ID + JSONL 落盘

检索用 Child（细），回答时给 Parent（全）——所以两层各自要有边界规则、ID 规则和字段契约。

```powershell
python -m unittest tests.test_parent_builder tests.test_stable_ids tests.test_jsonl_writer -v
```

**Parent 构建三段式**（对应方案 §2.3 的 6 条规则）：

1. **分组**：遇到 `heading` 块或 `section_path` 变化就开新组；标题块留在组内，所以 Parent 内容以标题开头。
2. **切长**：组内按 block 边界累积，超 `max_tokens` 就另起；单个 block 自己超限时用 D06 递归切分，并按下标累加 `source_start/end`。
3. **并短**：短于 `min_tokens` 的块并进**同一父章节**的前一块，合并后 `section_path` 取共同前缀；`section_path[0]` 不同绝不合并。

实测（40 篇真实 md）：

| 指标 | 结果 |
|---|---|
| 规模 | 40 document → **194 parent → 292 child** |
| parent token ≤ 2000 超限 | **0**（p50 256 / p90 1044 / max 2000） |
| child token ≤ 500 超限 | **0**（p50 308 / p90 441 / max 499） |
| chunk→parent 关联成功率 | **100.00%**（孤儿 0） |
| 两次运行 JSONL sha256 | **完全一致** |
| chunker_version 变更后 ID 复用 | **0** |

**一个值得注意的读数**：`292 child / 194 parent ≈ 1.5`，大量 Parent 只切出 1 个 Child，说明 Child 并没有比 Parent 更细。原因是 `min_tokens=200` 合并后剩下的章节本身就短。这不是 bug，是**配置该调**的信号——用 A–E 对照表（D11）验证，不要凭手感改。

**ID 的三条规矩**：

- `document_id` 认**来源路径**：改一个错别字不会变成「另一篇文档」，增量导入才能按 `document_id` 清旧版本；
- `document_version` 才是**内容哈希**（先归一 BOM/CRLF，否则 Windows 与仓库里的同一文件会得出两个版本）；
- `parent_id`/`chunk_id` 认**内容 + 版本**，任一段变化 ID 必变，用 `` 拼接防撞车（`a|b`+`c` 与 `a`+`b|c` 用可见字符拼会撞）。

**JSONL 字段契约**：`parents.jsonl` / `children.jsonl` / `documents.jsonl` 的字段集与方案 §3.1–3.3 完全一致，**多一个字段也报错**——这三个文件是 Qdrant payload、SQLite、评估脚本的共同输入，字段漂移没人发现比报错更贵。`manifest.json` 记录每个文件的 sha256，两次运行直接对比即可，不用手工 `sha256sum`。

**验收：** 能解释为什么 `document_id` 不含内容，并能用两次运行的 manifest sha256 证明输出可复现。

### 实验 12：Overlap 与质量报告——先量出问题，再谈调参

```powershell
python -m unittest tests.test_overlap tests.test_validator tests.test_quality_report -v

# 40 篇语料的 A–E 对照 + 8 项抽查（30–50 篇是 D11 的要求）
python -m src.rag.ingestion.quality_report --dir data/md --limit 40 --out experiments/reports/chunking_quality.md --json experiments/reports/chunking_quality.json
```

实测（40 篇真实 md）：

| 方案 | chunk 数 | token p50 | 重复 | 句子被切断 | 碎块(<80) | child/parent |
|---|---|---|---|---|---|---|
| A 固定长度 | 594 | 133 | 0.00% | **97.53%** | 4.71% | — |
| B Recursive | 209 | 412 | 0.00% | 0.00% | 1.91% | — |
| C Recursive+Overlap | 247 | 408 | 0.00% | 0.00% | 0.81% | — |
| D Parent-Child+Overlap | 332 | 314 | **1.81%** | 0.00% | **14.76%** | 1.64 |
| E Semantic | 未实现（语义切分第一版关闭） | | | | | |

Overlap：100 对相邻 chunk 的 overlap **100% 落在 [50,70]**，其中 **69.5%** 的起点在句子或行边界上——「句子边界优先」不是摆设。

**D 方案的短板在碎块率，不在断句**：`min_tokens=200` 下 202 个 Parent 只切出 332 个 Child，很多 Child 就是整个 Parent。参数扫描：

| `min_tokens` | child/parent | child p50 | 碎块率 | 重复 |
|---|---|---|---|---|
| 0 | 1.33 | 159 | 33.0% | 8.46% |
| **200（默认）** | 1.66 | 316 | 14.3% | 0.31% |
| **400（建议）** | 2.20 | 353 | **7.0%** | **0.00%** |

所以「Child 的粒度被 Parent 大小卡住」。这条写成了待定决策 B07：**必须等检索级消融确认收益再改默认值**，不能拿 chunking 指标拍板。

**三个会让你结论反向的指标坑**（都是实测踩出来的）：

1. 句子切断只能在**散文区**统计——把代码块/URL 的结尾算进去，D 会虚高到 77%，比不做结构的 B 还差；
2. `rstrip()` 之后判 `endswith("
")` 永远为假——以换行结尾的列表项全被误算成切断，虚高 40 个百分点；
3. 跨 Parent 的接缝不是句子切断——§2.3 规定 Child 不跨 Parent，按 parent 分组后 D 从 40.10% 归到 0.00%。

**验收：** 能解释「碎块率」为什么是 Parent 大小的问题，并说清为什么不能只凭 chunking 指标改默认参数。

### 实验 13：Qdrant——把向量检索变成带过滤的数据库

```powershell
docker compose up -d qdrant                                   # qdrant/qdrant:v1.15.0，内存限 2 GB
python -m src.rag.store.qdrant_store ping                     # {"ok": true, "collections": [...]}
python -m src.rag.store.qdrant_store create                   # rag_chunks_v1 + 6 个 payload index
python -m src.rag.store.qdrant_store info --json
curl -s localhost:6333/collections/rag_chunks_v1
```

验收读数（实测）：

```
size=1024  distance=Cosine  hnsw m=16  ef_construct=200
payload_schema: document_id / document_type / language / status / subject / tenant_id
```

**为什么不用 numpy 兜底**（决策 B04）：实测同一 query 的 Top-50 来自 **23 个文件、最大单文件仅 6 条**，
「只在这份文档内检索」用 post-filter 几乎捞不到东西（k=5 时常为 0 条）。
Qdrant 的 filter 是**先过滤再在子集里 ANN**，不会坍缩。numpy 精确检索保留下来只当 D19/D36 的
「标准答案」，不进产品路径。

**四个静默出错的坑**：

1. **本地模式 payload index 空转**：`QdrantClient(":memory:")` 建索引会"成功"，但客户端自己警告索引无效。
   验收必须连服务端。
2. **point id 只收 UUID 或整数**：`chk_<32 hex>` 直接写入被 400 拒绝 → 用 `point_id_for()`（UUID5）映射，
   这也是 D16 幂等的前提。
3. **`ef_construct` 是 collection 级**：写进 `VectorParams(hnsw_config=...)` 会被静默忽略成 100（实测）。
4. **枚举名与值不一致**：`Distance.COSINE` 的值是 `"Cosine"`，`models.Distance["COSINE"]` 才对。
5. **同名 collection 删了立刻重建会 400**（目录残留）：`recreate=True` 路径必须等名字真正消失，
   测试则改成每用例独占随机 collection 名。

**验收：** 能说出「payload index 为什么必须建在导入前」，并能解释 point id 为什么不能用行号。

### 实验 14：Embedding 服务——缓存不是优化，是硬需求

```powershell
python -m unittest tests.test_embedding_service -v          # 31 用例，全用假后端，不碰网络
RAG_OLLAMA_CHECK=1 python -m unittest tests.test_embedding_integration -v   # 3 用例，真 bge-m3

python -m src.rag.embedding.service stats                   # 缓存行数/体积/命中
python -m src.rag.embedding.service encode "HNSW 的 ef 参数"   # 看维度、耗时、是否命中
```

真实 24 条 chunk 实测（逐条调用，体现 Ollama 单次固定开销）：

| 场景 | p50 | p95 | 提速 |
|---|---|---|---|
| 冷启动 | 2,192 ms | 2,212 ms | 1× |
| 进程内 LRU 命中 | 0.007 ms | **0.019 ms** | **113,489×** |
| SQLite 命中 | 2.5 ms | **3.1 ms** | **713×** |

单条 query：冷 2176 ms → 命中 **6.78 ms**（ADR 要求 <10 ms）。缓存 4 KB/行（1024 × float32）。

**三个新坑**

1. **`with sqlite3.connect(...) as conn` 不关连接**，只提交事务。Windows 上锁住 db 文件 → 临时目录清理
   `PermissionError`。用 `@contextmanager` 显式 `close()`。
2. **`np.frombuffer` 遇到截断 blob 直接抛 ValueError**：先按字节数校验，坏行按 miss 重算。
3. **命中率验收要看「每次调用」**：100 条跑两遍，累计命中率是 49.8%，永远不达标也不反映缓存是否生效。
   `stats()["last_pass"]` 才是验收看的值。

还有一处与方案 §7.2 的偏差：签名是 `list[list[float]]`，实现返回 `np.ndarray(float32)`——
8 万 × 1024 个 Python float 的对象开销就是几百 MB，下游也全要数组。

**验收：** 能解释为什么"缓存命中率"必须按每次调用算，以及为什么全零向量要报错而不是静默返回。

### 实验 15：Document Store——命中之后要能翻回原文

```powershell
python -m src.rag.store.document_store import <chunks目录>   # 导入并自动与 JSONL 行数核对
python -m src.rag.store.document_store verify <chunks目录>
python -m src.rag.store.document_store show <chunk_id>       # 回溯父块原文 + 来源路径
python -m src.rag.store.document_store stats
```

真实 20 篇实测：documents 20 / parents **102** / children **182**，导入 **149 ms**，库 368 KB；
三表行数与 JSONL **完全一致**；任意 `chunk_id` 能查到父块原文 + `source_uri` + `section_path`。

三条设计约束：**Parent 原文只存这一份**（Qdrant payload 只放 Child 文本）、**外键强制**
（孤儿 Child 会让"命中却回溯不到父块"）、**换版本是一个事务**（半新半旧比失败更糟）。

### 实验 16：幂等 Upsert——重复导入不该长胖

```powershell
python -m unittest tests.test_indexer -v      # 14 用例（真 Qdrant，约 100 s）
```

真实 3 篇实测：

| 操作 | 结果 |
|---|---|
| 首轮索引 13 child | 13 点，18.6 s（含冷启 embedding） |
| 重复导入 | **13 点不变**，陈旧点 0，0.32 s（**57×**，embedding 全命中） |
| 版本变化（新 child 7） | 该文档 7 点，**陈旧点删除 6** |
| 限定单文档检索 | 命中 1 条，全部来自目标文档，可回溯 630 字符父块 |
| 按文档删除 | 过滤计数 **0** |

**两个坑**：

1. **payload 的 `null` 能写不能查**：Qdrant 允许写 `language: null`，但 `MatchValue(None)` 被 pydantic 拒绝
   （要 `IsNull` 条件）→ 字段"存进去了却过滤不到"。所以 payload 里 **None 直接剔除**。
2. **版本变化别用「按 document_id 全删再写」**：查询会有空窗，失败留下半篇数据。改成
   「现有该文档点集 − 本次点集 = 陈旧点集」精确删。写入顺序固定 **先 Store（事务）后 Qdrant（幂等）**，
   失败重跑能自愈；反过来会产生孤儿向量。

**验收：** 能解释幂等为什么靠「point id 由 chunk_id 决定」而不是「先删后插」。

### 实验 17：一条命令跑完 —— 断点恢复不是玄学

```powershell
python scripts/ingest.py --dir data/md --limit 200     # 小批量试跑（可反复执行）
python scripts/ingest.py --dir data                   # 全量：已完成的不重算
python scripts/ingest.py --status                     # 进度
python scripts/ingest.py --failures                   # 失败清单
python scripts/ingest.py --retry-failed               # 只重跑失败的
python scripts/ingest.py --dir data --only md/x.md    # 单条重试
python scripts/ingest.py --dir data --dry-run         # 只看不做
```

硬杀实测（6 篇 md，子进程 `os._exit(9)`）：

| 步骤 | 结果 |
|---|---|
| 第 1 轮全量 | 6 篇 → 6 child，**10.23 s** |
| 硬杀 | 状态表 3 篇 `COMPLETED`，退出码 9 |
| 重启续跑 | `[skip] doc1–3` + `[ok] doc4–6`，**1.55 s**（6.6×） |
| 幂等核对 | Qdrant 点数 **6**（不是 9） |
| 失败→修复→单条重试 | `FAILED_PARSING` → `--only` 成功，`resolution=completed` |

**「重启不重算」是三层保险叠出来的**：状态表（阶段 + 三处版本）跳过 `COMPLETED`；中断停在
EMBEDDING 时缓存让重算近乎免费；要重写时 point id 幂等所以不产生重复点。缺任何一层都不成立。

**两个坑**：

1. **窗口不能按文件大小猜子块数**——`size // 4000` 让小文件永远估成 1，`window_children` 形同虚设。
   改成切完再攒（按真实 child 数决定何时 flush）。
2. **源文件消失会留下修不好的幽灵失败项**——`prune_missing()` 同时清状态行与失败表未解决项，
   写 `resolution='source_removed'` 区分「重试成功」与「文件删了」。

**验收：** 能解释为什么「先删后插」不是幂等，以及断点恢复为什么需要三层保险而不是一个状态字段。

### 实验 18：索引版本化——同维不同模型最危险

```powershell
python -m src.rag.store.index_versions list
python -m src.rag.store.index_versions check rag_chunks_v1 --embedding-model bge-m3 --embedding-version bge-m3-v2 --vector-size 1024
# 退出码 1，报错里直接给出可用的新名字：rag_chunks_bge_m3_v2
```

两个 1024 维模型混写进同一个 collection **不会报任何错**，检索只会慢慢变差——所以要在 upsert
前对账。闸门接在 `Indexer` 里：拒绝时**一个点都不写**，并区分 `retired`（主动退役）与口径不符。

### 实验 19：ANN 召回——没有精确基准就是自欺

```powershell
python -m unittest tests.test_search -v          # 11 用例（含 1 万点真跑）
python -m src.rag.store.search --queries 20      # 直接算 ANN Recall
```

真实 bge-m3 向量 + 20 条真实 query：

| 点数 | recall@50 | 延迟 p50 |
|---|---|---|
| 10,000 | **1.000** | 12–14 ms |
| 81,027 | 0.994 | 16–28 ms |
| 100,000 | **0.990**（ef=256） | 11–22 ms |

**这一轮抓出的是 Qdrant 1.15.0 的 bug，不是参数问题**：81k 点 recall 只有 0.46 且 `ef` 完全无效
→ 查 `status` 发现 `red`、optimizer panic `OutputTooSmall`、点数报 133,382（实际插 81,027）。
升到 **1.16.2** 后同批数据 `green`、点数精确、recall 0.994。容器内存也从 2 GB 提到 6 GB。

三个排查教训：

1. **先看 `status` 和 `indexed_vectors_count`，再谈召回**——索引没建完时测的是半成品；
2. **压测脚本用带随机后缀的 collection 名**——同名删除后的残留段会被新库读到，点数虚高；
3. **`ef` 对结果毫无影响时别继续调参**，那是引擎异常信号。

**验收：** 能解释为什么「两个 1024 维模型混写」比「维度不一致」更危险，以及为什么 ANN 召回必须有精确基准。

### 实验 20：端到端 DoD —— 全语料 1467 篇

```powershell
python scripts/ingest.py --dir data --limit 10000      # 全链路
python scripts/ingest.py --status
python scripts/ingest.py --delete md/docs.nvidia.com/xxx.md   # 删除（三方一致）
```

| 环节 | 实测 |
|---|---|
| 摄取 | **1467 篇 → 25,411 parent → 38,865 child**，0 失败，1992 s |
| Qdrant | 38,865 点、`green`、HNSW 建好（36,914 索引）、6 个 payload index |
| Document Store | 1467 / 25,411 / 38,865，与 Qdrant 对齐 |
| 召回（中英各 5 条） | **hit@5 = 8/10**（中 4/5、英 4/5），**父块回溯 10/10** |
| 删除/重建 | 删 1 篇 → 38,859；重跑 → **38,865**（精确回到基线） |

两个未命中都查清了：`eval-002` 首次命中在 #22（排序问题，Phase 3/4 的活），
`en-1` 那个词在全库出现 **0 次**（是我出的 query 不成立，不是检索坏了）。

**又抓到一个真缺口**：删除文档后状态表仍标 `COMPLETED` → 重跑被跳过 → **永远重建不回来**。
修法：`forget_document_id` + `Indexer.delete_document(state=...)` + `--delete`。
「删除」从来不是单点操作，而是 **Qdrant + Document Store + 状态表三方一致**。

**一个会影响读数解释的细节**：Qdrant 默认 `indexing_threshold=20000`，低于它走**精确扫描**。
所以「1 万点 recall=1.000」是精确扫描的必然结果，不是 HNSW 的功劳。

**验收：** 能说出删除一篇文档需要同步哪三处，以及为什么 1 万点的召回数字不能当 ANN 的成绩。

### 实验 21：`embedding_text` 前缀到底有没有用（A/B，而不是"我觉得有用"）

```powershell
python -m src.rag.ingestion.embedding_text_report --dir data --stats-only
python -m src.rag.ingestion.embedding_text_report --dir data --from-gold --distractors 400 --queries 60 --k 10
```

同一批子块建**两个** collection（有前缀 / 纯正文），同一批 query 各查一遍，按**父块**算指标：

| 样本 | prefixed Recall@10 / hit@10 / MRR | plain Recall@10 / hit@10 / MRR | delta |
|---|---|---|---|
| 6,092 点 · 48 题 | 0.759 / 0.854 / **0.610** | 0.734 / 0.813 / 0.443 | +0.024 / +0.042 / **+0.166** |
| 10,028 点 · 48 题 | 0.734 / 0.833 / **0.554** | 0.726 / 0.813 / 0.429 | +0.009 / +0.021 / **+0.125** |

**前缀主要把「第一个正确结果」往前提（MRR +0.13~0.17），而不是让更多正确答案进 Top-10。**
对 RAG 来说这恰好是有用的那部分——上下文只放 4~8 条。

但差异来源很窄：48 题里只有 **2–5 题**不一致，其余 43–45 题完全相同。

- `eval-002`「为什么文档要切分成 chunk？」prefixed **1.000** vs plain **0.000**——正文讲的是"分段"，
  `chunk` 只出现在章节标题里，**前缀把标题关键词补进了向量**；
- 反例 `eval-030`（BMC 默认用户名）prefixed **0.25** vs plain **0.58**——同一文档里相似章节太多，
  前缀反而把它们推得更像。

所以：保留前缀（§2.8 不变），但别把它当"免费涨分"；同文档多相似章节要靠 reranker（D31）解决。

两条硬事实：全语料 25,411 父块**缺 title 0 / 缺 section_path 0**（前缀覆盖 100%），
且 `embedding_text` 与 `build_embedding_text(...)` **逐字节相等（10,028/10,028）**——
所以它**不用落库**，Qdrant payload 里没有它。

**验收：** 能解释为什么 MRR 的提升对 RAG 比 Recall@10 更有意义，以及前缀在什么场景下是负作用。

### 实验 22：评估集扩到 101 条中英成对题（D27）

```powershell
python -m experiments.expand_questions --dry-run     # 扩题前先看 gold 能不能定位
python -m experiments.expand_questions
python -m src.eval_bilingual draft                   # 机翻草稿（status=draft）
python -m src.eval_bilingual review --qid eval-009 --en "Why does the Agent Loop need a maximum step count?"
python -m src.eval_bilingual check --min-total 100   # CI：还有 draft 就失败
```

| 项 | 值 |
|---|---|
| 题数 | **101**（原 60 + 新增 41） |
| `query_en` 成对率 | **100.00%** |
| `topic` 覆盖 | **8/8**（concept_zh / zh_with_en_term / acronym_model / number_spec / paraphrase / multi_condition / cross_chunk / no_answer） |
| `difficulty` | easy 29 / normal 51 / hard 21 |
| `check` | 两套校验均 **0 error** |

**关键设计：`query_en` 有状态机。** 机翻只产出 `draft`，`check` 默认把 `draft` 当错误——
评估集里混着未审的机翻，消融结论就分不清是「检索不行」还是「翻译不行」。

新题集的 L1（旧 81k 索引）：

| 模式 | Recall@10 | hit@10 | MRR@10 |
|---|---|---|---|
| vector | 52.1% | 60.2% | 0.396 |
| bm25 | 41.9% | 48.2% | 0.358 |
| hybrid | 54.7% | 62.7% | 0.373 |
| hybrid_score | **60.4%** | **67.5%** | **0.421** |

**最能说明 Phase 3 必要性的一个数**：同一条中文 query，问中文语料 Recall@10 **79.6%**，问英文语料只有 **42.7%**——
差 36.9 个百分点。跨语言检索不是"锦上添花"，是当前最大的窟窿。

**踩到的坑**：`qwen3.5:9b` 是思考型模型，`/api/chat` 不传 `think:false` 时 `message.content` 是**空串**，
输出全在 `message.thinking` 里 → 「不报错但答案为空」或 300 s 超时。加上后单条 **1.1 s**。

**验收：** 能解释为什么 `query_en` 要分 `draft`/`reviewed` 两种状态，而不是生成完直接用。

### 实验 23：英文 lexical 检索帮谁、伤谁（D25）

```powershell
python -m src.rag.store.lexical --questions experiments/questions.json --k 10
python -m src.rag.store.lexical --questions experiments/questions.json --k 10 --keyword-only
```

先破一个常见误解：**Qdrant 的 full-text 不是 BM25**。`MatchText` 只回答"含不含这些词"，
没有 idf 和长度归一化，`scroll` 按点 id 返回——拿它当前 10 名等于随机挑。所以这一路是
**Qdrant 召回（过滤）+ 应用层 BM25 排序**，分词用英文规则而不用 jieba。

| 样本 | 中文 query hit@10 | 英文 query hit@10 | 差值 |
|---|---|---|---|
| 全部可判定 83 题 | 46/83 = 55.42% | 57/83 = **68.67%** | +11 条 |
| 关键词类 32 题 | 13/32 = 40.62% | 24/32 = **75.00%** | +11 条 / Recall **+32.55 pt** |

但 **+11 是「赢 22 输 11」的净值**，分 topic 看方向相反：

| topic | n | 中文 | 英文 |
|---|---|---|---|
| cross_chunk | 27 | 14 | **23** |
| number_spec | 24 | 8 | **17** |
| **zh_with_en_term** | 9 | **9** | 3 |
| **concept_zh** | 6 | **4** | 2 |

英文 lexical 在型号/数字/跨块题上大幅领先（`DGX B300`、`400 Gbps` 换语言写法不变），
但在中文语料题上反而更差——中文 query 命中 9/9，英文 query 只剩 3/9，因为金标正文是中文字。

**所以英文 lexical 是补充不是替代**，这就是 D24/D26 要三路并存的原因。

**修掉的 bug**：`fulltext_filter` 第一版用 AND 语义，英文问句候选 **0 条**
（要求 `per`、`direction` 同条出现）；改 OR (`should` + `min_should=1`) 后正常。

**验收：** 能说出为什么"英文 query 更好"不能推导出"应该换成英文 query"。

### 实验 24：Query 翻译服务（D23）

```powershell
python -m src.rag.translate.service --text "HNSW 是什么？"
python -m src.rag.translate.service --questions experiments/questions.json --json out.json
```

| 项 | 值 |
|---|---|
| 样本 | 202 条（101 条唯一中文问法 + 101 条前缀变体） |
| 成功率 | **100.00%**（验收线 98%） |
| 术语保留率 | **100%**（min = 1.0，`DGX B300`/`8192`/`HNSW`/`ef_construct` 全原样） |
| 冷跑 | 621.7 s（≈3.08 s/query） |
| 缓存 | 二次运行 **202/202 命中、0.2 s** → **3100×** |
| 与人工 `query_en` 词元 Jaccard | p50 **0.600** / p10 0.400，零重叠 0/101 |

结构上有五个出口，**永不抛异常**——翻译挂了最多少一路召回，不能把检索打挂：

| 情况 | 返回 | status |
|---|---|---|
| 正常 | 英文译文 | `ok` |
| 缓存命中 | 英文译文 | `cached` |
| 本来就是英文 | 原文 | `skipped` |
| 漏型号/数字 | **原文** | `passthrough_terms` |
| 超时/报错 | **原文** | `passthrough_error` |

**四个真 bug（都值得记住）：**

1. **提示词漏进译文**——把"这些 token 必须原样保留"塞进提示（user 或 system 都试过），
   模型直接当正文翻出来：`What is HNSW? (Must keep these tokens verbatim: HNSW)`。
   更糟的是**术语检查照样通过**（`HNSW` 在里面），脏译文能一路混进检索。
   → 提示只写通用规则，不给 token 清单，改成本地校验 + 重试。
2. **温度 0 重试必然无效** —— 同 prompt 同温度只会拿到同一段文本，重试改成 **温度 0.4**。
3. **Windows `with sqlite3.connect(...)` 只提交不关闭** → 临时目录清理 `WinError 32`。
4. **`cls.provider = 函数` 变成绑定方法** → 测试实例被当第一个参数塞进 provider，
   异常被降级逻辑吞掉，测试"通过"但结果全是原文。用 `staticmethod` 包一层。

**验收：** 能解释为什么"漏术语时要回退中文原文"而不是"用漏了术语的译文继续检索"。

### 实验 25：新库正式基线（D27b）

```powershell
# 1) 把 gold_spans 映射成新库的 parent/child id（写回 gold_points）
python -m src.eval_gold --write

# 2) 新库跑批（dense / lexical / hybrid），再算指标并落基线
python -m src.eval_store run --k 10
python -m src.evaluate measure latest --main-mode hybrid --write-baseline
python -m src.evaluate compare latest      # 门禁：结论 ok
```

**gold 映射（101 题）**：ok **82** / ambiguous **1**（`eval-030`，一个 span 命中 12 个父块）/
unmatched **0** / negative 18 → 可用 **83 题**；match_mode 全部 `exact`，gold 父块数 p50 = 1。

**基线（83 正 + 18 负，249 trace）**：

| 模式 | Recall@4 | Recall@10 | hit@10 | MRR@10 | p50 |
|---|---|---|---|---|---|
| dense | 52.91%±10.1 | 64.86%±8.9 | 77.11% | 0.506 | 12 ms |
| lexical（jieba BM25） | 43.47%±9.8 | 47.69%±9.9 | 53.01% | 0.379 | 5 ms |
| **hybrid（RRF k=60）** | 49.50%±9.8 | **72.59%±8.3** | **84.34%** | **0.509** | 17 ms |

**跨语言鸿沟（hybrid）**：中文语料 Recall@10 **98.1%** vs 英文语料（中文提问）**60.3%** → **37.8 pt**，
与旧索引上测到的 36.9 pt 一致：这是语料语言问题，不是索引实现问题。

**三处修复：**

1. **跨索引配对会虚报**——新库 run 的「上一份日志」是旧 81k 索引，逐题差值被算成
   `hybrid +0.578`（读作「改进 57.8 个点」）。→ `runs_comparable()` 先比 `content_hash`
   与 `dataset_sha256`，不可比就一个数都不报。
2. **报告里的复现命令指向旧入口**（`src.evaluate run` 读 `storage/index.json`）→
   run 事件带 `index.runner`，按 runner 输出 `src.eval_store run`。
3. **负样本 z 分数缺统计量** → `CorpusCosStats` 一次性扫全库算 μ 与二阶矩 `M`（4 MB 缓存），
   之后每题 `O(d²)` **精确**得到 E[cos] 与 std，不是抽样近似。

**两个反直觉的数字：**

- `hybrid` 的 **Recall@4 反而低于 `dense`**（49.5% vs 52.9%）：RRF 把两路都靠前的抬上来，
  代价是把单路第一名挤后。RAG 只放 4–8 条上下文，D26 调融合参数时要继续盯这个。
- `dense` vs `hybrid` 的 McNemar **p = 0.5034**：83 题上 7 pt 的 hit@10 差距**还没到统计显著**，
  现在不能说"混合检索比纯向量好"。

**验收：** 能解释为什么 `hybrid` 的 Recall@10 更高但 Recall@4 更低，以及为什么这不矛盾。

### 实验 26：新旧链路能不能直接比？（D27b 补充）

**不能比聚合指标**——两份 baseline 有 5 处不可比：题集（48 vs 83 可判定）、索引
（81,027 chunk vs 25,411 parent）、金标粒度（500-token chunk vs ~2k-token 父块）、
模式定义（`hybrid_score` 已不存在）、融合方式（加权 vs RRF）。
`evaluate compare --baseline experiments/baseline_old_81k.json` 只会说 `not_comparable`。

唯一站得住的比法是**同一批题逐题配对**：

```powershell
python experiments/compare_chains.py `
  --old-log logs/2026-11-02/20261102T143004-2188.jsonl `
  --new-log logs/2026-11-04/20261104T113427-12fd.jsonl `
  --map vector=dense,bm25=lexical,hybrid=hybrid
```

交集 **48 题**：

| 旧 → 新 | k | 旧 hit | 新 hit | 旧独中 / 新独中 | p |
|---|---|---|---|---|---|
| vector → dense | 10 | 52.1% | **79.2%** | 1 / 14 | **0.0010** |
| bm25 → lexical | 10 | 54.2% | 60.4% | 1 / 4 | 0.3750 |
| hybrid → hybrid | 10 | 60.4% | **87.5%** | 2 / 15 | **0.0023** |

- **语义路显著变好**，**词面路没显著变化**（jieba BM25 在 chunk 级就靠精确词匹配拿到大部分命中）
- **增益不能全记成「检索变好了」**：父块约 2k token，比 500-token chunk 大，命中本来就更松，
  方向对新链路有利 → 只能说"新链路在该口径下更强"
- 顺带修掉一个**静默归零**bug：按命中 id 前缀（`par_` vs `path::chunk-`）判断金标空间，
  否则用旧日志重算会得到全 0.0%，看着像「旧链路完全不可用」

**验收：** 能说清为什么"新链路 hit@10 涨了 27 个点"不能直接写成"检索质量提升 27 个点"。

### 实验 27：三路召回 + RRF + 消融矩阵 A–E（D24/D26/D28）

```powershell
python -m unittest tests.test_retrieval_routes tests.test_retrieval_fusion tests.test_eval_ablation
python -m src.eval_ablation run  --out experiments/reports/d28_ablation.md
python -m src.eval_ablation run --include-zh-lexical  --out experiments/reports/d28_ablation_zhlex.md
```

83 题首轮（父块粒度，不含翻译开销；`query_en` 用人工撰写）：

| 方案 | 配置 | R@10 | MRR@10 | 池/列表 | P50 |
|---|---|---|---|---|---|
| A | 中文 Query + Dense | 64.9% ±8.9 | 0.506 | 10.0 | 29 ms |
| B | 英文 Query + Dense | 61.0% ±9.3 | 0.478 | 10.0 | 30 ms |
| C | A ∪ B 候选池 | 66.5% ±8.9 | 0.507 | 13.5 | 58 ms |
| D | C + 英文 BM25 池 | **78.3%** ±7.4 | 0.513 | 20.2 | 126 ms |
| E | 三路 RRF | **72.4%** ±8.0 | **0.544** | 10.0 | 126 ms |

- **英文 query 单路 dense 不比中文原句强**（A vs B：5 胜 1 负、p=0.2188）→ bge-m3 的跨语言能力让「翻译再检索」没有独立增益
- **英文 BM25 是池子的主要贡献**（C→D：10 胜 0 负、p=0.0020）
- **三路 RRF 相对中文 dense 是 7 胜 1 负、p=0.0703**——方向明确但差一点显著，正是扩题到 200 的原因
- **E 的 MRR/nDCG 全场最高**（0.544/0.556）→ RRF 主要在「把对的抬到前面」
- **C/D 是候选池、E 是列表，不能直接比大小**；每路深度 = k（`E@5` = 3×5 候选融合取 5）

**验收：** 能解释「为什么 D 的 78.3% 不能当成比 E 的 72.4% 更强」，以及为什么 `E@5` 会随融合深度变化（63.6% vs 58.7%）。

### 实验 28：题库扩到 202 题 + D28 定案（D27c）

```powershell
python -m experiments.expand_questions_v2 dump --docs 26 --per-doc 7 --merge --only "dgx-spark/,bcm-"
python -m experiments.expand_questions_v2 append
python -m src.eval_bilingual check --min-total 190      # 成对率 100%、8/8 topic
python -m src.eval_gold --write                          # ok 174 / ambiguous 0 / unmatched 0
python -m src.eval_ablation run --out experiments/reports/d28_ablation_202.md
```

**扩题的两条机制**（不靠专注，靠机制）：

1. `dump` 把父块正文切成「句子候选」并给稳定 key（`c0001…`），写题只引用 key
   → `gold_spans` 100% 是语料原文，不可能抄错
2. 多版本文档重复的样板句用 `max_parents` 显式声明 + note，不再当作脏数据

**202 题最终口径（174 正 + 28 负）**：

| 方案 | R@10 | hit@10 | MRR@10 | P50 |
|---|---|---|---|---|
| A 中文 dense | 65.7% ±6.6 | 74.7% | 0.531 | 18 ms |
| B 英文 dense | 68.4% ±6.1 | 77.6% | 0.560 | 18 ms |
| C 两路并集池 | 72.4% ±5.7 | 82.2% | 0.537 | 36 ms |
| D +英文 BM25 池 | **85.1%** ±4.7 | **91.4%** | 0.543 | 147 ms |
| E 三路 RRF | 79.3% ±5.3 | 87.4% | **0.603** | 147 ms |

- **A vs E：R@10 +13.6 pt、23 胜 1 负、p<0.0001** —— 83 题时只有 p=0.0703，**扩题是它变显著的原因**
- **A vs B：p=0.3833 打平** —— 英文 query 的价值不是"更准"，而是给 RRF 一路独立信号（C vs A p=0.0002）
- **D 的池上限 85.1% vs E 的 79.3%** → 还有 5.8 pt 候选没被排进 top-10，这是 D31 reranker 的余量
- **B05 定案**：中文 BM25 保留为可选第四路（池内 7 胜 0 负 p=0.0156；进 RRF 后 p=0.7266）

**验收：** 能说清「为什么 83 题的 p=0.0703 变成 202 题的 p<0.0001 不能解读为效果变强了」。

### 实验 29：Phase 4 —— Reranker / Context Assembler / 引用规范 / 消融 F·G / L2 首版（D31–D35）

```powershell
python experiments/probe_reranker.py --candidates 50          # B08：延迟 + RSS 探针
python -m src.eval_rerank_ablation run --limit 202           # D34：E vs F + G 上下文层
python scripts/backfill_parent_index.py                      # N6：回填 25,411 行 parent_index
python -m src.eval_answer run --positives 20 --negatives 20 --k 8 --budget 4000 --resume
```

**B08 定案：不装 torch，走 onnxruntime + `mmarco-mMiniLMv2` fp32**

| 候选 | 体积 | 256 tok | 结论 |
|---|---|---|---|
| mmarco fp32 | 449 MB | 28–53 ms/对 | **选它** |
| jina-reranker-v2 int8 | 266 MB | 88 ms/对 | int8 在本机反而慢一倍 |

50 候选 × 1 query：**P50 1.83 s / P95 1.87 s，RSS +910 MB**；生产只重排 top-20 → **674 ms**。

**D34：F = E + 重排（174 题，池 20）**

| 方案 | k | Recall | hit | MRR | nDCG |
|---|---|---|---|---|---|
| E 三路 RRF | 10 | 76.09% | 83.91% | 0.6039 | 0.6190 |
| F + 重排 | 10 | 78.75% | **83.91%（不变）** | **0.6641** | **0.6730** |

- **MRR +6.02 pt、nDCG +5.40 pt 显著，hit@10 一点没变（p=1.0）**
- 这是第三次「召回不涨、排序变好」（前两次：D14 embedding 前缀、D28 RRF）
- 所以 **reranker 不是召回问题的解**：它把已进 top-10 的块往前挪，不会把第 11 名拉进来

**G = F + Context Assembler**：块 10.00 → 7.28、token 6,015 → **4,536（-24.6%）**、相邻重叠块 3.79% → **0.00%**

**D35：L2 首版（40 题 = 20 正 + 20 负）**

| 指标 | 值 |
|---|---|
| 引用越界率 | **0.00%**（D33 编号纪律成立） |
| 引用精确率（自动近似） | 21.67%（模型「讨好式全引」） |
| **错误自信率** | **5.00%**（修正弃答判定后；首版的 45% 是度量 bug，见 N12） |
| 过度弃答率 | 10.00%（人工复核后真实 5%） |
| 负样本弃答率 | 95.00% |
| 答案延迟 p50 | **53.5 s**（9B + 3k 上下文 + CPU，不可上线） |

**修掉的三个真问题**：`parents.parent_index` 25,411 行全 NULL（N6，已回填）、
`retrieval/__init__.__getattr__` 递归 `RecursionError`（N7）、
组装器相邻判定写成 `<=` 把「隔 20 个父块」也当重叠（虚报 17.9%）。

**验收：** 能解释「为什么 hit@10 完全没变、nDCG 却涨了 5.4 pt」，以及「为什么 D32 的 22.9% 和这里的 3.79% 不是一个口径」。

### 实验 30：单条 query 的端到端追踪（`experiments/trace_query.py`）

```powershell
python -m experiments.trace_query "智能体化 RAG 和传统 RAG 有什么区别？"
```

逐段打印「输入 → 输出」：语言识别（ratio 0.857 → zh）→ 翻译（cached，query_en 人工级）
→ 三路召回（dense_zh 154 ms / dense_en 43 ms / lexical_en 59 ms）→ RRF（`1/(60+rank)` 跨路累加，
第 1 名三路全中）→ 重排（池 20 → top 8，rank 14 被提到 rank 2）→ 组装（10 块 → 7 块 / 6000 token 封顶）
→ prompt → 生成（63.6 s）→ 引用体检（cited [1,3,2,5]，越界 0，parse_rate 1.00）。

**这条追踪一次抓出两个真 bug**：

| 编号 | 现象 | 修法 |
|---|---|---|
| **N11** | prompt 里 `section=[ > " > 深 > 入 > …` —— `section_path` 是 JSON 字符串，被 `list()` 拆成单字符 | `store_wiring._path_list()` + `assembler._path_list()` 双层规范化，6 用例 |
| **N12** | 完整答案被标成「弃答」——判定是「出现『证据不足』就算」，而模型转述时会写到这个词 | 改成「强信号只看首尾各 3 句 + 弱信号只认开头极简回答」，6 用例 |

N12 的代价最大：**D35 首版报的错误自信率 45% 是度量 bug**。修正后重新计分同一批 40 条答案
（未重跑模型）：错误自信率 **45% → 5%**、负样本弃答率 **55% → 95%**、「答了且引到 gold」50% → 65%。

**验收：** 能指出这条链路里哪一段在「变慢」而不是「变差」，以及为什么「弃答判定错了」比「没有弃答判定」更危险。

### 实验 31：产品入口 + 引用纪律（E07 / D33b）

```powershell
python -m src.ask "智能体化 RAG 和传统 RAG 有什么区别？"      # 完整问答
python -m src.ask "问题" --retrieve-only --sources all       # 只检索，不跑 9B
python -m src.ask "问题" --no-rerank --json                  # 关重排 + 机器可读输出
```

链路抽到 `src/rag/ask/pipeline.py`，`src/eval_answer.py` 改为调用同一份 `answer_query()`
—— **评估到的行为就是线上跑的行为**（此前"评估器是唯一入口"）。

D33b 把引用纪律从「全引」收紧到「每条结论只引最直接的一条」（system 第八条 + user 模板）：

| 指标（同 40 题、同参数） | D35 v1 | D33b |
|---|---|---|
| **引用精确率** | 21.67% | **38.46%** |
| 平均每条答案引用块数 | 3.00 | **1.95** |
| 引用越界率 | 0.00% | 0.00% |
| 正样本「答了且引到 gold」 | 60% | **75%** |
| 过度弃答率 | 10% | 10% |
| 逐题配对 | — | **33 更好 / 2 更差 / 5 持平** |

顺带修掉两个度量 bug：**N13** 弃答判定的固定短语清单漏了六种写法里的三种（改成
「否定词 + 缺失谓词 ≤6 字紧随 + 开头 3 句」）；**N14** `[证据 1-7]` 区间引用被判畸形（现已展开）。

**验收：** 能解释「为什么引用精确率涨 17 pt 而召回一点没变」，以及为什么「判据说得清」
比「判据准」更重要。


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
