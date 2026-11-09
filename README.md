# Local RAG Lab —— 一个带评估的本地 RAG（Ollama + Qdrant）

用 Ollama 做 embedding 与生成，用 Qdrant 存向量：文档 → 父子块 → 三路召回（中文 dense / 英文 dense / 英文词法）→ RRF 融合 → 交叉编码器重排 → 父块组装 → 带引用回答。

这个仓库的重点不是"能答"，而是**每个"更好"都有数字、每条数字都能重跑**：一份**随仓库发布的基准**（Kubernetes 官方文档中英对照 944 篇 + 100 道改写题）、三口径一起报的检索指标、正负样本分开计分、以及一份把所有失败留痕的导入状态机。

当前基线（`k=10`，含重排，`--query-en translate` = 生产口径）：`recall@10 0.8297` / `hit@10 0.8345` /
`full@10 0.8273` / `MRR 0.5648`，共 **139 个正样本**（144 题）。检索延迟 p50 1565 ms（重排占约 1.1 s）。**怎么算的、怎么复现、哪里最弱**都在下面「[基准数据集](#基准数据集)」「[基准结果](#基准结果)」两节。

## 它和常见 RAG demo 的差别

| 常见 demo | 这里 |
|---|---|
| 一个 `chain.invoke()` 看不到中间态 | 三路候选、RRF 分数、重排前后排名、组装后 token 数，每步都能单独打印或关掉对比 |
| "感觉效果变好了" | 语料和题库**随仓库发布**，别人跑出来的是同一组数；每个改动都是同题 A/B：逐题配对 + "更好/更差/持平计数" |
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

想解析 **PDF** 还要再装一个（**许可证不同，见文末「许可证」**）：

```bash
pip install -r requirements-pdf.txt     # PyMuPDF：AGPL-3.0 / 商业双授权，不装也能用 md/txt
```

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

> **同一台机器上放多套库**（测试 / 预发 / 生产）：用 `RAG_COLLECTION` 换集合名，
> 默认 `rag_chunks_v1`。换集合等于换一整套库 —— `ingest` 会发现状态表属于别的集合，
> 按「全新库」重建并在 stderr 说明；不换就不受影响。

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

### 跑测试：默认 34 秒、不联网

```bash
python -m unittest discover -s tests          # 909 用例，34 s，纯单测（不需要 Qdrant/Ollama）
RAG_RUN_SLOW=1 python -m unittest discover -s tests    # 全套含真实服务，约 6 分钟
```

分层规则只有一条：**碰真实服务或大模型的用例挂 `@slow`**（真实 Qdrant / Ollama / 17 MB 词表 /
449 MB 重排模型）。默认套件里它们是 `skipped`，所以：

| 场景 | 默认套件 | 加 `RAG_RUN_SLOW=1` |
|---|---|---|
| 刚克隆、什么都没起 | ✅ 全绿（skip 50） | ❌ 大量失败 |
| 有 Qdrant + Ollama | ✅ 全绿 | ✅ 全绿（skipped=5） |
| CI | ✅ 34 s | 不建议（要起服务） |

这条规矩由 `tests/test_suite_layering.py` 守着：它用 AST 检查有没有模块在**导入期**就探真实服务
（那会让「跳过」也变慢、并重新依赖外部进程）。你新写的用例如果这么干，会立刻红。

### 能调什么（全量在 `.env.example`，这里是最常改的几个）

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
| `RAG_RUN_SLOW` | `0` | 想在测试里连真实 Qdrant / Ollama / 大字表时设 `1`（`RAG_OLLAMA_CHECK` 是历史别名，仍认） |

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
- **流式与队列都已上线**（见上面的「流式问答（SSE）」与「并发与限流」两节）：`/api/v1/ask` 一次性返回，`/api/v1/ask/stream` 边检索边吐字；生成队列满了返 `429` + `Retry-After`，`retrieve_only=true` 不占生成名额。

### 日常运维：停服 / 看进度 / 删文档 / 重建 / 备份

**看进度**（建库中途也可以跑，只读）：

```bash
python scripts/ingest.py --status        # 集合名 + 状态表位置 + 库内点数 + 完成百分比
python scripts/ingest.py --failures      # 失败清单（--retry-failed 只重试失败的）
```

> `--status` 会把**三样一起报**（集合 / 状态表 / 库内点数）。如果它警告「库里有向量，
> 但状态表没有进度记录」，说明进度丢了、库没空 —— 重跑一次建库即可（点 id 幂等，embedding 有缓存）。
> 这种事不会因为换 `RAG_COLLECTION` 做实验再发生了：状态表现在按集合分文件（`state-<集合>.sqlite3`）。

**删掉一篇文档**——「删除」必须三方一致（向量点 + 文档表 + 状态表），只删 Qdrant 点会让
`--status` 和重建逻辑对不上。所以用命令而不是手工删集合：

```bash
python scripts/ingest.py --delete md/产品手册/安装.md
# [delete] md/产品手册/安装.md → 删点 12、删文档 1、清状态 1     ← 三处都动了才算删干净
```

> `--delete` 的 key 必须和建库时一致（相对 `--dir` 的路径），它靠状态表把 key 映射到
> `document_id`。状态表被清过就会打印「状态表里没有 document_id，跳过」。

**重建整个库**（比如换了 embedding 模型、或改了 `CHUNK_SIZE`）：

```bash
# 换模型/改切分 = 旧向量全部作废，必须整库重建；只重建一篇用上面的 --delete + 重新 ingest
python scripts/ingest.py --dir data --force
```

**停服与备份**：

```bash
docker compose stop qdrant        # 停但保留数据
cp -r storage storage.bak-$(date +%Y%m%d)   # 备份（先 stop，避免拷到写一半的段文件）
docker compose down               # 彻底停（数据仍在 ./storage/qdrant）
```

> 备份要一起拷 `storage/qdrant`（向量）、`storage/documents.sqlite3`（原文与元数据）、
> `storage/ingest/`（进度）——**只拷其中一个，恢复出来就是残缺的库**。

### 建库前先体检：`scripts/chunk_quality.py`

索引**你自己的文档**时没有题库可测（没有 `Recall@k`）。这时唯一能提前发现问题的办法是
**看切分有没有把段落/表格/代码切断** —— 它不需要任何标注，只用到 parser 给出的结构边界：

```bash
python scripts/chunk_quality.py --dir data/md --strict      # 结构体检，有不合格就退出码 1
python scripts/chunk_quality.py --dataset k8s-zh-en-v1 --json out.json   # 逐篇结果落文件
python scripts/ingest.py --dir data --precheck-only         # 同一个检查，但走建库入口
```

输出三样：`sc` 尺寸合规（有多少子块越界）、`bi` 结构完整（有多少段落/表格被切断）、
`gap` 丢字检查（父块内容有没有被子块漏掉，**这算 bug 不算评分**）。

> **`sc`/`bi` 不预测答对率**，只筛「明显结构有毛病」的语料；检索质量仍以 `scripts/eval_bench.py`
> 为准。仓库自带的基准语料实测：944 篇 / 9,795 子块 / `sc` 1.0 / `bi` 0.9607 / 丢字 0 篇。

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
| 首次问答「卡死」十几分钟 | HuggingFace 直连被墙，`huggingface_hub` 在重试 | 走镜像：`export HF_ENDPOINT=https://hf-mirror.com` 再跑一次；或先用 `--no-rerank` 把链路跑通 |

## 技术架构

### 两个阶段

系统只做两件事：**离线把文档变成可检索的向量 + 原文**，**在线把问题变成带引用的答案**。
两阶段之间只通过两个存储耦合（Qdrant 存向量、SQLite 存原文与元数据），所以建库和服务可以分开部署。

```text
【离线】建库
  data/*.md|txt|pdf
     │ ① 解析        parser/{markdown,pdf,txt} → Block（heading 层级、页码、source_start/end）
     │ ② 递归切分    splitter                 → Child（token 预算内、优先句子边界、0 超限）
     │ ③ 父子组装    parent_builder           → Parent（≈700 token，标题拼进正文头）
     │ ④ 校验/质检   validator + quality_report → 空块、超限、碎块率、孤儿率
     │ ⑤ 嵌入        embedding（bge-m3，批量 + SQLite 内容 hash 缓存）
     │ ⑥ 幂等写入    Qdrant upsert（父/子块 id 由 内容+版本 派生，天然幂等）+ Document Store
     │ ⑦ 状态机      ingest/state（六个阶段 + 四个失败态 + 断点续建 + 集合绑定）
     ▼
  Qdrant collection ── 一个点 = 一个 Child（向量 + parent_id 载荷）
  SQLite documents / parents / children / ingest_state / ingest_meta

【在线】问答
  question
     │ ① 三路召回   dense_zh（中文 query 直接编码，bge-m3 跨语言）
     │              dense_en（英文 query 编码：人工 query_en 或翻译服务）
     │              lexical_en（英文 BM25 over 父块正文；可选第四路 lexical_zh 用 jieba）
     │ ② 收敛父块   子块命中 → 按 parent 聚合
     │ ③ RRF 融合   score = Σ 1/(60+rank)，三路各取 k 条，不按分数绝对值混
     │ ④ 窗口重排   cross-encoder 只看「query 对齐的那 256 token 窗口」，不是整块
     │ ⑤ 预算组装   去重 → 邻接合并（相邻父块缝起来）→ token 预算截尾
     │ ⑥ 生成       Ollama（同步 / 流式），提示词写明引用规则与「答不了就说答不了」
     │ ⑦ 引用体检   解析 [n]、越界/畸形单独计数、弃答判定、来源回填
     ▼
  答案 + sources（chunk_id / 标题 / section_path / 分数）+ 降级信息
```

### 分层与职责

| 层 | 目录 | 职责 | 出问题时表现 |
|---|---|---|---|
| 解析 | `src/rag/ingestion/parser/` | 拿到结构（标题层级、页码、原文偏移），不只拿字符串 | 该篇标 `FAILED_PARSING`，其余继续 |
| 切分 | `ingestion/splitter.py`、`parent_builder.py` | token 预算内的 Child + 有上下文的 Parent | 超限率、碎块率进质量报告 |
| 嵌入 | `embedding/service.py`、`cache.py` | 批量编码 + 内容 hash 缓存 + 维度校验 | 整批失败进失败清单，可单条重试 |
| 存储 | `store/` | Qdrant 向量 + SQLite 原文 + 索引版本闸门 | 版本不配套直接拒写（同维不同模型最危险） |
| 召回 | `retrieval/routes.py`、`orchestrator.py` | 三路并行（可选第四路中文 BM25）、每路深度 = k、RRF 自算 | 某路抛异常 → 记 `degraded`，其余路继续 |
| 重排 | `retrieval/rerank.py` | 交叉编码器给「query × 窗口」打分 | 模型不可用 → 跳过重排并明示降级 |
| 组装 | `context/assembler.py` | 去重、邻接合并、预算截尾、碎片率 | 预算不足时截尾，不静默丢来源 |
| 作答 | `answer/prompt.py`、`citations.py` | 提示词、引用解析、弃答判定 | 引用畸形单独计数，不当作「有引用」 |
| 服务 | `api/` | `/ask`、SSE 流式、`/retrieve`、`/health`、聊天页 | 队列满 → 429 + `Retry-After`；`/health` 永远 200 |

### 一次提问发生了什么

| # | 步骤 | 典型耗时（本机） |
|---|---|---|
| 1 | 三路召回并发（`dense_zh` / `dense_en` / `lexical_en`，每路 `k` 条） | 30–80 ms |
| 2 | 子块收敛成父块排名 + RRF 融合（`pool=40`） | < 5 ms |
| 3 | 窗口重排（40 个候选 × 256 token） | **~1.1 s**（CPU ONNX；关掉重排则整条检索 p50 64 ms） |
| 4 | 父块组装（去重 + 邻接合并 + 预算） | < 10 ms |
| 5 | 生成（`qwen3.5:9b`） | 秒级到分钟级（p50 ≈ 53 s） |

所以**检索不是瓶颈，生成才是**：这也是为什么 `/retrieve` 与 `/ask` 用**两条独立的队列**，
`retrieve_only=true` 不占生成名额。

### 存储：三本账，各管一件事

| 存储 | 内容 | 为什么分开 |
|---|---|---|
| Qdrant collection | 子块向量 + `parent_id` / 语言 / 版本载荷 | 只做 ANN；换模型就换 collection（版本闸门拦同名不同维） |
| `documents.sqlite3` | 文档、父块正文、子块偏移、索引版本 | 命中之后要**翻回原文**，向量库不适合存长文本 |
| `state-<集合>.sqlite3` | 六个阶段进度 + 四个失败态 + 集合绑定 | 断点续建；**按集合分文件**，换集合不许清别人那本账 |

删一篇文档 = 三方一起动（`--delete` 打印 `删点 N、删文档 N、清状态 N`），只删一处就会出现
「库里有向量但翻不到原文」或「状态说没建过但点数还在」。

### 几条不打算改的取舍

| 取舍 | 为什么 |
|---|---|
| 父块 / 子块两层：检索打在子块、答在父块 | 子块保证向量精度，父块保证上下文完整；单层要么切太碎要么上下文被截断 |
| 点 id 稳定推导：`document_id` 认**来源**、`chunk/parent_id` 认**内容 + 版本 |
  「同一文档改个错别字」不会变成另一篇文档，而「切分规则改了但 ID 没变」在结构上不可能发生 |
| 三路各取 k 条后**自算 RRF** | 不同路的分数尺度不可比，绝不按原始分数相加 |
| 重排看「窗口」而不是整块 | 父块 p50 ≈ 700 token，直接截 256 只看得到标题 |
| 降级只认异常类型，不看错误文本 | 错误文案会变，异常类型不会；按文案判会静默失效 |
| 优化路径也必须走同一个受保护入口 | 否则「快的那条路」迟早绕过队列、限流、降级 |
| 默认测试 34 秒、完全不联网 | 碰真实服务的用例挂 `@slow`，要 `RAG_RUN_SLOW=1` 才跑 |
| 语料和数据集不进 git | 版权 + 体积 + 与你的业务无关；走 Release 资产 + sha256 |

### 扩展点

| 想做的事 | 改哪里 | 注意 |
|---|---|---|
| 换嵌入模型 | `RAG_EMBED_MODEL` + 新 collection 名 | 必须整库重建；`index_versions` 会拦住同名不同维 |
| 换聊天模型 | `OLLAMA_CHAT_MODEL` | 与嵌入模型无关，不用重建 |
| 加一路召回 | `retrieval/routes.py` 新 Route + `store_wiring.py` 装配（参照现成的 `include_zh_lexical`） | 每路深度仍 = k；RRF 自己算，别引入外部融合分数 |
| 关掉 / 换重排模型 | `RERANK_MODEL`、`--no-rerank` | 换模型后基线不可比，要重立基线 |
| 换语料 | `python scripts/ingest.py --dir 你的目录` | 换语料之后**旧的评估数字全部作废** |

## 项目结构

```text
local-rag-agentic-lab/
├── data/                     # 你的语料（.gitignore 排除，只保留 README 说明放置规则）
├── datasets/k8s-zh-en-v1/    # 基准数据集：manifest(sha256) + NOTICE + 题集；docs/ 不随 git（走 Release）
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
│   ├── eval_bench.py         # 基准评估（走和 /ask 同一条链：retrieve_with_meta）
│   ├── build_bench_questions.py  # 出题：候选句 dump + 草稿 append（key → 原文 span）
│   ├── build_bench_dataset.py    # 数据集打包（可复现 tar.gz + manifest sha256）
│   ├── fetch_dataset.py      # 下载 / --verify / --from-dir
│   └── backfill_parent_index.py
├── experiments/              # 探针、历史题库、消融脚本
├── legacy/                   # 旧链（单文件索引 index.json + embeddings.npy），仅存档
├── tests/                    # stdlib unittest，985 用例（默认套件 34 s、不联网）
└── docs/                     # 设计文档 + 本基线 JSON + 实验档案（experiments-archive.md）
```

## 基准数据集

仓库**自带一份语料和一个题库**，目的只有一个：让「效果变好了」这句话有参照物。
你换自己的文档可以随便换，但**所有指标都只在这份固定基准上互相比较**——
换了语料、换了题集、换了数据集版本，历史数字一个都不再可比。

`datasets/k8s-zh-en-v1/` 是 Kubernetes 官方文档的**中英对照**子集：

| 项目 | 值 |
|---|---|
| 上游 | [`kubernetes/website`](https://github.com/kubernetes/website) @ `380372f60632ed7ce70516ecaaab4710ffb8703a`（钉死 commit，不是 `main`） |
| 许可证 | **CC BY 4.0**（[`LICENSE-CC-BY-4.0`](datasets/k8s-zh-en-v1/LICENSE-CC-BY-4.0) + [`NOTICE`](datasets/k8s-zh-en-v1/NOTICE)，版权归 The Kubernetes Authors） |
| 选页规则 | 中英**同路径交集**页面 ∩ `concepts` / `tasks` / `tutorials` / `setup`；**排除 `reference`**（1,070 页自动生成的 API 字段清单，成段重复、不适合出题） |
| 规模 | **472 页 × 2 语言 = 944 个 Markdown**，正文 8.5 MB（压缩包 2.5 MB） |
| 格式归一 | 只剥 front matter、HTML 注释（`<!-- reviewers/overview/body -->`）和 Hugo 短代码，**没有改写任何一句正文**；见 `sources.json` 的 `transform` |
| 建库后 | 883 篇索引成功、61 篇空壳 `_index.md` 跳过（切分后无正文）→ **7,108 父块 / 9,795 子块**，本机耗时 453 s |
| 可校验 | `sources.json` 记 944 个文件的 sha256 + 压缩包的 `archive_sha256`，`--verify` 逐文件比对 |

```bash
python scripts/fetch_dataset.py             # 从 Release 下载并解包到 datasets/k8s-zh-en-v1/docs
python scripts/fetch_dataset.py --verify    # 逐文件校验 sha256（也抓「缺文件 / 内容变了 / 多文件」）
python scripts/fetch_dataset.py --from-dir D:/k8s-website   # 直接从本地克隆取（离线场景）
```

**题集**：`datasets/k8s-zh-en-v1/questions.jsonl`，**144 题**（覆盖 43 页文档，每页 ≥3 题）。

| 维度 | 分布 |
|---|---|
| 正 / 负样本 | 139 / **5**（负样本 = 语料里确实没有答案的问题，正确行为是弃答） |
| 题型 | `single_hop` 85 / `multi_hop` 11 / `numeric` 12 / `cross_lingual` 31 / `negative` 5 |
| 证据语言 | 英文 76 / 中文 63 |
| 跨语言方向 | 中文问→英文证据 16、英文问→中文证据 20（**两个方向都有**） |

出题纪律（都由用例守护，见 `tests/test_bench_questions.py`）：

- 题目**改写**，不照抄原文；`gold_spans` 必须是语料**原文**（由脚本从候选句 key 取，不手抄，所以不可能抄错字）；
- `query_en` **人工撰写**，不允许机翻回填；
- 一句 gold 命中超过 4 个块就算 ambiguous，直接拒收；
- 负样本必须留痕（`note` 以「负样本」开头），且不许带 gold；
- `gold_chunks`（父块，检索粒度）与 `gold_points`（子块，向量点）**是两个不相交的 id 空间**，用例断言前缀不许混。

```bash
# 出题（要能连上建好的基准库）
python scripts/build_bench_questions.py dump --pages 60 --per-page 6      # 导出候选句
python scripts/build_bench_questions.py dump --only-pages 页清单.txt     # 只抽指定页（扩题用）
python scripts/build_bench_questions.py append --drafts 草稿.jsonl --merge --dry-run  # --merge：只加不重写
python -m unittest tests.test_bench_questions tests.test_bench_dataset    # 32 个用例守护题集与数据
```

**数据本体不进 git**（`datasets/*/docs/` 被 `.gitignore` 排除）：走 Release 资产 + sha256 校验。
打包是**可复现**的（固定文件排序 + `mtime=0` + 属主 0 + `gzip -n`），两次打包 sha256 完全相同——
否则 manifest 里的 `archive_sha256` 永远对不上。数据集**只增不改**：内容一变就是 `v2`，旧版本永久保留。

## 基准结果

**当前唯一有效的数字**。产出物：`docs/baseline-k8s-v1.json`（汇总）+ `docs/baseline-k8s-v1.rows.jsonl`
（逐题，含每题检出了什么、首次命中排第几——排查错误直接看这个）。

### 口径

先讲清分母，否则数字没法读：

| 指标 | 定义 | 分母 |
|---|---|---|
| `hit@k` | 至少命中一条 gold | 全部正样本 |
| `full@k` | gold **全部**命中（multi_hop 只中一半算失败） | 全部正样本 |
| `recall@k` | 逐题 `|gold∩top-k| / |gold|` 的均值 | 全部正样本 |
| `MRR` | 首次命中排名的倒数均值，**未命中记 0** | 全部正样本 |

三条纪律：

1. **四个一起报**。只报一个综合分，会把「差一点」和「全跑偏」混成一个数；
2. **负样本不进 recall 分母**——它们的正确行为是弃答，混进去等于用「不回答」拉高召回；
3. **报延迟前先热身**（第一条 query 是冷调用，会把 p50 带偏几倍）。

### 结果（`k=10`、`pool=40`、含重排）

| 切片 | n | `recall@10` | `hit@10` | `full@10` | `MRR` |
|---|---|---|---|---|---|
| **全部（正样本）** | 139 | **0.8297** | **0.8345** | **0.8273** | **0.5648** |
| 英文证据 | 76 | 0.8860 | 0.8947 | 0.8816 | 0.5740 |
| 中文证据 | 63 | 0.7619 | 0.7619 | 0.7619 | 0.5538 |

按「提问语言 × 证据语言」拆开，问题就不是「中文差」这么模糊了：

| 提问 → 证据 | n | `hit@10` | `MRR` | 读法 |
|---|---|---|---|---|
| 英 → 英 | 54 | **0.9630** | **0.6817** | 命中最高、排序最稳 |
| 中 → 中 | 54 | 0.7963 | 0.6332 | 命中略低，排序和英英持平 |
| 英 → 中 | 9 | 0.5556 | 0.0778 | 找得到，但基本排不到前面 |
| 中 → 英 | 22 | 0.7273 | 0.3096 | **短板**：跨语言去英文池里捞 |

> 提问语言按「题干里有没有中日韩字符」判定（`tests/test_bench_questions.py` 用同一规则重算，
> 不抄手写数字）。所以英文题里引了中文术语的那几道会算进中文提问。

按题型：

| 题型 | n | `hit@10` | `full@10` | `MRR` |
|---|---|---|---|---|
| `multi_hop` | 11 | 0.9091 | 0.9091 | 0.8030 |
| `single_hop` | 85 | 0.8588 | 0.8471 | 0.6096 |
| `numeric` | 12 | 1.0000 | 1.0000 | 0.8625 |
| `cross_lingual` | 31 | 0.6774 | 0.6774 | **0.2423** |

**延迟**（已热身）：p50 **1565 ms**、p90 **1688 ms**（检索 + 重排，不含生成）。其中重排占 ~1.1 s：关掉重排 p50 约 64 ms，但命中率会明显下降（`--no-rerank` 可复现）。
首次命中排名分布：rank1 63 题、rank2 16、rank3 7、rank4–10 28、未命中 23。

**没测的**：答案层忠实度（`faithfulness`）需要人工判读，这里不给数——自动打分的
`faithfulness` 是自欺。引用质量（可解析率 / 越界率）逐次在 `/ask` 响应里返回，不在本基线里。

### 怎么读这张表

- **跨语言仍是最大短板**：`中→英` 0.7273、`英→中` 0.5556，而 `英→英` 是 0.9630。
  C0c 已把「谁拖后腿」查清（见下一条）：**不是翻译，是重排**；
- **翻译这一档要显式选**：`--query-en` 有三档 —— `translate`（生产口径，走和 `/ask` 同一个翻译服务）、
  `field`（用手写 `query_en`，翻译上界）、`none`（英文两路吃中文原句，修 N47 前的旧口径）。
  本表的数字是 `translate`。三档对比见 `docs/evidence/query-en-modes-k8s-v1.json`：
- **`hit` 高而 `MRR` 低**的地方（`英→中` 100% vs 0.28）说明「进得了 top-10，但排不到前面」——
  对 RAG 来说这比「完全没召回」更值得修，因为组装时它会掉出预算；
- **别拿百分比跟别的项目比**。换语料、换切分、换数据集版本之后，这些数一个都不再可比。

### 复现

```bash
# 1) 建到独立集合 + 独立文档表（别碰正式库）
RAG_COLLECTION=bench_k8s_v1 python scripts/ingest.py --dir datasets/k8s-zh-en-v1/docs \
    --document-db storage/bench/documents.sqlite3 \
    --versions storage/bench/index_versions.sqlite3 --cache storage/bench/embeddings.sqlite3

# 2) 跑评估（写汇总 JSON + 逐题 JSONL）
python scripts/eval_bench.py --k 10 --out docs/baseline-k8s-v1.json

# 3) 改任何参数后，都跟这份基线比，而不是跟记忆比
python scripts/eval_bench.py --k 10 --no-rerank --out /tmp/ab-norerank.json

# 4) 改了切分/参数要重建库时：建到**另一个**带随机后缀的集合，再指过去评估
#    （别覆盖唯一一份基线数据；注意改切分会让部分 chunk id 变化，gold_points 需重算）
RAG_COLLECTION=bench_k8s_v1_ab12 python scripts/ingest.py --dir datasets/k8s-zh-en-v1/docs \
    --document-db storage/bench/ab12/documents.sqlite3 --state storage/bench/ab12/state.sqlite3 \
    --versions storage/bench/ab12/index_versions.sqlite3 --cache storage/bench/embeddings.sqlite3
python scripts/eval_bench.py --k 10 --collection bench_k8s_v1_ab12 \
    --store storage/bench/ab12/documents.sqlite3 --out /tmp/ab12.json

# 5) 重建之后题集里的 gold id 会失效（id 由内容+版本派生），按 gold_spans 重算回来
python scripts/build_bench_questions.py refresh-ids --dry-run   # 先看会改几题
python scripts/build_bench_questions.py refresh-ids
```

评估脚本走的是**和 `/ask` 完全相同的一条链**（`retrieve_with_meta`），不另写一条——
否则测的就是另一套代码。

### 回归门禁：一条命令判「这次改动有没有把检索改坏」

```bash
python scripts/eval_bench.py --k 10 --compare docs/baseline-k8s-v1.json
echo $?                                   # 0 = 没掉出容差；1 = 有指标掉出容差
```

输出一张对照表 + 逐题计数：

```text
容差 = 基线一侧 95% bootstrap CI 半宽（MRR ±0.0698）；掉出容差才判红。

| 指标 | 基线 | 现在 | delta | 容差 | 判定 |
| hit@k    | 0.8345 | 0.9900 | -0.1555 | ±0.0612 | regression |
| MRR      | 0.5648 | 0.5648 | +0.0000 | ±0.0698 | ok |
逐题（首命中排名）：更好 0 · 更差 0 · 持平 95
```

三条设计取舍（都是踩过才定的）：

1. **容差不是拍脑袋的 1%**，而是**基线一侧**的 95% bootstrap CI 半宽。我们的指标在 139 个正样本上本身就抖
   ±0.06–0.07 量级，固定 1% 会把正常抖动全报成回归，最后没人看门禁。
   **代价**：小回归（1–5 个点）它抓不到 —— 那是「逐题更好/更差/持平」计数该看的事。
2. **不可比就不判红**：题集 / `k` / `pool` / 重排开关任一不同，打印 `not_comparable` + 原因并退 **0**。
   退出码非零在 CI 里等于「代码坏了」，而真实情况是「你换了题」。
3. **默认不写输出**：带 `--compare` 时不会覆盖 `docs/baseline-k8s-v1.json`；要存就显式 `--out`。

题集指纹（`questions_sha256`）写进基线 JSON —— 不认文件名、不认行数：改一个字、换一题，行数可能一样。

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

环境变量：`RAG_LOG_LEVEL=off|summary|detail|debug`、`RAG_LOG_DIR`、`RAG_LOG_TEXT=off`（只记 hash 不记正文）、`RAG_LOG_SAMPLE`。

## 许可证

**本仓库的源代码与文档是 Apache-2.0**（见 [`LICENSE`](LICENSE)）。选它而不是 MIT，
是为了那两条**明示的专利授权与专利报复终止**条款——企业法务不用自己推演能不能用；
你可以商业使用、修改、再分发，改动需标注并保留 [`NOTICE`](NOTICE)。

四件必须知道的事（细节都在 [`NOTICE`](NOTICE)）：

| 资产 | 许可证 | 说明 |
|---|---|---|
| `bge-m3` / `mmarco-mMiniLMv2` / Qwen 系列 | MIT / Apache-2.0 / Apache-2.0 | **运行时由你自己下载**（`ollama pull` / HuggingFace），本仓库不重新分发 |
| **PyMuPDF**（PDF 版面解析） | **AGPL-3.0 或 商业授权** | **不是默认依赖**：它只装在 `requirements-pdf.txt` 里，代码惰性导入。装它并对外提供服务 = 你的服务要遵守 AGPL；闭源商用请买 Artifex 授权，或只用 md/txt |
| `datasets/k8s-zh-en-v1/` 的内容 | **CC BY 4.0**（另一套许可证） | Kubernetes 官方文档，版权归 The Kubernetes Authors；可商用可再分发，但要按 CC BY 4.0 署名（`NOTICE` 已写好） |
| 你自己的语料 | — | **不携带**（版权 + 体积 + 与你的业务无关），只留 `data/README.md` 说明放置规则 |

为什么 PDF 这块要单独说：pypdf 只给「一串文本」，拿不到字号和坐标，标题层级与页眉页脚
剔除都无从判断；PyMuPDF 能拿到 `block → line → span` 三层结构，代价是 AGPL。
这个取舍写在 [`NOTICE`](NOTICE) 第三节，改动前请先读。

## Legacy：旧链（仅存档，不要用于新工程）

仓库早期版本用单文件索引（`storage/index.json` + `embeddings.npy` + 内存 BM25）跑通了 RAG 与 Agentic RAG 的全部教学实验（那些实验的过程与历史数值已归档到 [`docs/experiments-archive.md`](docs/experiments-archive.md)，**旧语料不在仓库里，那些数不可复现，只能当方法参考**）。新链改用 Qdrant + SQLite 之后，旧链被移到 `legacy/` 存档：

| 文件 | 说明 |
|---|---|
| `legacy/legacy_rag_cli.py` | 原来的 `src/rag.py`。**它被 `src/rag/` 包遮蔽，已无法作为 `src.rag` 导入**，所以移到这里并改绝对导入 |
| `legacy/legacy_agentic_rag.py` | 原来的 `src/agentic_rag.py`（受控 Agentic：模型可选检索/改写，有最大步数） |
| `src/build_index.py` `src/retrieve.py` `src/bm25.py` | 旧链的建库/检索/BM25，仍在 `src/`（[实验档案](docs/experiments-archive.md)里的教学实验要跑它），文件头标了 deprecated |

旧链需要先建索引，且指令与新链**不通用**：

```bash
python -m src.build_index        # 旧链建库（写 storage/index.json，慢）
python legacy/legacy_rag_cli.py  # 旧链问答
```

新链只需要记两条：`python scripts/ingest.py --dir data` 与 `python -m src.ask "问题"`。

## 重要说明

- **历史实验的百分比不是当前基线**：`docs/experiments-archive.md` 里的数字全部来自本项目自己的语料（1467 篇、81,027 chunk，未随仓库发布），换语料后不可比较；当前唯一可比的数字在 `docs/baseline-k8s-v1.json`。
- 本项目默认每次 `build_index` 都重建索引，方便实验；生产系统应使用增量更新、文档哈希、版本管理和删除策略。
- `bge-m3` 是独立的 Embedding 模型（1024 维、8192 上下文、多语言）。聊天模型与 Embedding 模型职责不同。
- 向量距离的数值取决于距离度量与向量库实现；不要跨实现直接比较绝对值。
- 索引分两个文件：`storage/index.json` 存元数据、`records`（id/source/chunk_id/text）和 `bm25` 倒排表；`storage/embeddings.npy` 存 float32 向量矩阵，第 i 行对应 `records[i]`。JSON 里存浮点文本会膨胀 3.3 倍，且必须整文件解析，所以向量单独用二进制。
- 换 Embedding 模型、chunk 参数或分词方案后必须重新 `build_index`；`index.json` 里的 `embedding_model`、`embedding_dim`、`tokenizer`、`n_records` 就是用来事后核对是否配套的。老格式索引（向量存在 `records[].embedding`、没有 `bm25` 字段）仍能读取，BM25 会现场建一次并缓存。
- 向量矩阵在进程内只加载一次（`retrieve._embedding_matrix` 按 `records` 对象身份缓存），`rag`/`agentic_rag` 启动时用 `warmup()` 提前加载，避免第一个问题出现冷启动尖峰。
- 建库参数可用环境变量调整：`EMBED_BATCH_SIZE`（默认 256）、`EMBED_400_RETRIES`（默认 3，Ollama 偶发 400 时的原地重试次数）、`CHUNK_SIZE`、`CHUNK_OVERLAP`；`RAG_DATA_DIR` / `RAG_STORAGE_DIR` 可把语料与索引指到别处，方便小批量试跑。实测 `bge-m3` 在 224–320 条/批都能跑，但约 30% 的请求会间歇性返回 400（runner 重启后 tokenize 代理连不上），所以 `build_index` 的策略是「400 先原地重试，仍失败再二分」。
- 检索结果对象不再携带 `embedding`，避免每条命中都复制上千维浮点数组。
- 不要把不可信文档中的指令当作 Agent 系统指令。当前实现只允许检索工具，不执行任意 shell、Python 或网络操作。
