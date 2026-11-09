# ROADMAP：剩余工作项

> 版本：2026-11-06 · 与 `docs/RAG数据切分与跨语言检索技术方案.md`（v2.37）配套。
> 设计文档记「为什么这样做」，本文件记「还剩什么没做、怎么算做完」。
>
> **验收一律写成「命令 + 期望值」**，不写「优化一下」「完善一下」这种无法判定的条目。

## 0. 当前位置

```
分支 main · 54 个提交 · remote origin（github.com/eekarot/local-rag-agentic-lab）· tag v0.1.0
默认套件 python -m unittest discover -s tests → 989 用例 OK (skipped=50)，30 s（不联网）
全套 RAG_RUN_SLOW=1 … → 真服务链路手跑（CI 只跑默认套件）

已落地链路：
  解析 → 切分 → 父块 → bge-m3 向量 → Qdrant + 英文 BM25
  → 三路召回 + RRF → ONNX 重排 → 引用体检 → 流式生成
  → CLI / HTTP 同步 / HTTP SSE / 浏览器聊天页
  → probes + doctor + 熔断降级 + 有界队列单飞
  → 基准数据集（k8s-zh-en-v1，944 文件）+ 100 题 + 基线（docs/baseline-k8s-v1.json）
```

## A. v0.1.0 发布阻塞（6 项，约 2.3 天）

| # | 项 | 验收（命令 → 期望值） | 状态 |
|---|---|---|---|
| A1 | **O5 `/health` 复用 probes** | ✅ 已完成：六项探针 + `fix`；停库 → `status=down` 且 HTTP **200**；热请求 0.65–0.81 s（`tests/test_api_health.py` 7 用例）| ✅ |
| A2 | **O6 依赖/配置清单** | ✅ 已完成：`.env.example` 36 变量 / `requirements-dev.txt` / `tests/test_dev_env_manifest.py` 7 用例（集合相等 + 默认值不漂移 + qdrant 版本纪律）| ✅ |
| A3 | **E03 测试分层** | ✅ 已完成：默认 **943 用例 / 33 s / 不联网**（把 URL 指到死端口仍全绿）；`RAG_RUN_SLOW=1` 全套 346 s 全绿；8 个模块挂 `@slow`；分层守卫（AST）防回归 | ✅ |
| A4 | **O8 法务与 CI** | ✅ 已完成：`LICENSE`(Apache-2.0 全文) + `NOTICE`(模型归属 + PyMuPDF AGPL 例外) + `CONTRIBUTING.md` + `.github/workflows/tests.yml`(3 平台，只跑默认套件，URL 指向死端口)；`tests/test_legal_and_ci.py` 10 用例 | ✅ |
| A5 | **干净克隆验收** | ✅ 已完成：clone + 全新 venv（装依赖 67 s）+ 默认套件 **941 用例 / 17 s / 无 storage、无 pymupdf、无 .env**；README 第 3–7 步全部走通（建库、检索、生成、`/`、`/health` 六项、SSE） | ✅ |
| A6 | **tag `v0.1.0`** | ✅ 已完成：`git tag -l` → `v0.1.0`（annotated，带能力边界）；`CHANGELOG.md` 写明包含/不包含/已知限制，版本号由 `ChangelogTest` 守护 | ✅ |

> A3 是 A5 的前置：不联网的套件才可能在 CI 上稳定跑绿。
> `E02`（FastAPI 四端点 + schema）实际已被 O3 / O4c / O4d 覆盖，随 A6 一并把文档状态改勾。

## B. 开源后可用性（O9–O13，约 3 天）

| # | 项 | 现状痛点 | 验收 |
|---|---|---|---|
| B1 | **O9 词表自动下载** | 现在必须手工把 17 MB `tokenizer` 放进 `storage/tokenizer`，新手第一步就卡 | 空 `storage/` 下 `python -m src.ask "x"` → 自动拉取并给出进度，或给出唯一一条可复制命令后 exit 0 |
| B2 | **O10 `RAG_QDRANT_MODE=local`** | 硬依赖 docker，安装门槛高 | `RAG_QDRANT_MODE=local` 下无容器也能 `ingest` + `ask`，且 `python -m src.doctor` 全绿 |
| B3 | **O11 `pyproject.toml` + console scripts** | 只能 `python -m src.ask`，装不成工具 | `pip install -e .` 后 `rag-ask` / `rag-doctor` / `rag-ingest` 可直接调用 |
| B4 | **O12 / E01 配置层收敛** | 18+ 个 `RAG_*` 散在 `os.getenv`，拼错静默、无范围校验 | 环境变量拼错 → 立刻报错并列出合法名字；`python -m src.config --show` 打印生效值与来源 |
| B5 | **O13 API Key + 配额** | HTTP **无鉴权**，只能绑 localhost | 无 key 请求 → 401；配有 key 后带 `Authorization` 通过；配额超限 → 429 |

## C. 检索与评估质量（D/E 系列）

| # | 项 | 说明 | 验收要点 |
|---|---|---|---|
| C0 | **基准数据集 + 第一版基线** | ✅ 已完成：`datasets/k8s-zh-en-v1`（Kubernetes 官方文档中英对照 472 页 × 2 语言，CC BY 4.0，上游 commit `380372f` 钉死）+ `questions.jsonl` **144 题**（139 正样本 + 5 负样本；覆盖 43 页、每页 ≥3 题；题型 single_hop 85 / multi_hop 11 / numeric 12 / cross_lingual 31 / negative 5）+ 基线 `docs/baseline-k8s-v1.json`：**recall@10 0.8297 / hit@10 0.8345 / full@10 0.8273 / MRR 0.5648**（生产口径 `--query-en translate`；延迟 p50 1565 ms）。历次重立：修 N45 后 0.8561/0.8632/0.8526/0.6075（95 题）→ 扩题到 139 个正样本 + 修 N47/N48 后为当前值；旧基线归档在 `docs/evidence/` | `python scripts/fetch_dataset.py --verify` → 944 文件一致；`python scripts/eval_bench.py --k 10 --query-en translate` → 落 baseline JSON + 逐题 rows；`python -m unittest tests.test_bench_questions tests.test_bench_dataset` → 31 OK | ✅ |
| C0b | **C0b：重排到底有没有用** | ✅ 已完成（2026-11-06）：修 N45（评测里重排一直喂空窗口）后重跑 —— 关重排 hit@10 0.7789 / MRR 0.5456、开重排（pool40）0.8632 / 0.6075，逐题 top-10 顺序确实变了；同时出 pool 20/40/60 与 k=20 消融（`docs/evidence/c0b-rerank-pool-ablation.json`） | `python scripts/eval_bench.py --k 10 --no-rerank --out /tmp/nr.json` → hit@10 显著低于含重排 | ✅ |
| C0c | **C0c：中→英 到底是谁拖后腿** | ✅ 已完成（2026-11-06）：三档口径对照 —— `none`（英文两路吃中文原句）中→英 hit 0.7273 / MRR 0.3096、`translate`（生产，真实翻译）相同、`field`（手写英文 query = 翻译上界）0.7500/0.3271，**只差 1 题 ⇒ 翻译不是瓶颈**。5 个漏检的 gold 在融合（关重排）里排 #1/#3/#4/#8/#14，重排后掉到 #16/#17/#31/#13/#14 **⇒ 瓶颈在重排**（重排拿中文 query 去给英文窗口打分）。副作用：真实翻译让中文证据题掉 2 题（中→中 0.8372→0.7963），整体 4 好 / 9 差 / 126 平 | `python scripts/eval_bench.py --k 10 --query-en field` → 与 `--query-en translate` 比，逐题更好 · 更差 · 持平 = 5 / 5 / 129（见 `docs/evidence/query-en-modes-k8s-v1.json`） | ✅ |
| C0d | **按候选块语言选重排 query** | 现在重排一律拿**中文原句**给英文窗口打分，这是 C0c 查出的真实瓶颈。试点：把重排 query 换成英文，中→英 的 x-009 从 #17 → #7、x-020 从 #16 → #3，但 x-010 从 #13 → #19 ⇒ 不能整题切换，要**按候选父块的语言**分别选 query | `python scripts/eval_bench.py --k 10 --compare docs/baseline-k8s-v1.json` → 中→英 hit@10 从 0.7273 提升且四条指标不掉出容差 | 2 |
| C0c | **中→英跨语言仍然最弱** | `中→英` hit@10 **0.6000** / MRR 0.1944（n=10），其余三格 0.75–0.95 | `python scripts/eval_bench.py --k 10 --limit 100` → 四格里的 `中→英` 行；改翻译或 dense_en 后看该行是否上升（题量 10，只认 ≥2 题的移动，别单题下结论） | ⬜ |

| C1 | **D29 语言不平衡治理** | 中文语料为主时，英文路可能淹没中文证据 | 同语言加权或来源配额，报 Recall@k / hit@k / full@k 三项 + 逐题更好/更差/持平计数 |
| C2 | **D30 跨语言门槛核算** | 跨语言命中该不该算、门槛怎么定 | 给出「算 / 不算」两套口径下的数字，结论写入设计文档 |
| C3 | **D36 ANN Recall@K 基准** | 目前只有单点 0.98 观测 | ANN vs 精确 Top-K 脚本，`ef_search` 扫描表 |
| C4 | **D37 HNSW 参数与量化实验** | 精度/内存/延迟三角未量化 | 不同 `m` / `ef_construct` / 量化组合的对比表 |
| C5 | **D38 百万级压测** | 当前 38,865 点，未见规模效应 | 生成 100 万点 → ingest 耗时、RSS、p50/p95 检索延迟 |
| C6 | **D39 16GB 内存预算** | reranker 945 MB + 可用 5.9 GB，多进程会 OOM | 明确 `--workers` 上限推导过程，写成表 |
| C7 | **E04 日志/评估资产接新 pipeline** | `raglog` 与 L1 评估仍走旧链 | 新链跑一次评估 → 日志字段与旧链可对齐比较 |
| C8 | **E05 新基线重立 + 回归门禁** | ✅ 已完成（2026-11-06）：基线重立两次（G2 修 overlap 超限 → 0.7737/0.7789/0.7684/0.5456；修 N45 重排空窗口 → **0.8561/0.8632/0.8526/0.6075**），旧基线归档在 `docs/evidence/`；门禁 `--compare` 已上线（容差 = 基线一侧 CI 半宽，不可比退 0） | `python scripts/eval_bench.py --k 10 --compare docs/baseline-k8s-v1.json` → 四条指标全 `ok`、逐题 0/0/95、退出码 **0**；把基线 hit@10 改成 0.99 再跑 → `regression`、退出码 **1**；`--no-rerank` 再比 → `not_comparable`（rerank 不同）、退出码 **0** | ✅ |

## D. 已知缺陷 / 明确 blocked

| # | 项 | 卡在哪 |
|---|---|---|
| D1 | **B07 `min_tokens` 200→400 消融** | 需随机后缀 collection + 全量 ingest ~2,000 s + 复跑 D28。**不能凭 chunking 指标改默认参数** |
| D2 | **N1 `language` 字段缺失** | 索引建了，38,865 点 payload 里没有该字段 → 无法按语言过滤 |
| D3 | **N9 PDF `source_start/end` 全 NULL** | PDF 无法定位到页/字符偏移 |
| D4 | **N10 `embedding_model` 空串** | payload 记不下「这份索引是哪个模型建的」 |
| D5 | **citation_precision 38.46%** | 真实水平。比任何新功能都更影响可用性：引用错比不给引用更糟 |
| D7 | **PyMuPDF 商业授权决策**（若产品要闭源商用 PDF 解析）| 技术侧已隔离（可选依赖 + 惰性导入），但要卖闭源产品必须二选一：买 Artifex 授权，或接受 AGPL 分发义务 |
| D6 | **Qdrant 里积了 42 个遗留测试集合**（共 45 个 collection，只有 `rag_chunks_v1` 是正式的）| 重启要恢复全部集合：实测 `docker start` 到可服务花 **>60 s**。需要：用例结束即删 + `scripts/cleanup_test_collections.py` |

## E. 性能账（已量化、未优化）

| 项 | 现状 | 目标 | 备注 |
|---|---|---|---|
| CLI 冷启动 setup | **13.5 s** | <3 s | 懒加载（25k 父块 + ONNX + Qdrant 装配都在启动路径上） |
| 翻译层 | 中文 query 常原样回填，白花 **~4 s** | 不需要翻译就跳过 | `question_en` 与原文相同即为信号 |
| 流式首字 | **8.4–9.7 s** | 结构上已改善体感（`sources` 4.99 s 先到） | 正文受 9B 模型限制，短时间难降 |

## F. 工程化与上手门槛（不含切片 / 检索算法）

> 这一节**只记工程与体验**：安装、首跑、文档、运维、贡献流程、平台适配。
> 改切分参数、换召回策略、调权重这类算法项不在本节（见 C 节）。
>
> 排序按「**别人能不能 5 分钟跑起来**」，不按实现难度。每条都给现状证据 +
> 验收（命令 → 期望值），否则不算做完。
>
> 与 B 节重复的项只记一次账：`B1`(词表自动下载) / `B2`(local 模式) / `B3`(pyproject) /
> `B4`(配置收敛) / `B5`(鉴权) 仍在 B 节维护，本节用「同 B3」指向，不复制。

### F-1 首次上手：从 clone 到第一条答案

| # | 项 | 现状证据 | 验收（命令 → 期望值） | 档 |
|---|---|---|---|---|
| F1 | **一条命令起全栈** | `docker-compose.yml` **只有 qdrant**；应用要手敲 `uvicorn`，还得自己记得 `--workers 1` | `docker compose up -d` → 60 s 内 `curl -s localhost:8000/health \| jq .status` = `"ok"`；compose 内两个服务都有 healthcheck | 2 |
| F2 | **仓库自带示例语料** | 干净克隆 `data/` 只有 `.gitkeep` → 第一条 `ingest` 就是「发现 0 个文件」 | `python scripts/ingest.py --dir samples` → 输出 `indexed>=5`；`python scripts/ingest.py --status` 显示库内点数 > 100；全程不联网 | 1 |
| F3 | **Release 资产自动上传** | `fetch_dataset` 默认从 Release 下载，但资产**目前是手传的**，一忘就 404 | push `bench-k8s-zh-en-v1` tag → `curl -sI <release asset url>` 返回 200；`python scripts/fetch_dataset.py --verify` → `OK：944 个文件全部与 manifest 一致` | 1 |
| F4 | **`.env` 自动加载** | README 自己写着「本仓库不自动读 `.env`」——用户 `cp` 完不生效且**没有任何提示** | 写一个 `.env` 设 `RAG_LOG_LEVEL=off` → `python -m src.doctor` 打印「已读取 .env（1 项生效）」；`python -c "print(os.environ['RAG_LOG_LEVEL'])"` 走 CLI 时应为 `off` | 1 |
| F5 | **依赖可复现** | `requirements.txt` 除 `qdrant-client` 外全是 `>=` → 今天能装，三个月后装出来的不是同一个环境 | 有 `requirements.lock`；CI 用 lock 装；`pip install -r requirements.lock` 两周一跑仍 `989 用例 OK` | 3 |
| F6 | **Python 版本声明对齐** | README 写「3.10+」，CI 只跑 3.12 —— 没验证过的承诺 | CI 矩阵加 `3.10` / `3.11` → 两个新格子 success；否则把 README 改成「3.12」 | 3 |
| F7 | **一键 bootstrap** | Windows / macOS / Linux 各 5 步，第 3 步还要 curl 一个 17 MB 文件 | `scripts/bootstrap.ps1` 与 `scripts/bootstrap.sh` 裸机跑完 → `python -m src.doctor` 退出码 0 且六项全 ✓ | 2 |
| F8 | **`pyproject.toml` + console scripts** | 同 B3 | 同 B3 | 1 |
| F9 | **免 Docker 模式** | 同 B2（最大安装门槛就是 Docker Desktop） | 同 B2 | 2 |

### F-2 让人不用读完 1500 行 README

| # | 项 | 现状证据 | 验收（命令 → 期望值） | 档 |
|---|---|---|---|---|
| F10 | **`doctor --fix` 真执行** | 现在只**打印**命令，用户还得复制粘贴 | `python -m src.doctor --fix --dry-run` → 列出将执行的 N 步；`--fix` 真跑，结束时 `python -m src.doctor` 退出码 0 | 2 |
| F11 | **`rag init` 引导** | 现在要用户自己决定 6 件事（语料 / 模型 / 是否起服务 / 集合名…） | `rag init` 问 3 个问题 → 写 `.env` + 起依赖 + 建库，结束打印 `打开 http://127.0.0.1:8000/`；`curl -s localhost:8000/health` = ok | 2 |
| F12 | **English README** | 全中文 —— 开源项目第一道门槛 | `README.en.md` 存在且覆盖：what / 3 条命令跑起来 / 基准数字 / 许可证 → `python -m unittest tests.test_readme_en` 通过（命令不许假） | 2 |
| F13 | **首屏有画面** | 现在 README 全是表格和代码块，**没有一张图** | `docs/images/` 下 3 张截图（聊天页 / `/retrieve?debug=true` / doctor 全绿）+ 1 个 ≤20 s 终端录屏 → README 首屏引用它们，`git ls-files docs/images` 有 4 个文件 | 1 |
| F14 | **5 分钟 quickstart** | 「快速开始」实际要 15 分钟 + 4 个外部前置（Docker / Ollama / 词表 / 模型） | `docs/quickstart-5min.md`：只用 compose + `samples/`，不碰词表下载 → 找个没跑过的人照做，`time` 显示 ≤5 min 且 `curl -s localhost:8000/health` = ok | 2 |
| F15 | **可直接跑的请求集** | 只有零散 `curl` 行 | `examples/api.http`（REST Client 可点）+ Postman collection → 每个端点 2 个示例，`httpyac send examples/api.http` 每个示例返回 2xx | 3 |
| F16 | **排障单独成页** | 常见故障只有 9 行表格 | `docs/troubleshooting.md` 按**症状**索引（端口占用 / 模型没拉 / 编码乱码 / 下载卡住 / 版本不匹配）→ 每条一个可复制命令，`tests/test_ops_probes.py::FixCommandsExistTest` 扩到扫这份文档后全绿 | 2 |

### F-3 运维与观测（别人接手要能看、能备、能恢复）

| # | 项 | 现状证据 | 验收（命令 → 期望值） | 档 |
|---|---|---|---|---|
| F17 | **`/metrics`** | 只有 `/health`（拉模式、给人看），Prometheus 抓不到 | `curl -s localhost:8000/metrics` → 出现 `# TYPE ask_latency_seconds`、`queue_inflight`、`degraded_total`；跑一个问题后计数 +1 | 2 |
| F18 | **应用日志落盘 + 轮转** | `raglog` 只服务**旧链**；新链日志在 stdout 里 | 设 `RAG_LOG_DIR=logs/app` → 跑一次 `/ask` 后该目录出现 `app-YYYYMMDD.jsonl`；`python -m src.raglog_report latest --failures` 对新链 run 也有效 | 2 |
| F19 | **镜像发布** | 没有镜像，只能源码跑 | `docker run -p 8000:8000 ghcr.io/eekarot/local-rag-agentic-lab:v0.1.0` → `docker image ls` 显示 <2 GB，容器内 `curl -s localhost:8000/health` = ok | 2 |
| F20 | **备份 / 恢复脚本** | README 让用户手 `cp -r storage`，还强调「只拷一个就残缺」——知道该拷却不给命令 | `rag backup out.tar.gz` → 含 `storage/qdrant` + `documents.sqlite3` + `ingest/`；`rag restore out.tar.gz` 后 `/health` 的父块数与备份前一致 | 3 |
| F21 | **遗留集合清理**（同 D6） | 45 个 collection，Qdrant 启动要约 2.5 min | `python scripts/cleanup_test_collections.py --dry-run` 列出 43 个 → 执行后 `curl -s localhost:6333/collections` 只剩正式集合；重启 Qdrant <10 s | 1 |
| F22 | **鉴权 + 配额** | 同 B5 | 同 B5 | 2 |
| F23 | **冷启动** | CLI setup 13.5 s、API 首请求 15–18 s（父块 25k + ONNX + Qdrant 都在启动路径） | 父块/ONNX 移后台线程预热 → `python -m src.ask --retrieve-only "x"` 计时 <3 s；`/health` 在 1 s 内可响应 | 3 |

### F-4 贡献者与 CI（决定这仓库能不能被别人改）

| # | 项 | 现状证据 | 验收（命令 → 期望值） | 档 |
|---|---|---|---|---|
| F24 | **把「干净克隆验收」固化进 CI** | 那次是**手工**做的，N26–N30 全靠它抓出来 | 新 job：`python -m venv` → 装依赖 → `python -m unittest discover -s tests` → `python scripts/fetch_dataset.py --verify`；故意改坏 README 路径 → job 必红 | 1 |
| F25 | **lint + type + coverage** | 989 用例，**零 lint、零类型检查、不知道覆盖率** | CI 跑 `ruff check .` / `mypy src/rag` / `coverage report` → 三个 job 绿，且覆盖率门槛写进 job 名（如 `coverage>=72%`），故意写一个未用变量 → ruff 必红 | 3 |
| F26 | **issue / PR 模板 + dependabot** | `.github/` 只有 1 个 `tests.yml` | `.github/ISSUE_TEMPLATE/bug.md`、`feature.md`、`PULL_REQUEST_TEMPLATE.md` 存在 → 新建 issue 时 GitHub 出现模板选择页；dependabot 每周开 PR 且改的是 lock 文件 | 3 |
| F27 | **pre-commit** | 全靠自觉 | `.pre-commit-config.yaml`：ruff / 行尾 / 大文件拦截 → `pre-commit run --all-files` 退出码 0 且无 diff | 3 |
| F28 | **任务入口** | 常用命令散在 README 各处 | `Makefile`：`make test` / `make bench` / `make ingest` / `make clean` → 四个目标逐个 `make <target>` 都能跑通（`make test` 打印 `OK (skipped=50)`） | 1 |
| F29 | **慢测夜间档** | 真服务链路只能手跑 | nightly job 跑 `RAG_RUN_SLOW=1 python -m unittest discover -s tests` → 连续 7 天全绿，失败时自动开 issue（Actions 里能看到记录） | 3 |
| F30 | **发版自动化** | tag → CHANGELOG → 附件 → push **全手工** | `git tag vX.Y.Z && git push --tags` → CI 自动打包、建 Release、上传资产、校验 `archive_sha256` | 3 |

### F-5 平台与离线

| # | 项 | 现状证据 | 验收（命令 → 期望值） | 档 |
|---|---|---|---|---|
| F31 | **Windows 专项** | README 里 `export` 与 PowerShell 混着写、`curl -N`、`taskkill` 各显神通 | 一份 `docs/windows.md` + `scripts/bootstrap.ps1` → 干净 Windows 照文档走完，`python -m src.doctor` 退出码 0 | 2 |
| F32 | **离线 / 内网安装** | 只提了 `HF_ENDPOINT` 镜像 | `docs/offline.md`：`pip download` + 预置 `storage/tokenizer` + `HF_HUB_OFFLINE=1` → 断网机器上走完一遍，`curl -s localhost:8000/health` = ok | 3 |
| F33 | **镜像里预置模型** | 首跑要下 449 MB 重排模型（还可能被墙） | 提供 `-full` 镜像 tag（含词表 + 449 MB 重排模型）→ 断网机器 `docker run` 后第一次 `/ask` 直接出答案（日志里无下载行）；slim tag 保持 `--no-rerank` 可用 | 3 |
| F34 | **ARM64 / Apple Silicon 说明** | 未验证、未声明 | 平台页给出 `onnxruntime` 平台表 + 一行实测（「M2 上 p50 xx ms」）→ 或者在 README 明确写「仅验证 x86_64」，二选一必须落到文档里 | 3 |

### F-6 评估产物的可读性（不改算法，只改「看得懂」）

| # | 项 | 现状证据 | 验收（命令 → 期望值） | 档 |
|---|---|---|---|---|
| F35 | **HTML 报告** | 现在只有 JSON + 逐题 JSONL，得自己写脚本看 | `python scripts/eval_bench.py --html out.html` → 单文件，失败题聚合、可点开看该题检索结果；不需要外网 | 3 |
| F36 | **一条命令出「基线 vs 现在」对比表**（同 C8） | ✅ 已完成：`scripts/eval_bench.py --compare`（容差 = 基线一侧 95% bootstrap CI 半宽、不可比退 0、默认不覆盖基线） | `python scripts/eval_bench.py --k 10 --compare docs/baseline-k8s-v1.json` → Markdown 对照表 + 「更好/更差/持平」计数 + 退出码 0；基线 hit@10 造高到 0.99 再跑 → `regression` + 退出码 1 | 1 |
| F37 | **bench profile** | 建基准库要手记 4 个路径参数（易写错，还可能污染正式库） | `docker compose --profile bench up` 或 `make bench` 一条命令完成「建库 → 评估 → 落 baseline」 | 2 |

### 建议动手顺序（工程线）

```
第一档（半天，收益最大）：F2 示例语料 → F4 .env 自动加载 → F8 console scripts
  理由：F2 把「第一条命令就报错」变成「第一条命令就出货」；
        F4/F8 是每个用户都会碰到的，且一下午能全做完

第二档（一天）：F1 compose 全栈 → F10 doctor --fix → F14 5 分钟 quickstart
  理由：三件合起来才是「clone → 一条命令 → 浏览器里有答案」

第三档（一天）：F24 干净克隆进 CI → F25 lint/coverage → F5 锁文件
  理由：先把门槛降低的成果锁住，否则下一个人改回去没人发现

随时可捡：F21 清集合（半小时，顺手把 Qdrant 启动从 2.5 min 拉回几秒）、
          F20 备份脚本、F28 Makefile、F13 截图
```

## G. 切片护栏与体检（借鉴 `adaptive-chunking`）

> 完整方案：[`docs/adaptive-chunking-integration-plan.md`](adaptive-chunking-integration-plan.md)
> （上游 `github.com/ekimetrics/adaptive-chunking`，LREC 2026，MIT —— 只做只读参考，不引入依赖）
>
> 一句话：**借思想不借代码**。上游核心依赖 torch / `maverick-coref`(CC BY-NC-SA) 与本仓库
> B08（只用 onnxruntime）、D39（16 GB 预算）冲突；金标也太薄（2.21 题/页），逐文档自适应选择不做。
>
> 本节只记「要做的事 + 怎么算做完」，判定口径见方案 §6。
>
> **进度（2026-11-06）**：P0–P2 已落地（G0–G4 ✅，43 个新用例：质量指标 20 + 不变量 9 +
> 体检命令 8 + 建库前体检 6）。P3/P4 等题集扩到 250+，P5（Docling）是旁路试验。

| # | 项 | 现状证据 | 验收（命令 → 期望值） | 状态 |
|---|---|---|---|---|
| G0 | **P0 冻结现状** | ✅ 已完成：`docs/evidence/chunk-quality-k8s-v1.json`（**944 篇 / 9,795 子块**；`sc_mean` 1.0、`bi_mean` **0.9607**、`bi<0.95` 的 **286 篇**、`bi=1.0` 的 270 篇、**丢字 0 篇**）；最低的 5 篇人工看过，都是短小节页（`_index.md` 与 `ingress-controllers.md`），确为真切断 | `python scripts/chunk_quality.py --dataset k8s-zh-en-v1 --json docs/evidence/chunk-quality-k8s-v1.json` → 同上数字（80 s） | ✅ |
| G1 | **P1 无缝隙覆盖不变量** | ✅ 已完成：`tests/test_ingest_invariants.py::NoTextLossTest` 4 用例（区间铺满 + 内容与偏移逐字对齐 + 短文 + 真实语料）；实现 `find_coverage_gaps()`。**变异验证**：把 `end` 减 1 字符 → 4 个用例全红 | `python -m unittest tests.test_ingest_invariants` → `OK`（9 用例）；改成少切 1 字符 → `FAILED (failures=4)` | ✅ |
| G2 | **P1 overlap 上限语义钉死** | ✅ 已完成且**逮到真问题**：`apply_overlap` 的默认 `max_overlap_tokens=70` 大于目标 60 → 基准库实测 **5/9,795 个子块超 500 token（501–505）**。定为 `chunk_size` 语义（含 overlap ≤ 上限），修法 = 收敛 overlap 上限 + `_fit_front()` 二分兜底裁剪。修复后 `sc_mean = 1.0`（0 块越界）。**检索端已做 A/B**（`docs/evidence/chunk-quality-ab-k8s-v1.json`）：A/A 两遍逐题完全一致；A/B 的 recall/hit/full **逐题不变**、MRR −0.0073（2 好 4 差 94 平）；同时发现 **1,067/9,795 个子块换了 id（10.9%）**，1 个 `gold_points` 成孤儿（N42） | `python scripts/chunk_quality.py --dataset k8s-zh-en-v1` → `sc 均值 1.0`；把 `split_max` 改回 `child_max_tokens` 并去掉裁剪 → `tests.test_ingest_invariants` 红（实测 147 > 120） | ✅ |
| G3 | **P1 SC + BI 指标** | ✅ 已完成：`src/rag/ingestion/quality_metrics.py`（`size_compliance` / `block_integrity` / `find_coverage_gaps`，纯整数区间，不碰 tokenizer 与向量库）+ `tests/test_quality_metrics.py` **20 用例**（含「一块被切两次只算一次」「容差内不算切断」「空输入给 None 不给 0%」） | `python -m unittest tests.test_quality_metrics` → `OK`（20 用例） | ✅ |
| G4 | **P2 体检命令**（面向用户，无金标也能用） | ✅ 已完成：`scripts/chunk_quality.py`（`--dir`/`--dataset`/`--metrics`/`--threshold`/`--limit`/`--json`/`--strict`）+ 编排层 `src/rag/ingestion/quality_inspect.py`；`ingest` 新增 `--precheck` / `--precheck-only` / `--precheck-strict` / `--precheck-threshold`（建库前一道闸，**不连 Qdrant、不写库**）。缺词表给 `curl` 不给堆栈；空目录退 0 | `python scripts/chunk_quality.py --dir data/md --strict` → 打印 3 指标；`python scripts/ingest.py --dir data --precheck-only`（`QDRANT_URL` 指死端口）→ 报告出现且**无 Qdrant 报错**，退 0；`tests.test_chunk_quality_cli` 8 用例 + `tests.test_ingest_precheck` 6 用例 | ✅ |
| G5 | **P3 结构桶对照实验** | 883 篇里 590 篇有代码块（66.8%）、87 篇有表格（9.9%）；但 95 正样本只盖 43 页（2.21 题/页） | 题集 ≥250 题 / ≥120 页后：`eval_bench --compare docs/baseline-k8s-v1.json` → `hit@10 +≥2 题`、`MRR 降 <0.02`、三计数「更好 > 更差」，三条全中才采纳 | ⬜ |
| G6 | **P4 与 B07 合并消融** | D1（`min_tokens` 200→400）要一次全量重建，别付两遍 2,000 s | 一张表同时给出 min_tokens 与结构保护的**单变量**效应 | ⬜ |
| G7 | **P5 Docling 评估（解 D7 + N9 的候选）** | PDF 走 PyMuPDF（AGPL-3.0，D7 卡着），且 `source_start/end` 全 NULL（N9）；Docling 实查 PyPI = **MIT** | `python scripts/probe_docling.py --pdf-dir data/pdf_md --limit 20` → 每篇给出 `has_bbox: true` + 耗时；`pip list \| grep -i pymupdf` 仍为空（不引入 AGPL） | ⬜ |

**明确不做**（理由见方案 §4-C）：逐文档自适应选择（金标 2.21 题/页）、语义切分（上游实测 SC 48.1 最差）、
RC 指标（`maverick-coref` CC BY-NC-SA + 仅英文代词）、LLM regex 切分（需 OpenAI）、抄上游 1100/600/100（口径不同）、
搬上游数字当基线（不同检索器）。

## 建议顺序

```
A2 → A1 → A3 → A4 → A5 → A6(tag v0.1.0)      发布线，已全部完成
→ F2 → F4 → F8（工程第一档：示例语料 / .env / console scripts）
→ F1 → F10 → F14（工程第二档：一条命令起全栈 / doctor --fix / 5 分钟 quickstart）
→ F24 → F25 → F5（工程第三档：把成果锁进 CI）
→ B1 → B2（拆安装门槛）→ B4 → B5
→ G0 → G1 → G2 → G3 → G4（切片护栏与体检，不改默认参数，1.5 天）
→ D5（引用精度）与 C1/C2（语言平衡）并行
→ G5 → G6（结构桶实验，等题集扩到 250+）· G7（Docling，可能一箭双雕解 D7+N9）
性能账（E 表）随时可捡，但都不阻塞发布
```

工程线（F 节）与安装线（B 节）**不重叠**：F 管「跑起来顺不顺、看得懂看不懂、能不能被人改」，
B 管三个具体功能（词表自动下载 / local 模式 / 鉴权等）。F 先做，因为它的每一项都会被每个新用户碰到。

## 判定口径（不许临时改）

- 指标同时报 `Recall@k` + `hit@k` + `full@k`，且报「逐题更好/更差/持平计数」，不许只报综合分。
- 评估基线只在「同一检索器 + 同一题集」下可比；`not_comparable` 时不返回非零退出。
- 压测/单测一律用带随机后缀的 collection 名。
- 「删除文档」= Qdrant + Document Store + 状态表三方一致才算删。
- 降级判定只认异常类型（`CapabilityError` / `CircuitOpenError.capability`），不看错误文本。
