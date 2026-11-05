# ROADMAP：剩余工作项

> 版本：2026-11-05 · 与 `docs/RAG数据切分与跨语言检索技术方案.md`（v2.24）配套。
> 设计文档记「为什么这样做」，本文件记「还剩什么没做、怎么算做完」。
>
> **验收一律写成「命令 + 期望值」**，不写「优化一下」「完善一下」这种无法判定的条目。

## 0. 当前位置

```
分支 master · 34 个提交 · 无 remote · 无 tag
默认套件 python -m unittest discover -s tests → 909 用例 OK (skipped=50)，33.9 s（不联网）
全套 RAG_RUN_SLOW=1 … → 909 用例 OK (skipped=5)，346 s

已落地链路：
  解析 → 切分 → 父块 → bge-m3 向量 → Qdrant(38,865 点) + 英文 BM25
  → 三路召回 + RRF → ONNX 重排 → 引用体检 → 流式生成
  → CLI / HTTP 同步 / HTTP SSE / 浏览器聊天页
  → probes + doctor + 熔断降级 + 有界队列单飞
```

## A. v0.1.0 发布阻塞（6 项，约 2.3 天）

| # | 项 | 验收（命令 → 期望值） | 状态 |
|---|---|---|---|
| A1 | **O5 `/health` 复用 probes** | ✅ 已完成：六项探针 + `fix`；停库 → `status=down` 且 HTTP **200**；热请求 0.65–0.81 s（`tests/test_api_health.py` 7 用例）| ✅ |
| A2 | **O6 依赖/配置清单** | ✅ 已完成：`.env.example` 36 变量 / `requirements-dev.txt` / `tests/test_dev_env_manifest.py` 7 用例（集合相等 + 默认值不漂移 + qdrant 版本纪律）| ✅ |
| A3 | **E03 测试分层** | ✅ 已完成：默认 **909 用例 / 33.9 s / 不联网**（把 URL 指到死端口仍全绿）；`RAG_RUN_SLOW=1` 全套 346 s 全绿；8 个模块挂 `@slow`；分层守卫（AST）防回归 | ✅ |
| A4 | **O8 法务与 CI** | ✅ 已完成：`LICENSE`(Apache-2.0 全文) + `NOTICE`(模型归属 + PyMuPDF AGPL 例外) + `CONTRIBUTING.md` + `.github/workflows/tests.yml`(3 平台，只跑默认套件，URL 指向死端口)；`tests/test_legal_and_ci.py` 10 用例 | ✅ |
| A5 | **干净克隆验收** | ✅ 已完成：clone + 全新 venv（装依赖 67 s）+ 默认套件 **941 用例 / 17 s / 无 storage、无 pymupdf、无 .env**；README 第 3–7 步全部走通（建库、检索、生成、`/`、`/health` 六项、SSE） | ✅ |
| A6 | **tag `v0.1.0`** | 依赖 A1–A5 全绿；`git tag -l` 出现 `v0.1.0` | ⬜ |

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
| C1 | **D29 语言不平衡治理** | 中文语料为主时，英文路可能淹没中文证据 | 同语言加权或来源配额，报 Recall@k / hit@k / full@k 三项 + 逐题更好/更差/持平计数 |
| C2 | **D30 跨语言门槛核算** | 跨语言命中该不该算、门槛怎么定 | 给出「算 / 不算」两套口径下的数字，结论写入设计文档 |
| C3 | **D36 ANN Recall@K 基准** | 目前只有单点 0.98 观测 | ANN vs 精确 Top-K 脚本，`ef_search` 扫描表 |
| C4 | **D37 HNSW 参数与量化实验** | 精度/内存/延迟三角未量化 | 不同 `m` / `ef_construct` / 量化组合的对比表 |
| C5 | **D38 百万级压测** | 当前 38,865 点，未见规模效应 | 生成 100 万点 → ingest 耗时、RSS、p50/p95 检索延迟 |
| C6 | **D39 16GB 内存预算** | reranker 945 MB + 可用 5.9 GB，多进程会 OOM | 明确 `--workers` 上限推导过程，写成表 |
| C7 | **E04 日志/评估资产接新 pipeline** | `raglog` 与 L1 评估仍走旧链 | 新链跑一次评估 → 日志字段与旧链可对齐比较 |
| C8 | **E05 新基线重立 + 回归门禁** | 正式基线要等 v0.1.0 后、题集与检索器固定时立 | 基线文件 + 门禁（容差 = 基线一侧 CI 半宽），`not_comparable` 不返回非零退出 |

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

## 建议顺序

```
A2 → A1 → A3 → A4 → A5 → A6(tag v0.1.0)      发布线，不插队
→ B3 → B1 → B2（拆安装门槛）→ B4 → B5
→ D5（引用精度）与 C1/C2（语言平衡）并行
性能账（E 表）随时可捡，但都不阻塞发布
```

## 判定口径（不许临时改）

- 指标同时报 `Recall@k` + `hit@k` + `full@k`，且报「逐题更好/更差/持平计数」，不许只报综合分。
- 评估基线只在「同一检索器 + 同一题集」下可比；`not_comparable` 时不返回非零退出。
- 压测/单测一律用带随机后缀的 collection 名。
- 「删除文档」= Qdrant + Document Store + 状态表三方一致才算删。
- 降级判定只认异常类型（`CapabilityError` / `CircuitOpenError.capability`），不看错误文本。
