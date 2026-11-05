# 贡献指南

> 这个仓库的每一行代码背后都有一条**被实测撑着的理由**。改动前请先读
> [`docs/ROADMAP.md`](docs/ROADMAP.md)（还剩什么、怎么算做完）和
> [`docs/RAG数据切分与跨语言检索技术方案.md`](docs/RAG数据切分与跨语言检索技术方案.md)
> （为什么这样做，含 24 条踩坑记录）。

## 1. 五条不可商量的规矩

| 规矩 | 为什么 |
|---|---|
| **测试先行**（RED → GREEN） | 先写一个会失败的用例，再让它通过。没有失败过的用例，你不知道它在守什么 |
| **默认套件必须 <60 s 且不联网** | 别人克隆下来第一件事就是跑测试；碰真实服务的用例挂 `@slow`（见 §3） |
| **实现与文档不一致时，先改文档并记 changelog** | 文档是开发标准，不是事后总结 |
| **每条结论都要能复现**（命令 + 期望值） | 「优化了一下」「应该更快了」不算结论 |
| **不给错药** | 报错信息/修复命令必须真的能跑；写错一条比不写更浪费时间（N17/N22 就是这类事故） |

## 2. 本地怎么跑

```bash
# 运行必备（Ollama + Qdrant + 词表，见 README 第 0–3 步）
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt -r requirements-dev.txt

# 默认套件：909 用例，约 34 s，不需要任何外部服务
python -m unittest discover -s tests

# 全套（真实 Qdrant + Ollama + 真词表 + 449 MB 重排模型），约 6 分钟
RAG_RUN_SLOW=1 python -m unittest discover -s tests

# 单跑一个文件 / 一个用例
python -m unittest tests.test_api_stream
python -m unittest tests.test_api_stream.StreamBasicsTest.test_event_order
```

**PDF 解析另需**（许可证不同，见 [`NOTICE`](NOTICE) 第三节）：

```bash
pip install -r requirements-pdf.txt      # PyMuPDF 是 AGPL-3.0/商业双授权，不装也能用 md/txt
```

## 3. 新增用例时：它该不该默认跑

判据只有一条——**它会不会碰外部世界**：

| 会碰 | 例子 | 做法 |
|---|---|---|
| 真实 Qdrant / Ollama | `make_client()`、`ollama_client.embed_texts()` | 挂 `@slow`，并用 `SERVER = run_slow() and _server_available()` 短路导入期探测 |
| 17 MB 真词表 / 449 MB 重排模型 | `TokenCounter()` 默认路径 | 挂 `@slow`，或改用临时构造的小词表 |
| 本地文件系统 | 临时目录、合成 PDF | 默认跑（用 `TemporaryDirectory`，不要写仓库目录） |
| 纯函数/假件 | 切分、融合、引用解析、HTTP 层（注入假 retriever） | 默认跑 |

`tests/test_suite_layering.py` 会用 AST 检查**导入期**有没有探真实服务（那会让「跳过」也变慢）。
你新写的用例如果这么干，会立刻红。

## 4. 加一个开关 / 改一个默认值时

会连带三处，缺一个 `tests/test_dev_env_manifest.py` 就红：

1. `src/...` 里读它（`os.getenv("RAG_XXX", "默认值")`）；
2. `.env.example` 里写一行（值必须与代码默认值一致，数值按数值比）；
3. 若属于「慢/联网」开关，同步 `tests/_slow.py`。

## 5. 改检索/切分/评估的行为时

- **指标同时报 `Recall@k` + `hit@k` + `full@k`**，并给出「逐题更好 / 更差 / 持平」计数；
  只报一个综合分等于把「哪一类题变好了」藏起来；
- **基线只在「同一检索器 + 同一题集」下可比**；不可比时输出 `not_comparable` 且**不返回非零退出码**；
- **不许凭 chunking 指标就改默认参数**（例如 `min_tokens`），必须做检索级消融（B07 就是这么被卡住的）；
- 压测/单测的 collection 名一律带随机后缀，跑完删掉（否则下次 `docker start` 要恢复几十个集合）。

## 6. 提交前自检

```bash
python -m unittest discover -s tests    # 必须 OK 且 <60 s
python -m src.doctor                    # 七行必须全 ✓（有服务时）
git status --short                      # storage/ logs/ data/ baseline*.json 不该出现在里面
```

提交信息请写**「做了什么 + 为什么 + 实测数字」**，例如：

```
A1/O5 /health 复用探针：六项 + fix + 永不 5xx（顺带修 N21 每次 Ollama 调用白等 2 s）
- 现象：/health 热请求 4.7 s，里面 embed 2150 ms
- 根因：Windows 上 localhost 先解析 ::1，Ollama 只听 IPv4（2083 ms vs 49 ms）
- 修法：每线程一个 requests.Session + 默认地址改 127.0.0.1
- 实测：/health 热 4.6→0.65 s，CLI setup 13.9→7.0 s
```

## 7. 不要提交的东西

`storage/`、`logs/`、`data/` 下的文档、`baseline*.json`、`reports/`、`probes/` 全部被
`.gitignore` 排除。仓库**不携带语料**（版权 + 体积 + 与你的业务无关），
只保留 `data/README.md` 说明放置规则。

## 8. 许可证

贡献即表示同意以 **Apache-2.0** 授权（见 [`LICENSE`](LICENSE)）。涉及第三方资产
（模型、PDF 解析库）的改动，请同步更新 [`NOTICE`](NOTICE)。
