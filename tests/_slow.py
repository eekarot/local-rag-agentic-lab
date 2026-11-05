"""慢且联网的用例分层开关：默认套件不跑它们。

为什么需要这一层（不是整理洁癖，是两次真实事故的结论）：

- **N16/N20**：全套件偶发 1 例 ERROR，全是「连真实 Qdrant」的用例在系统代理 /
  服务抖动时超时。它们验证的其实是「Qdrant 还活着」，保护价值远低于带来的不稳定；
- **发布门槛**：干净克隆验收与 CI 都要求「默认套件 <60 s 且不联网」，否则
  别人克隆下来第一件事就是等 6 分钟，还会因为没起 Qdrant 看到一片红。

分层规则只有三条：

1. **默认套件只跑纯单测**（假件、临时目录、内存模式），不许出现真实网络调用；
2. 碰真实 Qdrant / Ollama / 17 MB 真词表 / 449 MB 重排模型的用例，一律挂 `@slow`；
3. 开关是 `RAG_RUN_SLOW=1`（老的 `RAG_OLLAMA_CHECK=1` 继续认，兼容旧文档）。

```bash
python -m unittest discover -s tests                      # 快，默认
RAG_RUN_SLOW=1 python -m unittest discover -s tests       # 全套，需 Qdrant + Ollama
```
"""

from __future__ import annotations

import os
import unittest


def run_slow() -> bool:
    """是否跑慢/联网用例。`RAG_OLLAMA_CHECK` 是历史开关，继续认。"""
    return os.getenv("RAG_RUN_SLOW") == "1" or os.getenv("RAG_OLLAMA_CHECK") == "1"


SLOW_HINT = ("慢/联网用例已跳过（默认套件不联网）。要跑：RAG_RUN_SLOW=1 "
             "python -m unittest discover -s tests")

#: 挂在 TestCase 上：`@slow`
slow = unittest.skipUnless(run_slow(), SLOW_HINT)
