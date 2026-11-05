"""`python -m src.doctor` —— 自检入口。

真正的逻辑在 `src/rag/ops/doctor.py`（可测），这里只做入口：

    python -m src.doctor          # 人类可读（失败项带可复制的修复命令）
    python -m src.doctor --json   # 给脚本/CI 用，stdout 只留 JSON

退出码：`0` = ok / degraded（能检索），`1` = down（检索不可用）。
"""
from src.rag.ops.doctor import main

if __name__ == "__main__":
    raise SystemExit(main())
