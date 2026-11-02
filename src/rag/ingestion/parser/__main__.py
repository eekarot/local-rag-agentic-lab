"""`python -m src.rag.ingestion.parser` 入口（实现在 cli.py，避免 __main__ 被 import 两次）。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
