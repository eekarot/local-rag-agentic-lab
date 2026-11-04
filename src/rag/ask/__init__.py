"""E07 产品层问答编排（CLI 入口见 `src/ask.py`，链路见 `pipeline.py`）。"""

from __future__ import annotations

from .pipeline import (
    DEFAULT_BUDGET,
    DEFAULT_K,
    DEFAULT_POOL,
    AskResult,
    answer_query,
    render_cli,
    retrieve_ranked,
)

__all__ = [
    "AskResult",
    "answer_query",
    "render_cli",
    "retrieve_ranked",
    "DEFAULT_POOL",
    "DEFAULT_K",
    "DEFAULT_BUDGET",
]
