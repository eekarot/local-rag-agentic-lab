"""检索层：三路召回（D24）+ RRF 融合（D26）+ cross-encoder 精排（D31）。

**不做 eager import**：子模块会牵到 numpy / Qdrant client / jieba / onnxruntime，
`import src.rag.retrieval` 必须保持轻量（`runpy` 场景下会告 RuntimeWarning）。

⚠️ `__getattr__` 里**不能用 `from . import fusion`**：那会走一次 `getattr(pkg, "fusion")`，
在子模块尚未进 `sys.modules` 时又打回本函数 → `RecursionError`（`from src.rag.retrieval import rerank`
这条以前没被用例覆盖，因为测试都是直接 import 子模块）。必须用 `importlib.import_module`。
"""

import importlib

__all__ = [
    "DEFAULT_ROUTES",
    "ROUTE_DENSE_EN",
    "ROUTE_DENSE_ZH",
    "ROUTE_LEXICAL_EN",
    "ROUTE_LEXICAL_ZH",
    "CrossEncoderReranker",
    "DenseRoute",
    "FusedHit",
    "JiebaLexicalRoute",
    "LexicalRoute",
    "MultiRouteRetriever",
    "RerankConfig",
    "RerankHit",
    "RetrievalResult",
    "RouteHit",
    "RouteResult",
    "best_window",
    "rerank_tiles",
    "tile_windows",
    "collapse_to_parents",
    "rerank_hits",
    "rrf_fuse_routes",
    "union_pool",
]

#: 子模块名 → 相对模块路径。新增子模块**只加一行**，不许在 `__getattr__` 里写 `from . import`。
_SUBMODULES = {
    "fusion": ".fusion",
    "orchestrator": ".orchestrator",
    "rerank": ".rerank",
    "routes": ".routes",
    "store_wiring": ".store_wiring",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(_SUBMODULES[name], __name__)
    for path in _SUBMODULES.values():
        module = importlib.import_module(path, __name__)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module 'src.rag.retrieval' has no attribute {name!r}")
