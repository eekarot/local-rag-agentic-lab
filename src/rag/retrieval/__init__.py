"""检索层：三路召回（D24）+ RRF 融合（D26）。

**不做 eager import**：子模块会牵到 numpy / Qdrant client / jieba，
`import src.rag.retrieval` 必须保持轻量（`runpy` 场景下会告 RuntimeWarning）。
"""

__all__ = [
    "DEFAULT_ROUTES",
    "ROUTE_DENSE_EN",
    "ROUTE_DENSE_ZH",
    "ROUTE_LEXICAL_EN",
    "ROUTE_LEXICAL_ZH",
    "DenseRoute",
    "FusedHit",
    "JiebaLexicalRoute",
    "LexicalRoute",
    "MultiRouteRetriever",
    "RetrievalResult",
    "RouteHit",
    "RouteResult",
    "collapse_to_parents",
    "rrf_fuse_routes",
    "union_pool",
]


def __getattr__(name: str):
    from . import fusion, orchestrator, routes

    for module in (routes, fusion, orchestrator):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module 'src.rag.retrieval' has no attribute {name!r}")
