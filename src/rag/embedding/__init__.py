"""Embedding 层（D13）：批量编码 + 维度/NaN 校验 + 内容 hash 缓存。"""

from .errors import (
    EmbeddingBackendError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingValueError,
)
from .service import (
    DEFAULT_MODEL,
    DEFAULT_VERSION,
    VECTOR_SIZE,
    EmbeddingService,
    OllamaBackend,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_VERSION",
    "VECTOR_SIZE",
    "EmbeddingBackendError",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingService",
    "EmbeddingValueError",
    "OllamaBackend",
]
