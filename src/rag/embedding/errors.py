"""Embedding 层的错误类型。

原则：**宁可抛错，也不静默产生空向量**（§7.2 明文要求）。全零向量在余弦相似度下
对所有 query 都是 0 分，检索会静默失效——这类失败必须炸出来。
"""


class EmbeddingError(RuntimeError):
    """Embedding 相关错误基类。"""


class EmbeddingInputError(EmbeddingError):
    """输入不合法：空字符串、非字符串、超长。"""


class EmbeddingBackendError(EmbeddingError):
    """后端（Ollama / 模型）调用失败。"""


class EmbeddingDimensionError(EmbeddingError):
    """返回维度不是约定维度（BGE-M3 Dense = 1024）。"""


class EmbeddingValueError(EmbeddingError):
    """向量数值非法：NaN / Inf / 全零。"""
