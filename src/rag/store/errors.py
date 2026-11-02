"""向量库相关的错误类型。"""


class VectorStoreError(RuntimeError):
    """向量库操作的基类异常。"""


class QdrantUnavailableError(VectorStoreError):
    """连不上 Qdrant（服务没起、端口不对、镜像没跑）。"""


class CollectionMismatchError(VectorStoreError):
    """已存在的 collection 与本次写入的规格不一致（维度/距离）。

    D21 的地基：换了 embedding 模型（维度变了）就必须换 collection，
    否则向量塞进旧维度里，检索出来是静默的垃圾。
    """
