"""D15：Qdrant 部署形态、Collection 与 Payload Index。

三个由实测确定、不能凭直觉写的地方：

1. `ef_construct` / `m` 是 **collection 级** 参数。写进
   `VectorParams(hnsw_config=...)` 会被服务端静默忽略（实测仍是默认 100），
   而 D15 的验收要 `ef_construct=200`，所以必须放在 `create_collection(hnsw_config=...)`。
2. 本地模式（`QdrantClient(":memory:")` / `path=`）**payload index 是空转**，
   client 会警告 "Payload indexes have no effect in the local Qdrant"。
   所以「6 个索引存在」只能在服务端验；本地模式不要拿来当验收环境。
3. Qdrant 的 point id 只收 **UUID 或整数**，我们的 `chunk_id`（`chk_<32 hex>`）
   塞进去直接 400。`point_id_for()` 用带命名空间的 UUID5 做确定性映射，
   保证「同一 chunk 重复导入 = 同一个点」（D16 幂等的前提）。

部署（本机已验通）：

    docker run -d --name rag-qdrant -p 6333:6333 -p 6334:6334 \\
      -v "<repo>/storage/qdrant:/qdrant/storage" --memory 2g qdrant/qdrant:v1.15.0

    python -m src.rag.store.qdrant_store ping
    python -m src.rag.store.qdrant_store create
    python -m src.rag.store.qdrant_store info
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from urllib.parse import urlsplit
from typing import Any, Iterable, Mapping

from qdrant_client import QdrantClient, models

from .errors import CollectionMismatchError, QdrantUnavailableError, VectorStoreError

# --- 与方案 §3.5 / D15 验收口径绑定的常量 -------------------------------

COLLECTION_NAME = "rag_chunks_v1"



#: 集合名的环境变量（部署时用它把多套知识库放在同一个 Qdrant 上）：
#:
#: 踩坑（A5 干净克隆验收）：集合名原来只有常量，于是 clone 到临时目录按 README
#: 建库，写进的是**正式集合** —— 正式库从 38,865 点变 38,871 点，而且不报任何错，
#: 只是检索结果里混进了别的语料。同一台机器上放「测试 / 预发 / 生产」三套库很常见，
#: 撞库必须能被配置解开，不能靠「记得先改代码」。
COLLECTION_ENV = "RAG_COLLECTION"


def default_collection() -> str:
    """当前生效的集合名：`RAG_COLLECTION` > 历史默认 `rag_chunks_v1`。"""
    return (os.getenv(COLLECTION_ENV) or "").strip() or COLLECTION_NAME

VECTOR_SIZE = 1024                      # bge-m3 dense 维度
DISTANCE = "Cosine"
HNSW_M = 16
EF_CONSTRUCT = 200

# 方案 D15 要求 6 个过滤字段。其中 subject / tenant_id / document_type 目前在
# §3.3 的 Child 结构里还没有（多租户、学科分类是后续阶段的事），先按索引建好：
# Qdrant 的 payload index 是 schema 级的，不要求字段立刻有值，等字段上线即可生效。
PAYLOAD_INDEXES: dict[str, str] = {
    "document_id": "keyword",
    "language": "keyword",
    "status": "keyword",
    "subject": "keyword",
    "tenant_id": "keyword",
    "document_type": "keyword",
}

# point id 的 UUID5 命名空间：裸 chunk_id 直接做 UUID5 会和别的项目撞车
POINT_ID_NAMESPACE = "local-rag-agentic-lab/chunk"

DEFAULT_URL = "http://localhost:6333"

# 进 payload 的字段。注意 `embedding_text` **不在**里面：它只用于编码，
# 存进 payload 等于在库里多留一份重复文本（Parent 原文另有 Document Store）。
PAYLOAD_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "parent_id",
    "document_id",
    "document_version",
    "language",
    "title",
    "section_path",
    "token_count",
    "source_start",
    "source_end",
    "chunker_version",
    "embedding_model",
    "embedding_version",
    "content_hash",
    "status",
)


def qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", DEFAULT_URL)


#: 本地地址：**不许走代理**（见 `make_client` 的注释）
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"})


def is_local_url(url: str) -> bool:
    """判断是不是本机地址（`localhost` / `127.x` / `::1`）。"""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return False
    return host in LOCAL_HOSTS or host.startswith("127.")


def make_client(url: str | None = None, timeout: float | int = 10, **kwargs: Any) -> QdrantClient:
    """建客户端。`timeout` 会被测试用来快速判活（3 秒）。

    本机地址默认 `trust_env=False`：httpx 会**信任系统代理**（Windows 读注册表 IE 设置，
    国内机器上常被 Clash/v2ray 之类设成 `127.0.0.1:7897`），于是 `localhost:6333`
    的请求也会绕到代理上，表现为随机 `ConnectTimeout`（N16/N20 就是这么来的）。
    连本机 Qdrant 走代理永远是错的，所以这里强制直连；远端地址保持默认（云上可能真需要代理）。
    """
    resolved = url or qdrant_url()
    if is_local_url(resolved):
        kwargs.setdefault("trust_env", False)
    return QdrantClient(url=resolved, timeout=timeout, **kwargs)


def point_id_for(chunk_id: str) -> str:
    """`chk_<32 hex>` → 确定性 UUID。同一 chunk 永远落在同一个点。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POINT_ID_NAMESPACE}:{chunk_id}"))


def payload_from_child(child: Mapping[str, Any]) -> dict[str, Any]:
    """Child 记录 → Qdrant payload（`text` 取 `content`）。

    **None 字段整个剔除**：实测 Qdrant 接受 `null` 值写入，但 `MatchValue` 的
    value 不接受 None（pydantic 校验失败，要用 IsNull 条件），于是这类字段
    "存进去了却过滤不到"，等于给索引埋了个假字段。宁可缺席。
    """
    payload = {field: child.get(field) for field in PAYLOAD_FIELDS if child.get(field) is not None}
    payload["text"] = child.get("content")
    return payload


def _distance_value(distance: Any) -> str:
    return getattr(distance, "value", str(distance))


def _collection_names(client: QdrantClient) -> list[str]:
    return [c.name for c in client.get_collections().collections]


def collection_exists(client: QdrantClient, collection: str = COLLECTION_NAME) -> bool:
    return collection in _collection_names(client)


def collection_config(client: QdrantClient, collection: str = COLLECTION_NAME) -> dict[str, Any]:
    """把 collection 的现状读成扁平 dict（CLI 与验收命令都用它）。"""
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    # 命名向量（dense/sparse 共存）时 vectors 是 dict，这里只支持单向量
    if isinstance(vectors, dict):
        raise VectorStoreError(f"{collection} 使用命名向量配置，当前实现只支持单 dense 向量")
    hnsw = info.config.hnsw_config
    return {
        "collection": collection,
        "vector_size": vectors.size,
        "distance": _distance_value(vectors.distance),
        "hnsw_m": hnsw.m,
        "ef_construct": hnsw.ef_construct,
        "payload_indexes": sorted(info.payload_schema or {}),
        "points_count": client.count(collection, exact=True).count,
    }


def _wait_collection_gone(client: QdrantClient, collection: str, timeout: float = 15.0) -> None:
    """等 collection 真正消失。

    实测：`delete_collection` 返回后服务端目录 `./storage/collections/<name>` 可能还在，
    这时同名 `create_collection` 会 400
    "Collection data already exists at ./storage/collections/<name>"。
    `recreate=True` 的产品路径正好是「删了再建同名」，必须在这里等。
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not collection_exists(client, collection):
            return
        time.sleep(0.1)


def ensure_collection(
    client: QdrantClient,
    collection: str = COLLECTION_NAME,
    *,
    recreate: bool = False,
    vector_size: int = VECTOR_SIZE,
    distance: str = DISTANCE,
    hnsw_m: int = HNSW_M,
    ef_construct: int = EF_CONSTRUCT,
) -> dict[str, Any]:
    """建 collection + 6 个 payload index。幂等；`recreate=True` 会先删。"""
    if collection_exists(client, collection):
        if recreate:
            _delete_with_retry(client, collection)
            _wait_collection_gone(client, collection)
        else:
            existing = collection_config(client, collection)
            if existing["vector_size"] != vector_size:
                raise CollectionMismatchError(
                    f"collection {collection} 已存在且维度为 {existing['vector_size']}，"
                    f"本次写入维度 {vector_size}。换 embedding 模型必须换 collection 名，"
                    f"不能复用（D21）"
                )
            _ensure_payload_indexes(client, collection)
            return collection_config(client, collection)

    _create_with_retry(
        client,
        collection=collection,
        vector_size=vector_size,
        distance=distance,
        hnsw_m=hnsw_m,
        ef_construct=ef_construct,
    )
    _ensure_payload_indexes(client, collection)
    return collection_config(client, collection)


def _delete_with_retry(client: QdrantClient, collection: str, attempts: int = 3) -> None:
    """删除重试。

    实测：整套测试连跑时，`delete_collection` 偶发瞬时失败（服务端正在 flush 段），
    单独跑同一个用例却 3/3 通过。删除是幂等操作，重试是安全的；
    不重试会让 `recreate=True` 这种产品路径随机崩。
    """
    import time

    for attempt in range(1, attempts + 1):
        try:
            client.delete_collection(collection)
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(0.3 * attempt)


def _create_with_retry(
    client: QdrantClient,
    *,
    collection: str,
    vector_size: int,
    distance: str,
    hnsw_m: int,
    ef_construct: int,
    attempts: int = 5,
) -> None:
    import time

    for attempt in range(1, attempts + 1):
        try:
            _create_collection(client, collection, vector_size, distance, hnsw_m, ef_construct)
            return
        except Exception as exc:
            # 只对「目录残留」这一种竞态重试，其它错误立刻抛
            if "already exists at" not in str(exc) or attempt == attempts:
                raise
            time.sleep(0.5 * attempt)


def _create_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    distance: str,
    hnsw_m: int,
    ef_construct: int,
) -> None:
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=vector_size,
            # 枚举成员名是 COSINE/DOT/EUCLID，枚举**值**是 "Cosine"/"Dot"——
            # 用值去构造不能用 .upper()（实测 'COSINE' 不是合法值）
            distance=models.Distance[distance.upper()],
        ),
        # 注意：必须在 collection 级给 hnsw_config，放进 VectorParams 会被忽略
        hnsw_config=models.HnswConfigDiff(m=hnsw_m, ef_construct=ef_construct),
    )


def _ensure_payload_indexes(client: QdrantClient, collection: str) -> list[str]:
    existing = set(client.get_collection(collection).payload_schema or {})
    created = []
    for field, schema in PAYLOAD_INDEXES.items():
        if field in existing:
            continue
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            # 枚举：名字是大写 KEYWORD，值是 "keyword"（构造要用名字）
            field_schema=models.PayloadSchemaType[schema.upper()],
            wait=True,
        )
        created.append(field)
    return created


def ping(client: QdrantClient | None = None) -> dict[str, Any]:
    client = client or make_client(timeout=5)
    try:
        collections = _collection_names(client)
        return {"ok": True, "url": qdrant_url(), "collections": sorted(collections)}
    except Exception as exc:  # 连不上、镜像没跑、端口不通都归到这里
        raise QdrantUnavailableError(f"连接 Qdrant 失败（{qdrant_url()}）：{exc}") from exc


# --------------------------------------------------------------------- CLI


def cli_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qdrant_store", description="Qdrant collection 管理（D15）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping", help="探活并列出 collections")
    create = sub.add_parser("create", help="建 collection + payload index")
    create.add_argument("--collection", default=COLLECTION_NAME)
    create.add_argument("--recreate", action="store_true", help="先删再建（丢数据）")
    create.add_argument("--vector-size", type=int, default=VECTOR_SIZE)
    info = sub.add_parser("info", help="读 collection 现状")
    info.add_argument("--collection", default=COLLECTION_NAME)
    info.add_argument("--json", action="store_true", help="输出 JSON")
    drop = sub.add_parser("drop", help="删 collection（要 --yes）")
    drop.add_argument("--collection", default=COLLECTION_NAME)
    drop.add_argument("--yes", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    client = make_client()
    try:
        if args.cmd == "ping":
            print(json.dumps(ping(client), ensure_ascii=False))
        elif args.cmd == "create":
            cfg = ensure_collection(
                client, args.collection, recreate=args.recreate, vector_size=args.vector_size
            )
            print(f"[create] {args.collection} → size={cfg['vector_size']} "
                  f"distance={cfg['distance']} m={cfg['hnsw_m']} ef_construct={cfg['ef_construct']} "
                  f"payload_index={len(cfg['payload_indexes'])} 个")
        elif args.cmd == "info":
            cfg = collection_config(client, args.collection)
            if args.json:
                print(json.dumps(cfg, ensure_ascii=False, indent=2))
            else:
                for key, value in cfg.items():
                    print(f"  {key:16s} {value}")
        elif args.cmd == "drop":
            if not args.yes:
                print("拒绝执行：删 collection 会丢数据，确认请加 --yes", file=sys.stderr)
                return 2
            client.delete_collection(args.collection)
            print(f"[drop] {args.collection} 已删除")
    except VectorStoreError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
