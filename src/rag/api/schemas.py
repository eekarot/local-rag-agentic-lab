"""HTTP 层的请求/响应契约（Pydantic v2）。

契约定在这里而不是写在路由里，原因有两个：

1. **`/openapi.json` 就是文档**：别人用这个仓库时，先看 schema 比读 README 快；
2. **校验边界即安全边界**：`question` 长度、`k` 上限、`budget` 上限都在这里挡住，
   路由函数里就不用再写一遍 if，也不会出现"某个端点忘了校验"。

字段口径与 `AskResult.as_dict()` 保持一致（CLI 的 `--json` 与 HTTP 返回可以互相翻译），
只有一处刻意不同：HTTP 返回**完整 blocks**（`blocks` 是列表 + `block_count` 是数量），
因为网络另一端的客户端要自己渲染引用，只能给它块本体。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

#: 单次请求的硬上限。不是"性能参数"，是防止有人用一条 20 万字的 prompt 把 9B 打死
MAX_QUESTION_CHARS = 2000
MAX_K = 50
MAX_POOL = 200


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS,
                          description="用户问题；中英文均可，中文会自动翻译出一路英文 query")
    k: int = Field(default=10, ge=1, le=MAX_K,
                   description="交给上下文组装器的块数上限")
    pool: int = Field(default=20, ge=1, le=MAX_POOL,
                      description="每路检索深度，也是 RRF 融合池大小")
    budget: int = Field(default=6000, ge=500, le=32000,
                        description="上下文 token 预算（超出会截尾）")
    rerank: bool = Field(default=True, description="是否过交叉编码器重排；关掉快约 0.7 s/次")
    retrieve_only: bool = Field(default=False,
                                description="只检索不生成：不调用聊天模型（省 50 s+）")
    lang_quota: int = Field(default=0, ge=0, le=MAX_K,
                            description="最终 top-k 里每种语言至少占几个名额（0 = 关闭）")

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "智能体化 RAG 和传统 RAG 有什么区别？",
                          "k": 10, "pool": 20, "budget": 6000}]
        }
    }


class SourceOut(BaseModel):
    index: int = Field(description="答案里 [证据 n] 的 n")
    document_id: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    parent_indexes: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list,
                                    description="章节路径；已规范化成 list（不再是 JSON 字符串）")
    source_uri: str | None = None
    title: str | None = None


class AskResponse(BaseModel):
    question: str
    question_en: str | None = Field(default=None, description="中文提问翻译出的英文 query")
    language: str = Field(description="zh / en / other")
    answer: str
    abstained: bool = Field(description="是否弃答（无证据 或 模型明确说答不了）")
    abstain_reason: str = ""
    cited: list[int] = Field(default_factory=list, description="答案引用到的证据编号")
    out_of_range: list[int] = Field(default_factory=list, description="引用了不存在的编号")
    malformed: list[str] = Field(default_factory=list, description="无法解析的引用片段")
    parse_rate: float = 0.0
    sources: list[SourceOut] = Field(default_factory=list)
    blocks: list[dict[str, Any]] = Field(default_factory=list, description="组装后的上下文块全文")
    block_count: int = 0
    context_tokens: int = 0
    candidates: int = Field(default=0, description="RRF 融合后的候选数")
    route_hits: dict[str, int] = Field(default_factory=dict, description="每路召回命中数")
    degraded: dict[str, Any] | None = Field(
        default=None,
        description="检索降级信息（缺向量模型 / 某一路故障）；null = 全部正常。"
                    "非 null 时 `answer` 的可信度会下降，客户端应提示用户。",
    )
    timings: dict[str, float] = Field(default_factory=dict, description="分段耗时（毫秒）")
    llm_called: bool = False


class RetrieveHit(BaseModel):
    parent_id: str
    rank: int = Field(description="在本次响应里的位置（从 1 开始）")
    rrf_rank: int = Field(default=0, description="RRF 融合阶段的原始名次（重排前）")
    score: float = Field(description="RRF 融合分（关掉重排时这就是排序依据）")
    rerank_score: float | None = None
    section_path: list[str] = Field(default_factory=list)
    source_uri: str | None = None
    title: str | None = None
    preview: str = Field(default="", description="正文前 200 字符，便于人眼快速判断")


class RetrieveResponse(BaseModel):
    query: str
    question_en: str | None = None
    language: str = "zh"
    hits: list[RetrieveHit] = Field(default_factory=list)
    route_hits: dict[str, int] = Field(default_factory=dict)
    degraded: dict[str, Any] | None = Field(
        default=None,
        description="检索降级信息（缺向量模型 / 某一路故障）；null = 全部正常。"
                    "`reason=embed_unavailable` 时只剩 lexical 路，结果会明显变差。",
    )
    candidates: int = 0
    timings: dict[str, float] = Field(default_factory=dict)


class CheckOut(BaseModel):
    """一条探针结果（与 `src/rag/ops/probes.Check` 一一对应）。"""

    name: str = Field(description="qdrant / embed_model / chat_model / tokenizer / reranker / parents")
    ok: bool
    detail: str = Field(description="一句话现状：点数量、维度、版本号，或失败原因")
    fix: str = Field(default="", description="可直接复制的修复命令（失败时非空）")
    optional: bool = Field(default=False, description="True = 挂了只是不能生成，检索仍可用")
    info: bool = Field(default=False, description="True = 纯信息项，不影响可用性")


class HealthResponse(BaseModel):
    status: str = Field(description="ok / degraded（能检索不能生成）/ down（检索不可用）")
    version: str
    checks: list[CheckOut] = Field(
        default_factory=list,
        description="六项外部依赖探针；未注入探针的场合为空列表（单测装配）",
    )
    blocking_fixes: list[str] = Field(default_factory=list, description="硬失败的修复命令")
    degraded_fixes: list[str] = Field(default_factory=list, description="可降级的修复命令")
    parents: int = Field(description="已加载的父块数（启动时一次性读入内存）")
    routes: list[str] = Field(default_factory=list)
    rerank_enabled: bool = True
    generation_enabled: bool = True
    queue: dict[str, Any] = Field(
        default_factory=dict,
        description="生成队列快照：inflight / waiting / served / rejected_full / rejected_timeout",
    )
