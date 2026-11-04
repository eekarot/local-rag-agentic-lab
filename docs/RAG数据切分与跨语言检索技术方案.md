# RAG 数据切分与跨语言检索系统技术方案

> **文档版本**：2.7\
> **日期**：2026-11-02\
> **目标**：形成一套可在 Windows 本地运行、可逐步扩展到百万级 Chunk 的
> RAG 数据处理与跨语言检索工程。\
> **主要场景**：用户使用中文提问，知识库主要为英文文档；文档切分需要尽量保持语义完整；检索需要兼顾召回率、准确率、延迟和本地资源限制。

------------------------------------------------------------------------

## 0. 文档使用说明

本文档面向直接开发，包含架构决策、模块边界、数据结构、处理流程、核心算法、接口约定、配置建议、测试方案和实施顺序。

### 0.1 已确定的工程约束

  -----------------------------------------------------------------------
  项目                                决策
  ----------------------------------- -----------------------------------
  运行环境                            Windows 本地优先

  内存                                16GB 或以下作为设计约束

  LLM                                 已安装 Ollama，使用本地聊天模型

  Embedding                           独立加载多语言 Embedding，不依赖
                                      Ollama 必须提供 Embedding API

  语料语言                            第一阶段以英文语料为主

  用户语言                            第一阶段以中文提问为主

  向量规模                            从小规模验证，设计上支持 100 万级
                                      Child Chunk

  向量数据库                          Qdrant

  ANN 索引                            Qdrant HNSW

  第一阶段目标                        单机、可运行、可测量；不引入
                                      Kubernetes 或分布式集群
  -----------------------------------------------------------------------

### 0.2 本方案的核心设计原则

1.  **先保留文档结构，再做长度约束**：不能只按固定字符数截断。
2.  **小块召回、大块供上下文**：采用 Parent-Child Chunking。
3.  **语义切分是增强能力，不是第一版强依赖**：第一版先用结构化递归切分，避免切分阶段引入过高的模型成本。
4.  **不翻译整个知识库**：保留英文原文，在 Query 阶段做跨语言处理。
5.  **多路召回、统一融合、精排**：Dense Retrieval + 英文关键词检索 +
    RRF + 多语言 Reranker。
6.  **ANN 只负责候选召回**：HNSW
    不保证精确最近邻，必须用精确检索基准测量 Recall@K。
7.  **所有重要参数可配置、可评估**：Chunk 大小、Overlap、HNSW
    参数、召回数量、融合方式均不得散落在代码中。
8.  **索引可重建、数据可追踪**：文档版本、切分版本、Embedding
    模型版本必须进入元数据或索引版本。

------------------------------------------------------------------------

# 1. 总体技术架构

## 1.1 系统逻辑架构

``` text
                         ┌─────────────────────────┐
                         │       用户 / API        │
                         │ 中文 Query / 英文 Query │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Query Processing     │
                         │ 语言识别 / 术语保护      │
                         │ 中文问题 / 英文翻译      │
                         └────────────┬────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  │                   │                   │
                  ▼                   ▼                   ▼
          ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
          │ Dense Search │    │ Dense Search │    │ Lexical Search│
          │ 中文原问题   │    │ 英文翻译问题 │    │ 英文 BM25     │
          └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
                 │                   │                   │
                 └───────────────────┼───────────────────┘
                                     ▼
                           ┌──────────────────┐
                           │   RRF Fusion     │
                           │ 去重 / 排名融合  │
                           └────────┬─────────┘
                                    ▼
                           ┌──────────────────┐
                           │    Reranker      │
                           │ 中文 Query       │
                           │ 英文 Candidate   │
                           └────────┬─────────┘
                                    ▼
                           ┌──────────────────┐
                           │ Context Assembler│
                           │ Parent / 邻接块  │
                           │ Token Budget     │
                           └────────┬─────────┘
                                    ▼
                           ┌──────────────────┐
                           │   Ollama LLM     │
                           │ 中文答案 + 引用  │
                           └──────────────────┘


离线数据处理：

文件输入
  │
  ▼
文档解析 / OCR（按需）
  │
  ▼
结构识别（标题、段落、列表、表格、代码）
  │
  ▼
Parent 构建
  │
  ▼
Child 递归切分 + Overlap
  │
  ▼
元数据、ID、版本生成
  │
  ▼
BGE-M3 Dense Embedding
  │
  ├──────────────► Qdrant Dense Vector + HNSW
  │
  ├──────────────► Qdrant Text Payload Index（英文关键词检索）
  │
  └──────────────► Document Store（Parent 原文、文档结构）
```

## 1.2 推荐技术栈

  --------------------------------------------------------------------------------------------------
  模块                    技术                                               用途
  ----------------------- -------------------------------------------------- -----------------------
  开发语言                Python 3.11（建议）                                主工程

  API                     FastAPI + Pydantic                                 本地 HTTP 接口

  文档解析                pymupdf、python-docx、python-pptx、BeautifulSoup   PDF、Office、HTML 解析
                          等按需选择                                         

  Markdown                markdown-it-py 或自定义解析                        保留标题层级

  文本切分                自定义结构化 Recursive Splitter                    结构优先、Token 约束

  Token 计数              BGE-M3 对应 tokenizer                              以模型 Token 数控制长度

  Embedding               `BAAI/bge-m3` + FlagEmbedding                      多语言 Dense Embedding

  向量数据库              Qdrant                                             向量和检索元数据

  ANN                     Qdrant HNSW                                        大规模近似向量搜索

  关键词检索              Qdrant Full-text / BM25                            英文词项检索

  融合                    应用层 RRF                                         合并多路召回

  Reranker                `BAAI/bge-reranker-v2-m3`                          中文 Query
                                                                             与英文文档精排

  LLM                     Ollama 本地聊天模型                                查询翻译、最终答案

  配置                    YAML + Pydantic Settings                           参数管理

  日志                    Python logging / structlog                         运行观测

  测试                    pytest                                             单元、集成、性能测试
  --------------------------------------------------------------------------------------------------

> 第一版不需要同时引入 FAISS、Milvus、Elasticsearch。Qdrant 负责
> Dense/HNSW 和文本检索，应用层负责 Query Processing、RRF、Reranker
> 和上下文组装。FAISS 可作为后续 HNSW/IVF 算法对照实验工具。

------------------------------------------------------------------------

# 2. 数据切分架构

## 2.1 数据处理总流程

``` text
原始文件
  │
  ▼
Document Parser
  │  输出统一 Document AST / Block 列表
  ▼
Structure Extractor
  │  标题、段落、列表、表格、代码块、页码
  ▼
Parent Builder
  │  按章节 / 主题组织父块
  ▼
Child Chunker
  │  递归结构切分
  │  Token 长度约束
  │  Overlap
  ▼
Chunk Validator
  │  长度、空文本、重复、边界、元数据检查
  ▼
Metadata & ID Builder
  │
  ▼
Embedding Batch
  │
  ▼
Qdrant Upsert + Document Store
```

## 2.2 文档解析统一接口

不同格式不能直接采用同一种切分方法。首先将各类文档转换成统一的 Block
结构。

建议内部数据结构：

``` python
from dataclasses import dataclass, field
from typing import Literal, Optional

BlockType = Literal[
    "heading", "paragraph", "list",
    "table", "code", "quote", "caption"
]

@dataclass
class DocumentBlock:
    block_id: str
    block_type: BlockType
    text: str
    heading_level: Optional[int] = None
    page_number: Optional[int] = None
    section_path: list[str] = field(default_factory=list)
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    metadata: dict = field(default_factory=dict)
```

解析器必须尽量保留：

-   原始文档 ID、文件名、来源路径。
-   标题及标题层级。
-   段落顺序。
-   页码或原始偏移量（解析器可提供时）。
-   表格行列关系。
-   代码块边界和语言标识。
-   图片说明、图注和 OCR 文本（如果有）。

### 解析规则

  文档类型   解析策略                     切分注意事项
  ---------- ---------------------------- -----------------------
  Markdown   按标题层级解析               不要把标题和正文分离
  TXT        段落、句子识别               递归切分
  HTML       DOM 结构解析                 保留标题、列表、表格
  PDF        按页和版面块解析             保留页码、标题、表格
  DOCX       按段落、标题样式、表格解析   保留章节关系
  PPTX       按幻灯片、标题、文本框解析   每页保留 slide number
  代码文件   按语言 AST / 函数 / 类切分   尽量不拆函数和类

PDF 解析质量不稳定时，必须将解析结果抽样检查；不能把解析错误归咎于
Chunking。

## 2.3 Parent-Child 分块策略

采用两层结构：

-   **Parent Chunk**：提供完整上下文，通常对应一个小节或连续主题。
-   **Child Chunk**：用于向量检索，粒度更细。

初始配置：

  参数                               初始值 说明
  ---------------------- ------------------ ------------------------------
  child_target_tokens                   400 Child 目标长度
  child_max_tokens                      500 Child 最大长度
  child_overlap_tokens                   60 相邻 Child 重叠长度
  parent_target_tokens                 1500 Parent 目标长度
  parent_max_tokens                    2000 Parent 最大长度
  min_child_tokens                       80 小于该长度时尝试与相邻块合并
  semantic_chunking                   false 第一版关闭
  tokenizer                BGE-M3 tokenizer 统一 Token 计量口径

以上是起始参数，必须通过评估集调整。对标题、短定义、表格、代码块等特殊结构允许合理例外，不应为了达到固定长度而破坏结构。

### Parent 构建规则

1.  优先以文档章节或标题层级作为 Parent 边界。
2.  同一章节内的连续段落可以合并。
3.  Parent 超过最大 Token 数时，在子标题或段落边界拆分。
4.  Parent 太短时，可与同一父章节下相邻段落合并。
5.  不跨越明显不同的章节主题合并 Parent。
6.  Parent
    保存完整原文和来源定位信息，不直接作为第一阶段向量检索的主要对象。

### Child 构建规则

1.  在 Parent 内部切分，不跨 Parent。
2.  优先保留完整段落。
3.  超长段落再按句子切分。
4.  超长句子才进入 Token 级兜底切分。
5.  相邻 Child 使用 Overlap，但尽量在句子边界上重叠。
6.  每个 Child 必须关联唯一 `parent_id`。
7.  Child 的 `section_path`、标题、语言、来源信息继承自 Parent。

## 2.4 Recursive Chunking 算法

分隔符优先级建议：

``` python
SEPARATORS = [
    "\n\n",       # 段落
    "\n",         # 行
    "。", "！", "？",
    ". ", "! ", "? ",  # 英文句子，注意小数和缩写
    "；", "; ",
    "，", ", ",
    " ",          # 词间空格
    ""            # 最终 Token/字符兜底
]
```

实际项目中，Markdown 标题、代码块、表格应先由 Parser
识别，不要只依赖以上分隔符。

核心伪代码：

``` python
def recursive_split(text, separators, max_tokens, tokenizer):
    if token_count(text, tokenizer) <= max_tokens:
        return [text]

    if not separators:
        return split_by_token_window(text, max_tokens, tokenizer)

    separator = separators[0]
    parts = split_preserving_separator(text, separator)

    result = []
    for part in parts:
        if token_count(part, tokenizer) <= max_tokens:
            result.append(part)
        else:
            result.extend(
                recursive_split(
                    part,
                    separators[1:],
                    max_tokens,
                    tokenizer
                )
            )

    return merge_small_parts(result, max_tokens, tokenizer)
```

工程实现必须补充：

-   `split_preserving_separator()`：避免丢失句号、标题符号等分隔信息。
-   `merge_small_parts()`：将相邻短片段合并至目标长度。
-   `split_by_token_window()`：只作为最后兜底。
-   Overlap 应在最终片段合并后执行，避免重复膨胀。
-   所有长度均通过 tokenizer 计算，不得将字符数误当成 Token 数。

## 2.5 Overlap 实现规则

Overlap 的目标是缓解边界信息丢失，而不是让所有 Chunk 大量重复。

建议：

-   初始 60 Tokens。
-   先尝试回退到完整句子边界。
-   若句子边界无法满足长度要求，再按 Token 窗口截取。
-   保存 `source_start`、`source_end`，用于识别重叠区域。
-   检索融合后，对高度重叠的相邻 Child 去重或合并。

不建议把 Overlap 设置得过大。过大将导致向量数量、存储量和重复召回增加。

## 2.6 语义切分（第二阶段增强）

第一版先关闭语义切分。后续可实现为可插拔策略：

1.  将段落拆成句子。
2.  使用 Embedding 模型计算句向量。
3.  计算相邻句子相似度。
4.  根据相似度下降幅度或局部分布确定语义断点。
5.  将相邻语义相关的句子组合。
6.  再执行最大 Token 长度约束。

相邻句相似度：

\[ s_i = `\frac{e_i \cdot e_{i+1}}`{=tex} {\|e_i\|\|e\_{i+1}\|} \]

不要把固定阈值（例如
0.75）写死为通用规则。应使用语料样本分布确定阈值，并与结构化递归切分做对照实验。

语义切分需要额外的 Embedding
计算；如果文档更新频繁或语料很大，必须评估离线构建成本。

## 2.7 Late Chunking（可选研究项）

Late Chunking 的顺序是：

``` text
长文档 → Transformer Token Embeddings → 按边界聚合 Token 向量
```

传统方法是先切分，再独立生成每个 Chunk 的向量。Late Chunking 让 Token
Embedding 在切分前已经获得更长上下文。

它依赖模型输出 Token-level Embedding 和长上下文能力，不是所有 Embedding
模型都支持。第一版不实现，作为独立实验，不与普通 BGE-M3 句向量模式混用。

## 2.8 Chunk Embedding 输入格式

为 Child Embedding 添加少量上下文前缀：

``` text
Document: Agent Engineering Guide
Section: Retrieval / HNSW
Content:
HNSW uses a hierarchical graph to accelerate approximate nearest neighbor search...
```

建议：

-   标题和章节路径可以加入 Embedding 输入。
-   不要将过多无关 Parent 内容拼入 Child Embedding。
-   保留 `content` 原文和 `embedding_text` 两个字段。
-   `content` 用于最终上下文；`embedding_text` 用于生成向量。
-   Query 和 Document 必须使用同一套 Embedding 模型版本。

------------------------------------------------------------------------

# 3. 数据结构与存储设计

## 3.1 Document 元数据

``` json
{
  "document_id": "doc_01H...",
  "source_uri": "data/raw/agent_guide.pdf",
  "file_name": "agent_guide.pdf",
  "file_type": "pdf",
  "title": "Agent Engineering Guide",
  "language": "en",
  "document_version": "sha256:...",
  "parser_version": "parser_v1",
  "created_at": "2026-11-02T10:00:00Z",
  "status": "active"
}
```

## 3.2 Parent Chunk 结构

``` json
{
  "parent_id": "par_...",
  "document_id": "doc_...",
  "document_version": "sha256:...",
  "language": "en",
  "title": "Agent Engineering Guide",
  "section_path": ["Retrieval", "HNSW"],
  "content": "完整父块原文……",
  "token_count": 1450,
  "source_start": 1200,
  "source_end": 6800,
  "chunker_version": "recursive_parent_child_v1"
}
```

Parent 原文可存放在 SQLite 或其他 Document Store 中。第一版可以用 SQLite
保存 Parent、Document 和 Chunk 的文本及来源信息。

## 3.3 Child Chunk 结构

``` json
{
  "chunk_id": "chk_...",
  "parent_id": "par_...",
  "document_id": "doc_...",
  "document_version": "sha256:...",
  "language": "en",
  "title": "Agent Engineering Guide",
  "section_path": ["Retrieval", "HNSW"],
  "content": "HNSW uses a hierarchical graph...",
  "embedding_text": "Document: Agent Engineering Guide\nSection: Retrieval / HNSW\nContent: ...",
  "token_count": 420,
  "source_start": 1800,
  "source_end": 3200,
  "chunker_version": "recursive_child_v1",
  "embedding_model": "BAAI/bge-m3",
  "embedding_version": "bge-m3-v1",
  "content_hash": "sha256:...",
  "status": "active"
}
```

## 3.4 ID 设计

建议使用 UUID 或基于内容的确定性 ID。

建议规则：

``` text
document_id = UUID/业务 ID

parent_id = hash(
  document_id +
  document_version +
  parent_index +
  chunker_version
)

chunk_id = hash(
  parent_id +
  child_index +
  content_hash +
  chunker_version
)
```

要求：

-   同一文档同一版本重复导入时，ID 应保持稳定。
-   文档内容变化后，document_version 必须变化。
-   Chunker 规则变化后，chunker_version 必须变化。
-   Embedding 模型变化时，必须重建对应向量，不得混用不同模型向量。
-   删除或更新文档时，按 `document_id` 清理旧版本数据。

## 3.5 Qdrant Collection 设计

第一阶段建议一个 Child Chunk Collection：

``` text
rag_chunks_v1
```

向量：

``` text
dense: 1024 dimensions, cosine
```

Payload 至少包含：

``` json
{
  "chunk_id": "chk_...",
  "parent_id": "par_...",
  "document_id": "doc_...",
  "language": "en",
  "title": "Agent Engineering Guide",
  "section_path": ["Retrieval", "HNSW"],
  "text": "HNSW uses a hierarchical graph...",
  "document_version": "sha256:...",
  "chunker_version": "recursive_child_v1",
  "embedding_version": "bge-m3-v1",
  "status": "active"
}
```

`text` 字段用于 Qdrant 全文检索和候选文档读取。Parent 的完整原文保存在
Document Store，避免每个 Child 重复存储完整 Parent。

### Payload 索引

在批量导入前，为常用过滤字段创建 Payload Index：

-   `document_id`
-   `language`
-   `status`
-   `subject`（如有）
-   `tenant_id`（未来多租户时）
-   `document_type`（如有）

过滤字段必须根据实际查询场景选择，不要为所有字段盲目建索引。Qdrant
官方文档建议在使用过滤检索时建立相应 Payload Index；若需要过滤条件参与
HNSW 图优化，应尽量在导入数据前创建索引。

------------------------------------------------------------------------

# 4. 跨语言检索架构

## 4.1 设计目标

场景：

-   用户输入中文。
-   知识库主要为英文。
-   检索结果需要是英文原文。
-   最终回答使用中文。
-   不翻译和覆盖原始英文知识库。

总体策略：

``` text
中文 Query
   │
   ├── 原始中文 Query ──► BGE-M3 Dense ──► Dense Results A
   │
   └── 翻译英文 Query
          │
          ├── BGE-M3 Dense ──► Dense Results B
          │
          └── English BM25 ──► Lexical Results C

A + B + C
   │
   ▼
RRF Fusion
   │
   ▼
Multilingual Reranker
   │
   ▼
Parent / Neighbor Context Expansion
   │
   ▼
Ollama 中文生成
```

## 4.2 Embedding 模型

第一阶段采用：

``` text
BAAI/bge-m3
```

主要原因：

-   多语言能力，支持 100 多种语言。
-   Dense 向量维度为 1024。
-   支持最长 8192 Tokens 的输入长度。
-   同一模型还支持 Sparse 和 Multi-vector 表示。
-   官方建议将 Hybrid Retrieval 与 Reranking 组合使用。

官方模型说明： https://huggingface.co/BAAI/bge-m3

第一阶段只启用 Dense 输出，降低工程复杂度。后续可以评估 BGE-M3 的 Sparse
输出，与 BM25 进行对照实验。

**重要约束：**

-   文档向量和 Query 向量必须使用同一模型版本。
-   模型版本或向量维度变化时，使用新 Collection 或完整重建索引。
-   不得把旧模型生成的向量与新模型 Query Vector 混合比较。
-   不需要为了 BGE-M3 额外给 Query 添加英文检索指令；以模型卡说明为准。

## 4.3 Query Processing

输入：

``` json
{
  "query": "HNSW 算法是如何实现快速向量检索的？",
  "target_corpus_languages": ["en"]
}
```

输出：

``` json
{
  "original_query": "HNSW 算法是如何实现快速向量检索的？",
  "source_language": "zh",
  "translated_queries": [
    "How does the HNSW algorithm enable fast vector search?"
  ],
  "keywords": ["HNSW", "vector search", "approximate nearest neighbor"],
  "entities": ["HNSW"],
  "target_languages": ["en"]
}
```

### Query Processing 规则

1.  识别用户问题语言。
2.  根据知识库语言配置决定是否翻译。
3.  翻译时保留专有名词、缩写、模型名称、代码符号、数字和版本号。
4.  不允许翻译器擅自扩大问题范围。
5.  英文 Query 作为 BM25 检索输入。
6.  原始中文 Query 和英文 Query 都可以进入 Dense Retrieval。
7.  翻译失败时，不阻断检索，退化为原始 Query 的 Dense Retrieval。
8.  对标准化 Query 做缓存，避免重复调用 Ollama。

翻译 Prompt：

``` text
你是 RAG 系统的检索查询翻译器。

将用户的查询转换为适合检索英文技术知识库的英文查询。

要求：
1. 保留原始问题的语义和约束，不要扩展问题。
2. 保留技术术语、产品名、模型名、缩写、代码标识符和数字。
3. 如果有标准英文术语，优先使用标准术语。
4. 不要回答用户问题，只生成英文检索查询。
5. 输出合法 JSON，不要输出 Markdown。

JSON 格式：
{
  "translated_query": "...",
  "keywords": ["..."],
  "entities": ["..."]
}

用户问题：
{{query}}
```

### 翻译服务接口

``` python
class QueryTranslator:
    def translate(
        self,
        query: str,
        target_language: str = "en"
    ) -> TranslationResult:
        ...
```

`TranslationResult` 至少包含：

``` python
class TranslationResult:
    original_query: str
    source_language: str
    translated_query: str | None
    keywords: list[str]
    entities: list[str]
    success: bool
```

## 4.4 Dense Retrieval

对原始中文 Query 和英文翻译 Query 分别生成 BGE-M3 Dense Vector：

``` python
q_zh = embedder.encode_dense(original_query)
q_en = embedder.encode_dense(translated_query)
```

分别执行向量检索：

``` text
Search A:
query = q_zh
limit = 50

Search B:
query = q_en
limit = 50
```

说明：

-   BGE-M3 的跨语言语义能力允许中文 Query 直接检索英文文档。
-   翻译 Query 提供额外检索路径。
-   两条结果可能高度重叠，后续必须融合和去重。
-   不应假设翻译 Query 一定优于原始 Query，最终权重由评估集决定。

## 4.5 英文关键词检索

第一版使用 Qdrant Full-text / BM25 检索英文 Child 文本。

逻辑：

``` text
translated_query
      │
      ▼
English tokenizer / full-text index
      │
      ▼
BM25 search
      │
      ▼
Top 50 Child Chunks
```

英文知识库使用英文翻译 Query，可改善关键词匹配，例如：

``` text
中文：向量数据库的 HNSW 索引
英文：HNSW index in vector databases
```

BM25 主要补充精确术语、缩写、数字、产品名称等匹配能力。BM25
本身不理解"向量数据库"和"vector
database"是跨语言同义词，因此英文语料场景需要先翻译 Query。

如果未来需要更复杂的全文检索，可以将 BM25 替换或补充为 Elasticsearch /
OpenSearch / Tantivy；第一版不需要额外部署这些服务。

## 4.6 RRF 结果融合

不同检索器的分数尺度不同，不直接把 Cosine Similarity 和 BM25 分数相加。

采用 Reciprocal Rank Fusion：

\[ RRF(d)=`\sum`{=tex}\_{r `\in `{=tex}R}`\frac{1}{k+rank_r(d)}`{=tex}
\]

其中：

-   (d)：候选 Chunk。
-   (R)：不同检索结果列表。
-   (rank_r(d))：Chunk 在某检索列表中的排名，从 1 开始。
-   (k)：平滑参数，应用层初始取 60。

三个检索结果：

-   中文 Dense Top 50。
-   英文 Dense Top 50。
-   英文 BM25 Top 50。

应用层 RRF 伪代码：

``` python
from collections import defaultdict

def rrf_fusion(result_lists, k=60):
    scores = defaultdict(float)

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            scores[item.chunk_id] += 1.0 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
```

注意：

-   同一 Chunk 在不同列表出现时累加排名贡献。
-   不同列表中的重复 Chunk 必须按 `chunk_id` 去重。
-   RRF 输出的是融合分，不是概率。
-   初始不使用手工权重；有评估集后再调权重。
-   如果使用 Qdrant 原生 RRF，需按当前服务版本的默认参数和 API
    行为核对；本方案应用层 RRF 便于统一控制和测试。

## 4.7 Reranker 精排

使用：

``` text
BAAI/bge-reranker-v2-m3
```

它是多语言 Reranker，输入为 Query 和候选 Passage，直接计算相关性分数。

第一版：

-   RRF 后保留 Top 50。
-   使用原始中文 Query + 英文 Candidate 进行精排。
-   返回 Top 5～10。
-   若需要，可将英文翻译 Query 的精排结果作为独立对照实验。
-   Reranker 不对 100 万 Chunk 全量运行，只对召回候选运行。

接口：

``` python
class Reranker:
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 10
    ) -> list[RankedChunk]:
        ...
```

资源受限时：

-   第一阶段允许关闭 Reranker，先验收召回。
-   第二阶段启用 Reranker。
-   对候选分批推理。
-   限制 Reranker 并发。
-   不要同时无约束加载 Embedding、Reranker 和大型 Ollama 模型。

## 4.8 Context Expansion

Reranker 得到 Child Top-K 后，不直接简单拼接全部 Child。

按以下顺序组装：

1.  读取 Child 对应的 Parent。
2.  检查同一 Parent 是否已加入上下文。
3.  如果 Parent 太大，可读取命中 Child 前后相邻 Child。
4.  合并重叠文本，避免重复。
5.  按 Reranker 排名和来源可信度排序。
6.  根据 LLM 上下文预算截断。
7.  每个上下文块保留 `document_id`、`chunk_id`、标题、页码或来源路径。

最终上下文示例：

``` text
[Source 1]
Document: Agent Engineering Guide
Section: Retrieval / HNSW
Chunk ID: chk_001
Content: ...

[Source 2]
Document: Vector Search Handbook
Section: Approximate Nearest Neighbor
Chunk ID: chk_102
Content: ...
```

## 4.9 最终回答

Ollama 负责根据检索上下文回答，不能把 Query Translation 的结果当作答案。

系统 Prompt 核心要求：

``` text
你是一个基于知识库回答问题的助手。

规则：
1. 优先依据提供的 Context 回答。
2. 不得将 Context 中没有的信息编造成事实。
3. 如果证据不足，明确说明知识库未提供足够依据。
4. 使用用户原始提问的语言回答。
5. 技术术语可以保留英文，并在首次出现时给出中文解释。
6. 关键结论应标注来源编号。
7. 不要把检索相似度分数当作事实置信度。
```

------------------------------------------------------------------------

# 5. 百万级 Chunk 检索与 HNSW 设计

## 5.1 为什么需要 ANN

暴力检索的复杂度约为：

\[ O(N `\times `{=tex}d) \]

N 为向量数，d 为向量维度。百万级向量逐一计算会增加查询成本。

HNSW 通过多层图导航，避免每次都扫描所有向量。

Qdrant 的 Dense Vector Index 使用 HNSW。核心参数：

  参数             作用                 初始建议
  ---------------- ------------------ ----------
  `m`              图连接度                   16
  `ef_construct`   建图搜索候选规模          200
  `hnsw_ef`        查询时搜索宽度        64～128
  Dense Top-K      ANN 候选数            50～100

参数不是固定最优值。必须根据 Recall@K、P95
Latency、内存和构建时间进行测试。

## 5.2 HNSW 的运行逻辑

``` text
插入阶段：
新向量
  │
  ▼
随机分配最高层级
  │
  ▼
从当前入口节点开始
  │
  ▼
由高层向低层搜索
  │
  ▼
在各层选择邻居
  │
  ▼
建立双向连接并修剪超限边
  │
  ▼
必要时更新入口节点

查询阶段：
Query Vector
  │
  ▼
最高层入口
  │
  ▼
逐层 Greedy Search
  │
  ▼
Layer 0 使用更宽的候选搜索
  │
  ▼
返回 Top-K
```

## 5.3 百万级内存估算

假设：

-   1,000,000 个 Child Chunk。
-   BGE-M3 Dense 维度 1024。
-   float32，每维 4 Bytes。

原始向量空间：

\[ 1,000,000 `\times 1024`{=tex} `\times 4`{=tex} =4.096GB \]

这是原始向量数据，不包括 HNSW 图、Payload、文本、数据库和运行时开销。

因此：

-   不能把 4.096GB 当作整个系统的内存需求。
-   Parent 文本不要在每个 Child 中重复存储。
-   原始文档和 Parent 内容放在 Document Store。
-   Qdrant Payload 只保留检索必需字段和 Child 文本。
-   使用 Qdrant 量化和磁盘存储能力降低内存压力。
-   必须在目标机器上实测 RSS、磁盘空间和 P95 延迟。

## 5.4 16GB 或以下内存的部署策略

建议按阶段执行：

**阶段 A：10 万级**

-   BGE-M3 CPU 推理。
-   Qdrant 本地单机。
-   float32 向量。
-   HNSW 默认参数附近。
-   验证数据链路和 Recall。

**阶段 B：30 万～50 万级**

-   测量索引内存。
-   开启标量量化实验。
-   评估向量 On-disk。
-   控制 Embedding 和 Reranker 并发。
-   将 Parent 原文放 SQLite。

**阶段 C：100 万级**

-   使用 Qdrant HNSW。
-   对比 float32 与 int8 Scalar Quantization。
-   开启量化 Rescore（若当前配置支持且测试有效）。
-   评估向量磁盘存储。
-   限制每批导入数量。
-   记录构建时间、内存峰值、查询 P95 和 Recall@10。

不要在 16GB 机器上同时无约束运行 BGE-M3、BGE Reranker 和大型 Ollama
模型。可以将离线
Embedding、在线检索和最终生成分阶段运行，或为不同模型设置独立进程与并发上限。

## 5.5 Qdrant 索引配置原则

创建 Collection 时：

-   Dense Vector：1024 维，Cosine。
-   HNSW：初始 `m=16`、`ef_construct=200`。
-   为常用过滤字段创建 Payload Index。
-   全文检索字段建立 Full-text Index。
-   批量导入完成后检查 Collection 索引状态。
-   查询时通过 Search Params 设置 `hnsw_ef`。

过滤字段索引应尽量在批量导入前创建，尤其当需要过滤条件与 HNSW
搜索协同工作时。

## 5.6 量化策略

建议实验顺序：

1.  float32 原始向量作为基准。
2.  Scalar Quantization INT8。
3.  在可用时启用原始向量 Rescore。
4.  对比内存、磁盘、Recall@10、P95。

不要第一步就使用 IVF-PQ。Qdrant 的主 Dense ANN 路径采用 HNSW；IVF、PQ
可以用 FAISS 单独学习和对比，不要把 FAISS 的索引参数误认为 Qdrant HNSW
配置。

------------------------------------------------------------------------

# 6. 项目目录结构

建议项目名称：

``` text
local-rag-crosslingual/
│
├── README.md
├── pyproject.toml
├── .env.example
├── config/
│   ├── app.yaml
│   ├── chunking.yaml
│   ├── retrieval.yaml
│   └── models.yaml
│
├── data/
│   ├── raw/                  # 原始文档
│   ├── parsed/               # 解析后的中间数据
│   ├── chunks/               # Parent / Child JSONL
│   ├── eval/                 # 评估集
│   └── sqlite/               # Document Store
│
├── scripts/
│   ├── ingest.py
│   ├── rebuild_index.py
│   ├── evaluate_chunking.py
│   ├── evaluate_retrieval.py
│   └── benchmark_hnsw.py
│
├── src/
│   └── rag/
│       ├── api/
│       │   ├── app.py
│       │   └── schemas.py
│       │
│       ├── ingestion/
│       │   ├── parser/
│       │   │   ├── base.py
│       │   │   ├── markdown_parser.py
│       │   │   ├── pdf_parser.py
│       │   │   ├── docx_parser.py
│       │   │   └── html_parser.py
│       │   ├── structure.py
│       │   ├── parent_builder.py
│       │   ├── recursive_chunker.py
│       │   ├── semantic_chunker.py
│       │   ├── overlap.py
│       │   ├── validator.py
│       │   └── pipeline.py
│       │
│       ├── embedding/
│       │   ├── base.py
│       │   └── bge_m3.py
│       │
│       ├── query/
│       │   ├── language_detector.py
│       │   ├── translator.py
│       │   ├── query_processor.py
│       │   └── cache.py
│       │
│       ├── retrieval/
│       │   ├── dense_retriever.py
│       │   ├── lexical_retriever.py
│       │   ├── fusion.py
│       │   ├── reranker.py
│       │   ├── context_assembler.py
│       │   └── pipeline.py
│       │
│       ├── storage/
│       │   ├── qdrant_store.py
│       │   ├── document_store.py
│       │   └── models.py
│       │
│       ├── generation/
│       │   └── ollama_client.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── dataset.py
│       │   └── runner.py
│       │
│       ├── config.py
│       └── logging.py
│
└── tests/
    ├── unit/
    │   ├── test_recursive_chunker.py
    │   ├── test_overlap.py
    │   ├── test_metadata.py
    │   ├── test_rrf.py
    │   └── test_query_translation.py
    ├── integration/
    │   ├── test_ingestion_pipeline.py
    │   ├── test_qdrant_retrieval.py
    │   └── test_crosslingual_retrieval.py
    └── performance/
        ├── test_hnsw_recall.py
        └── test_retrieval_latency.py
```

------------------------------------------------------------------------

# 7. 核心模块接口

## 7.1 Ingestion Pipeline

``` python
class IngestionPipeline:
    def ingest_file(self, file_path: str) -> IngestionResult:
        """解析、切分、向量化、写入存储。"""
        ...

    def ingest_directory(self, directory: str) -> BatchIngestionResult:
        """批量导入目录。"""
        ...

    def delete_document(self, document_id: str) -> None:
        """删除文档及其 Parent / Child / Vector。"""
        ...

    def rebuild_document(self, document_id: str) -> IngestionResult:
        """按当前 Parser、Chunker、Embedding 版本重建。"""
        ...
```

导入流程必须具备：

-   批处理。
-   失败重试。
-   进度记录。
-   幂等 Upsert。
-   失败任务清单。
-   文档版本管理。
-   旧版本清理。
-   可从上次成功批次恢复。

## 7.2 Embedding Service

``` python
class EmbeddingService:
    def encode_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        ...

    def encode_queries(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        ...
```

要求：

-   固定模型路径和版本。
-   Batch 推理。
-   输入长度检查。
-   输出维度检查（BGE-M3 Dense 为 1024）。
-   NaN / Inf 检查。
-   模型加载失败时明确报错，不得静默产生空向量。
-   对同一文本可以使用内容 Hash 缓存 Embedding。

## 7.3 Query Processor

``` python
class QueryProcessor:
    def process(self, query: str) -> ProcessedQuery:
        ...
```

输出：

``` python
@dataclass
class ProcessedQuery:
    original_query: str
    source_language: str
    translated_query: str | None
    keywords: list[str]
    entities: list[str]
    translation_success: bool
```

## 7.4 Retriever

``` python
class RetrievalPipeline:
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> list[RankedChunk]:
        ...
```

内部执行：

1.  Query Processing。
2.  原始 Query Dense Search。
3.  翻译 Query Dense Search。
4.  翻译 Query BM25 Search。
5.  RRF。
6.  Reranker。
7.  Parent / Neighbor Expansion。
8.  返回结构化结果。

建议每个 Retriever 独立实现，方便关闭某一路做消融实验。

## 7.5 Generation

``` python
class RAGService:
    def ask(
        self,
        query: str,
        top_k: int = 8
    ) -> AnswerResult:
        ...
```

`AnswerResult` 应包含：

``` json
{
  "answer": "中文回答……",
  "sources": [
    {
      "document_id": "doc_001",
      "chunk_id": "chk_001",
      "title": "Agent Engineering Guide",
      "section_path": ["Retrieval", "HNSW"]
    }
  ],
  "retrieval_debug": {
    "translation_success": true,
    "dense_candidates": 50,
    "lexical_candidates": 50,
    "fused_candidates": 80,
    "reranked_candidates": 10
  }
}
```

生产或正式使用时，是否向终端用户暴露 `retrieval_debug` 应由配置控制。

------------------------------------------------------------------------

# 8. API 设计

第一阶段提供本地 FastAPI 服务。

## 8.1 导入文档

`POST /api/v1/documents`

请求：

``` json
{
  "file_path": "data/raw/agent_guide.pdf",
  "language": "en",
  "rebuild": false
}
```

响应：

``` json
{
  "document_id": "doc_001",
  "status": "accepted",
  "message": "Document ingestion started"
}
```

## 8.2 查询检索结果

`POST /api/v1/retrieve`

请求：

``` json
{
  "query": "HNSW 算法如何实现快速向量检索？",
  "top_k": 10,
  "debug": true
}
```

响应：

``` json
{
  "query": "HNSW 算法如何实现快速向量检索？",
  "translated_query": "How does HNSW enable fast vector search?",
  "results": [
    {
      "chunk_id": "chk_001",
      "parent_id": "par_001",
      "document_id": "doc_001",
      "score": 0.82,
      "rerank_score": 0.91,
      "content": "..."
    }
  ]
}
```

分数仅用于排序和调试，不应直接解释为答案正确概率。

## 8.3 RAG 问答

`POST /api/v1/ask`

请求：

``` json
{
  "query": "HNSW 算法如何实现快速向量检索？",
  "top_k": 8,
  "include_sources": true
}
```

响应：

``` json
{
  "answer": "HNSW 通过多层图结构实现近似最近邻搜索……",
  "sources": [
    {
      "document_id": "doc_001",
      "chunk_id": "chk_001",
      "title": "Agent Engineering Guide"
    }
  ]
}
```

## 8.4 查询导入状态

`GET /api/v1/documents/{document_id}/status`

返回解析、切分、Embedding、索引写入各阶段状态。

------------------------------------------------------------------------

# 9. 配置文件

## 9.1 chunking.yaml

``` yaml
chunking:
  strategy: recursive_parent_child

  tokenizer:
    model: BAAI/bge-m3

  parent:
    target_tokens: 1500
    max_tokens: 2000

  child:
    target_tokens: 400
    max_tokens: 500
    min_tokens: 80
    overlap_tokens: 60

  semantic:
    enabled: false
    threshold_mode: percentile
    threshold_percentile: 20

  preserve:
    headings: true
    tables: true
    code_blocks: true
    page_numbers: true
```

## 9.2 models.yaml

``` yaml
embedding:
  provider: flag_embedding
  model_name: BAAI/bge-m3
  dimension: 1024
  device: cpu
  batch_size: 8
  max_length: 8192
  normalize_embeddings: true

reranker:
  enabled: true
  model_name: BAAI/bge-reranker-v2-m3
  device: cpu
  batch_size: 4
  max_candidates: 50

llm:
  provider: ollama
  base_url: http://localhost:11434
  model: ${OLLAMA_MODEL}
  temperature: 0
```

## 9.3 retrieval.yaml

``` yaml
retrieval:
  target_corpus_languages:
    - en

  dense:
    enabled: true
    top_k_per_query: 50
    original_query: true
    translated_query: true

  lexical:
    enabled: true
    engine: qdrant_full_text
    query_language: en
    top_k: 50

  fusion:
    method: rrf
    k: 60

  reranker:
    enabled: true
    top_k_input: 50
    top_k_output: 10
    query_mode: original_query

  context:
    expansion: parent
    max_parent_per_document: 3
    max_context_tokens: 6000
```

## 9.4 app.yaml

``` yaml
app:
  environment: local
  log_level: INFO

qdrant:
  url: http://localhost:6333
  collection: rag_chunks_v1
  distance: cosine
  hnsw:
    m: 16
    ef_construct: 200
    ef_search: 100

storage:
  sqlite_path: data/sqlite/documents.db
  raw_dir: data/raw
  parsed_dir: data/parsed
  chunks_dir: data/chunks

ingestion:
  batch_size: 64
  max_embedding_workers: 1
  retry_times: 3
```

注意：配置文件中的 `device`、批大小和并发数要根据实际机器调整。Windows
CPU 环境优先从小批次开始。

------------------------------------------------------------------------

# 10. 异常处理与数据一致性

## 10.1 导入失败处理

每个文档导入分阶段记录状态：

``` text
PENDING
  ↓
PARSING
  ↓
CHUNKING
  ↓
EMBEDDING
  ↓
INDEXING
  ↓
COMPLETED
```

失败状态：

``` text
FAILED_PARSING
FAILED_CHUNKING
FAILED_EMBEDDING
FAILED_INDEXING
```

要求：

-   每个阶段记录开始时间、结束时间、错误信息。
-   失败任务可重试。
-   已成功处理的 Batch 不重复计算。
-   Upsert 使用稳定 ID，保证重复提交不产生重复点。
-   只有新版本成功写入后，才将旧版本标记为 inactive 并清理。
-   对文档删除操作，确保 Qdrant 和 Document Store 都完成清理。

## 10.2 模型与索引版本一致性

必须记录：

``` text
parser_version
chunker_version
embedding_model
embedding_model_revision
vector_dimension
collection_version
```

以下变化需要重新处理：

  变化                            是否重建 Chunk         是否重新 Embedding
  ------------------------------- ---------------------- -----------------------
  Parser 规则变化                 通常需要               需要
  Chunker 规则变化                需要                   需要
  Embedding 模型变化              不一定                 必须
  Embedding 维度变化              不一定                 必须使用新 Collection
  Reranker 变化                   不需要                 不需要
  Query Translation Prompt 变化   不需要                 不需要
  HNSW 参数变化                   不需要重新 Embedding   通常需要重建/优化索引

## 10.3 查询失败降级

  故障                        降级策略
  --------------------------- -----------------------------------------
  Ollama 翻译失败             只用原始 Query Dense 检索
  英文 BM25 失败              使用两路 Dense 检索
  中文 Dense 失败             使用英文翻译 Dense + BM25（若翻译成功）
  Reranker 失败               返回 RRF 排序结果
  Document Store 暂时不可用   返回错误，不生成无来源答案
  Qdrant 不可用               返回检索服务不可用，不伪造结果

------------------------------------------------------------------------

# 11. 评估体系与验收标准

必须建立独立的 Evaluation Dataset，不能只通过主观提问判断效果。

## 11.1 Chunking 评估

准备 30～50 篇不同结构文档，人工抽查：

-   标题是否和正文关联。
-   句子是否被不合理切断。
-   表格是否保留行列关系。
-   代码块是否完整。
-   Parent 是否覆盖正确章节。
-   Child 是否能够回溯 Parent。
-   Overlap 是否过大。
-   是否出现大量空 Chunk 或重复 Chunk。

对比方案：

  实验   Chunking
  ------ ------------------------------------
  A      固定长度
  B      Recursive
  C      Recursive + Overlap
  D      Parent-Child + Recursive + Overlap
  E      Semantic + Parent-Child

## 11.2 跨语言检索评估集

至少建立 100 条中英文问题对，推荐 200 条以上。

每条数据包含：

``` json
{
  "query_zh": "什么是 HNSW？",
  "query_en": "What is HNSW?",
  "relevant_chunk_ids": [
    "chk_001",
    "chk_017"
  ],
  "relevant_document_ids": [
    "doc_001"
  ],
  "topic": "vector_search",
  "difficulty": "normal"
}
```

问题类型至少覆盖：

-   中文概念问题。
-   中文问题包含英文术语。
-   缩写和模型名。
-   数字、参数、版本号。
-   同义表达。
-   多条件问题。
-   需要跨多个 Chunk 才能回答的问题。
-   知识库没有答案的问题。

## 11.3 检索指标

### Recall@K

\[ Recall@K = `\frac{\text{Top-K 中相关文档数}}`{=tex}
{`\text{全部相关文档数}`{=tex}} \]

### MRR

\[ MRR = `\frac{1}{|Q|}`{=tex}
`\sum`{=tex}\_{q`\in `{=tex}Q}`\frac{1}{rank_q}`{=tex} \]

### nDCG@K

衡量多条相关结果的排序质量。

### ANN Recall@K

以精确向量检索结果为 Ground Truth，衡量 HNSW 找回了多少精确近邻：

\[ ANN Recall@K = `\frac{|ANN\_TopK \cap Exact\_TopK|}{K}`{=tex} \]

需要区分：

-   **ANN Recall**：HNSW 是否找回精确向量近邻。
-   **RAG Retrieval Recall**：是否找到了人工标注的语义相关 Chunk。

两者不是同一个指标。

## 11.4 消融实验

至少运行以下实验：

  实验   检索配置
  ------ --------------------------------
  A      中文 Query + Dense
  B      英文翻译 Query + Dense
  C      中文 Dense + 英文 Dense
  D      中文 Dense + 英文 Dense + BM25
  E      D + RRF
  F      E + Reranker
  G      F + Parent Expansion

记录：

-   Recall@5 / Recall@10。
-   MRR@10。
-   nDCG@10。
-   P50 / P95 延迟。
-   翻译耗时。
-   Embedding 耗时。
-   Reranker 耗时。
-   最终答案引用正确率。

## 11.5 初步验收门槛

以下是项目启动阶段的目标值，不是模型保证值：

  指标                                        初始验收目标
  --------------------------- ----------------------------
  Chunk 最大 Token 超限率        0（结构性例外需明确记录）
  Child → Parent 关联成功率                           100%
  导入重复执行的重复点数量                               0
  中文问题跨语言 Recall@10        ≥ 0.85（在自建评估集上）
  HNSW Recall@10                ≥ 0.95（相对精确检索基准）
  查询结果来源可追踪率                                100%
  检索失败时伪造答案                                     0

如果评估集规模较小，应同时报告命中数量和样本量，不能只报告百分比。

------------------------------------------------------------------------

# 12. 分阶段开发计划

## Phase 1：文档解析与 Chunking

**目标：** 输入一份 Markdown/PDF 文档，生成结构合理的 Parent / Child
JSONL。

任务：

1.  建立统一 DocumentBlock 数据结构。
2.  实现 Markdown Parser。
3.  实现 TXT Parser。
4.  实现 Recursive Chunker。
5.  实现 Token 计数。
6.  实现 Overlap。
7.  实现 Parent Builder。
8.  实现 Chunk Validator。
9.  输出 JSONL。
10. 编写 Chunking 单元测试。

验收：

-   能够处理 Markdown 和 TXT。
-   Parent / Child 关系完整。
-   Token 数量符合配置。
-   标题、段落和来源信息保留。
-   可重复运行并生成稳定 ID。

## Phase 2：Embedding 与 Qdrant

**目标：** 将 Child Chunk 转成向量并完成本地检索。

任务：

1.  加载 BGE-M3。
2.  实现 Batch Embedding。
3.  启动 Qdrant。
4.  创建 Collection。
5.  创建 Payload Index。
6.  批量 Upsert。
7.  实现 Dense Search。
8.  实现 Document Store。

验收：

-   可以导入 1 万条 Chunk。
-   可以通过中文或英文 Query 检索。
-   能够读取命中 Chunk 对应 Parent。
-   能够根据 document_id 删除数据。

## Phase 3：跨语言检索

**目标：** 中文 Query 能够有效检索英文语料。

任务：

1.  实现语言识别。
2.  接入 Ollama Query Translation。
3.  实现中文 Dense Search。
4.  实现英文 Dense Search。
5.  实现英文 BM25。
6.  实现 RRF。
7.  建立跨语言评估集。
8.  对比不同检索路径。

验收：

-   翻译失败能够降级。
-   三路检索结果可以合并。
-   RRF 能够正确去重和排序。
-   可以输出每一路的召回结果用于调试。

## Phase 4：Reranker 与上下文组装

**目标：** 提高最终候选排序质量，减少上下文断裂。

任务：

1.  加载 BGE-Reranker-v2-m3。
2.  实现 Query-Document Pair 批量评分。
3.  实现 Parent Expansion。
4.  实现相邻 Child 合并。
5.  实现 Token Budget。
6.  接入 Ollama Answer Generation。
7.  返回引用来源。

验收：

-   Reranker 只处理候选集。
-   最终上下文不包含大量重复内容。
-   回答可定位到原始文档。
-   证据不足时明确拒答或说明不足。

## Phase 5：百万级检索性能

**目标：** 评估单机 HNSW 对百万级 Child Chunk 的承载能力。

任务：

1.  生成或整理百万级测试数据。
2.  批量导入。
3.  测量索引构建时间。
4.  测量内存峰值。
5.  测量查询 P50/P95。
6.  调整 M、ef_construct、ef_search。
7.  测试量化和磁盘向量。
8.  与精确检索结果比较 ANN Recall。

验收：

-   1,000,000 个向量导入完成。
-   HNSW 查询结果可复现。
-   输出性能报告。
-   报告包含硬件、模型、维度、参数、数据量和测试并发数。
-   不以单次查询耗时作为性能结论。

------------------------------------------------------------------------

# 13. 开发时需要特别避免的错误

1.  **用字符数代替 Token 数。** 不同语言的
    Token/字符比例不同，必须使用模型 Tokenizer。
2.  **只做固定长度切分。** 应优先保留标题、段落、表格和代码结构。
3.  **Overlap 过大。** 会产生重复向量和重复上下文。
4.  **只用 Child 检索结果直接回答。** 应通过 Parent 或相邻块恢复上下文。
5.  **把 Query 翻译当成唯一检索路径。** 保留原始中文 Dense
    检索，避免翻译丢失细节。
6.  **认为 BM25 理解跨语言语义。** 中文"向量数据库"和英文"Vector
    Database"需要翻译或术语映射才能可靠匹配。
7.  **把不同模型生成的向量混在一个空间。** Embedding
    模型版本变化必须重新向量化。
8.  **把 RRF 分数当成概率。** RRF 只是排名融合分。
9.  **让 Reranker 扫描全库。** Reranker 只处理召回候选。
10. **认为 HNSW 返回的一定是精确 Top-K。** 必须用 Flat/Brute Force 建立
    Ground Truth。
11. **只测试 100 万条数据的查询速度，不测构建和内存。**
    离线构建同样是系统成本。
12. **在低内存机器上同时加载多个大型模型。**
    必须控制批大小和并发，必要时分阶段运行。

------------------------------------------------------------------------

# 14. 最终推荐配置

这是第一版建议采用的基线配置：

  类别                  配置
  --------------------- ---------------------------------
  Parser                Markdown + TXT 优先
  Chunk Strategy        Parent-Child + Recursive
  Child Target          400 Tokens
  Child Max             500 Tokens
  Overlap               60 Tokens
  Parent Target         1500 Tokens
  Semantic Chunking     第一版关闭
  Embedding             BAAI/bge-m3
  Embedding Dimension   1024
  Vector Store          Qdrant
  Distance              Cosine
  ANN                   HNSW
  M                     16
  ef_construct          200
  ef_search             100（基线，需压测）
  Query Translation     Ollama
  Dense Retrieval       中文原 Query + 英文翻译 Query
  Lexical Retrieval     英文 BM25
  Fusion                RRF，应用层 k=60
  Reranker              BAAI/bge-reranker-v2-m3
  Reranker Input        Top 50
  Final Context         Top 5～10，受 Token Budget 约束
  Document Store        SQLite（本地 MVP）
  API                   FastAPI

------------------------------------------------------------------------

# 15. 官方资料与参考链接

以下资料用于开发时核对模型能力、API
和索引行为。实现前应查看对应项目当前版本文档，不要直接复制旧版本 API。

1.  **BGE-M3 模型卡**\
    https://huggingface.co/BAAI/bge-m3\
    多语言、Dense/Sparse/Multi-vector、向量维度和最大输入长度。

2.  **FlagEmbedding 项目**\
    https://github.com/FlagOpen/FlagEmbedding\
    BGE-M3 的模型加载、Embedding 和 Reranking 示例。

3.  **BGE-Reranker-v2-m3 模型卡**\
    https://huggingface.co/BAAI/bge-reranker-v2-m3\
    多语言 Reranker 的使用方法和许可证。

4.  **Qdrant 官方文档**\
    https://qdrant.tech/documentation/\
    Collection、Payload、向量存储、客户端和服务配置。

5.  **Qdrant HNSW / Indexing**\
    https://qdrant.tech/documentation/manage-data/indexing/\
    HNSW、Payload Index、Sparse Index 和索引配置。

6.  **Qdrant Hybrid Queries**\
    https://qdrant.tech/documentation/hybrid-queries/\
    Dense、Sparse、BM25、RRF、DBSF 和多阶段查询。

7.  **Qdrant Quantization**\
    https://qdrant.tech/documentation/guides/quantization/\
    量化和内存优化。

8.  **LangChain Recursive Text Splitter**\
    https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter\
    结构优先递归切分的参考实现。注意其默认长度计量可能是字符数，本方案要求使用模型
    Tokenizer 控制 Token 长度。

9.  **HNSW 原始论文**\
    https://arxiv.org/abs/1603.09320\
    *Efficient and Robust Approximate Nearest Neighbor Search Using
    Hierarchical Navigable Small World Graphs.*

------------------------------------------------------------------------

# 16. 给 Codex 的实施要求

开始开发时，必须遵循以下约束：

1.  先阅读本方案，按照模块边界实现，不要擅自合并
    Parser、Chunker、Embedding、Retriever 和 Generator。
2.  第一阶段只实现 Markdown/TXT、Recursive Parent-Child Chunking、BGE-M3
    Dense、Qdrant。
3.  每个模块提供独立接口和单元测试。
4.  不要一开始就实现 Semantic Chunking、Late Chunking、分布式检索和复杂
    Agentic RAG。
5.  跨语言检索阶段再增加 Query Translation、BM25、RRF。
6.  Reranker 必须可配置启停。
7.  所有关键参数从 YAML 加载，不允许散落在业务代码中。
8.  所有数据都必须保留来源和版本信息。
9.  所有导入过程必须支持幂等、失败重试和进度恢复。
10. 先完成 1 万条数据的端到端验收，再逐步扩展到 10 万和 100 万。
11. 不得声称达到百万级性能，除非完成目标机器上的实际压测并提供指标。
12. 每完成一个 Phase，更新 README、测试结果和已知问题。

**第一项交付物：** 完成 Phase 1，提供可运行的文档解析与 Parent-Child
Chunking 程序、示例输入、JSONL 输出、单元测试和 Chunking 质量报告。

------------------------------------------------------------------------

# 17. 现状差距清单（可执行，v1.1 追加；决策已更新至 v1.2）

> 本章由 2026-11-02 对现有实现（`local-rag-agentic-lab`）与本方案 v1.0 的逐模块核对产出。
> 每一条都是**可执行、可验收**的：`验收方式` 列写的是命令 + 期望结果，不是"做得更好"这类口号。
> 图例：❌ 缺（整层不存在）｜⚠️ 部分（有实现但与方案目标不符）｜✅ 已达成（无需重做）

## 17.0 使用说明

**编号规则**：`D` = 功能差距，`E` = 跨阶段工程项，`B` = 阻塞决策（决策未定时，下游 D 不得开工）。

**命令映射**：本清单的验收命令按方案 §6 的目标目录书写（`scripts/`、`src/rag/`、`config/`）。
在目录迁移（E06）完成前，同一功能在当前仓库里对应 `python -m src.<module>`；两套写法在 E06 完成后统一。

**排序规则**：同一 Phase 内严格按依赖顺序排列，不允许并行开工下游项。

---

## 17.1 现状快照（作为一切改造前的对照基线）

| 项 | 当前实测值 | 方案目标 | 状态 |
|---|---|---|---|
| 切分单位 | **字符**：`CHUNK_SIZE=500` / `CHUNK_OVERLAP=80`；实测 chunk 平均 **134 token**（bge-m3 tokenizer 口径）/ 453 字符 / 79 词 | **Token**：Child 400–500 token、overlap 60 token | ⚠️ 差 3×（B02 已决定重切分） |
| 层级 | 单层 chunk，无 Parent | Parent 1500–2000 token + Child | ❌ |
| 文档解析 | md/txt 直读 + pypdf 抽文本，仅拼 `[PDF 第 n 页]` 标记 | `DocumentBlock` AST（标题/段落/列表/表格/代码 + section_path + 页码） | ❌ |
| Chunk 数 | 81,027（1,467 个源文件） | 首阶段 1 万条验证，终态支持 100 万 | — |
| Embedding | Ollama `/api/embed` + `bge-m3`，1024 维 | `BAAI/bge-m3` + FlagEmbedding，本地批量 | ⚠️ |
| 向量存储 | `storage/embeddings.npy`（81,027×1024 float32，332 MB，进程内常驻） | Qdrant collection + HNSW | ❌ |
| 向量检索 | 暴力全扫（`matrix @ q` 4.2 ms，56 GB/s 已贴内存带宽墙） | Qdrant HNSW `m=16 / ef_construct=200 / ef_search=100` | ❌ |
| 关键词检索 | 自研 Okapi BM25 + jieba，输入为**中文 query** | **英文** BM25（Qdrant full-text），输入为翻译后英文 query | ⚠️ |
| Query 处理 | 无 | 语言识别 + 翻译 + 关键词/实体 + 缓存 + 降级 | ❌ |
| 融合 | 应用层 RRF（`hybrid`）+ min-max 加权（`hybrid_score`） | 应用层 RRF，三路（zh dense / en dense / en BM25） | ⚠️ |
| 精排 | 无 | `bge-reranker-v2-m3`，Top50 → Top5–10 | ❌ |
| 上下文组装 | `format_context()` 全量拼接，无展开/去重/预算；top-4 中同文档多 chunk **79.6%**、相邻重叠块 **22.9%**；全库重复文本 **23.3%** | Parent 展开 + 邻接合并 + Token Budget 6000 | ❌ |
| 对外接口 | CLI REPL（`python -m src.rag`） | FastAPI 四端点 + Pydantic | ❌ |
| 参数管理 | 环境变量 + 代码内硬编码（`depth=3`、`rrf_constant`、`alpha`） | YAML + Pydantic Settings，禁止散落 | ⚠️ |
| 测试 | 45 个 stdlib `unittest`（覆盖指标/日志/建库重试） | pytest：unit / integration / performance | ⚠️ |
| 评估集 | 60 题（48 正 + 12 负），**单向中文 query** | 100–200 条**中英成对**题 | ⚠️ |
| 检索指标 | Recall@4/@10 + hit + full + MRR@10 + nDCG@10 + bootstrap CI + McNemar + 切片 + 负样本 z 分数 + 基线门禁 | Recall@K / MRR / nDCG + ANN Recall@K + 消融 A–G | ⚠️ |
| Chunking 评估 | 无 | 30–50 篇人工抽查 8 项 + A–E 对照 | ❌ |
| 跨语言实测 | 中文问英文语料：Recall@4 **15.9%** / @10 **31.9%**（`hybrid`；`hybrid_score` 为 23.0% / 34.3%） | 验收门槛 Recall@10 ≥ **0.85** | ❌ 差 2.7× |
| 语料语言构成 | 英文 79,244 chunk / 中文 1,783 chunk（97.8% 英文） | 方案未涉及 | ❌ 方案缺口 |
| 建库韧性 | 断点续建（fingerprint 校验）+ 400 原地重试，实战中断 3 次无损 | 可从上次成功批次恢复 | ✅ |

**基线标识（改造前的对照点）**：评估 run `20261102T143004-2188`，`content_hash = 3e08ef4c1c4f058e`，dataset `05ba575e4f1ea96e`。

---

## 17.2 Phase 1：文档解析与 Parent-Child 切分

| 编号 | 依赖 | 任务 | 验收方式 |
|---|---|---|---|
| D01 | — | ✅ **已完成（2026-11-02）** `DocumentBlock` 数据结构 + Parser 基类接口（方案 §2.2） | `python -m unittest tests.test_document_block tests.test_parser_base`（14 个用例）全绿：7 种 `block_type` 均可构造、未知类型报错且列出合法值、heading 缺 level 报错、`source_start>source_end` 报错、`to_dict/from_dict` 往返一致、缺省容器不共享、注册表大小写不敏感分派、未知后缀报错并列出已注册后缀。（原计划用 pytest + md 样本计数，改为 unittest；「heading 数 == `^#+ ` 行数」移入 D02，因为那需要真实 Markdown 解析器） |
| D02 | D01 | ✅ **已完成（2026-11-02）** Markdown Parser（标题层级、列表、表格、代码块、引用、图注、setext 标题；标题不脱离正文） | `python -m unittest tests.test_markdown_parser`（32 用例）全绿，含「表格紧跟标题」「代码块内含 `#`」两个陷阱；`python -m src.rag.ingestion.parser check data/md` → **section_path 非空率 100.00%、空 block 0**（1,227 文件 / 99,479 block）；heading 块 31,446 == 文内 `^#+ ` 行 35,741 − 代码块内 `#` 行 4,295 |
| D03 | D01 | ✅ **已完成（2026-11-02）** TXT Parser（空行分段、软换行保留、句子数统计、BOM/CRLF 归一） | `python -m unittest tests.test_txt_parser`（16 用例）全绿；段落数 == 空行分隔段数、空 block 0；`check` 命令在混合语料上回报空 block 0 |
| D04 | D01 | ✅ **已完成（2026-11-02）** PDF Parser（pymupdf `get_text("dict")`：页码 + bbox + 字号；按字号相对比识别标题；重复页眉页脚剔除并留痕） | `python -m unittest tests.test_pdf_parser`（22 用例）全绿；`check data` → **PDF 页码覆盖率 100.00%**；30 页字符数：pypdf 62,779 vs pymupdf 59,874（0.954），**去空白后 47,687 vs 47,464（0.995）**——差异来自 pypdf 给每个 span 补空白（其空白占比 24.0% vs 20.7%），非内容丢失；人工抽查表用 `python -m src.rag.ingestion.parser.pdf_parser report <pdf> --pages 1-30 --out report.csv` 生成 |
| D05 | B01 | ✅ **已完成（2026-11-02）** 统一 Token 计数（BGE-M3 tokenizer 口径） | `python -m src.rag.ingestion.tokenizer count "文本"` 输出整数；`verify --n 8` 返回 `aligned: true`（实测逐条中位误差 0%、最大绝对差 0）；17 个单测覆盖空串/未知词/批量一致/`truncate` 边界/词表缺失提示。**原「与 FlagEmbedding 计数差 0」已作废**（B01 决定不引入 FlagEmbedding），改为「与 Ollama `prompt_eval_count` 逐条对齐」 |
| D06 | D05 | ✅ **已完成（2026-11-02）** Recursive Splitter（分隔符优先级 + `split_preserving_separator` + `merge_small_parts` + token 窗口兜底） | `python -m unittest tests.test_recursive_splitter tests.test_recursive_splitter_real`（35 用例）全绿，6 类边界（超长段落/超长句/无分隔符/超长单词/代码块/表格）逐类断言；`python -m src.rag.ingestion.recursive_chunker check --n 2000` → **超限率 0.00%、还原失败 0**；不变量 `"".join(pieces) == text` 在 300 条真实 chunk + 100 篇拼长文档上全部成立 |
| D07 | D06 | ✅ **已完成（2026-11-02）** Overlap（`overlap.py`：句子/行边界优先挑重叠起点，越界退回 token 窗口；`TokenCounter.truncate_tail` 保证后缀是原文子串） | `python -m unittest tests.test_overlap`（16 用例）全绿；真实语料 **前 100 对相邻 chunk：overlap ∈ [50,70] 占比 100.0%**（min 50 / p50 60 / max 70）；**69.5%** 的重叠起点落在句子或行边界之后；child token ≤ 500 超限 0 |
| D08 | D06 | ✅ **已完成（2026-11-02）** Parent Builder（标题/章节边界分组 → block 边界累积 → 单块超限用 D06 递归切分 → 短块并进同一父章节） | `python -m unittest tests.test_parent_builder`（23 用例）全绿；40 篇真实 md：**parent 194 / child 292，超限 0，孤儿 child 0（关联成功率 100.00%）**；parent token p50 256 / p90 1044 / max 2000，child p50 308 / p90 441 / max 499 |
| D09 | D06,D08 | ✅ **已完成（2026-11-02）** Chunk Validator（`validator.py`：空/超限/重复/边界/元数据/孤儿 + 父块主键重复自检；报告四项数字 + `*.invalid.jsonl`） | `python -m unittest tests.test_validator`（23 用例）全绿；40 篇真实 md：**空 0、超限 0、重复占比 0.31%、边界完整率 100.00%**；gold 收敛统计输出 0/1/2/3+ 直方图与 `converged_ratio` |
| D10 | D01,D08,D09 | ✅ **已完成（2026-11-02）** 稳定 ID（`ids.py`：document 认来源路径、parent/chunk 认内容+版本，`` 分隔防拼接撞车） | `python -m unittest tests.test_stable_ids`（7 用例）全绿；40 篇语料两次运行 **三个 JSONL 的 sha256 完全一致**；chunker_version 变更后 **ID 复用数 0** |
| D11 | D09,D10 | ✅ **已完成（2026-11-02）** 质量报告（`quality_report.py`：A–E 方案对照 + 8 项抽查自动指标 + 人工评分列 + 逐文档明细，E 显式标 not_implemented） | `python -m src.rag.ingestion.quality_report --dir data/md --limit 40 --out <md> --json <json>`；`tests.test_quality_report`（25 用例）全绿；40 篇真实语料报告：A 断句 97.53% / B 0% / C 0% / D 0%，D 的碎块率 **14.76%**、重复 1.81%、heading 关联 87.65%、overlap 合规 99.24% |
| D12 | D10 | ✅ **已完成（2026-11-02）** JSONL 输出（`jsonl_store.py`：严格字段契约，多一个字段也报错；UTF-8 无 BOM / LF / 一行一 JSON；manifest 记 sha256） | `python -m unittest tests.test_jsonl_writer`（11 用例）全绿；40 篇语料写出 documents 40 / parents 194 / children 292 行，**逐行可解析且字段集与 §3.1–3.3 完全一致**；`jq` 存在时用例会真跑 `jq -c .`（本机无 jq 时跳过并注明） |

**Phase 1 出厂标准（DoD）**：D01–D12 全部通过 → 交付"可运行的解析+切分程序 + 示例输入 + JSONL + 单测 + 质量报告"，即方案 §16 要求的第一项交付物。

---

## 17.3 Phase 2：Embedding、Qdrant 与数据一致性

| 编号 | 依赖 | 任务 | 验收方式 |
|---|---|---|---|
| D13 | D05,B01 | ✅ **已完成（2026-11-02）** Embedding 服务抽象（`src/rag/embedding/`：`encode_documents` / `encode_queries`、批量与二分重试、维度/NaN/Inf/全零校验、内容 hash 缓存（SQLite + 进程内 LRU）、`stats` 与 CLI） | `python -m unittest tests.test_embedding_service`（31 用例）全绿；真实 24 条 chunk：**冷启 p95 2212 ms → 内存命中 p95 0.019 ms（113,489×）、SQLite 命中 p95 3.1 ms（713×）**，二次调用命中率 **100%**；单条 query 冷 2176 ms → 命中 6.78 ms（<10 ms）；维度≠1024 / NaN / Inf / 全零各自抛错 |
| D14 | D13 | ✅ **已完成（2026-11-02）** `embedding_text` 与 `content` 分离 + 前缀可统计 + 有/无前缀 A/B（`src/rag/ingestion/embedding_text_report.py`：`prefix_stats`/`reconstruction_stats`/`query_metrics`/`ab_summary`/`run_ab` + CLI；`DocumentStore.parents_containing`/`documents_containing` 供金标选样） | `python -m unittest tests.test_embedding_text_report` **18 用例**；`--stats-only` 全量前缀覆盖 **100%**、可重建 **100%**；A/B：48 题、10,028 点 → **MRR 0.429 → 0.554（+0.125）**、hit@10 +2.1pt、Recall@10 +0.9pt（两次独立运行同向） |
| D15 | B04 | ✅ **已完成（2026-11-02）** Qdrant 部署 + Collection + Payload Index（6 个过滤字段）；`docker-compose.yml` 固定 `qdrant/qdrant:v1.15.0`，`src/rag/store/qdrant_store.py` 管建/读/删 + CLI | `docker compose up -d qdrant && python -m src.rag.store.qdrant_store create`；`curl -s localhost:6333/collections/rag_chunks_v1` → **size=1024、distance=Cosine、hnsw m=16、ef_construct=200、payload_schema 6 个**（实测已核对）；`python -m unittest tests.test_qdrant_store` 18 用例全绿 |
| D16 | D10,D15 | ✅ **已完成（2026-11-02）** 幂等 Upsert + 版本管理 + 按文档删除（`src/rag/store/indexer.py`：point id = `point_id_for(chunk_id)` 覆盖写、陈旧点集差集删除、按 `document_id` 过滤删除、先 Store 后 Qdrant 的失败可自愈顺序） | `python -m unittest tests.test_indexer` 14 用例；真实 3 篇：首轮 13 点 → **重复导入 13 点不变、陈旧点 0**；版本变化时 **陈旧点 6 个被删、点数 = 新 child 数**；`delete_document` 后 **按 document_id 过滤计数 0** |
| D17 | D08 | ✅ **已完成（2026-11-02）** SQLite Document Store（`src/rag/store/document_store.py`：documents / parents / children 三表 + 外键强制 + 整篇替换事务 + `import_chunks_dir` / `verify_against_jsonl` / `get_parent_for_chunk` + CLI） | `python -m unittest tests.test_document_store` 18 用例；真实 20 篇：**`SELECT count(*) FROM parents` = 102 == JSONL 行数 102**（documents 20/20、children 182/182）；任意 `chunk_id` → `python -m src.rag.store.document_store show <chunk_id>` 可查到父块原文 + `source_uri` + `section_path`；重复导入计数不变 |
| D18 | D16,D17 | ✅ **已完成（2026-11-02）** 导入状态机 + 失败清单 + 断点恢复（`src/rag/ingest/state.py` 10 个状态 + `resolution` 留痕、`pipeline.py` 编排（窗口攒批 + 逐阶段落库）、`scripts/ingest.py` 一条命令入口） | `python -m unittest tests.test_ingest_state tests.test_ingest_pipeline`（33 用例）；**子进程 `os._exit(9)` 硬杀后重启：3 篇 `[skip]` 不重算、3 篇续跑，用时 10.23 s → 1.55 s**，Qdrant 点数 6（不是 9）；失败文件进 `--failures` 清单、修好后 `--only` 单条重试成功并标 `resolution=completed` |
| D19 | D15 | ✅ **已完成（2026-11-02）** Dense 检索 + 精确基准对照（`src/rag/store/search.py`：`qdrant_search` 带过滤与 `ef`、`exact_topk` numpy 暴力、`ann_recall`、`measure_ann_recall`、CLI） | `python -m unittest tests.test_search` 11 用例；真实 bge-m3 向量 + 20 条真实 query：**1 万点 Recall@50 = 1.000**（≥0.99 ✓）、**10 万点 = 0.990**（≥0.95 ✓，ef=256）；`python -m src.rag.store.search --queries 20` 可直接复跑 |
| D20 | D12,D13,D16,D17,D19 | ✅ **已完成（2026-11-02）** 端到端 DoD：`scripts/ingest.py` 一条命令 全语料 → Qdrant + Document Store；删除/重建闭环（补上了「删除时同步清状态表」的缺口） | `python scripts/ingest.py --dir data --limit 10000` → **1467 篇 / 25,411 parent / 38,865 child / 0 失败 / 1992 s**；**中英文 query 各 5 条：hit@5 = 8/10、父块回溯 10/10**；删除 1 篇后 38,865 → 38,859，重跑恢复 **38,865**（精确回到基线） |
| D21 | D15 | ✅ **已完成（2026-11-02）** 索引版本化（`src/rag/store/index_versions.py`：collection↔模型口径登记表 + `assert_compatible` 写入闸门 + `status` active/retired + `collection_name_for` 建议新名；已接进 `Indexer` 与 `scripts/ingest.py`） | `python -m unittest tests.test_index_versions` 13 用例 + `tests.test_indexer` 4 个闸门用例；**负向实测：`python -m src.rag.store.index_versions check rag_chunks_v1 --embedding-model bge-m3 --embedding-version bge-m3-v2 --vector-size 1024` → 退出码 1**，报错含建议名 `rag_chunks_bge_m3_v2`；同口径检查退出码 0 |

**Phase 2 出厂标准**：D13–D21 通过 → 1 万条数据可从原始文档一路走到 Qdrant 检索并可删除/重建。

---

## 17.4 Phase 3：跨语言检索

| 编号 | 依赖 | 任务 | 验收方式 |
|---|---|---|---|
| D22 | — | ✅ **已完成（2026-11-04）** 语言识别（`src/rag/lang/detect.py`：汉字数 /（汉字数 + 英文单词数）比 + 0.20 阈值） | `python -m unittest tests.test_lang_detect` **15 用例**；`python -m src.rag.lang.detect --questions experiments/questions.json` → **202 条样本（zh 101 / en 101）准确率 100.00%**（验收线 95%）；`HNSW 是什么？` → **zh**（ratio 0.75）、`DGX B300 装了几张 GPU？` → zh（0.5714）、`What is RRF?` → en、`8192` → unknown |
| D23 | D22 | Query 翻译服务（Ollama + 严格 JSON + 术语/型号/数字保护 + 缓存 + 失败降级） | 200 条中文 query 翻译成功率 ≥98%；`DGX B300`、`8192`、`HNSW` 原样保留率 == 100%；模拟超时后仍返回原 query 且检索不中断（单测） |
| D24 | D19,D23 | 三路召回编排（zh dense / en dense / en BM25） | 一次请求输出三路各自 Top50 与融合结果（debug 字段）；任一路可单独关闭且其余路结果与单独运行逐条一致 |
| D25 | D15,D23 | ✅ **已完成（2026-11-04）** 英文 lexical（`src/rag/store/lexical.py`：`ensure_text_index` + `fulltext_filter`(OR) + `load_payloads`/`build_bm25`/`BM25Index` + CLI 中英对比）。**依赖已解除**：直接用评估集的 `query_en`（§17.11.3），不等 D23 翻译服务 | `python -m src.rag.store.lexical --questions experiments/questions.json --k 10` → 83 题可判定：中文 query **46/83（55.42%）** vs 英文 query **57/83（68.67%）**，**+11 条**；关键词类子集 32 题：**13/32（40.62%）vs 24/32（75.00%）**，+11 条 / Recall **+32.55 pt** |
| D26 | D24 | 应用层 RRF 三路融合 + 跨列表按 `chunk_id` 去重 | 输出无重复 `chunk_id`；同 chunk 多路命中分数正确累加；单测覆盖 k=60、并列名次、空列表 |
| D27 | D11,B02,B06 | ✅ **已完成（2026-11-04）** 评估集扩到 **101 条中英成对题**（`src/eval_bilingual.py` 校验骨架 + `experiments/expand_questions.py` 增量扩题） | `python -m src.eval_bilingual check --min-total 100` → **0 error**；`python -m src.eval_dataset check` → **0 error**；**成对率 100.00%（101/101）**、`topic` **8/8 覆盖**（concept_zh 6 / zh_with_en_term 9 / acronym_model 8 / number_spec 24 / paraphrase 6 / multi_condition 3 / cross_chunk 27 / no_answer 18）、`difficulty` easy 29 / normal 51 / hard 21、待审草稿 **0** |
| D28 | D27,D26 | 消融矩阵 A–E 跑通并出报告 | A–E 每格给出 Recall@5/@10、MRR@10、nDCG@10、P50/P95、翻译耗时；配 McNemar / bootstrap CI |
| D29 | D28 | 语言不平衡治理（同语言加权或来源配额） | 中文语料切片不低于改造前水平（`hybrid` 70.0% / `hybrid_score` 75.0%），同时英文切片下降不超过基线 CI 半宽 |
| D30 | D28 | 跨语言门槛核算 | 中文问英文语料 Recall@10 报告值（当前 31.9%）与门槛 0.85 的差距归因表（翻译/精排/Parent/语料失衡各占多少） |

**Phase 3 出厂标准**：D22–D30 通过 → 三路召回 + RRF 可用，且能说清"离 0.85 还差多少、差在哪一环"。

---

## 17.5 Phase 4：Reranker 与上下文组装

| 编号 | 依赖 | 任务 | 验收方式 |
|---|---|---|---|
| D31 | D19 | Reranker 服务（`bge-reranker-v2-m3`，可开关、批量、并发上限） | 50 候选 × 1 query 的 P50/P95 延迟与 CPU 峰值 RSS 实测记录；关闭开关时结果与 D26 逐字节一致 |
| D32 | D17,D31 | Context Assembler（Parent 展开 + 邻接合并 + 去重 + token budget 6000 + 来源字段） | 上下文内"同文档相邻重叠块"占比 **≤5%**（当前 22.9%）；上下文 token ≤6000；每块含 `document_id/chunk_id/section_path` |
| D33 | D32 | Prompt 与引用规范（§4.9 七条 + 兼容现有 `[证据 n]`） | 引用越界（n>k）计数 == 0；证据不足时输出"证据不足"；引用解析成功率 ≥99% |
| D34 | D32 | 消融 F/G（+Reranker、+Parent Expansion） | F/G 相对 E 的配对检验；rerank 前后 nDCG@10 差值写入报告 |
| D35 | D33 | L2 答案层评估（引用正确率、忠实度、弃答三角） | 40 题人工打分表；给出四个数：引用精确率、faithfulness、错误自信率、过度弃答率；负样本弃答率单列 |

**Phase 4 出厂标准**：D31–D35 通过 → 上下文可复现、可定位、无重复膨胀，且答案层有第一版人工数据。

---

## 17.6 Phase 5：百万级与 ANN

| 编号 | 依赖 | 任务 | 验收方式 |
|---|---|---|---|
| D36 | D19 | ANN Recall@K 基准脚本（ANN vs 精确 Top-K） | `scripts/benchmark_ann.py --n 100000` 输出 Recall@10 与交集明细；固定 seed 可复现 |
| D37 | D36 | HNSW 参数与量化实验 | `m/ef_construct/ef_search` 网格表；float32 vs int8（±rescore）的 Recall@10 / 内存 / P95 三列对照 |
| D38 | D37 | 百万级数据生成与压测 | 1,000,000 点导入完成；构建时间、内存峰值、P50/P95、Recall@10 全部记录，硬件与参数写入报告 |
| D39 | D31,D38 | 16GB 内存预算与进程隔离 | 同时加载 embedder + reranker + LLM 的 RSS 实测；给出分阶段运行方案与并发上限；超限时脚本明确报错而不是 OOM |

---

## 17.7 跨阶段工程项

| 编号 | 依赖 | 任务 | 验收方式 |
|---|---|---|---|
| E01 | — | YAML + Pydantic Settings 收敛全部参数 | `depth / rrf_k / alpha / top_k / ef_search / chunk_size` 均来自 YAML；`rg -n "depth=3" src/` 与 `rg -n "alpha=" src/` 均无命中 |
| E02 | D20 | FastAPI 四端点 + Pydantic schema | `POST /api/v1/ask` 返回 `answer` + `sources`；`/retrieve?debug=true` 含三路候选数；OpenAPI 文档可访问 |
| E03 | B03 | 测试体系（pytest：unit / integration / performance） | 三类目录齐全；现有 45 个 `unittest` 迁移后仍全绿；新增 chunking / RRF / translation 单测 |
| E04 | D20 | 日志与评估资产迁移（保留 `raglog` 与 L1 评估，接新 pipeline） | 新 pipeline 产出同类事件（`retrieve`/`prompt`/`generate`）；`measure` 能对新日志出报告 |
| E05 | E04,D27 | 新基线重立 + 回归门禁 | 生成新 `baseline.json`；人为劣化时 `compare` 返回 exit 1 |
| E06 | — | 目录结构与脚本迁移（`src/rag/*`、`scripts/*`、`config/*`） | 目录与 §6 一致；README 命令同步；旧入口保留一个版本周期并在 README 标注 deprecated |

---

## 17.8 依赖序（执行批次）与里程碑

下表是按下表的依赖关系算出的**最长路径分层**：同一批次内的项互不依赖，可并行开工；下一批次必须等上一批次对应依赖完成。
（例如 D09 需要 D06 与 D08，而 D08 在 W2，所以 D09 落到 W3。）

| 批次 | 可并行开工 | 前置来自 |
|---|---|---|
| **W0** | D01（Block/Parser 接口）、D05（Token 计数，需 B01）、D13（Embedding 服务，需 D05/B01）、D15（Qdrant，需 B04）、D22（语言识别）、E01（YAML 收敛）、E03（测试框架，需 B03）、E06（目录迁移） | — |
| **W1** | D02、D03、D04、D06、D14、D19、D21、D23 | W0 |
| **W2** | D07、D08、D24、D25、D31、D36 | W0/W1 |
| **W3** | D09、D17、D26、D37 | W2 |
| **W4** | D10、D32、D38 | W3（D32 还需 D17） |
| **W5** | D11、D12、D16、D33、D34、D39 | W4 |
| **W6** | D18、D20、D27、D35 | W5（D27 还需 B02/B06） |
| **W7** | D28、E02、E04 | W6 |
| **W8** | D29、D30、E05 | W7 |

> B01/B02/B06 已于 2026-11-02 定案（见 17.11），W0 批次可直接开工。仍待定：B05（中文 BM25 去留，须在 W2 前定，它是 D25/D26/D28 的设计前提）、B03（测试框架）、B04（Qdrant 时机）。

**关键路径（决定总工期）**：`D05 → D06 → D09 → D10 → D16 → D20 → D27 → D28 → E05`，以及旁支 `D06 → D08 → D17 → D32 → D33 → D35`。
这两条链上的任何一项延期都会直接推迟 M3/M4，因此 B01、B02、B06 三个决策必须在 W0 之前定下来。

| 里程碑 | 内容 | 通过判据 |
|---|---|---|
| **M1** | Phase 1 完成（解析 + Parent-Child + 质量报告） | 17.2 全部 D 项通过 |
| **M2** | Phase 2 完成（1 万条 → Qdrant → 可删可重建） | 17.3 全部 D 项通过 + ANN Recall@10 ≥0.99（1 万点） |
| **M3** | Phase 3 完成（三路召回 + 成对评估集 + 消融 A–E） | 17.4 全部 D 项通过；给出跨语言 Recall@10 与门槛差距归因 |
| **M4** | Phase 4 完成（Reranker + Context Assembler + L2 首版） | 17.5 全部 D 项通过；上下文重叠块占比 ≤5% |
| **M5** | Phase 5 完成（百万级压测报告） | 17.6 全部 D 项通过；报告含硬件/参数/构建/内存/延迟/Recall |

---

## 17.9 阻塞决策与结论

| 编号 | 决策问题 | 结论 | 影响的下游 |
|---|---|---|---|
| **B01** | Embedding 走 Ollama API 还是 FlagEmbedding 本地 | ✅ **已定（2026-11-02）：走 Ollama API**（保持现状，不引入 FlagEmbedding / torch 依赖）。细则见 17.11.1 | D05、D13、D39 |
| **B02** | 是否按方案重切分（500 字符 → Child 400–500 token） | ✅ **已定（2026-11-02）：按方案重切分**（Parent-Child + Token 口径）。影响评估见 17.11.2 | D05–D12、D27、E05 |
| B03 | 测试框架 pytest 还是 stdlib unittest（现状环境无 pytest） | ⏳ 未定。建议：不阻塞交付，先用 unittest 保持目录分层 | E03 |
| B04 | Qdrant 引入时机 | ✅ **已定（2026-11-02，用户拍板）：Phase 2 起直接用 Qdrant**，不做「先 numpy 后切换」。理由见 §17.11.4：过滤能力是硬需求（实测 Top-50 分散在 23 个文件、单文件最多 6 条）、numpy 兜底等于做两套、本机 Docker + `qdrant/qdrant:v1.15.0` 已验通 | D15–D21、D25、D36（**解开 24 项**） |
| B05 | 中文 BM25 是否保留（当前 jieba BM25 对 1,783 条中文 chunk 有效，方案只要英文 lexical） | ⏳ 未定。建议：两套并存，消融单列一路 | D25、D26、D28 |
| **B06** | 评估集重建成本与方式 | ✅ **已定（2026-11-02）：复用 `experiments/build_questions.py`**，新增 `query_en` 字段，不重写工具链。细则见 17.11.3 | D27、D28、E05 |
| **B07** | Parent `min_tokens` 是否从 200 提到 400 | ⏳ **待定（D11 数据已出）**。40 篇实测：碎块率 14.3% → **7.0%**、重复 0.31% → **0**、child/parent 1.66 → 2.20；代价是合并后 `section_path` 变粗。**必须等 D28 检索级消融确认收益后再改默认值** | D08 默认配置、D20 端到端门槛、D28 消融 |

---

## 17.10 迁移影响：改造后哪些资产会失效

| 资产 | 影响 | 处置 |
|---|---|---|
| `experiments/questions.json` 的 `gold_chunks` | 重切分后 chunk id 全变（B02） | 用 `build_questions.py` 重新定位；`gold_spans` 不变，可复用 |
| `experiments/baseline.json`（`3e08ef4c1c4f058e`） | `content_hash` 变，`compare` 会判 `not_comparable` | 按 E05 重立基线；旧基线归档保留，禁止混比 |
| `storage/index.json` + `embeddings.npy` | 被 Qdrant 取代 | 保留为"精确检索基准"，D19/D36 需要它算 ANN Recall |
| `src/retrieve.py` 的 4 个模式 | 由三路召回 + RRF 取代 | `hybrid`/`hybrid_score` 保留作消融对照组（方案 §11.4 的 E 组） |
| `src/raglog.py`、`src/eval_*.py`、`tests/`（45 用例） | **不失效，是差异化资产** | 直接接入新 pipeline（E04）；指标/日志口径继续沿用 |
| 60 题评估集与 5 组探针结果 | 指标不可与改造后直接比较 | 作为"改造前基线"存档，仅用于说明改造收益的方向与量级 |


## 17.11 决策记录（ADR）

> 记录 2026-11-02 已确认的三项决策：背景、结论、实测依据、对下游条目的具体调整。
> 每条决策末尾的「实施细则」是本清单的执行口径，改动时须同步改这里和对应 D 项。

### 17.11.1 B01：Embedding 走 Ollama API（不引入 FlagEmbedding）

**背景**：方案 §7.2 建议 `BAAI/bge-m3` + FlagEmbedding 本地推理；现状是 Ollama `/api/embed`。两条路的能力差异主要在「单条 query 延迟」与「tokenizer 可控性」。

**结论**：**走 Ollama API**（现状）。理由与实测：

| 项 | 实测值 | 说明 |
|---|---|---|
| 单条 query embedding 延迟 | **2.1 s**（nomic 2095 ms / bge-m3 2144 ms） | 与模型大小无关，是 Ollama 单次调用固定开销 → **必须有 query 向量缓存**，否则每次提问都吃 2 s |
| 批量吞吐 | batch 256 ≈ **24 ms/条**；全量 81,027 条 ≈ 40 min | 已实现断点续建；batch ≥224 时有 ~30% 概率返回 400，已实现「原地重试 → 再二分」 |
| 依赖面 | 仅 `requests` | 不引入 torch / FlagEmbedding（Windows + 16GB 内存约束下收益低） |
| 已测能力 | 1024 维、8192 上下文、跨语言尾部（gold 排名 p90 15,394 → 145） | 与方案对 bge-m3 的预期一致 |

**代价与补偿**：

1. **无法拿到逐条 token 数**。`/api/embed` 只返回整个 batch 的 `prompt_eval_count`（如 200 条 chunk → 26,867 token，平均 134 token/条），逐条调用等于每条 2 s，81k 条不可行。
   → **D05 执行口径改为**：token 计数走**本地 tokenizer**（`tokenizers` 库读 bge-m3 的 `tokenizer.json`，纯 CPU、毫秒级，不跑模型推理），Ollama 的 `prompt_eval_count` 只用于**逐条抽样校验**。已实施并对齐（2026-11-02）：单条 `prompt_eval_count == count(text) + 2`（BOS+EOS），逐条中位相对误差 **0%**、最大绝对差 **0 token**；用 nomic 冒充时中位 **5.97%**、最大差 **107 token**（负向对照，证明校验有分辨力）。
   → 注意**不能用聚合总量校验**：200 条 chunk 上换错模型的聚合差只有 **0.35%**，正负误差相互抵消，测不出来。
   → 注意：这不违反 B01 —— B01 约束的是"向量计算走 API"，不是"tokenizer 也必须走 API"。
   → 用字符/词数**估算** token 不可接受：实测相对误差 p50 15.4% / **max 43.1%**，无法支撑「max 500 token 硬上限」的验收（D06 要求超限率 == 0）。
2. **query embedding 缓存成为硬需求**（新增到 D13）：`query hash → 向量` 的 LRU 或 SQLite 缓存，命中率目标 ≥60%（评估集场景可达 ~100%）。
3. **P95 预算要扣掉 2.1 s**（D39）：这条固定开销与语料规模无关，百万级时它仍是延迟主项。

**对下游条目的调整**

| 条目 | 调整 |
|---|---|
| D05 | token 计数改为本地 tokenizer；新增验收：与 Ollama `prompt_eval_count` 抽样对比误差 ≤2% |
| D13 | 新增"query embedding 缓存"（命中率 ≥60%，命中时延迟 <10 ms） |
| D39 | 延迟预算明确包含 2.1 s 的 embed 固定开销 |

---

### 17.11.2 B02：按方案重切分（Parent-Child + Token 口径）

**背景**：当前是 500 **字符**单层切分，实测平均 **134 token**；方案要求 Child 400–500 token + Parent 1500–2000 token。差 3 倍，不重切分则 Phase 3/4 的收益（翻译、Reranker、Parent 展开）都会被错误切分掩盖。

**结论**：**按方案重切分**。基于实测 token 数据（81,027 条 × 平均 134 token ≈ **10.88 M token**）的规模预估：

| 指标 | 当前 | 重切分后（预估） | 变化 |
|---|---|---|---|
| Child 数 | 81,027 | **≈ 31,000**（10.88M ÷ 400 × 1.15 重叠系数） | ÷2.6 |
| Parent 数 | 无 | ≈ 7,200（10.88M ÷ 1500） | 新增 |
| 向量存储 | 332 MB（81,027×1024×4） | ≈ 127 MB | ÷2.6 |
| 全量建库耗时 | 40 min | ≈ 15 min | ÷2.6 |
| 暴力全扫延迟 | 4.2 ms | ≈ 1.6 ms | ÷2.6 |

**对评估体系的 4 个具体影响（必须一起处理）**

1. **`gold_chunks` 全部失效**：chunk id 由内容 hash 生成，切分变化后 85 条 gold 的 31 个唯一 id 全部改变。处置：`gold_spans` 不变（仍是语料原文），用 D10 的 `locate()` 重新定位（B06 已决定复用工具链）。
2. **多 gold 题会收敛**：现在 24 题有 2–5 条 gold（因为 500 字符块把一句话拆到了相邻块）。切到 400 token 后同一句通常只落在 1 个块内 → 多 gold 题数量下降，`hit` 与 `full` 的差值缩小，`Recall@k` 的"部分命中"语义基本消失。
   → 报告必须同时给出「重切分前后多 gold 题数量」，禁止直接用旧的新口径对比；`full@k` 将退化为与 `hit@k` 接近，需在报告里注明。
3. **旧基线作废**：`content_hash 3e08ef4c1c4f058e`、`dataset 05ba575e4f1ea96e` 都变，`compare` 会判 `not_comparable`（正确行为）。按 E05 重立基线，旧基线归档只作方向性参考。
4. **题集的 `gold_points` 更关键**：单块变大后，一个 chunk 内可能同时包含 gold 与干扰内容，`Recall` 看不出差异，必须靠 L2 的要点覆盖（D35）区分。

**执行顺序约束**：B02 重切分必须在 D27（扩题集）**之前**完成，否则题集 gold 要定位两次。

**对下游条目的调整**

| 条目 | 调整 |
|---|---|
| D05–D12 | 按 Token 口径重写验收（Child 400/500、Parent 1500/2000、overlap 60） |
| D09 | 校验报告新增「多 gold 收敛统计」：同一 `gold_span` 落在几个 chunk 里的分布 |
| D27 | 出题顺序改为：重切分 → 重新 locate → 再加 `query_en` |
| E05 | 新基线必须记录切分版本（`chunker_version`）与平均 token/child |

---

### 17.11.3 B06：复用 `experiments/build_questions.py` 重建评估集

**背景**：需要 100–200 条中英成对题（方案 §11.2）。现有工具链已实现「只写语料原文句子 → 自动定位 gold chunk」，并在 60 题上验证过（`check` 0 error、负样本 12/12 整句 0 命中）。

**结论**：**复用现有工具链**，增量扩展，不重写。

具体做法：

1. `questions.json` 每条新增 `query_en`（见评估设计文档 §3.1）：与 `question` 同义的英文检索 query，仅用于三路消融的英文 Dense / 英文 BM25 路，**不参与主指标**。
2. 扩题到 100–200 条：沿用 `type` 四分类配额（fact / multi_hop / concept / negative），新增 `topic` 与 `difficulty` 两个切片字段（方案 §11.2 要求），切片轴仍以**证据语言**为主（`corpus_lang`）。
3. 负样本继续保留 12 条以上，`note` 里记录三条留痕（整句 0 命中 + 语义 top-1 cos + 人工核对结论）。
4. `query_en` 的产出方式：**人工撰写**（不从 `question` 机翻），避免把翻译器的缺陷写进评估集；撰写规则：保留型号/数字/缩写原样、不扩写问题、不使用语料原文整句。

**对下游条目的调整**

| 条目 | 调整 |
|---|---|
| D27 | 验收新增：`query_en` 非空率 100%（negative 题也必须有，用于测"没答案时英文路是否也安全"）；`topic`/`difficulty` 字段齐全 |
| D28 | 消融 A–D 直接用 `query_en` 作英文路输入，不需要再调用翻译（D23 的翻译质量单独用 200 条样本评测） |
| E05 | 新基线记录题集 sha256 与 `query_en` 覆盖率 |

---


### 17.11.4 B04：直接使用 Qdrant（不做「先 numpy 后切换」）

**决策（2026-11-02，用户拍板）**：Phase 2 起向量库直接用 Qdrant，不保留 numpy 全矩阵作为产品路径。

**为什么放弃原方案的「接口先行 + numpy 兜底」**

1. **过滤能力是硬需求，不是优化**。现有 `retrieve.py` 只有 `matrix @ qvec` 全库打分，
   连 `filter` 这个概念都没有。实测同一 query 的 **Top-50 来自 23 个文件、最大单文件仅 6 条**，
   意味着「只在这份文档内检索」用 post-filter 几乎捞不到东西（k=5 时大概率 0 条）。
   D29 语言过滤、D25 英文 lexical、单文档问答全卡在这。
2. **numpy「兜底」的价值很低**：没有持久化、没有并发写、没有版本管理，
   D16/D18/D21 都要重写一遍，等于维护两套。
3. **本机可落地性已验通**：Docker 29.8.2 + `qdrant/qdrant:v1.15.0`（292 MB）+ 内存限 2 GB，
   `docker compose up -d` 一条命令起停，数据卷 `./storage/qdrant`。

**numpy 精确检索没有取消，只是换了角色**：D19/D36 算 ANN Recall@K 必须有「精确 Top-K 标准答案」，
所以保留 `np` 暴力实现作为**评估工具**，不进产品路径。

**影响**

- D13 编码后直接 upsert 进 Qdrant，**不再产出 `embeddings.npy`**（旧索引保留用于回归对照）。
- D16 幂等靠 `point_id_for(chunk_id)`（UUID5）保证，不靠行号对齐。
- D19 的「精确基准」= 同一批向量在内存里暴力算，与 Qdrant ANN 结果求交集。
- 版本纪律：镜像 minor 与 `qdrant-client` minor 必须对齐（实测 client 1.20 连 server 1.15 会告警不兼容）。

---

## 17.12 执行记录（滚动更新）

| 条目 | 状态 | 落地内容 | 验证 |
|---|---|---|---|
| D01 | ✅ 2026-11-02 | `src/rag/ingestion/block.py`（`DocumentBlock` + `BLOCK_TYPES` + 字段校验 + `to_dict/from_dict`）、`src/rag/ingestion/parser/base.py`（`BaseParser` 抽象基类 + `can_parse` + `describe`）、`src/rag/ingestion/parser/__init__.py`（后缀注册表，冲突显式报错）、`src/rag/ingestion/errors.py` | `python -m unittest tests.test_document_block tests.test_parser_base` → 21 用例全绿 |
| D05 | ✅ 2026-11-02 | `src/rag/ingestion/tokenizer.py`：`TokenCounter`（`count` / `count_batch` / `truncate` / `verify_against_ollama` / `default_counter`）+ CLI（`count` / `verify`）；词表落 `storage/tokenizer/bge-m3/tokenizer.json`（ModelScope 镜像，17.1 MB，不入库） | `python -m src.rag.ingestion.tokenizer verify --n 6` → `aligned: true`、`median_rel_diff: 0.0`、`max_abs_diff: 0`；集成测试含 nomic 负向对照 |
| D06 | ✅ 2026-11-02 | `src/rag/ingestion/recursive_chunker.py`：`SEPARATORS`（照抄 §2.4）、`recursive_split`、`split_preserving_separator`（分隔符留在前段末尾）、`split_by_token_window`（token 窗口兜底，收缩到硬上限内）、`merge_small_parts`（只在最后做一次合并）；CLI：`split` / `check` | `python -m src.rag.ingestion.recursive_chunker check --n 2000` → 超限率 0.00%、还原失败 0、碎块率 3.95%；`python -m unittest tests.test_recursive_splitter tests.test_recursive_splitter_real` → 35 用例全绿 |
| D02 | ✅ 2026-11-02 | `src/rag/ingestion/parser/markdown_parser.py`：ATX/setext 标题、fenced code（先于标题识别）、GFM 表格、列表、引用、图注；`section_path` 栈式继承，无标题时回退文件名 | `check data/md` → section_path 非空率 **100.00%**、空 block **0**；陷阱统计：代码块内 `#` 行 4,295 处未误判为标题、表格紧跟标题 454 处仍为 table |
| D03 | ✅ 2026-11-02 | `src/rag/ingestion/parser/txt_parser.py`：空行分段、软换行原样保留、`n_sentences` 统计、BOM/CRLF 归一；不做启发式标题猜测 | `python -m unittest tests.test_txt_parser` 16 用例全绿；段落数 == 空行分隔段数 |
| D04 | ✅ 2026-11-02 | `src/rag/ingestion/parser/pdf_parser.py`：pymupdf 版面块 → 行 → span；按字号相对比定标题层级；`block` 内按字号再切 run（pymupdf 会把标题和正文聚进一个 block）；重复页眉页脚归一化识别并记入 `stats` | `check data` → PDF 页码覆盖率 **100.00%**；30 页字符数去空白后与 pypdf 差 **0.47%**；`report` 子命令产出人工抽查 CSV |
| D08 | ✅ 2026-11-02 | `src/rag/ingestion/parent_builder.py`：`ParentConfig`(1500/2000/200)、三段式构建（分组 → 切长 → 并短）、`build_children` 回填 `parent_id`；超长单块用 D06 递归切分并按 piece 长度累加偏移 | 40 篇真实 md：parent 超限 **0**、孤儿 child **0**；`tests.test_parent_builder` 23 用例 |
| D10 | ✅ 2026-11-02 | `src/rag/ingestion/ids.py`：`make_document_id`（按 source_uri）、`document_version`（归一化后内容哈希）、`make_parent_id`/`make_chunk_id`（按 §3.4 公式，`` 拼接） | 两次运行 JSONL sha256 完全一致；chunker_version 变更后 ID 复用 **0**；`tests.test_stable_ids` 7 用例 |
| D12 | ✅ 2026-11-02 | `src/rag/ingestion/jsonl_store.py`：`PARENT_FIELDS`/`CHILD_FIELDS`/`DOCUMENT_FIELDS` 严格字段契约、`write_jsonl`（先全量校验再开文件）、`write_chunks`（`data/chunks/` + manifest 记 sha256） | `tests.test_jsonl_writer` 11 用例；40 篇语料 3 个 JSONL 逐行解析 + 字段集完全一致 |
| D07 | ✅ 2026-11-02 | `src/rag/ingestion/overlap.py`：`apply_overlap` 在上一块里挑句子/行起点作为重叠起点，找不到合适长度就退回 token 窗口；`TokenCounter.truncate_tail` 按 token offset 回切原文后缀 | 100 对相邻 chunk 的 overlap 100% 落在 [50,70]；双测（`tests.test_overlap`） |
| D09 | ✅ 2026-11-02 | `src/rag/ingestion/validator.py`：六类检查 + `(document_id, parent_index)` 关联键 + 父块主键重复自检 + `gold_convergence` 直方图；`write_invalid` 无问题时**不生成**文件 | 40 篇真实语料：空 0 / 超限 0 / 重复 0.31% / 边界 100%；`tests.test_validator` 23 用例 |
| D11 | ✅ 2026-11-02 | `src/rag/ingestion/quality_report.py`：A–E 五方案（E 标 not_implemented）+ 8 项抽查自动指标 + 人工评分列 + 逐文档明细 + CLI（`--out`/`--json`） | 40 篇真实语料报告见 §17.12 实测表；`tests.test_quality_report` 25 用例 |
| D15 | ✅ 2026-11-02 | `src/rag/store/qdrant_store.py`（`ensure_collection` / `collection_config` / `point_id_for` / `payload_from_child` + CLI create/info/drop/ping）+ `docker-compose.yml`（镜像钉 v1.15.0，内存限 2 GB）+ `errors.py`（`CollectionMismatchError` 为 D21 铺路） | `curl -s localhost:6333/collections/rag_chunks_v1`：size=1024 / Cosine / m=16 / ef_construct=200 / payload index 6 个；`tests.test_qdrant_store` 18 用例 |
| D13 | ✅ 2026-11-02 | `src/rag/embedding/`：`service.py`（EmbeddingService：批量、400 原地重试 + 二分、维度/NaN/Inf/全零校验、进程内 LRU、stats、CLI encode/stats/purge）、`cache.py`（SQLite 内容 hash 缓存，4 KB/行）、`errors.py`；`tests/test_embedding_integration.py` 真模型用例挂 `RAG_OLLAMA_CHECK=1` | `python -m unittest tests.test_embedding_service` 31 用例；`RAG_OLLAMA_CHECK=1 python -m unittest tests.test_embedding_integration` 3 用例；实测冷 p95 2212 ms → 内存命中 0.019 ms、SQLite 命中 3.1 ms |
| D17 | ✅ 2026-11-02 | `src/rag/store/document_store.py`：三表 schema（外键 ON DELETE CASCADE + 显式 `PRAGMA foreign_keys=ON`）、`replace_document` 单事务整篇替换、`get_parent_for_chunk` 三表 join、`import_chunks_dir`、`verify_against_jsonl`、`delete_document`、CLI import/verify/show/delete/stats | `tests.test_document_store` 18 用例；真实 20 篇：parents 102==102、children 182==182、chunk→父块回溯成功 |
| D16 | ✅ 2026-11-02 | `src/rag/store/indexer.py`：`index_document`（幂等 upsert + 陈旧点差集删除）、`index_chunks_dir`、`delete_document`、`count(document_id)`、`chunk_ids`、`stats`；payload 剔除 None 字段 | `tests.test_indexer` 14 用例；真实 3 篇：重复导入点数不变、版本变化删 6 陈旧点、按文档删除后过滤计数 0 |
| D18 | ✅ 2026-11-02 | `src/rag/ingest/state.py`（Stage 枚举严格照 §10.1 + 状态表 + 失败表 + `resolution` 区分「重试成功」/「源文件已移除」+ `is_done` 三版本判据 + `prune_missing`）、`pipeline.py`（发现→窗口攒批→逐阶段落库→失败留痕，`crash_hook` 供 kill 测试）、`scripts/ingest.py`（run/dry-run/status/failures/retry-failed/only/force/limit） | `tests.test_ingest_state` 15 + `tests.test_ingest_pipeline` 18；真实 6 篇：冷跑 10.23 s → 硬杀 → 重启 3 skip + 3 续跑 1.55 s，点数 6 无重复 |
| D21 | ✅ 2026-11-02 | `src/rag/store/index_versions.py`（登记表 + 只读 `check` + `assert_compatible` 闸门 + retire/activate + 建议名）、`Indexer._assert_versions`（upsert 前对账，失败一个点都不写）、`scripts/ingest.py --versions`、`tests tests.test_indexer::VersionGateTest` | 负向：v2 写 v1 collection → `CollectionVersionMismatch` 且点数不变 |
| D19 | ✅ 2026-11-02 | `src/rag/store/search.py`（`qdrant_search`/`exact_topk`/`ann_recall`/`measure_ann_recall`/`scroll_vectors` + CLI） | 真实向量：1 万点 1.000、10 万点 0.990；`tests.test_search` 11 用例 |
| D20 | ✅ 2026-11-02 | `scripts/ingest.py --dir data --limit 10000`（全链路）；`IngestState.forget_document_id` + `Indexer.delete_document(state=...)` + `scripts/ingest.py --delete` 补上「删除→状态表同步→可重建」闭环 | 全语料 1467 篇 → 38,865 点 / 0 失败 / 1992 s；hit@5 8/10、回溯 10/10；删除后重建精确回到 38,865 |
| D14 | ✅ 2026-11-02 | `src/rag/ingestion/embedding_text_report.py`（前缀统计 / 可重建性 / 父块 Recall@k / A-B 汇总 / 双 collection 跑分 + CLI）、`DocumentStore.parents_containing` / `documents_containing` | 前缀覆盖 100%（全语料 25,411 父块无缺 title/section）、可重建 100%；A/B 48 题：MRR +0.125、hit@10 +2.1pt、Recall@10 +0.9pt |
| D27 | ✅ 2026-11-04 | `src/eval_bilingual.py`（8 类 topic 分类 / 成对率 / 术语保留率 / 草稿→过审状态机 / `check`）、`experiments/expand_questions.py`（+41 题，gold 由 `locate()` 自动定位） | 101 题、成对率 **100%**、topic **8/8**、两套 `check` 均 0 error；新题集 L1：`hybrid_score` Recall@10 **60.4%**，中文语料 79.6% vs 英文语料 42.7% |
| D22 | ✅ 2026-11-04 | `src/rag/lang/detect.py`（`cjk_char_count` / `latin_word_count` / `language_ratio` / `detect` / `detect_batch` / `summarize` + CLI） | 评估集 202 条**准确率 100%**；混排样例 `HNSW 是什么？`→zh、`What is RRF?`→en、`8192`→unknown |
| D25 | ✅ 2026-11-04 | `src/rag/store/lexical.py`（`tokenize_en` / `BM25Index` / `parent_hits` / `ensure_text_index` / `fulltext_filter` / `load_payloads` / CLI） | 83 题：中文 46/83 vs 英文 57/83（**+11**）；关键词类 32 题 **13/32 vs 24/32**；Qdrant text 索引 38,865 点建成功；修 AND→OR 语义 bug |

**Token 口径（D05 产出，后续 D06–D12 直接用）**

- `count(text)` = BGE-M3 tokenizer 的**内容 token**（不含 BOS/EOS），空白文本为 0；
- `truncate(text, n)` 返回**原文的逐字符前缀**（按第 n 个 token 的字符 offset 回切），不是 `decode(ids[:n])`——后者会规范化空格/标点，用它算 `source_start/source_end` 会错位（D07 依赖这一点）；
- Ollama 单条 `prompt_eval_count` = `count(text) + 2`；批量 = `Σcount + 2×条数`；
- 因此切分预算（child 400/500、overlap 60、parent 1500/2000）用 `count()` 衡量，与 Ollama 计数天然差 2/条，这是定义而不是误差；
- 校验用**逐条**中位相对误差（≤2%）+ 逐条最大绝对差（≤2 token），**不要用聚合总量**（见 §17.11.1）。

**新增依赖**：`tokenizers>=0.20.0`（实装 0.23.3），只用于加载词表，不跑模型推理。

**D06 实测（用于 D07/D08/D09 的参数决策）**

| 测量 | 结果 |
|---|---|
| 2,000 条真实 chunk（target 400 / max 500） | 超限率 **0.00%**、还原失败 **0**、碎块率（<80 token）**3.95%** |
| 100 篇「拼 8 条 chunk」的长文档 → 699 段 | token p50 **318**、p95 **479**、max **499**、碎块率 **0.1%** |
| 切分吞吐 | ≈ **3,000 chunk/s**（纯 CPU，含 token 计数；81k 全量约 27 s） |
| 现有索引 chunk 的 token 分布（300 条抽样） | p50 **128**、max **434** —— **全部低于 500 token 上限** |

最后一行是 B02 的直接证据：旧切分是「500 **字符**」，实际只用到 500 token 预算的 1/4，最长的 chunk 也只有 434 token。「按 token 重切分」不是把块变碎，而是让块真正达到设计粒度——每块塞进更多上下文，Child 数从 81,027 降到约 3 万。

**D02–D04 实测（解析层）**

| 测量 | 结果 |
|---|---|
| 全语料 `check data`（md + pdf + txt） | 1,467 文件 → **217,950 block**；section_path 非空率 **100.00%**；空 block **0**；PDF 页码覆盖率 **100.00%** |
| `check data/md` | 1,227 文件 → **99,479 block**：paragraph 37,557 / list 36,754 / heading 13,430 / code 8,783 / table 2,442 / caption 513 |
| 陷阱 1：代码块内含 `#` | 真实语料里 **4,295** 处，全部留在 `code` 块内，未被当成标题 |
| 陷阱 2：表格紧跟标题 | **454** 处，全部仍是 `table` |
| PDF 30 页字符数 | pypdf 62,779 / pymupdf 59,874（0.954）；**去空白后 47,687 / 47,464（0.995）** |

PDF 那 4.6% 的差不是丢内容，而是 pypdf 会给每个 span 补空白：它的空白字符占比 24.0%，pymupdf 20.7%。去掉空白后两边差 0.47%（223 字符 / 30 页），来源是重复页眉页脚被剔除（306 字符）与个别连字符断行，属预期。

**实现与 §6 目录的对齐**：`src/rag/ingestion/parser/`（单数，按 §6）、`recursive_chunker.py`（§6 命名，D06 的实现文件）。§6 未列出但必需的三个文件：`block.py`（D01 数据结构）、`errors.py`（异常层）、`tokenizer.py`（D05 token 口径）。

**pymupdf 的一个坑**：它按版面把行聚成 block，但标题和正文**可能落在同一个 block 里**（排版紧、行距一致时）。只按 block 首行字号判标题会整段漏掉标题——实测合成 PDF 里 `1.1 向量检索` 就这样消失了。修法是 block 内部再按「是否标题 + 字号是否变化」切 run，逐 run 判类型。

**D08/D10/D12 实测（40 篇真实 md）**

| 测量 | 结果 |
|---|---|
| 规模 | document 40 → **parent 194 → child 292** |
| parent token ≤ 2000 超限数 | **0**（p50 256 / p90 1044 / max 2000） |
| child token ≤ 500 超限数 | **0**（p50 308 / p90 441 / max 499） |
| chunk→parent 关联成功率 | **100.00%**（孤儿 child 0） |
| 两次运行 JSONL sha256 | **完全一致**（documents / parents / children 三个文件） |
| chunker_version 变更后 ID 复用 | **0** |
| 字段契约 | 三个 JSONL 逐行可解析，字段集与 §3.1–3.3 **完全一致** |

**一个必须记下来的信号**：parent p50 只有 256 token，child p50 308 token——**292 child / 194 parent ≈ 1.5**，也就是说大量 parent 只切出 1 个 child，Child 粒度并没有比 Parent 更细。原因是 `min_tokens=200` 把短章节合并后，剩下的章节本身就不长。这不是 bug，但它意味着「Parent 提供上下文、Child 负责召回」的设计在当前语料上没完全成立，**配置该在 D11 里用 A–E 对照表调**（调 `target_tokens`/`min_tokens`，看 child/parent 比与召回效果），不要凭手感改。

**为什么 document_id 不含内容**：`document_id` 由 `source_uri` 派生，`document_version` 才是内容哈希。这样同一篇文档改一个错别字不会变成「另一篇文档」，增量导入可以按 `document_id` 清旧版本（§3.4 最后一条）；反过来若把内容塞进 `document_id`，每次改字都会留下永远删不掉的孤儿向量。

**D07/D09/D11 实测（40 篇真实 md）**

A–E 方案对照（`python -m src.rag.ingestion.quality_report --dir data/md --limit 40`）：

| 方案 | chunk 数 | token p50 | p90 | max | 重复占比 | 句子被切断 | 碎块(<80) | Parent | child/parent |
|---|---|---|---|---|---|---|---|---|---|
| A 固定长度 | 594 | 133 | 186 | 260 | 0.00% | **97.53%** | 4.71% | — | — |
| B Recursive | 209 | 412 | 454 | 499 | 0.00% | 0.00% | 1.91% | — | — |
| C Recursive+Overlap | 247 | 408 | 446 | 489 | 0.00% | 0.00% | 0.81% | — | — |
| D Parent-Child+Overlap | 332 | 314 | 420 | 475 | **1.81%** | 0.00% | **14.76%** | 202 | 1.64 |
| E Semantic+Parent-Child | — | — | — | — | — | — | — | — | — |

D 的其余七项：heading 关联 **87.65%**、Parent 章节纯度 **100.00%**、Child 可回溯 **100.00%**、overlap 合规 **99.24%**、表格完整 **72.73%**（B 81.82% 最好）、代码完整 **95.28%**、空 chunk **0**。

Overlap 实测：前 100 对相邻 chunk 的 overlap **100% 落在 [50,70]**（p50 60），其中 **69.5%** 的起点落在句子或行边界上——说明「句子边界优先、越界退窗口」确实在工作，不是全靠窗口兜底。

**参数敏感性（D 方案，40 篇）**

| `min_tokens` | parent | child | child/parent | child p50 | 碎块率 | 重复占比 |
|---|---|---|---|---|---|---|
| 0 | 338 | 449 | 1.33 | 159 | 33.0% | 8.46% |
| 100 | 278 | 389 | 1.40 | 199 | 21.6% | 4.37% |
| **200（当前默认）** | 194 | 322 | 1.66 | 316 | 14.3% | 0.31% |
| **400（建议）** | 129 | 284 | 2.20 | 353 | **7.0%** | **0.00%** |
| 600 | 114 | 280 | 2.46 | 360 | 6.4% | 0.00% |

结论：**Child 的粒度被 Parent 的大小卡住了**——`min_tokens=200` 下 202 个 Parent 只切出 332 个 Child（1.64），大量 Child 就是整个 Parent，碎块率因此 14.76%。把 `min_tokens` 提到 **400** 能把碎块率砍半（14.3% → 7.0%）、重复归零，代价是 `section_path` 变粗（合并取共同前缀）。这条作为 **B07 待定决策**提交，必须等 D28 的检索级消融确认收益后再改默认值，不能凭 chunking 指标拍板。

**D25 实测：英文 lexical 帮英文语料、伤中文语料**

做法上有个坑先说清楚：**Qdrant 的 full-text 是布尔匹配，不是 BM25**。`TextIndexParams` + `MatchText`
只回答「这条 payload 含不含这些词」，没有 idf/长度归一化分数，`scroll` 还是按点 id 顺序返回——
直接拿它当前 10 名等于随机挑。所以这一路拆成**Qdrant 召回（过滤）+ 应用层 BM25 排序**。
分词刻意**不用 jieba**：英文路径用英文分词，中文问法进来只会剩型号/缩写（`HNSW 是什么` → `["hnsw"]`），
这恰好就是 D25 要量的事。

**修掉一个真 bug**：`fulltext_filter` 第一版用 `must`（AND 语义），英文问句
`What is the peak speed of NDR InfiniBand per direction in a DGX SuperPOD?` 候选 **0 条**——
它要求 `per`、`direction` 这些词在同一条 payload 里同时出现。改成 `should` + `min_should=1`（OR）后正常。
过滤只负责「别把无关点捞进来」，排序交给 BM25。

**验收（同一批题、同一条英文 lexical 路，只换 query 语言）**：

| 样本 | 中文 query hit@10 | 英文 query hit@10 | 差值 |
|---|---|---|---|
| 全部可判定 83 题 | 46/83 = **55.42%**（Recall 50.00%） | 57/83 = **68.67%**（Recall 57.93%） | **+11 条** / Recall +7.93 pt |
| 关键词类 32 题（number_spec + acronym_model） | 13/32 = **40.62%**（Recall 36.46%） | 24/32 = **75.00%**（Recall 69.01%） | **+11 条** / Recall **+32.55 pt** |

**但这 11 条净增益背后是「赢 22 输 11」，不是单调变好**——分 topic 看方向完全不同：

| topic | n | 中文 hit | 英文 hit | 仅中文中 | 仅英文中 |
|---|---|---|---|---|---|
| cross_chunk | 27 | 14 | **23** | 0 | 9 |
| number_spec | 24 | 8 | **17** | 1 | 10 |
| acronym_model | 8 | 5 | **7** | 0 | 2 |
| **zh_with_en_term** | 9 | **9** | 3 | **6** | 0 |
| concept_zh | 6 | **4** | 2 | 2 | 0 |
| paraphrase | 6 | **4** | 3 | 1 | 0 |
| multi_condition | 3 | 2 | 2 | 1 | 1 |

* **英文 lexical 在 `cross_chunk` / `number_spec` / `acronym_model` 上大幅领先**（型号、数字、参数名跨语言后写法不变，英文问法直接命中）；
* **在 `zh_with_en_term` / `concept_zh` 上反而更差**：这些题的金标在中文语料（`rag_basics.md` / `agent_basics.md`）里，
  英文问法根本匹配不到中文字——中文 query 一次命中 9/9，英文 query 只剩 3/9。

**结论**：英文 lexical 是**补充而不是替代**，这正好是 D24/D26 要三路并存（中文 dense 覆盖中文语料 + 英文 dense 跨语言 + 英文 lexical 抓型号数字）而不是「换成英文就完事」的理由。

**D22 实测：语言识别为什么不能用「字符占比」**

混排的形态是「中文问句里夹型号/缩写」，不是均衡双语。实测这些问法的**字母数并不比汉字少**：

| 问法 | 汉字 | 字母 | 汉字字符占比 |
|---|---|---|---|
| `HNSW 是什么？` | 3 | 4 | 42.9% |
| `DGX B300 装了几张 GPU？` | 4 | 7 | 36.4% |
| `NVLink 和 PCIe 有什么区别？` | 6 | 10 | 37.5% |

按字符占比判语言，阈值稍高就会把这些中文问法判成 en；路由判错 → 中文问法被送去英文 BM25 → 关键词全部丢失。
所以判据改成**汉字数 /（汉字数 + 英文单词数）**，按「词」而不是「字母」计：

| 问法 | 汉字 | 英文单词 | ratio | 判定 |
|---|---|---|---|---|
| `HNSW 是什么？` | 3 | 1 | 0.75 | **zh** |
| `DGX B300 装了几张 GPU？` | 5 | 3 | 0.5714 | **zh** |
| `Why should documents be split into chunks?` | 0 | 7 | 0.0 | en |
| `8192` | 0 | 0 | None | unknown |

阈值 0.20；验收样本 **202 条（评估集 101 条中文问法 + 101 条 `query_en`）→ 准确率 100.00%**（验收线 95%）。
`unknown`（既无汉字也无英文单词）单独成一类而不是硬塞进 zh/en——路由遇到它就不该假装知道语言。

**D27 实测：评估集从 60 条扩到 101 条中英成对题**

原来 60 题全是中文问法、无英文问法、无 `topic`/`difficulty` 切片。D27 补三件事（工具见 `src/eval_bilingual.py`）：

1. **成对**：101 题全部有 `query_en`（方案 §11.2 的 `query_zh`/`query_en`，落地时不改 `question` 字段名，见 §17.11.3）；
2. **切片**：`topic` 按 §11.2 的 8 类问题落成枚举，`difficulty` 分 easy/normal/hard；
3. **过审状态机**：`query_en_status ∈ {draft, reviewed, edited}`。机器草稿一律 `draft`，`check` 默认把 `draft` 当**错误**——
   评估集里混着未审的机翻，跑出来的消融结论就分不清是「检索不行」还是「翻译不行」。

| 项 | 值 |
|---|---|
| 题数 | **101**（原 60 + 新增 41） |
| 题型配额 | concept 23 / fact 33 / multi_hop 27 / negative 18 |
| `query_en` 成对率 | **100.00%（101/101）** |
| `topic` 覆盖 | **8/8**：concept_zh 6、zh_with_en_term 9、acronym_model 8、number_spec 24、paraphrase 6、multi_condition 3、cross_chunk 27、no_answer 18 |
| `difficulty` | easy 29 / normal 51 / hard 21 |
| 待审草稿 | **0** |
| `check` | `src.eval_bilingual check --min-total 100` → **0 error**；`src.eval_dataset check` → **0 error**（原有 gold 定位/配额校验仍然通过） |

新题的 `gold_spans` 全部是语料原文，由 `locate()` 自动定位成 chunk id（41 题**一次零失败**）；`query_en` 由人写，
旧 60 题先用 `python -m src.eval_bilingual draft` 出机翻草稿、再逐条改写标 `edited`。

**顺带修掉一个会影响主链路的 bug**：`qwen3.5:9b` 是**思考型模型**，`/api/chat` 不显式传 `think:false` 时
`message.content` 是**空串**、真正的输出全在 `message.thinking` 里——表现就是「调用不报错但答案是空的」，
或者推理写太长直接 300 s 超时（实测一条草稿 >700 s 未返回）。加上 `think=False` 后单条 **1.1 s**。
`src/ollama_client.py` 的 `chat_full/chat` 已加 `think` / `num_predict` 两个透传参数（默认 `None`，不改变既有调用行为）。

**新题集的 L1 现状（旧 81k chunk 索引，`--modes vector,bm25,hybrid,hybrid_score --k 10`）**：

| 模式 | Recall@10 | hit@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| vector | 52.1%±9.9 | 60.2%±10.3 | 0.396 | 0.401 |
| bm25 | 41.9%±9.6 | 48.2%±10.8 | 0.358 | 0.345 |
| hybrid | 54.7%±10.0 | 62.7%±10.3 | 0.373 | 0.393 |
| hybrid_score | **60.4%±9.8** | **67.5%±10.2** | **0.421** | **0.445** |

**最有信息量的一个数**（也正是 Phase 3 存在的理由）：

| 切片 | n | Recall@10 | MRR@10 |
|---|---|---|---|
| corpus=中文语料 | 27 | **79.6%±14.8** | 0.590 |
| corpus=英文语料（中文提问） | 56 | **42.7%±10.5** | 0.268 |

同一条中文 query，问中文语料能到 79.6%，问英文语料掉到 42.7%——**差距 36.9 个百分点**，这就是 D23/D24/D26 要填的坑。

**D14 实测：`embedding_text` 前缀到底有没有用**

设计意图是「子块脱离父块后仍知道自己在哪篇文档、哪一节」，但**意图不等于收益**，得测。做法：同一批子块建两个 collection（有前缀 / 纯正文），同一批 query 各查一遍，按**父块**算 Recall@k / hit@k / MRR。

样本规模（`--from-gold`：按题目 `gold_spans` 挑含金标的文件，再均匀抽干扰文件，避免"按字母序取前 N 个"的偏样）：

| 样本 | 点数 | 可判定题数 | prefixed Recall@10 / hit@10 / MRR | plain Recall@10 / hit@10 / MRR | delta |
|---|---|---|---|---|---|
| 19 含金标 + 200 干扰 | 6,092 | 48 | 0.7587 / 0.8542 / **0.6095** | 0.7344 / 0.8125 / 0.4431 | +0.0243 / +0.0417 / **+0.1664** |
| 19 含金标 + 400 干扰 | 10,028 | 48 | 0.7344 / 0.8333 / **0.5541** | 0.7257 / 0.8125 / 0.4288 | +0.0087 / +0.0208 / **+0.1253** |

**结论：前缀的主要收益是把"第一个正确结果"往前提，而不是让更多正确答案进 Top-10。** MRR 两次独立运行分别 +0.166 / +0.125，hit@10 与 Recall@10 只小幅变化（其中一次 Recall@10 是 −0.0167，负的）。对 RAG 来说这恰恰是有用的那部分——上下文窗口只放 4~8 条，第 1 名比第 9 名重要得多。

**但必须说清差异的来源有多窄**（不然就是拿 2 道题冒充 48 道题的结论）：

| 运行 | prefixed 更好 | 更差 | 完全持平 | 差异题 |
|---|---|---|---|---|
| 6,092 点 | 2 题 | 1 题 | 45 题 | eval-002 ↑、eval-030 ↓、eval-045 ↑ |
| 10,028 点 | 2 题 | 3 题 | 43 题 | eval-002 ↑、eval-019 ↑、eval-020 ↓、eval-030 ↓、eval-034 ↓ |

差异几乎全部集中在「**问题用词和正文用词不同、但和章节标题相同**」的题上：

- `eval-002`「为什么文档要切分成 chunk？」：prefixed **1.000** vs plain **0.000**（两次运行都是 0）。正文里讲的是"分段/子块"，`chunk` 这个词只出现在章节标题里——**前缀把标题里的关键词补进了向量**，这正是 §2.8 想要的效果；
- `eval-045`「更换 embedding 模型要重建索引，那新模型的 chunk 还受什么长度限制？」：prefixed 0.500 vs plain 0.000；
- 反例 `eval-030`「DGX RHEL8 安装指南建议如何处理 BMC 的默认用户名？」：prefixed **0.250/0.333** vs plain **0.583/0.667**。前缀在这里是干扰——同一篇文档里 BMC 相关章节太多，`Document:`/`Section:` 把不同章节的子块推得更像了。

所以**保留前缀（方案 §2.8 不变）**，但记下两条：① 收益幅度小、题间方差大，不是"免费大涨分"；② 前缀对"同一文档内多个相似章节"的场景有副作用——这类问题应该靠 reranker（D31）而不是靠 embedding 前缀解决。

另外两条可验收的硬事实：

- **前缀覆盖 100%**：全语料 25,411 个父块里缺 `title` 的 **0** 个、缺 `section_path` 的 **0** 个；即使两者都缺，最差也只是退化成 `Content: ...` 一行，不会变成"无前缀"。
- **`embedding_text` 100% 可重建**：10,028 个抽样子块的 `embedding_text` 与 `build_embedding_text(title, section_path, content)` 逐字节相等（mismatched = 0）。这条直接支撑现有设计——**不用把它落库**（Qdrant payload 里没有它），少存一份原文，索引体积不膨胀。

**D20 端到端实测（全语料 1467 篇）**

```
python scripts/ingest.py --dir data --limit 10000
→ 发现 1467 个文件，本次处理 1467 个
→ 完成：skipped=0 indexed=1467 failed=0 children=38865 用时 1992.44s
```

| 环节 | 实测 |
|---|---|
| 摄取 | **1467 篇 → 25,411 parent → 38,865 child**，0 失败，1992 s（冷 cache 起步，后续重跑命中缓存） |
| Qdrant | 38,865 点，`status=green`，HNSW 已建（`indexed_vectors_count=36,914`），6 个 payload index |
| Document Store | documents 1467 / parents 25,411 / children 38,865，与 Qdrant 逐一对齐 |
| 召回（中英各 5 条） | **hit@5 = 8/10**（中文 4/5、英文 4/5），**父块回溯 10/10** |
| 删除/重建 | 删 1 篇（6 点）→ 38,865→38,859、Store 1467→1466；重跑 → **38,865**（精确回到基线） |

两个未命中都查清了原因，不是"检索坏了"：
- `eval-002`（中文）首次命中在 **#22**——排序问题，正是 Phase 3/4（lexical + reranker）要解决的；
- `en-1`（英文，「ef_construct 参数」）全库含该词的父块 **0 个**——NVIDIA 文档里根本没有这个词，**是我出的 query 不成立**。

**这一轮又抓到一个真缺口**：删除文档后状态表仍标 `COMPLETED`，重跑会被 `is_done()` 跳过，**永远重建不回来**（实测：删了之后 `--only` 重跑显示 `发现 0 个文件 / indexed 0`）。修法：`IngestState.forget_document_id()` + `Indexer.delete_document(state=...)` 同步清理 + `scripts/ingest.py --delete`。这也说明「删除」不是单点操作，而是 **Qdrant + Document Store + 状态表三方一致**。

**关于 `indexed_vectors_count`（影响 D19 的读数解释）**：Qdrant 默认 `indexing_threshold=20000`，点数低于阈值时 HNSW 不建索引、走**精确扫描**。所以 D19 里「1 万点 recall=1.000」其实是精确扫描的结果，不是 HNSW 的功劳；38,865 点建好索引后才是真 ANN（81k/100k 的 0.99 是 ANN 的真实数字）。

**D21 + D19 实测（真实 bge-m3 向量 + 20 条真实 query）**

ANN Recall（Qdrant Top-K ∩ numpy 精确 Top-K，K=50）：

| 点数 | ef=default | ef=256 | ef=512 | 检索延迟 p50 |
|---|---|---|---|---|
| 10,000 | **1.000** | — | 1.000 | 12–14 ms |
| 81,027 | 0.994 | — | 0.997 | 16–28 ms |
| 100,000 | 0.988 | **0.990** | 0.990 | 11–22 ms |

验收线：1 万 ≥0.99 ✓、10 万 ≥0.95 ✓。Recall@1 在 10 万点、default ef 下是 0.900，`ef=256` 提到 1.000——**RAG 只取 Top-4~8，`ef` 比 `m` 更值得先调**。

**这一轮的真正收获是抓出一个 Qdrant 1.15.0 的 bug（不是调参）**

| 现象 | 证据 |
|---|---|
| 81k 点 recall 只有 0.46~0.52，且 `ef` 完全无效 | 正常 ANN 不会对 `ef` 无反应 → 先怀疑数据/索引，不是参数 |
| 集合状态 `red` | `optimizer_status=error: Optimization task panicked ... OutputTooSmall { expected: 4, actual: 0 }`（`gridstore.rs:53`） |
| 点数对不上 | 新 collection 插 81,027 点，`points_count` 报 **133,382**；`indexed_vectors_count` 只有 17,920 |

换 `qdrant/qdrant:v1.16.2`（+ `qdrant-client==1.16.2`）后同一批数据：`status=green`、`points=81,027` 精确、`recall@50=0.994`。**结论：镜像与客户端一起升级，并把容器内存从 2 GB 提到 6 GB**（2 GB 下优化期会被内核回收 mmap 页，正好触发该 bug 的读取路径）。

三个排查教训：
1. **先看 `status` 和 `indexed_vectors_count`，再谈召回**。索引没建完时测召回，测的是半成品；
2. **collection 名不要复用**。同名 collection 删除后残留的段会被新库读到（点数虚高），压测脚本一律用带随机后缀的名字；
3. **`ef` 对结果毫无影响时，不要继续调参**——这是引擎异常的信号，不是参数不敏感。

**D18 实测（6 篇 md，真 Qdrant + 真 bge-m3）**

| 步骤 | 实测 |
|---|---|
| 第 1 轮全量 | 6 篇 → 6 child，**10.23 s**（含冷启 embedding 固定开销） |
| 子进程 `os._exit(9)` 硬杀 | 杀掉时状态表 3 篇 `COMPLETED`，退出码 9 |
| 重启续跑 | `[skip] doc1–doc3`（不重算）+ `[ok] doc4–doc6`，**1.55 s**（6.6×，embedding 全命中缓存） |
| 幂等核对 | Qdrant 点数 **6**（不是 9）——point id 由 chunk_id 决定，重写是覆盖 |
| 失败清单 | `bad.xyz` → `FAILED_PARSING` + 失败表一行（`resolved=0`） |
| 单条重试 | 改成 `bad.md` 后 `--only bad.md` → 成功，失败记录标 `resolution=completed` |

**三层保险共同成就「重启不重算」**：状态表（阶段 + 三处版本）跳过 `COMPLETED`；中断停在 EMBEDDING 时缓存让重算近乎免费（0.019 ms/条）；即使要重写向量，point id 幂等所以不产生重复点。**缺任何一层都不成立**。

**两个坑**

1. **窗口不能按文件大小猜子块数**。第一版用 `size // 4000` 估窗口大小，结果 `window_children` 形同虚设（小文件永远估成 1）。改成**切完再攒**：解析+切分后按真实 child 数决定何时 flush。
2. **源文件消失会留下永远修不好的幽灵失败项**。改名/删除后，它的 `FAILED_*` 行仍挂在重试清单里，但它再也不可能被修好。`prune_missing()` 同时清状态行与失败表中的未解决项，并写 `resolution='source_removed'` 以区分「重试成功」——否则审计时分不清这两种情况。

**D17 + D16 实测（真实语料，Qdrant 1.15.0）**

| 步骤 | 实测 |
|---|---|
| 20 篇 md → JSONL → SQLite | documents 20 / parents **102** / children **182**，导入 **149 ms**，库 368 KB；三表行数与 JSONL **完全一致** |
| 按 `chunk_id` 回溯 | 父块原文 + `source_uri` + `section_path` + 偏移 `0–9`；父块原文包含子块全文 |
| 重复导入（D17） | doc 20→20、parent 102→102、child 182→182（不重复） |
| 首轮索引 3 篇 / 13 child | Qdrant 13 点，18.6 s（含冷启 embedding），Store 9 parent / 13 child |
| 重复索引（D16 幂等） | 13 → **13 点不变**，陈旧点 **0**，耗时 0.32 s（**57×**，embedding 全命中缓存） |
| 版本变化 | 该文档新 child 7 → 点数 7，**陈旧点删除 6**，总点数 14 |
| 限定单文档检索 | 命中 1 条且全部来自目标文档；命中可回溯到 630 字符的父块原文 |
| 按文档删除 | 删点 1 / 删文档 1 → 该文档过滤计数 **0** |

**两个设计上的坑**

1. **payload 里的 `null` 能写不能查**。Qdrant 接受 `language: null` 写入，但 `MatchValue(value=None)` 会被 pydantic 拒绝（要改用 `IsNull` 条件），于是字段"存进去了却过滤不到"，等于给索引埋假字段。`payload_from_child` 因此**剔除 None 字段**（D15 用例已同步改成断言"字段缺席"）。
2. **版本变化不能靠「按 document_id 全删再写」**。那样查询会有一段空窗期，且失败后留下半篇数据。改成「现有该文档的点集 − 本次点集 = 陈旧点集」精确删除；写入顺序固定为**先 Document Store（事务）后 Qdrant（幂等）**，失败重跑即可自愈——反过来会产生"命中却回溯不到父块"的孤儿向量。

**D13 实测（bge-m3 / Ollama 0.40.5）**

真实 24 条 chunk（取自 `data/md`，逐条调用以体现 Ollama 的单次固定开销约 2.1 s）：

| 场景 | p50 | p95 | 相对冷启动 |
|---|---|---|---|
| 冷启动（调模型） | 2,192 ms | 2,212 ms | 1× |
| 进程内 LRU 命中 | 0.007 ms | **0.019 ms** | **113,489×** |
| SQLite 命中（新实例/新进程） | 2.5 ms | **3.1 ms** | **713×** |

二次调用命中率 **100%**；单条 query 冷 2176 ms → 命中 **6.78 ms**（ADR 要求 <10 ms）。缓存占用 1024 维 × float32 = **4 KB/行**（25 行 = 100 KB）。

**与方案 §7.2 的一处偏差（已记 changelog）**：签名写的是 `list[list[float]]`，实现返回 `np.ndarray(float32)`。8 万 chunk × 1024 维 = 8.3e7 个 Python float，光对象开销就是几百 MB，且下游（Qdrant upsert / numpy 基准）全要数组。需要 list 时 `.tolist()`。

**三个新坑**

1. **`with sqlite3.connect(...) as conn` 不关闭连接**——它只提交事务。Windows 上会一直锁住 `.sqlite3` 文件，临时目录清理直接 `PermissionError`，长跑脚本里则是句柄泄漏。缓存层用 `@contextmanager` 显式 `close()`。
2. **`np.frombuffer` 遇到被截断的 blob 会抛 `ValueError("buffer size must be a multiple of element size")`**，缓存坏行不该让整条摄取链挂掉。改为**先按字节数校验**（`len(blob) == dim*4`）再解析，坏行按 miss 处理、重算覆盖。
3. **命中率验收必须看「每次调用」而不是累计**：100 条跑两遍，累计命中率是 100/201 = **49.8%**，拿它当验收指标永远不达标也不反映缓存是否生效。`stats()` 同时给累计值与 `last_pass`，D13 验收看后者。

另外 SQLite 命中实测 3.1–6.8 ms，离 ADR 的 10 ms 只剩 30% 余量（换块慢盘就破），所以在 SQLite 之上加了**进程内 LRU**（默认 1024 条 ≈ 4 MB，`memo_size` 可调），命中打到 0.019 ms。

**D15 实测（Qdrant 1.15.0 / Docker 29.8.2）**

```
docker compose up -d qdrant                 # 镜像 qdrant/qdrant:v1.15.0，数据卷 ./storage/qdrant，内存限 2 GB
python -m src.rag.store.qdrant_store create # rag_chunks_v1 + 6 个 payload index
curl -s localhost:6333/collections/rag_chunks_v1
  → size=1024  distance=Cosine  hnsw m=16  ef_construct=200
  → payload_schema 6 个：document_id / document_type / language / status / subject / tenant_id
```

`tests.test_qdrant_store` 18 用例：4 个纯单测（UUID5 确定性、payload 映射、常量对齐方案）+ 7 个服务端用例（配置、6 索引、幂等、recreate 清点、维度不一致报错、过滤检索命中）。服务端不可达时自动 skip。

**五个实测坑（每个都会让"看起来对"的代码静默出错）**

1. **本地模式的 payload index 是空转**。`QdrantClient(":memory:")` 下 `create_payload_index` 会成功返回，但 client 自己警告 "Payload indexes have no effect in the local Qdrant"。所以「6 个索引存在」只能在服务端验；拿本地模式当验收环境等于自欺。
2. **point id 只收 UUID 或整数**。我们的 `chunk_id = chk_<32 hex>` 直接塞进去被 400 拒绝。必须用 `point_id_for()`（带命名空间的 UUID5）映射——这同时是 D16「同一 chunk 重复导入不增点」的前提，不能用行号或自增 id。
3. **`ef_construct` 是 collection 级参数**。写进 `VectorParams(hnsw_config=...)` 会被**静默忽略**，读回来仍是默认 `100`（实测）。要满足验收必须放在 `create_collection(hnsw_config=...)`。
4. **枚举的名字和值不一致**。`Distance.COSINE` 的值是 `"Cosine"`、`PayloadSchemaType.KEYWORD` 的值是 `"keyword"`——所以 `models.Distance(name.upper())` 抛 `'COSINE' is not a valid Distance`，正确写法是下标 `models.Distance[name.upper()]`。

5. **同名 collection 删除后立刻重建会 400**。`delete_collection` 返回后，服务端目录 `./storage/collections/<name>` 可能还在，此时 `create_collection` 同名会报 `Collection data already exists at ./storage/collections/<name>`。这条正好打在 `recreate=True` 的产品路径上（`rag_chunks_v1` 删了重建），所以 `ensure_collection` 里加了 `_wait_collection_gone()`（轮询到名字消失）+ `_create_with_retry()`（只对这一种竞态重试）。测试侧则改成**每个用例独占一个随机后缀的 collection 名**，不跟竞态较劲。

**版本纪律**：镜像 `v1.15.0` 配 `qdrant-client==1.15.1`。实测 client 1.20 连 server 1.15 会告警 "client version is incompatible with server version（major 相同、minor 差 ≤1）"，所以两侧 minor 必须对齐并写进 `requirements.txt` 与 `docker-compose.yml`。

**三个指标定义上的坑（都会让结论反向）**

1. **句子切断只能在散文区统计**。第一版把代码块和 URL 的内部结尾也算进去，D 方案虚高到 77%，反而比不做结构的 B 还差——代码/表格完整性该由 `code_intact_ratio` / `table_intact_ratio` 单独负责。
2. **`rstrip()` 之后判 `endswith("
")` 永远为假**。以换行结尾的列表项全被误算成切断，指标虚高 40 个百分点。判据要先只去掉行尾空格/制表符。
3. **跨 Parent 的接缝不是句子切断**。§2.3 规定 Child 不跨 Parent，章节边界处的"没写完"是结构边界的正常现象；按 parent 分组统计后 D 从 40.10% 归到 0.00%。

**与方案伪代码的一处差异**：伪代码在递归每一层都调 `merge_small_parts`，实现改为**只在最后合并一次**。原因：内层已按 target 拼好的片段到外层还会被再拼一次，既重复 token 计数（最坏 O(n²)），又让「按哪一级目标切」不可预测；末尾合并结果等价、成本更低。


---

---

## 版本变更记录

  ------------------------------------------------------------------------------------
  版本                    日期                    说明
  ----------------------- ----------------------- ------------------------------------
  1.0                     2026-11-02              初版：数据切分、跨语言检索、Qdrant
                                                  HNSW、工程实施和验收方案
  1.1                     2026-11-02              追加第 17 章「现状差距清单」：
                                                  39 个功能差距（D01–D39）、6 个工程项
                                                  （E01–E06）、6 个阻塞决策（B01–B06），
                                                  含依赖批次表、里程碑与迁移影响
  1.2                     2026-11-02              确认三项决策：B01 Embedding 走 Ollama API、
                                                  B02 按方案重切分（Token 口径）、B06 复用
                                                  build_questions.py；新增 17.11 决策记录
                                                  （含实测依据与下游条目调整）
  1.3                     2026-11-02              D01/D05 落地：DocumentBlock + Parser 注册表、
                                                  TokenCounter（本地 BGE-M3 词表，与 Ollama
                                                  逐条对齐）；新增 17.12 执行记录与 Token 口径
  1.4                     2026-11-02              D06 落地：Recursive Splitter（分隔符优先级、
                                                  分隔符保留、token 窗口兜底、单次合并）+ check CLI；
                                                  补 D06 实测（超限率 0、碎块率 3.95%、3k chunk/s）
                                                  与「现有 chunk 全部 <500 token」这一 B02 证据
  1.5                     2026-11-02              D02/D03/D04 落地：Markdown/TXT/PDF 解析器 + 语料级
                                                  验收命令 `python -m src.rag.ingestion.parser`；
                                                  补解析层实测（217,950 block、section_path 100%、
                                                  PDF 页码 100%、去空白字符差 0.47%）；
                                                  目录命名对齐 §6（parser/、recursive_chunker.py）
  1.6                     2026-11-02              D08/D10/D12 落地：Parent Builder（三段式）、稳定 ID、
                                                  JSONL 严格字段契约 + manifest；补实测（parent 超限 0、
                                                  关联 100%、两次运行 sha256 一致、ID 复用 0），
                                                  并记录「child/parent ≈ 1.5」这个待 D11 调的信号
  1.7                     2026-11-02              D07/D09/D11 落地：Overlap（句子边界回退）、Chunk
  1.8                     2026-11-02              B04 定案：直接使用 Qdrant（ADR §17.11.4）；D15 落地
  1.9                     2026-11-02              D13 落地：Embedding 服务（批量 + 二分重试、维度/NaN/Inf/
  2.0                     2026-11-02              D17 + D16 落地：SQLite Document Store（三表 + 外键 + 整篇
  2.1                     2026-11-02              D18 落地：导入状态机（§10.1 十个状态）+ 失败清单 +
  2.2                     2026-11-02              D21 + D19 落地：索引版本闸门（同维不同模型拒绝混写）与
  2.3                     2026-11-02              D20 端到端 DoD：全语料 1467 篇 → 38,865 点 / 0 失败 /
  2.4                     2026-11-02              D14 落地：前缀覆盖/可重建性统计 + 有/无前缀 A/B。实测前缀主要
  2.5                     2026-11-04              D27 落地：评估集 60 → 101 条中英成对题（成对率 100%、topic 8/8、
  2.6                     2026-11-04              D22 落地：语言识别按「汉字数/(汉字数+英文单词数)」判定，
  2.7                     2026-11-04              D25 落地：英文 lexical = Qdrant full-text 召回 + 应用层 BM25 排序
                                                  （Qdrant full-text 只有布尔匹配、没有 BM25 分数）。修 AND→OR 语义 bug。
                                                  83 题：中文 46/83 vs 英文 57/83（+11）；但赢 22 输 11——英文 lexical
                                                  帮 cross_chunk/number_spec，伤 zh_with_en_term/concept_zh
                                                  → 三路并存（D24/D26）而不是替换
                                                  阈值 0.20；评估集 202 条准确率 100%，HNSW 是什么？→zh。
                                                  记录「字符占比法在混排上不可靠」的实测数据
                                                  待审草稿 0）。新增 eval_bilingual（8 类 topic / 术语保留率 /
                                                  draft→reviewed 状态机 / check）与 expand_questions 增量扩题。
                                                  修 qwen3.5:9b 思考型模型需 think:false（否则答案为空/超时）。
                                                  新题集 L1：hybrid_score Recall@10 60.4%；中文语料 79.6% vs
                                                  英文语料 42.7%（差 36.9pt）
                                                  提升 MRR（+0.125~0.166）而非 Recall@10，差异集中在 2–5 道
                                                  「问题用词=章节标题」的题；记下同文档多相似章节的副作用。
                                                  Phase 2 全部完成（D13/D14/D15/D16/D17/D18/D19/D20/D21）
                                                  1992 s；中英 query hit@5 8/10、回溯 10/10；删除重建
                                                  精确回到基线。补「删除时同步清状态表」缺口，记录
                                                  indexed_vectors_count < 20000 时走精确扫描这件事
                                                  ANN vs 精确召回基准；实测 1 万 1.000 / 10 万 0.990。
                                                  抓出 Qdrant 1.15.0 gridstore panic（optimizer red、点数
                                                  虚高），升级镜像与客户端到 1.16.2、容器内存 2→6 GB
                                                  断点恢复 + `scripts/ingest.py` 一条命令入口；硬杀重启
                                                  实测 3 skip + 3 续跑、点数无重复；记两个坑（窗口别猜
                                                  子块数、源文件消失会留幽灵失败项）
                                                  替换事务 + JSONL 核对）与幂等 Upsert（point id 覆盖写、
                                                  陈旧点差集删除、按文档删除）；记两个坑（payload null 能写
                                                  不能查、版本变化别用全删再写）
                                                  全零校验、内容 hash 缓存 SQLite + 进程内 LRU、stats/CLI）；
                                                  记签名偏差（返回 ndarray）与三个新坑（连接不关锁文件、
                                                  frombuffer 遇截断 blob、命中率要看每次调用）
                                                  （compose + collection + 6 payload index + CLI，
                                                  验收 curl 对齐）；记四个实测坑（本地模式索引空转、
                                                  point id 只收 UUID、ef_construct 是 collection 级、
                                                  枚举名与值不同）
                                                  Validator（六类检查 + gold 收敛）、质量报告与 A–E
                                                  对照；补 40 篇实测与参数敏感性（建议 min_tokens
                                                  200→400，记作 B07 待定），记录三个指标定义坑
  ------------------------------------------------------------------------------------
