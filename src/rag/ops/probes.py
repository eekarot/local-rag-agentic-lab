"""运行时就绪探测：**一份实现，三处复用**（`/health`、`python -m src.doctor`、检索降级判定）。

设计动机（来自开源评估的实测）：用户起不来的时候，最没用的是「500 Internal Server Error」，
第二没用的是「qdrant 不可用」。有用的是：

    ✗ embed_model  Ollama 未响应：无法调用 Ollama API
      修复：ollama serve

所以每条失败都带 `fix`（可复制的命令）。这不是装饰——B12 定案「不做一条命令安装」的
前提就是「每一步都能自己验证」，而验证必须有地方看。

三个约定：

1. **探测不许抛**：它恰好在「很多东西都坏了」的时候被调用，任何未捕获异常都会把
   诊断信息变成堆栈；
2. **`optional` 区分「挂了」与「降级」**：`embed_model` / `qdrant` / `tokenizer` / `parents`
   缺一即检索不可用（`down`）；`chat_model` 缺席只是不能生成，检索仍可用（`degraded`）；
3. **全部依赖注入**：单测不连 Qdrant / Ollama / HF，也不读真词表。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.rag.store.qdrant_store import COLLECTION_NAME, VECTOR_SIZE


@dataclass(frozen=True)
class Check:
    """一项探测结果。

    三级严重度（决定 `ProbeReport.status`，也决定输出里那个符号）：
    - 硬失败（`optional=False, info=False`）→ 服务不可用，`down`；
    - 可降级（`optional=True`）→ 能检索不能生成，`degraded`；
    - 信息项（`info=True`）→ **不影响可用性**（例：`data/` 为空只意味着不能再建库）。
    """

    name: str
    ok: bool
    detail: str
    fix: str = ""
    optional: bool = False
    info: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def render(self) -> str:
        mark = "✓" if self.ok else ("·" if (self.optional or self.info) else "✗")
        line = f"  {mark} {self.name:<12} {self.detail}"
        if not self.ok and self.fix:
            line += "".join(f"\n      修复：{row}" for row in self.fix.splitlines())
        return line


@dataclass
class ProbeReport:
    checks: list[Check] = field(default_factory=list)

    def of(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    def failures(self, *, optional: bool | None = None) -> list[Check]:
        """`optional=None` 时只返回**影响可用性**的失败（`info` 项永远不算）。"""
        rows = [c for c in self.checks if not c.ok and not c.info]
        if optional is None:
            return rows
        return [c for c in rows if c.optional is optional]

    @property
    def blocking_fixes(self) -> list[str]:
        return [c.fix for c in self.failures(optional=False) if c.fix]

    @property
    def degraded_fixes(self) -> list[str]:
        return [c.fix for c in self.failures(optional=True) if c.fix]

    @property
    def status(self) -> str:
        """`ok` / `degraded`（能检索，不能生成）/ `down`（检索也做不了）。"""
        if self.failures(optional=False):
            return "down"
        if self.failures(optional=True):
            return "degraded"
        return "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": [c.as_dict() for c in self.checks],
            "blocking_fixes": self.blocking_fixes,
            "degraded_fixes": self.degraded_fixes,
        }

    def render(self) -> str:
        head = {"ok": "全部就绪", "degraded": "可检索，但不能生成答案", "down": "检索不可用"}[self.status]
        return "\n".join([f"[{self.status}] {head}", *[c.render() for c in self.checks]])


def _first_line(text: str) -> str:
    return (text or "").strip().splitlines()[0][:200] if (text or "").strip() else ""


# ---------------------------------------------------------------- 单项探测


def probe_qdrant(*, client, collection: str = COLLECTION_NAME) -> Check:
    """向量库：连得上 + collection 存在 + 点数 > 0。"""
    fix = (
        f"docker compose up -d qdrant  # 起库（镜像 v1.16.2，与 qdrant-client 同 minor）\n"
        f"python scripts/ingest.py --dir data  # 建库（先把文档按 data/README.md 放进 data/）"
    )
    try:
        names = [c.name for c in client.get_collections().collections]
        if collection not in names:
            return Check("qdrant", False,
                         f"没有 collection {collection}（现有：{', '.join(names) or '无'}）", fix)
        count = getattr(client.count(collection), "count", 0)
    except Exception as exc:                      # noqa: BLE001 - 探测不许抛
        return Check("qdrant", False, f"连不上 Qdrant：{type(exc).__name__}: {_first_line(str(exc))}", fix)
    if not count:
        return Check("qdrant", False, f"{collection} 是空的（0 点）", fix)
    return Check("qdrant", True, f"{collection}：{count} 点")


def probe_embed_model(
    *, embed: Callable[[Sequence[str]], Any],
    expected_dim: int = VECTOR_SIZE,
    model: str = "bge-m3",
) -> Check:
    """查询向量模型。**维度不对必须抓出来**：换 768 维模型不会报错，只会让相似度全错。"""
    pull = f"ollama pull {model}  # 向量模型必须与建库时一致：{model}（换模型需整库重建）"
    try:
        vectors = embed(["健康检查"])
        dim = len(vectors[0])
    except Exception as exc:                      # noqa: BLE001
        message = str(exc)
        unreachable = any(k in message for k in
                          ("无法调用 Ollama", "Connection", "refused", "timed out", "Max retries"))
        # 服务没起和模型没拉是两件事，给错药比不给药更浪费时间
        fix = f"ollama serve  # 先确认 Ollama 在跑（桌面版/服务均可）" + chr(10) + pull if unreachable else pull
        return Check("embed_model", False, f"调用失败：{type(exc).__name__}: {_first_line(message)}", fix)
    if dim != expected_dim:
        return Check("embed_model", False,
                     f"维度 {dim} ≠ 索引要求 {expected_dim}：这个模型不能用于当前索引", pull)
    return Check("embed_model", True, f"{model} 可用，维度 {dim}")


def probe_chat_model(*, list_models: Callable[[], Sequence[str]], model: str) -> Check:
    """生成模型。**optional**：缺席不阻断检索（降级为只检索）。"""
    def norm(name: str) -> str:
        return name[:-7] if name.endswith(":latest") else name

    try:
        names = [str(n) for n in list_models()]
    except Exception as exc:                      # noqa: BLE001
        return Check("chat_model", False,
                     f"Ollama 未响应：{type(exc).__name__}: {_first_line(str(exc))}",
                     "ollama serve  # 启动 Ollama（或检查 OLLAMA_BASE_URL）", optional=True)
    wanted = norm(model)
    hit = wanted in {norm(n) for n in names} or wanted.split(":")[0] in {norm(n).split(":")[0] for n in names}
    if not hit:
        return Check("chat_model", False,
                     f"未安装 {model}（已安装：{', '.join(names) or '无'}）",
                     f"ollama pull {model}", optional=True)
    return Check("chat_model", True, f"{model} 已安装", optional=True)


def probe_tokenizer(*, counter) -> Check:
    """分词器（建库与预算共用）。缺文件时把错误里自带的 curl 原样搬到 `fix`。"""
    try:
        tokens = counter.count("健康检查")
    except Exception as exc:                      # noqa: BLE001
        message = str(exc)
        rows = [row.strip() for row in message.splitlines()
                if any(mark in row for mark in ("curl", "http", "RAG_TOKENIZER_PATH"))]
        fix = chr(10).join(rows) if rows else "见 README「第 3 步：下载 tokenizer.json」（约 17 MB）"
        return Check("tokenizer", False, f"分词器不可用：{_first_line(message)}", fix)
    return Check("tokenizer", True, f"分词器可用（「健康检查」→ {tokens} tokens）")


def probe_reranker(*, reranker) -> Check:
    """重排器。禁用是**合法状态**（`--no-rerank` / 消融实验），不算失败。"""
    config = getattr(reranker, "config", None)
    enabled = bool(getattr(config, "enabled", True))
    if not enabled:
        return Check("reranker", True, "未启用（--no-rerank：精度下降、延迟下降）")
    return Check("reranker", True, "已启用（mmarco-mMiniLMv2，首次调用自动下载 449 MB；离线需 HF_HUB_OFFLINE）")


def probe_parents(parents: Mapping[str, Any]) -> Check:
    """父块（上下文组装的原料）。`parents` 只在启动时读一次 → 更新需重启。"""
    try:
        count = len(parents or {})
    except Exception as exc:                      # noqa: BLE001 - 探测不许抛
        return Check("parents", False, f"父块加载失败：{type(exc).__name__}: {_first_line(str(exc))}",
                     "python scripts/ingest.py --dir data  # 建库后重启服务")
    if not count:
        return Check("parents", False, "父块 0 个：建库没做或加载失败",
                     "python scripts/ingest.py --dir data  # 然后把文档放进 data/ 再重启服务")
    return Check("parents", True, f"父块 {count} 个（启动时加载，重建库后需重启）")


def probe_data_dir(path: str | Path, *, patterns: Sequence[str] = ("**/*.md", "**/*.txt", "**/*.pdf"),
                   info: bool = True) -> Check:
    """语料目录。**不进 `run_probes` 的必查项**：它只影响「再建库」，不影响已在跑的检索。"""
    root = Path(path)
    fix = "把 .md/.txt/.pdf 放进 data/（格式与体积见 data/README.md）"
    if not root.exists():
        return Check("data_dir", False, f"目录不存在：{root}", fix, info=info)
    hits: set[Path] = set()
    for pattern in patterns:
        hits.update(p for p in root.glob(pattern) if p.is_file())
    if not hits:
        return Check("data_dir", False, f"{root} 里没有 .md/.txt/.pdf（0 个）", fix, info=info)
    return Check("data_dir", True, f"{len(hits)} 个文件（.md/.txt/.pdf）", info=info)


# ---------------------------------------------------------------- 汇总


def run_probes(
    *,
    qdrant,
    embed,
    list_models,
    model: str,
    counter,
    reranker,
    parents: Mapping[str, Any],
    collection: str = COLLECTION_NAME,
    expected_dim: int = VECTOR_SIZE,
    embed_model: str = "bge-m3",
) -> ProbeReport:
    """跑「服务能不能用」的六项必查（顺序即输出顺序，`data/` 不在此列）。"""
    return ProbeReport([
        probe_qdrant(client=qdrant, collection=collection),
        probe_embed_model(embed=embed, expected_dim=expected_dim, model=embed_model),
        probe_chat_model(list_models=list_models, model=model),
        probe_tokenizer(counter=counter),
        probe_reranker(reranker=reranker),
        probe_parents(parents),
    ])
