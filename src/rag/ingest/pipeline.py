"""D18：导入编排——一条命令从「一堆文件」跑到「Qdrant + Document Store 双写完成」。

流程（每个 window）：

    发现文件 → 逐篇 PARSING → CHUNKING →（写 Document Store）
             → 攒够 window_children / window_docs → 一次性 EMBEDDING
             → 逐篇 INDEXING → COMPLETED

为什么按 **window 攒批** 而不是一篇一调：Ollama `/api/embed` 的单次固定开销实测
约 2.1 s，与批量大小几乎无关（见 ADR §17.11.1）。一篇一调的话，8 万篇文档就是
8 万 × 2.1 s ≈ 47 小时，全花在固定开销上；按窗口攒到几百条后每条约 24 ms。

断点恢复的三个保险（缺一不可）：

1. 状态表持久化「阶段 + 三处版本」，重启直接跳过 `COMPLETED` 且版本未变的文档；
2. 中断若停在 EMBEDDING，embedding 缓存让重算近乎免费（实测命中 0.019 ms/条）；
3. 就算要重写向量，point id 由 chunk_id 决定，upsert 是覆盖而不是新增。

所以「重启不重算」不是靠一个开关，而是三层机制叠出来的。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from ..embedding.service import DEFAULT_VERSION as DEFAULT_EMBEDDING_VERSION
from ..ingestion.errors import IngestionError
from ..lang.detect import detect as detect_language  # noqa: E402  语言在建库时判一次（N49）
from ..ingestion.ids import document_version, make_document_id
from ..ingestion.parser import get_parser, register_builtin_parsers
from ..ingestion.parent_builder import ParentBuilder, build_children
from ..ingestion.tokenizer import TokenCounter
from .state import IngestState, Stage

DEFAULT_PATTERNS = ("**/*.md", "**/*.txt", "**/*.pdf")


#: dry-run 预览时最多列出多少个待处理文件（够看清命名习惯，又不刷屏）
PREVIEW_KEYS = 10


def _pattern_labels(patterns: Sequence[str]) -> list[str]:
    """`("**/*.md", "**/*.txt")` → `[".md", ".txt"]`。给用户看扩展名，不给看 glob。"""
    labels: list[str] = []
    for pattern in patterns:
        suffix = Path(pattern).suffix
        label = suffix or pattern
        if label not in labels:
            labels.append(label)
    return labels


#: 仓库自带的占位文件：它们不是语料。`data/README.md` 是放置规则说明，
#: `.gitkeep` 是空目录占位。不排除的话，用户第一次建库会把说明文档也塞进知识库，
#: 而「空目录引导」也会误判成「有文件」（A5 真·干净环境实测）。
PLACEHOLDER_NAMES = ("README.md", ".gitkeep")


def is_placeholder(path: Path, data_dir: Path) -> bool:
    """只排除**扫描根目录下**的占位文件；子目录里的 README.md 是正当语料。"""
    path = Path(path)
    try:
        if path.parent.resolve() != Path(data_dir).resolve():
            return False
    except OSError:                                   # pragma: no cover - 路径不可解析
        return False
    return path.name in PLACEHOLDER_NAMES


def discover_corpus(data_dir: Path | str, patterns: Sequence[str] = DEFAULT_PATTERNS) -> list[Path]:
    """按扩展名发现语料，**相对路径排序**（断点恢复要可复现），并跳过占位文件。"""
    root = Path(data_dir)
    found: set[Path] = set()
    for pattern in patterns:
        found.update(p for p in root.glob(pattern) if p.is_file())
    return sorted((p for p in found if not is_placeholder(p, root)),
                  key=lambda p: p.relative_to(root).as_posix())


def empty_corpus_hint(
    data_dir: Path | str, patterns: Sequence[str] = DEFAULT_PATTERNS
) -> list[str]:
    """语料目录为空时的可复制指引。

    开源场景下这是**第一次运行最常见的失败**：仓库不携带语料（版权 + 体积），
    用户克隆下来 `data/` 是空的。只打印「发现 0 个文件」等于把人扔在半路，
    所以这里必须一次说清三件事：**该放哪**、**支持什么格式**、**放完跑哪条命令**。

    空目录不是错误：`failed` 仍为 0，退出码仍为 0（见 `tests/test_ingest_empty_corpus.py`）。
    """
    exts = "、".join(_pattern_labels(patterns))
    globs = " ".join(patterns)
    target = f'python scripts/ingest.py --dir "{data_dir}"'
    return [
        f"[ingest] {data_dir} 里没有可导入的文件（空目录不算错误，退出码仍是 0）",
        f"[ingest] 支持格式：{exts}（递归扫描子目录）",
        f"[ingest] 默认 patterns：{globs}（需要别的后缀用 --patterns 覆盖）",
        "[ingest] 放法：把文档放进 data/ 下，例如 data/md/我的笔记.md；建议按主题分子目录",
        f"[ingest] 放好之后重跑：{target}",
        f"[ingest] 想先看会切出什么、不落库：{target} --dry-run",
    ]


def missing_only_keys_hint(only_keys: Iterable[str]) -> list[str]:
    """`--only` 一个都没匹配到时给**另一套**指引。

    这跟「空目录」是两件事：目录里其实有文件，是 key 写法不对（key 必须是相对 `--dir`
    的路径）。给错药比不给药更浪费时间，所以两种提示分开生成。
    """
    keys = list(only_keys)
    shown = "、".join(keys[:5]) + ("…" if len(keys) > 5 else "")
    return [
        f"[ingest] --only 指定的文件一个都没匹配到：{shown}",
        "[ingest] key 是相对 --dir 的路径，如 md/nvidia/hnsg.md：不带盘符、不带 --dir 前缀",
        "[ingest] 想知道本次会处理哪些文件：加 --dry-run 跑一遍，会列出前 10 个文件名",
    ]


def _preview_keys(keys: Sequence[str], limit: int = PREVIEW_KEYS) -> list[str]:
    """dry-run 的文件名预览（`--only` 写错时靠它自查）。"""
    head = list(keys[:limit])
    if len(keys) > limit:
        head.append(f"…（共 {len(keys)} 个）")
    return head


@dataclass
class IngestConfig:
    data_dir: Path
    patterns: Sequence[str] = DEFAULT_PATTERNS
    limit: int | None = None
    force: bool = False
    dry_run: bool = False
    window_docs: int = 64
    window_children: int = 256
    continue_on_error: bool = True


@dataclass
class IngestSummary:
    discovered: int = 0
    skipped: int = 0
    indexed: int = 0
    failed: int = 0
    documents: int = 0
    parents: int = 0
    children: int = 0
    embedding_calls: int = 0
    elapsed_s: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)
    #: 空语料 / `--only` 落空时给用户的可复制指引（`--json` 下日志会被吞，所以也放 summary）
    hint: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class IngestPipeline:
    """编排器。所有协作者都可注入，便于测试与替换（假 store / 假 indexer / 假 embedder）。"""

    def __init__(
        self,
        *,
        data_dir: str | os.PathLike[str],
        state: IngestState,
        store: Any,
        indexer: Any,
        embedder: Any,
        counter: TokenCounter | None = None,
        builder: ParentBuilder | None = None,
        log: Callable[[str], None] | None = None,
        crash_hook: Callable[[int], None] | None = None,
        patterns: Sequence[str] = DEFAULT_PATTERNS,
        embedding_version: str = DEFAULT_EMBEDDING_VERSION,
        window_docs: int = 64,
        window_children: int = 256,
    ):
        self.data_dir = Path(data_dir)
        self.state = state
        self.store = store
        self.indexer = indexer
        self.embedder = embedder
        self.counter = counter or TokenCounter()
        self.builder = builder or ParentBuilder(counter=self.counter)
        self.log = log or (lambda message: None)
        self.crash_hook = crash_hook
        self.patterns = tuple(patterns)
        self.embedding_version = embedding_version
        self._window_docs = window_docs
        self._window_children = window_children

    # -- 发现 -----------------------------------------------------------

    def discover(self) -> list[Path]:
        """按扩展名发现文件（相对路径排序；跳过根目录的占位文件）。"""
        return discover_corpus(self.data_dir, self.patterns)

    # -- 主流程 ---------------------------------------------------------

    def run(
        self,
        *,
        limit: int | None = None,
        force: bool = False,
        dry_run: bool = False,
        window_docs: int | None = None,
        window_children: int | None = None,
        crash_hook: Callable[[int], None] | None = None,
        continue_on_error: bool = True,
        only_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        self._force = force
        self._continue_on_error_flag = continue_on_error
        summary = IngestSummary()
        hook = crash_hook or self.crash_hook
        # 先清掉源文件已消失的状态行（改名/删除），否则会留下永远修不好的幽灵失败项
        pruned = self.state.prune_missing(self.data_dir)
        if pruned:
            self.log(f"[ingest] 清理 {pruned} 条已消失文件的状态记录")

        files = self.discover()
        summary.discovered = len(files)
        if only_keys is not None:
            wanted = set(only_keys)
            files = [p for p in files if p.relative_to(self.data_dir).as_posix() in wanted]
            if not files:
                # 目录里有文件但 key 全落空 → 教路径怎么写，别说「请放文档」
                hint = missing_only_keys_hint(only_keys)
                for line in hint:
                    self.log(line)
                summary.hint = hint
                summary.elapsed_s = round(time.perf_counter() - started, 2)
                return summary.as_dict()
        if limit is not None:
            files = files[:limit]
        self.log(f"[ingest] 发现 {summary.discovered} 个文件，本次处理 {len(files)} 个"
                 f"{'（dry-run）' if dry_run else ''}")
        if not files:
            # 空语料：不是错误，但必须给出「放哪 / 支持什么 / 放完跑哪条」
            hint = empty_corpus_hint(self.data_dir, self.patterns)
            for line in hint:
                self.log(line)
            summary.hint = hint
        if dry_run:
            for key in _preview_keys([p.relative_to(self.data_dir).as_posix() for p in files]):
                self.log(f"[dry-run] {key}")
            summary.elapsed_s = round(time.perf_counter() - started, 2)
            return summary.as_dict()

        processed = 0
        docs_per_window = window_docs or self._window_docs
        children_per_window = window_children or self._window_children

        # 窗口按「真实子块数」切，不按文件大小猜：只有切完才知道一篇文档有多少 child
        buffer: list[dict[str, Any]] = []
        buffered_children = 0
        for path in files:
            key = path.relative_to(self.data_dir).as_posix()
            try:
                item = self._prepare(path, key, summary)
            except (_Skipped, _DocFailed):
                item = None
            if item is not None:
                buffer.append(item)
                buffered_children += len(item["children"])
            if buffer and (buffered_children >= children_per_window or len(buffer) >= docs_per_window):
                self._flush(buffer, summary)
                processed += len(buffer)
                buffer, buffered_children = [], 0
                if hook is not None:
                    hook(processed)
        if buffer:
            self._flush(buffer, summary)
            processed += len(buffer)
            if hook is not None:
                hook(processed)

        summary.elapsed_s = round(time.perf_counter() - started, 2)
        self.log(
            f"[ingest] 完成：skipped={summary.skipped} indexed={summary.indexed} failed={summary.failed} "
            f"children={summary.children} 用时 {summary.elapsed_s}s"
        )
        summary.failures = self.state.failed_documents()
        return summary.as_dict()

    def _flush(self, prepared: Sequence[dict[str, Any]], summary: IngestSummary) -> None:
        if not prepared:
            return
        # 一个窗口一次 embedding 调用（内部按 batch_size 切批 + 缓存）
        texts = [text for item in prepared for text in item["embedding_texts"]]
        try:
            summary.embedding_calls += 1
            matrix = self.embedder.encode_documents(texts)
        except Exception as exc:  # noqa: BLE001 - 失败要进清单，不能让整轮崩
            message = f"{type(exc).__name__}: {exc}"
            for item in prepared:
                self.state.fail(item["key"], Stage.EMBEDDING, message)
                summary.failed += 1
                self.log(f"[fail] {item['key']} @EMBEDDING {message}")
            if not self._continue_on_error():
                raise
            return

        offset = 0
        for item in prepared:
            count = len(item["children"])
            vectors = {
                child["chunk_id"]: matrix[offset + i]
                for i, child in enumerate(item["children"])
            }
            offset += count
            self._index(item, vectors, summary)

    def _prepare(self, path: Path, key: str, summary: IngestSummary) -> dict[str, Any] | None:
        """PARSING → CHUNKING → 写 Store。返回待 embedding 的条目。"""
        # 文本格式用「归一化文本」算版本（换行/BOM 不影响）；二进制格式用原始字节，
        # 否则 PDF 被 decode 成乱码再 hash，改了字体也会"版本不变"
        version = _version_of(path)

        if not self._force and self.state.is_done(
            key,
            document_version=version,
            chunker_version=self.builder.ingest_version,  # 父+子，见 N43
            embedding_version=self.embedding_version,
        ):
            summary.skipped += 1
            self.log(f"[skip] {key}：已完成且版本未变")
            raise _Skipped

        self.state.begin(key)
        self.state.advance(key, Stage.PARSING)
        try:
            # 取解析器也要在 try 里：不支持的后缀必须落 FAILED_PARSING，
            # 而不是让 UnsupportedFormatError 冒出去把整轮打断
            parser = self._parser_for(path)
            blocks = parser.parse(str(path))
        except Exception as exc:  # noqa: BLE001
            self._fail(key, Stage.PARSING, exc, summary)
            raise _DocFailed from exc

        self.state.advance(key, Stage.CHUNKING)
        try:
            document_id = make_document_id(key)
            title = _guess_title(blocks, path)
            parents = self.builder.build(blocks, document_id=document_id,
                                         document_version=version, title=title)
            children = build_children(parents, counter=self.counter,
                                      embedding_version=self.embedding_version)
            if not children:
                raise IngestionError("切分后没有任何 child（空文档或解析结果为空）")
            parser_version = getattr(parser, "parser_version", "") or f"{path.suffix.lstrip('.').lower()}_v1"
            # 语言在建库时判一次就够（整篇同语言），别留给下游每次现判（N49）。
            language = detect_language(" ".join(p.content for p in parents)[:4000])
            for parent in parents:
                parent.language = language
            document = {
                "document_id": document_id,
                "source_uri": key,
                "file_name": path.name,
                "file_type": path.suffix.lstrip(".").lower(),
                "title": title,
                "language": language,
                "document_version": version,
                "parser_version": parser_version,
                "created_at": None,
                "status": "active",
            }
            parent_records = [p.to_record() for p in parents]
            child_records = [c.to_record() for c in children]
            self.store.replace_document(document, parent_records, child_records)
        except Exception as exc:  # noqa: BLE001
            self._fail(key, Stage.CHUNKING, exc, summary)
            raise _DocFailed from exc

        self.state.advance(key, Stage.EMBEDDING)
        summary.documents += 1
        summary.parents += len(parent_records)
        summary.children += len(child_records)
        return {
            "key": key,
            "document": document,
            "parents": parent_records,
            "children": child_records,
            "embedding_texts": [c.get("embedding_text") or c["content"] for c in child_records],
        }

    def _index(self, item: dict[str, Any], vectors: dict[str, Any], summary: IngestSummary) -> None:
        self.state.advance(item["key"], Stage.INDEXING)
        try:
            self.indexer.index_document(item["document"], item["parents"], item["children"], vectors=vectors)
        except Exception as exc:  # noqa: BLE001
            self._fail(item["key"], Stage.INDEXING, exc, summary)
            return
        self.state.complete(
            item["key"],
            document_id=item["document"]["document_id"],
            document_version=item["document"]["document_version"],
            chunker_version=self.builder.ingest_version,  # 父+子，见 N43
            embedding_version=self.embedding_version,
        )
        summary.indexed += 1
        self.log(f"[ok] {item['key']}：{len(item['children'])} child / {len(item['parents'])} parent")

    # -- 辅助 -----------------------------------------------------------

    _force = False
    _continue_on_error_flag = True

    def _continue_on_error(self) -> bool:
        return self._continue_on_error_flag

    def _fail(self, key: str, stage: Stage, exc: BaseException, summary: IngestSummary) -> None:
        message = f"{type(exc).__name__}: {exc}"
        self.state.fail(key, stage, message)
        summary.failed += 1
        self.log(f"[fail] {key} @{stage.value} {message}")
        if not self._continue_on_error_flag:
            raise exc

    def _parser_for(self, path: Path):
        global _PARSERS_READY
        if not _PARSERS_READY:
            # 幂等注册：注册表在进程内是全局的，重复注册会抛「后缀已被占用」
            register_builtin_parsers(replace=True)
            _PARSERS_READY = True
        return get_parser(str(path))

    def retry_failed(self, **kwargs) -> dict[str, Any]:
        """只重跑处于 FAILED_* 的文档（D18 验收的「单条重试」用 run(only_keys=[...])）。"""
        failed = {row["doc_key"] for row in self.state.failed_documents()}
        return self.run(only_keys=failed, **kwargs)


_PARSERS_READY = False


class _Skipped(Exception):
    """内部信号：文档已完成，跳过。"""


class _DocFailed(Exception):
    """内部信号：文档失败已记入状态表，继续下一篇。"""


def _version_of(path: Path) -> str:
    if path.suffix.lower() in {".pdf"}:
        import hashlib

        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return document_version(_read_text(path))


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _guess_title(blocks, path: Path) -> str:
    for block in blocks:
        if block.block_type == "heading" and block.heading_level == 1 and block.text.strip():
            return block.text.strip()
    return path.stem
