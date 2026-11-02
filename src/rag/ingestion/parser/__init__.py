"""解析器注册表：按文件后缀分派（方案 §2.2）。

注册是显式的：``register_parser(MarkdownParser)`` 才会生效，
不做目录扫描式的自动注册——导入顺序影响行为是这类注册表最常见的坑。
"""

from pathlib import Path

from src.rag.ingestion.errors import UnsupportedFormatError
from src.rag.ingestion.parser.base import BaseParser

__all__ = [
    "BaseParser",
    "available_suffixes",
    "get_parser",
    "register_parser",
    "reset_registry",
]

_REGISTRY: dict[str, type[BaseParser]] = {}


def register_parser(parser_cls: type[BaseParser], *, replace: bool = False) -> type[BaseParser]:
    """注册解析器。后缀冲突时默认报错（显式 replace=True 才覆盖）。"""
    suffixes = [s.lower() for s in getattr(parser_cls, "supported_suffixes", ())]
    if not suffixes:
        raise ValueError(f"{parser_cls.__name__} 没有声明 supported_suffixes，无法注册")

    conflicts = [s for s in suffixes if s in _REGISTRY and not replace]
    if conflicts:
        owners = ", ".join(f"{s} → {_REGISTRY[s].__name__}" for s in conflicts)
        raise ValueError(
            f"后缀已被占用：{owners}；如需替换请用 register_parser(..., replace=True)"
        )

    for suffix in suffixes:
        _REGISTRY[suffix] = parser_cls
    return parser_cls


def get_parser(path: str) -> BaseParser:
    """按后缀返回解析器实例。"""
    suffix = Path(str(path)).suffix.lower()
    parser_cls = _REGISTRY.get(suffix)
    if parser_cls is None:
        supported = ", ".join(available_suffixes()) or "(空)"
        raise UnsupportedFormatError(
            f"没有解析器支持后缀 {suffix or '(无后缀)'}；已注册后缀：{supported}。路径：{path}"
        )
    return parser_cls()


def available_suffixes() -> list[str]:
    return sorted(_REGISTRY)


def reset_registry() -> None:
    """清空注册表（测试用，也用于「只加载需要的解析器」的批处理场景）。"""
    _REGISTRY.clear()


def register_builtin_parsers(*, replace: bool = False) -> list[str]:
    """注册内置解析器（Markdown / TXT / PDF），返回可用后缀列表。

    刻意做成显式函数而不是 import 时自动注册：注册表被测试反复 reset，
    副作用式注册只会让 bug 表现为「跑单个测试能过、跑全量就挂」。
    """
    from .markdown_parser import MarkdownParser
    from .pdf_parser import PdfParser
    from .txt_parser import TxtParser

    for parser_cls in (MarkdownParser, TxtParser, PdfParser):
        register_parser(parser_cls, replace=replace)
    return available_suffixes()
