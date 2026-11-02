"""Parser 基类（方案 §2.2「文档解析统一接口」）。

每个格式一个子类，只负责「文件 → DocumentBlock 列表」；
切分、embedding、索引都不认识原始格式。
"""

from abc import ABC, abstractmethod
from typing import Any

from src.rag.ingestion.block import DocumentBlock


class BaseParser(ABC):
    """解析器抽象基类。

    子类必须提供两项类属性：
    - ``parser_version``：写进 Document 元数据，切分结果可追溯到解析器版本；
    - ``supported_suffixes``：小写后缀元组，注册表据此分派。
    """

    #: 解析器版本号，进入 Document 元数据（方案 §3.1 的 parser_version）
    parser_version: str = ""

    #: 认领的小写后缀，例如 (".md", ".markdown")
    supported_suffixes: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: str, text: str | None = None) -> list[DocumentBlock]:
        """把文件解析成 Block 列表。

        ``text`` 用于调用方已经读过文件内容的场景（测试、批量导入避免重复 IO），
        为 None 时由解析器自行读取 ``path``。
        """

    @classmethod
    def can_parse(cls, suffix: str) -> bool:
        return suffix.lower() in {s.lower() for s in cls.supported_suffixes}

    def describe(self) -> dict[str, Any]:
        return {
            "parser": type(self).__name__,
            "parser_version": self.parser_version,
            "supported_suffixes": list(self.supported_suffixes),
        }
