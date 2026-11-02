"""DocumentBlock：所有格式解析后的统一中间结构（方案 §2.2）。

解析器不直接产出文本，而是产出一串带类型与定位信息的 Block；
切分器只认 Block，不认原始格式。
"""

from dataclasses import dataclass, field
from typing import Any

#: 方案 §2.2 规定的七种块类型。顺序固定，错误信息里按此顺序列出。
BLOCK_TYPES: tuple[str, ...] = (
    "heading",
    "paragraph",
    "list",
    "table",
    "code",
    "quote",
    "caption",
)

_BLOCK_TYPE_SET = frozenset(BLOCK_TYPES)

_HEADING_LEVEL_RANGE = (1, 6)

FIELDS: tuple[str, ...] = (
    "block_id",
    "block_type",
    "text",
    "heading_level",
    "page_number",
    "section_path",
    "source_start",
    "source_end",
    "metadata",
)


@dataclass
class DocumentBlock:
    """解析产物的一块内容。

    校验放在 ``__post_init__`` 里而不是交给下游：切分/建库阶段的报错点离现场太远，
    一个 ``block_type`` 拼错会让整条流水线在几千行之后才崩。
    """

    block_id: str
    block_type: str
    text: str
    heading_level: int | None = None
    page_number: int | None = None
    section_path: list[str] = field(default_factory=list)
    source_start: int | None = None
    source_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise ValueError(f"block_id 必须是非空字符串，收到 {self.block_id!r}")

        if self.block_type not in _BLOCK_TYPE_SET:
            raise ValueError(
                f"未知 block_type={self.block_type!r}，合法取值：{', '.join(BLOCK_TYPES)}"
            )

        if not isinstance(self.text, str):
            raise ValueError(f"text 必须是字符串，收到 {type(self.text).__name__}")

        if self.block_type == "heading":
            if self.heading_level is None:
                raise ValueError(f"heading 块必须给 heading_level（{_HEADING_LEVEL_RANGE[0]}–{_HEADING_LEVEL_RANGE[1]}）：block_id={self.block_id}")
            low, high = _HEADING_LEVEL_RANGE
            if not isinstance(self.heading_level, int) or not low <= self.heading_level <= high:
                raise ValueError(
                    f"heading_level 必须是 {low}–{high} 的整数，收到 {self.heading_level!r}"
                )

        if self.page_number is not None:
            if not isinstance(self.page_number, int) or self.page_number < 1:
                raise ValueError(f"page_number 必须是 ≥1 的整数，收到 {self.page_number!r}")

        if self.source_start is not None and self.source_end is not None:
            if self.source_start > self.source_end:
                raise ValueError(
                    f"source_start({self.source_start}) 不能大于 source_end({self.source_end})"
                )

    @property
    def is_empty(self) -> bool:
        """只有空白字符的块。切分前应被过滤（D09）。"""
        return not self.text.strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "text": self.text,
            "heading_level": self.heading_level,
            "page_number": self.page_number,
            "section_path": list(self.section_path),
            "source_start": self.source_start,
            "source_end": self.source_end,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentBlock":
        missing = [name for name in ("block_id", "block_type", "text") if name not in data]
        if missing:
            raise ValueError(f"DocumentBlock 缺少必填字段：{', '.join(missing)}")
        return cls(
            block_id=data["block_id"],
            block_type=data["block_type"],
            text=data["text"],
            heading_level=data.get("heading_level"),
            page_number=data.get("page_number"),
            section_path=list(data.get("section_path") or []),
            source_start=data.get("source_start"),
            source_end=data.get("source_end"),
            metadata=dict(data.get("metadata") or {}),
        )
