"""D33 Prompt 与引用规范（方案 §4.9 的七条规则 + 兼容现有 `[证据 n]`）。

三条设计约束：

1. **规则必须逐条落在 system prompt 里**，不能只写在文档里——否则改 prompt 时没人知道漏了哪条；
2. **证据编号只能由组装器给**（`AssembledContext.blocks[].index`），prompt 里不重排、不重复编号，
   这样 `[证据 n]` 和来源字段是同一个 n；
3. **没有证据就不要调模型**：`should_abstain()` 为真时直接返回固定的弃答话术，
   省一次 9B 推理，也避免模型在空上下文上编造。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .citations import ABSTAIN_PATTERNS

#: 方案 §4.9 的七条规则（原文照搬，只在前面加一句「编号从哪来」）
SYSTEM_RULES: tuple[str, ...] = (
    "优先依据提供的 Context 回答。",
    "不得将 Context 中没有的信息编造成事实。",
    "如果证据不足，明确说明知识库未提供足够依据。",
    "使用用户原始提问的语言回答。",
    "技术术语可以保留英文，并在首次出现时给出中文解释。",
    "关键结论应标注来源编号，格式为 [证据 n]，n 必须是 Context 里给出的编号。",
    "不要把检索相似度分数当作事实置信度。",
)

SYSTEM_PROMPT = (
    "你是一个基于知识库回答问题的助手。\n\n"
    "规则：\n"
    + "\n".join(f"{number}. {rule}" for number, rule in enumerate(SYSTEM_RULES, start=1))
    + "\n\nContext 里每条的编号（如 [证据 3]）就是你可以引用的来源编号；"
      "不要改动编号，也不要引用 Context 里没有出现的编号。\n"
      "如果 Context 为空或不支持回答，请直接说明证据不足，不要猜测。"
)

#: 弃答话术：证据为空、或模型自己判断证据不足时用。**写死**，不进模型自由发挥。
ABSTAIN_TEXT_ZH = "证据不足：知识库未提供足够依据回答这个问题。"
ABSTAIN_TEXT_EN = "Insufficient evidence: the knowledge base does not provide enough support to answer."

#: 用户提问是中文 → 要求中文回答；否则要求英文
_ZH_HINT = "请用中文回答（技术术语可保留英文，首次出现时给中文解释）。"
_EN_HINT = "Answer in English (keep technical terms in English where appropriate)."

USER_TEMPLATE = """用户问题：{question}

检索到的证据：
{context}

{language_hint}
请逐条依据证据回答；每条关键结论后标注来源编号，例如 [证据 1]。
如果证据不足，请只说明证据不足以及缺什么，不要臆测。"""


@dataclass
class PromptBundle:
    """一次问答要送进模型的东西（含可复现的元信息）。"""

    messages: list[dict[str, str]]
    question: str
    context: str
    blocks: list[dict[str, Any]]
    tokens: int
    language: str
    abstained: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "context": self.context,
            "blocks": self.blocks,
            "tokens": self.tokens,
            "language": self.language,
            "abstained": self.abstained,
            "reason": self.reason,
        }


def looks_chinese(text: str) -> bool:
    """极简语言判定：够用于「回答用哪种语言」，正式语言识别见 `src/rag/lang/detect.py`。"""
    han = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return han >= 2


def should_abstain(blocks: Sequence[Any], *, min_tokens: int = 1) -> bool:
    """没有可用证据就弃答。`blocks` 可以是 dict 或 `ContextBlock`。"""
    total = 0
    for block in blocks:
        total += int(getattr(block, "tokens", 0) or (block.get("tokens", 0) if isinstance(block, dict) else 0))
    return total < min_tokens


def build_messages(
    question: str,
    assembled,
    *,
    language: str | None = None,
) -> PromptBundle:
    """把 `AssembledContext` 变成 messages。

    `assembled` 需要 `.render()` 与 `.blocks`（`ContextBlock` 列表）。
    没有证据时不生成 user 消息里的空 Context，而是直接给弃答话术（`abstained=True`）。
    """
    language = language or ("zh" if looks_chinese(question) else "en")
    context = assembled.render() if assembled is not None else ""
    blocks = [block.as_dict() for block in (assembled.blocks if assembled is not None else [])]
    tokens = int(getattr(assembled, "tokens", 0) or 0)

    if should_abstain(blocks):
        answer = ABSTAIN_TEXT_ZH if language == "zh" else ABSTAIN_TEXT_EN
        return PromptBundle(
            messages=[{"role": "assistant", "content": answer}],
            question=question, context="", blocks=[], tokens=tokens, language=language,
            abstained=True, reason="no_evidence",
        )

    hint = _ZH_HINT if language == "zh" else _EN_HINT
    return PromptBundle(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                question=question, context=context, language_hint=hint)},
        ],
        question=question, context=context, blocks=blocks, tokens=tokens,
        language=language, abstained=False,
    )


def detect_abstain(answer: str) -> bool:
    """模型自己说「证据不足」也算弃答（与 `PromptBundle.abstained` 合起来统计弃答率）。"""
    lowered = (answer or "").lower()
    return any(pattern in lowered for pattern in ABSTAIN_PATTERNS)
