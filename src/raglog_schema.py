"""日志事件与字段常量。

埋点侧（raglog.py）与读取侧（raglog_report.py）共用同一份定义，
避免"写日志的字段名"和"读日志的字段名"两处漂移。
字段规范见 docs/eval-and-logging-design.md 第 7 章。
"""

SCHEMA_VERSION = "raglog/1"

# 日志级别：off 完全关闭；summary 只记请求级摘要；detail 记证据全文与分路排名；debug 追加候选分布
LEVELS = ("off", "summary", "detail", "debug")
DEFAULT_LEVEL = "detail"
LEVEL_ORDER = {"off": 0, "summary": 1, "detail": 2, "debug": 3}

# 事件名
EV_RUN = "run"
EV_TRACE = "trace"
EV_SPAN = "span"
EV_RETRIEVE = "retrieve"
EV_PROMPT = "prompt"
EV_GENERATE = "generate"
EV_ERROR = "error"
EV_EVENT = "event"          # 通用事件（agent 决策等）
EV_LABEL = "label"          # 人工标注，写在独立的 *.labels.jsonl

# 文本字段上限：超过则外置到 artifacts/，主日志只留引用与 sha256
TEXT_LIMIT = 8000

# review 告警阈值（raglog_report 使用，可被命令行覆盖）
# 注意：不能用分数差的绝对值。RRF 的分数就是 1/(k+rank) 的累加，相邻名次天然只差 ~2.7e-4，
# 用绝对值会把每一条 RRF 结果都判成「平坦」；只用相对值，且 RRF 仅在完全同分时才算平坦。
FLAT_SCORE_RATIO = 0.02      # (top1 - topk) / top1 小于此值视为「分数平坦，排序不可信」
SOURCE_COLLAPSE_RATIO = 0.5  # 不同来源数 / k 低于此值视为「来源坍缩」

# 负样本离群阈值：用「相对全库的 z 分数」而不是绝对余弦。
# 余弦尺度随 embedding 模型变（换 bge-m3 后负样本 top1 绝对余弦 0.682→0.619，
# 正样本 gold 也从 0.682→0.590），拿 0.75 这种绝对值跨模型比较会得出反向结论。
NEG_Z_ALERT = 3.0        # top1 z 分数 ≥ 此值 = 「检索器很自信地凑了个答案」
NEG_SEPARATION_MIN = 1.0 # 正样本 top1 z 中位数要至少比负样本高这么多，才谈得上用阈值弃答

# 人工标注枚举（固定，不允许自由文本）
LABELS = (
    "ok",
    "retrieval_miss",
    "ranking_bad",
    "answer_wrong",
    "hallucinated_citation",
    "over_abstain",
    "under_abstain",
)

# 弃答启发式：只用于打标记，命中记录必须带 abstained_heuristic=True。
# 分强弱两级的原因：正文里「提到了证据不足这个概念」不等于「这一问弃答了」。
# 实测 s0-04 的答案在讲 Top-K 风险时写了「其余证据未提供…」，被宽泛匹配误判成弃答。
ABSTAIN_STRONG = (
    "无法回答",
    "无法确定",
    "不能回答",
    "无法可靠回答",
    "没有足够证据",
    "insufficient evidence",
    "cannot answer",
)
ABSTAIN_WEAK = (
    "证据不足",
    "没有找到",
    "未提供",
    "找不到",
)
# 兼容旧名
ABSTAIN_PATTERNS = ABSTAIN_STRONG + ABSTAIN_WEAK

# 引用解析不要求方括号：实测 agentic_rag 的回答写成「（证据 1）」，只认 [证据 1] 会漏计
CITATION_PATTERN = r"证据\s*(\d+)"
