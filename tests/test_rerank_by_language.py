"""C0d：重排时按**候选块自己的语言**选 query（中文块用中文问、英文块用英文问）。

C0c 查到的事实（`docs/evidence/query-en-modes-k8s-v1.json`）：中→英 的 5 个漏检，
在融合（关重排）里排 **#1 / #3 / #4 / #8 / #14**，重排后掉到 **#16 / #17 / #31 / #13 / #14**。
原因是重排拿**中文原句**去给**英文窗口**打分 —— cross-encoder 不是跨语言模型，
它只会看「这段文本像不像在回答这句话」。

试点也证明不能整题切换：把重排 query 换成英文，x-009 从 #17 → #7、x-020 从 #16 → #3，
但 x-010 从 #13 → #19（中文证据的题被打下去了）。所以必须**逐候选**按块语言选。

三条不许忘的边界：
1. 没有英文 query（翻译失败 / 旧口径 `none`）→ 全部退回中文 query，不许两边都空；
2. 父块记录里没有 `language` 字段 → 按中文处理（中文 query 至少不会比现状更差）；
3. 关掉重排时一行代码都不许多跑（别为了拿语言把重排路径强行打开）。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _LangAwareEncoder:
    """只有当 query 与窗口同语言、且窗口里有 TARGET 时才给 1.0。

    这样「用错 query」在分数上必然显形：中文 query 打英文窗口 = 0 分。
    """

    def __init__(self):
        self.seen = []

    @staticmethod
    def _is_zh(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def encode_pairs(self, query, texts):
        import numpy as np

        self.seen.append((query, list(texts)))
        scores = []
        for text in texts:
            same_lang = self._is_zh(query) == self._is_zh(text)
            scores.append(1.0 if ("TARGET" in text and same_lang) else 0.0)
        return np.asarray(scores, dtype=np.float32)


def _parents():
    return {
        "par_zh_gold": {"text": "TARGET 中文证据", "content": "TARGET 中文证据",
                        "language": "zh-cn", "title": "t", "section_path": "s"},
        "par_en_gold": {"text": "TARGET english evidence", "content": "TARGET english evidence",
                        "language": "en", "title": "t", "section_path": "s"},
        "par_noise": {"text": "无关内容 noise", "content": "无关内容 noise",
                      "language": "zh-cn", "title": "t", "section_path": "s"},
    }


def _retriever():
    class _Route:
        def __init__(self, ids):
            self.ids = ids
            self.enabled = True

    class _Result:
        def __init__(self):
            self.routes = {"dense_zh": _Route(["par_noise", "par_en_gold", "par_zh_gold"])}
            self.degraded = None

    class _Retriever:
        def retrieve(self, query, k, query_en=None):
            return _Result()

    return _Retriever()


class RerankByLanguageTest(unittest.TestCase):
    def setUp(self):
        from src.rag.ask.pipeline import retrieve_with_meta
        from src.rag.ingestion.tokenizer import TokenCounter
        from src.rag.retrieval.rerank import CrossEncoderReranker, RerankConfig

        self.retrieve = retrieve_with_meta
        self.counter = TokenCounter()
        self.encoder = _LangAwareEncoder()
        self.reranker = CrossEncoderReranker(config=RerankConfig(enabled=True),
                                             encoder=self.encoder)

    def _run(self, *, mode, query_en):
        return self.retrieve("中文问题？", retriever=_retriever(), parents=_parents(),
                             counter=self.counter, reranker=self.reranker, pool=3, rerank=True,
                             query_en=query_en, rerank_query=mode)

    def test_by_lang_scores_each_candidate_with_its_own_language(self):
        out = self._run(mode="by_lang", query_en="english question?")
        order = [hit.parent_id for hit in out.hits]
        # 两块 gold 都该被自己语言的 query 打中，噪音垫底
        self.assertEqual(set(order[:2]), {"par_zh_gold", "par_en_gold"}, order)
        self.assertEqual(order[-1], "par_noise")

    def test_single_query_mode_keeps_old_behaviour(self):
        out = self._run(mode="question", query_en="english question?")
        order = [hit.parent_id for hit in out.hits]
        # 只用中文 query：英文 gold 拿 0 分，排到中文噪音后面
        self.assertEqual(order[0], "par_zh_gold")
        self.assertEqual(order[-1], "par_en_gold")

    def test_without_english_query_falls_back_to_the_question(self):
        out = self._run(mode="by_lang", query_en=None)
        self.assertEqual(out.hits[0].parent_id, "par_zh_gold")
        # 只喂过中文 query：不许出现空 query（英文块也用中文 query，分数必然是 0）
        self.assertTrue(all(self.encoder.seen), self.encoder.seen)

    def test_missing_language_key_is_treated_as_question_language(self):
        parents = _parents()
        parents["par_en_gold"].pop("language")
        out = self.retrieve("中文问题？", retriever=_retriever(), parents=parents,
                            counter=self.counter, reranker=self.reranker, pool=3, rerank=True,
                            query_en="english question?", rerank_query="by_lang")
        # 没语言字段就当中文块 → 用中文 query（不许静默用英文 query，那样会两边都不对）
        first_query = self.encoder.seen[0][0]
        self.assertIn("中", first_query)
        self.assertTrue(out.hits)

    def test_rerank_off_does_not_touch_language_logic(self):
        out = self.retrieve("中文问题？", retriever=_retriever(), parents=_parents(),
                            counter=self.counter, reranker=self.reranker, pool=3, rerank=False,
                            query_en="english question?", rerank_query="by_lang")
        self.assertEqual([hit.parent_id for hit in out.hits],
                         ["par_noise", "par_en_gold", "par_zh_gold"])
        self.assertEqual(self.encoder.seen, [], "关掉重排还在调编码器 = 白烧 CPU")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run(mode="auto", query_en="x")


class ParentLanguageTest(unittest.TestCase):
    """生产链路的父块记录必须带 `language` —— 没有它，C0d 只能靠正则猜语言。"""

    def test_load_parent_records_selects_language(self):
        import inspect

        from src.rag.retrieval.store_wiring import load_parent_records

        source = inspect.getsource(load_parent_records)
        self.assertIn("p.language", source)
        self.assertIn('"language"', source)


if __name__ == "__main__":
    unittest.main()


class CliAndPipelineWiringTest(unittest.TestCase):
    """口径要能从命令行选，且默认不变（默认不变 = 基线可比）。"""

    def test_parser_has_rerank_query_with_safe_default(self):
        from scripts.eval_bench import build_parser

        args = build_parser().parse_args([])
        self.assertEqual(args.rerank_query, "question", "默认必须是旧行为，否则基线数字会悄悄漂移")
        self.assertEqual(build_parser().parse_args(["--rerank-query", "by_lang"]).rerank_query,
                         "by_lang")

    def test_answer_query_forwards_rerank_query(self):
        import inspect

        from src.rag.ask.pipeline import answer_query

        source = inspect.getsource(answer_query)
        self.assertIn("rerank_query=rerank_query", source)


class ResolveLanguageTest(unittest.TestCase):
    """`language` 这一列在**所有历史库**里都是 NULL（N49）—— 不能只靠它。

    实测：`storage/documents.sqlite3` 25,411 条父块、`storage/bench` 7,108 条，`language` 全 NULL，
    因为 ingest 写文档时把 `"language": None` 写死了，也没传给 `ParentBuilder.build()`。
    所以规则是：**列里有就用列**（快、且是建库时的判断），没有才按正文现判（正确优先于快）。
    """

    def test_stored_value_wins(self):
        from src.rag.lang.detect import resolve_language

        self.assertEqual(resolve_language("en", "这是中文"), "en")
        self.assertEqual(resolve_language("zh-cn", "English text"), "zh-cn")

    def test_missing_value_falls_back_to_text(self):
        from src.rag.lang.detect import resolve_language

        self.assertEqual(resolve_language(None, "这是一段中文正文。"), "zh")
        self.assertEqual(resolve_language("", "This is English body text."), "en")

    def test_empty_text_stays_unknown(self):
        from src.rag.lang.detect import resolve_language

        self.assertIn(resolve_language(None, ""), ("", None))

    def test_block_lang_uses_text_when_column_is_null(self):
        from src.rag.ask.pipeline import _block_lang

        self.assertEqual(_block_lang({"language": None, "text": "English evidence here"}), "en")
        self.assertEqual(_block_lang({"language": None, "text": "中文证据"}), "zh")
        self.assertEqual(_block_lang({}), "zh")


class IngestWritesLanguageTest(unittest.TestCase):
    """建库时就要把语言写进去 —— 否则「按块语言选 query」只能每次现判。

    这里刻意**不跑整条 ingest**（文件发现 → 解析 → 切分 → 嵌入 → 写库）：
    那条路径的每个环节都有独立用例覆盖，而它依赖全局解析器注册表与词表替身，
    在 CI 与本地会走出不同分支（本次 CI 只在那边红）。这里只钉三件事：
    语言算得对、算出的语言传给了每条父块与文档、以及 run() 里确实接了线。
    """

    class _Parent:
        def __init__(self, content):
            self.content = content
            self.language = None

    def test_document_language_detects_chinese_and_english(self):
        from src.rag.ingest.pipeline import document_language

        self.assertEqual(document_language([self._Parent("这是第一段中文正文，讲 HNSW 与检索。")]), "zh")
        self.assertEqual(document_language([self._Parent("This is an English paragraph about HNSW.")]), "en")

    def test_document_language_tolerates_empty(self):
        from src.rag.ingest.pipeline import document_language

        self.assertIn(document_language([]), ("zh", "en", "", "unknown"))

    def test_parents_get_the_computed_language(self):
        from src.rag.ingest.pipeline import apply_document_language

        parents = [self._Parent("中文一"), self._Parent("中文二")]
        language = apply_document_language(parents)
        self.assertEqual(language, "zh")
        self.assertEqual([p.language for p in parents], ["zh", "zh"])

    def test_run_wires_language_into_document_and_parents(self):
        """源码断言：run() 里必须用算出来的 language 同时喂父块与文档字典。"""
        source = (ROOT / "src" / "rag" / "ingest" / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("language = apply_document_language(parents)", source)
        self.assertIn('"language": language', source)
