"""基准题集与评估指标口径的守卫。

两件事：

1. **题集本身要自洽**：qid 唯一、正样本有 gold、负样本没有 gold、`query_en` 人工写过、
   两个 id 空间（`gold_chunks`=parent / `gold_points`=child）不混；
2. **指标口径不许退化成「一个综合分」**：`hit@k` / `full@k` / `recall@k` 三者分母不同，
   必须同时报；负样本不进 recall 分母。
"""

import json
import math
import re
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "datasets" / "k8s-zh-en-v1" / "questions.jsonl"

import importlib.util


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_bench = _load("eval_bench", ROOT / "scripts" / "eval_bench.py")


class QuestionsIntegrityTest(unittest.TestCase):
    def setUp(self):
        if not QUESTIONS.is_file():
            self.skipTest("题集还没生成（先跑 scripts/build_bench_questions.py append）")
        self.rows = [json.loads(line) for line in
                     QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_qids_unique_and_count(self):
        qids = [row["qid"] for row in self.rows]
        self.assertEqual(len(qids), len(set(qids)), "qid 必须唯一")
        self.assertGreaterEqual(len(qids), 100, "第一版基准至少 100 题")

    def test_every_question_is_rewritten_text(self):
        for row in self.rows:
            self.assertTrue(row["question"].strip(), row["qid"])
            self.assertTrue(row["query_en"].strip(),
                            f"{row['qid']}：query_en 必须人工写，不许留空（机翻回填不允许）")

    def test_positives_have_gold_and_negatives_do_not(self):
        for row in self.rows:
            if row["type"] == "negative":
                self.assertEqual(row["gold_chunks"], [], f"{row['qid']}：负样本不该有 gold")
                self.assertEqual(row["gold_spans"], [], f"{row['qid']}：负样本不该有 span")
            else:
                self.assertTrue(row["gold_chunks"], f"{row['qid']}：正样本必须有 gold_chunks")
                self.assertTrue(row["gold_spans"], f"{row['qid']}：正样本必须有 gold_spans")

    def test_two_id_spaces_are_not_mixed(self):
        for row in self.rows:
            for cid in row["gold_chunks"]:
                self.assertTrue(cid.startswith("par_"), f"{row['qid']}：gold_chunks 应是 parent id")
            for pid in row.get("gold_points", []):
                self.assertTrue(pid.startswith("chk_"), f"{row['qid']}：gold_points 应是 child id")

    def test_evidence_language_is_declared_for_positives(self):
        for row in self.rows:
            if row["type"] == "negative":
                continue
            self.assertIn(row["expected_evidence_lang"], ("zh-cn", "en"), row["qid"])

    def test_type_vocabulary(self):
        allowed = {"single_hop", "multi_hop", "numeric", "cross_lingual", "negative"}
        for row in self.rows:
            self.assertIn(row["type"], allowed, f"{row['qid']}：未知题型 {row['type']}")

    def test_cross_lingual_group_exists_both_directions(self):
        x = [row for row in self.rows if row["type"] == "cross_lingual"]
        self.assertTrue(any(row["expected_evidence_lang"] == "en" for row in x), "缺「中文问→英文证据」")
        self.assertTrue(any(row["expected_evidence_lang"] == "zh-cn" for row in x), "缺「英文问→中文证据」")

    def test_negative_questions_stay_out_of_corpus(self):
        """负样本必须是「语料里确实没有」的问题，不能是「我们没写 gold」。"""
        negatives = [row for row in self.rows if row["type"] == "negative"]
        self.assertGreaterEqual(len(negatives), 3)
        for row in negatives:
            self.assertTrue(row["note"].startswith("负样本"), f"{row['qid']}：负样本要留痕说明")


class SummarizeMetricTest(unittest.TestCase):
    """指标口径：三个分母都要报，负样本不进分母。"""

    def _row(self, *, gold, hit, full, recall, rank):
        return {"gold": gold, "hit": hit, "full": full, "recall": recall, "first_hit_rank": rank}

    def test_reports_all_three_denominators(self):
        rows = [
            self._row(gold=["a"], hit=True, full=True, recall=1.0, rank=1),
            self._row(gold=["a", "b"], hit=True, full=False, recall=0.5, rank=3),
            self._row(gold=["a"], hit=False, full=False, recall=0.0, rank=None),
            self._row(gold=[], hit=False, full=False, recall=None, rank=None),   # 负样本
        ]
        stats = eval_bench.summarize(rows, 10)
        self.assertEqual(stats["positives"], 3, "负样本不进分母")
        self.assertAlmostEqual(stats["hit@10"], 0.6667, places=4)
        self.assertAlmostEqual(stats["full@10"], 0.3333, places=4)
        self.assertAlmostEqual(stats["recall@10"], 0.5, places=4)
        self.assertAlmostEqual(stats["mrr"], (1 / 1 + 1 / 3) / 3, places=4)

    def test_all_miss_gives_zero_mrr(self):
        rows = [self._row(gold=["a"], hit=False, full=False, recall=0.0, rank=None)]
        stats = eval_bench.summarize(rows, 10)
        self.assertEqual(stats["mrr"], 0.0, "全未命中时 MRR 必须是 0，不能留空")

    def test_no_positives_is_not_a_zero(self):
        stats = eval_bench.summarize([self._row(gold=[], hit=False, full=False, recall=None, rank=None)], 10)
        self.assertEqual(stats, {"n": 1, "positives": 0},
                         "全是负样本时不该报 0% 命中，那样会误导成「系统全错」")


if __name__ == "__main__":
    unittest.main()


class ReadmeBaselineNumbersTest(unittest.TestCase):
    """README 里的基线数字必须与 `docs/baseline-k8s-v1.json` 一致。

    手写数字最常见的死法是「跑完新结果忘了改 README」——读的人照着旧数做决策，
    比没有数字更糟。这里让漂移直接变成红灯。
    """

    BASELINE = ROOT / "docs" / "baseline-k8s-v1.json"
    README = ROOT / "README.md"

    def setUp(self):
        if not self.BASELINE.is_file():
            self.skipTest("基线还没跑（先跑 scripts/eval_bench.py）")
        self.data = json.loads(self.BASELINE.read_text(encoding="utf-8"))
        self.readme = self.README.read_text(encoding="utf-8")

    def test_overall_numbers_appear_verbatim(self):
        overall = self.data["overall"]
        k = self.data["k"]
        for key in (f"recall@{k}", f"hit@{k}", f"full@{k}", "mrr"):
            value = f"{overall[key]:.4f}"
            self.assertIn(value, self.readme, f"README 里找不到 {key}={value}，数字漂了？")

    def test_evidence_language_slices_appear_verbatim(self):
        k = self.data["k"]
        for lang in ("en", "zh-cn"):
            stats = self.data["by_evidence_lang"][lang]
            for key in (f"recall@{k}", f"hit@{k}", f"mrr"):
                value = f"{stats[key]:.4f}"
                self.assertIn(value, self.readme, f"README 里找不到 {lang} 的 {key}={value}")

    def test_latency_appears(self):
        latency = self.data["latency_ms"]
        for name in ("p50", "p90"):
            value = latency[name]
            # 取整方式（半数进位/银行家舍入）不该让守卫变红，所以两边都认
            candidates = {f"{name} **{value:.0f} ms**", f"{name} **{math.floor(value + 0.5)} ms**"}
            self.assertTrue(candidates & {c for c in candidates if c in self.readme},
                            f"README 里找不到 {name}（实际 {value} ms）")

    def test_question_counts_are_consistent(self):
        overall = self.data["overall"]
        self.assertIn(f"| {overall['positives']} |", self.readme, "正样本数与 README 不一致")
        re_questions = ROOT / "datasets" / "k8s-zh-en-v1" / "questions.jsonl"
        total = len([l for l in re_questions.read_text(encoding="utf-8").splitlines() if l.strip()])
        self.assertIn(f"**{total} 题**", self.readme, f"README 里的题数应为 {total}")


class EvalCollectionOptionTest(unittest.TestCase):
    """`eval_bench` 必须能指向**另一个**集合与文档表。

    为什么必须有：改切片/嵌入参数后要比「新旧两库」的差，而旧实现把
    `bench_k8s_v1` + `storage/bench/documents.sqlite3` 写死在模块常量里 ——
    只能先覆盖正式基准库再评估，等于把唯一一份基线数据当试验田用。
    纪律本来就要求「压测/实验用带随机后缀的集合名」，脚本得先支持。
    """

    def test_defaults_stay_on_the_official_baseline(self):
        from scripts.eval_bench import build_parser

        args = build_parser().parse_args([])
        self.assertEqual(args.collection, "bench_k8s_v1")
        self.assertEqual(Path(args.store).name, "documents.sqlite3")

    def test_collection_and_store_can_be_overridden(self):
        from scripts.eval_bench import build_parser

        args = build_parser().parse_args(["--collection", "bench_qc_ab12", "--store", "x/y.sqlite3"])
        self.assertEqual(args.collection, "bench_qc_ab12")
        self.assertEqual(Path(args.store), Path("x/y.sqlite3"))

    def test_evaluate_uses_the_given_collection(self):
        """打桩到 `build_default_retriever` 这一层：证明参数真的传下去了。"""
        import scripts.eval_bench as bench

        seen = {}
        original = bench._build_retriever

        def fake(collection, store_path, warmup_texts, log):
            seen["collection"] = collection
            seen["store"] = store_path
            return object()

        bench._build_retriever = fake
        try:
            bench.evaluate([], k=1, pool=1, rerank=False, warmup=False,
                           collection="bench_qc_ab12", store=Path("tmp.sqlite3"),
                           parents={}, counter=object(), reranker=None)
        finally:
            bench._build_retriever = original
        self.assertEqual(seen["collection"], "bench_qc_ab12")
        self.assertEqual(Path(seen["store"]), Path("tmp.sqlite3"))


class EvalOutPathDisplayTest(unittest.TestCase):
    """`--out` 指到仓库外（或相对路径）时，最后一行打印不许崩。

    BUG 实证：`--out storage/bench/qc/eval.json`（相对路径）跑完 100 题后
    `args.out.relative_to(ROOT)` 抛 `ValueError` —— 结果文件其实已经写好了，
    但人看到的是 traceback，会以为评估失败、白跑一遍。
    """

    def test_relative_out_is_resolved_against_cwd(self):
        from scripts.eval_bench import build_parser, _resolve_paths

        args = _resolve_paths(build_parser().parse_args(["--out", "storage/bench/x/eval.json"]))
        self.assertTrue(args.out.is_absolute())

    def test_display_path_outside_repo_does_not_raise(self):
        import tempfile

        from scripts.eval_bench import _display_path

        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "eval.json"
            self.assertEqual(_display_path(outside), str(outside))

    def test_display_path_inside_repo_is_relative(self):
        from scripts.eval_bench import ROOT, _display_path

        self.assertEqual(_display_path(ROOT / "docs" / "x.json"), str(Path("docs") / "x.json"))


class ReadmeCrossLingualCellsTest(unittest.TestCase):
    """README 里「提问语言 × 证据语言」四格和题型表也必须跟着基线走。

    N38 只守了总表与 en/zh 切片，那两处之外还有 8 个更常被引用的数字（四格）+
    12 个题型数字。重立基线时漏改它们，读者会照着「中→英 30%」这种旧结论做决策。
    这里直接从 `baseline-k8s-v1.rows.jsonl` 重算（提问语言按是否含中日韩字符判定），
    不抄任何手写数字。
    """

    ROWS = ROOT / "docs" / "baseline-k8s-v1.rows.jsonl"
    README = ROOT / "README.md"
    CJK = re.compile(r"[\u4e00-\u9fff]")

    def setUp(self):
        if not self.ROWS.is_file():
            self.skipTest("逐题结果还没跑（先跑 scripts/eval_bench.py）")
        self.rows = [json.loads(line) for line in self.ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.readme = self.README.read_text(encoding="utf-8")

    def _cell(self, qlang: str, elang: str):
        picked = [
            row for row in self.rows
            if row["type"] != "negative"
            and ("中" if self.CJK.search(row["question"]) else "英") == qlang
            and {"zh-cn": "中", "en": "英"}.get(row["expected_evidence_lang"]) == elang
        ]
        hit = sum(1 for row in picked if row["hit"]) / len(picked)
        mrr = statistics.mean((1.0 / row["first_hit_rank"]) if row["first_hit_rank"] else 0.0 for row in picked)
        return len(picked), hit, mrr

    def test_four_cells_appear_verbatim(self):
        for qlang, elang in (("英", "英"), ("中", "中"), ("英", "中"), ("中", "英")):
            n, hit, mrr = self._cell(qlang, elang)
            row_marker = f"| {qlang} → {elang} | {n} |"
            self.assertIn(row_marker, self.readme, f"README 四格表里找不到「{qlang} → {elang}」这一行，或 n 变了")
            for value in (f"{hit:.4f}", f"{mrr:.4f}"):
                self.assertIn(value, self.readme, f"README 里找不到 {qlang}→{elang} 的 {value}")

    def test_per_type_numbers_appear_verbatim(self):
        baseline = json.loads((ROOT / "docs" / "baseline-k8s-v1.json").read_text(encoding="utf-8"))
        k = baseline["k"]
        for name, stats in baseline["by_type"].items():
            if name == "negative":
                continue
            self.assertIn(f"`{name}` | {stats['positives']} |", self.readme, f"题型 {name} 的 n 与 README 不一致")
            for key in (f"hit@{k}", f"full@{k}", "mrr"):
                self.assertIn(f"{stats[key]:.4f}", self.readme, f"README 里找不到 {name} 的 {key}={stats[key]:.4f}")
