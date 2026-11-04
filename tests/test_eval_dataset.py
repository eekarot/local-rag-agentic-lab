"""`src.eval_dataset.validate` 的用例：重点锁住「已有 gold_points 时不再报硬错」这条。

背景（D27c）：题库从 101 扩到 202 题后，新题的 gold 走新库的父块 id（`gold_points`），
而 `validate` 用的是旧 81k 索引。旧索引有两个与新库不一致的地方：

- 同一段文字在多个版本文档里重复 → 旧索引里命中 ≥5 处，原来按「需要人工指定 id」报错；
- 有些文档只进了新库、没进旧索引 → 原样报「定位不到」。

但这两件事在**已有显式 `gold_points` 映射**时都已经人工确认过，不应该再算错误，
否则每加一批题都要手工改校验器阈值。
"""

import json
import tempfile
import unittest
from pathlib import Path


def _write(items) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(items, handle, ensure_ascii=False)
    handle.close()
    return Path(handle.name)


def _item(**overrides):
    base = {
        "qid": "eval-001",
        "question": "Slurm 用什么管理计算资源？",
        "query_en": "What does Slurm use to manage compute resources?",
        "query_en_status": "edited",
        "type": "fact",
        "topic": "acronym_model",
        "difficulty": "normal",
        "lang": "zh",
        "gold_spans": ["这段文字在语料里根本不存在"],
        "gold_chunks": [],
        "reference_answer": "分区、节点、作业。",
        "note": "",
    }
    base.update(overrides)
    return base


class ValidateGoldPointsToleranceTest(unittest.TestCase):
    def setUp(self):
        from src import eval_dataset

        self.module = eval_dataset

    def test_missing_span_is_error_without_gold_points(self):
        path = _write([_item()])
        errors, warnings = self.module.validate(path, verbose=False)
        self.assertTrue(any("定位不到" in error for error in errors), errors)
        self.assertFalse(any("只进了新库" in warning for warning in warnings))

    def test_missing_span_is_warning_when_gold_points_exist(self):
        path = _write([_item(gold_points={"parent_ids": ["par_" + "a" * 32]})])
        errors, warnings = self.module.validate(path, verbose=False)
        self.assertFalse(any("定位不到" in error for error in errors), errors)
        self.assertTrue(any("只进了新库" in warning for warning in warnings), warnings)

    def test_negative_still_requires_note(self):
        path = _write([_item(type="negative", topic="no_answer", gold_spans=[],
                             reference_answer="", note="")])
        errors, _warnings = self.module.validate(path, verbose=False)
        self.assertTrue(any("查证方式" in error for error in errors), errors)


class ThresholdTest(unittest.TestCase):
    """span 命中多处的口径：有 gold_points 时降级为警告。"""

    def test_threshold_constant_is_documented(self):
        from src import eval_dataset

        # 阈值本身不变（≥5 篇重复才需要人看），变的是「已有映射时是否算错误」
        self.assertEqual(eval_dataset.SPAN_HIT_ERROR_THRESHOLD, 5)


if __name__ == "__main__":
    unittest.main()
