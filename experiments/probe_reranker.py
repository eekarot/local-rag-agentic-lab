"""B08 探针：在 **不装 torch** 的前提下，用 onnxruntime + tokenizers 跑 cross-encoder 重排。

为什么要探针（而不是直接照文档写 `bge-reranker-v2-m3`）：

1. 本机没有 torch / sentence-transformers，`BAAI/bge-reranker-v2-m3` 只发布了 PyTorch 权重，
   硬盘上跑不动；装 CPU 版 torch 又要多 ~1–2 GB RSS，与 D39 的 16 GB 预算正面冲突。
2. 已经在的 `onnxruntime`（1.31.1）+ `tokenizers`（0.23.3）能不能顶住，只有实测才知道。
3. D31 的验收要的是 **P50/P95 延迟 + CPU 峰值 RSS**，这两个数不测就得猜。

候选（都是 ONNX 有货的多语 cross-encoder）：

- `jinaai/jina-reranker-v2-base-multilingual` int8（266 MB，中英都覆盖）
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` fp32（449 MB，mMARCO 中文段训练，118M 参数）

用法：

    python experiments/probe_reranker.py --candidates 50 --repeat 3
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = {
    "jina-int8": ("jinaai/jina-reranker-v2-base-multilingual", "onnx/model_int8.onnx"),
    "mmarco-fp32": ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "onnx/model.onnx"),
}


def rss_mb() -> float:
    """当前进程 RSS（MB）。Windows 用 GetProcessMemoryInfo，POSIX 退回 /proc。"""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.WorkingSetSize / 2**20
    return int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**20


class OnnxReranker:
    """cross-encoder 重排：`(query, passage)` 拼成一对送进模型，取 logits 当分数。"""

    def __init__(self, model_id: str, onnx_name: str, max_length: int = 512,
                 cache_dir: str = "storage/models", threads: int | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.max_length = max_length
        self.tokenizer = Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json", cache_dir=cache_dir))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        started = time.perf_counter()
        self.session = ort.InferenceSession(
            hf_hub_download(model_id, onnx_name, cache_dir=cache_dir),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.load_ms = (time.perf_counter() - started) * 1000
        self.input_names = {i.name for i in self.session.get_inputs()}

    def score(self, query: str, passages: list[str], batch_size: int = 8) -> np.ndarray:
        scores = []
        for start in range(0, len(passages), batch_size):
            batch = passages[start:start + batch_size]
            encoded = self.tokenizer.encode_batch([(query, text) for text in batch])
            feed = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            }
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
            logits = self.session.run(None, feed)[0]
            logits = np.asarray(logits, dtype=np.float32).reshape(len(batch), -1)
            scores.extend(logits[:, 0].tolist())
        return np.array(scores, dtype=np.float32)


def sample_passages(limit: int) -> list[str]:
    """从 Document Store 抽真实父块，长度按线上分布（截断到 max_length 由 tokenizer 负责）。"""
    from src.rag.store.document_store import DocumentStore

    store = DocumentStore("storage/documents.sqlite3")
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT content FROM parents WHERE length(content) > 800 ORDER BY parent_id LIMIT ?", (limit,)
        ).fetchall()
    store.close()
    return [row[0] for row in rows]


def percentiles(samples: list[float]) -> dict:
    ordered = sorted(samples)
    pick = lambda q: ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
    return {"p50": round(pick(0.50), 1), "p95": round(pick(0.95), 1),
            "mean": round(statistics.fmean(samples), 1), "n": len(samples)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--models", default="jina-int8,mmarco-fp32")
    parser.add_argument("--out", default="experiments/probes/b08_reranker.json")
    args = parser.parse_args()

    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__} | providers {ort.get_available_providers()}")
    passages = sample_passages(args.candidates)
    print(f"父块样本 {len(passages)} 条，字符长度 p50={statistics.median(len(p) for p in passages):.0f}")

    query_zh = "在 BCM 里怎么把镜像的内核版本换成新的？"
    query_en = "How do you set a new kernel version for an image in BCM?"

    report = {"onnxruntime": ort.__version__, "max_length": args.max_length,
              "batch": args.batch, "n_passages": len(passages), "models": {}}

    for label in args.models.split(","):
        model_id, onnx_name = MODELS[label]
        base_rss = rss_mb()
        reranker = OnnxReranker(model_id, onnx_name, max_length=args.max_length)
        peak_rss = rss_mb()
        print(f"\n=== {label} ({model_id} / {onnx_name}) ===")
        print(f"  加载 {reranker.load_ms:.0f} ms；RSS {base_rss:.0f} → {peak_rss:.0f} MB（+{peak_rss-base_rss:.0f}）")

        entry = {"model_id": model_id, "onnx": onnx_name,
                 "load_ms": round(reranker.load_ms, 1),
                 "rss_base_mb": round(base_rss, 1), "rss_loaded_mb": round(peak_rss, 1),
                 "rss_delta_mb": round(peak_rss - base_rss, 1)}

        for name, query in (("zh", query_zh), ("en", query_en)):
            latencies = []
            scores = None
            for _ in range(args.repeat):
                started = time.perf_counter()
                scores = reranker.score(query, passages, batch_size=args.batch)
                latencies.append((time.perf_counter() - started) * 1000)
            entry[f"latency_{name}_ms"] = percentiles(latencies)
            entry[f"score_{name}"] = {"min": round(float(scores.min()), 3),
                                      "max": round(float(scores.max()), 3),
                                      "std": round(float(scores.std()), 3)}
            print(f"  [{name}] 50 候选 {entry[f'latency_{name}_ms']}；分数 {entry[f'score_{name}']}")

        report["models"][label] = entry
        del reranker

    report["rss_final_mb"] = round(rss_mb(), 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
