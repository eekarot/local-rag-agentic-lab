""".. deprecated:: 2.0
   旧链（单文件索引 `index.json` + `embeddings.npy` + 内存 BM25），仅作历史对比保留。

   新链已迁到 Qdrant + SQLite Document Store，**本模块不参与新链**，不要在新代码里引用：

       建库：python scripts/ingest.py --dir data
       问答：python -m src.ask "问题"
       说明：README「快速开始」、设计文档 §17
"""
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

from .config import (
    INDEX_PATH, STORAGE_DIR, EMBEDDINGS_PATH, EMBED_BATCH_SIZE,
    CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_EMBED_MODEL, BM25_K1, BM25_B,
)
from .bm25 import build_bm25
from .ollama_client import embed_texts
from .text_utils import load_documents, split_text
from .tokenizer import tokenizer_name

# 断点续建：向量先写进 embeddings.part.npy，进度写进 .build_progress.json，
# 全部成功后才替换成正式文件名。大库建到一半中断不会白跑。
PROGRESS_PATH = STORAGE_DIR / ".build_progress.json"
PART_PATH = STORAGE_DIR / "embeddings.part.npy"
PRINT_EVERY = 10  # 每 N 批打一次进度（落盘是每批都做，否则可能丢一批已算好的向量）

# Ollama 偶发 400：bge-m3 在 batch=256 时实测约 30% 的请求返回 400
# （runner 重启后 tokenize 代理暂时连不上），同一批再发一次就成功。
# 所以先原地重试，重试完还 400 再认为是「批太大」去二分。
EMBED_400_RETRIES = int(os.getenv("EMBED_400_RETRIES", "3"))
EMBED_400_BACKOFF = float(os.getenv("EMBED_400_BACKOFF", "1.0"))  # 秒，按次数线性退避

# ollama_client 抛的原文是「Ollama API 返回 HTTP 400：…」，requests 自己是「400 Client Error」。
# 旧实现只匹配后者，导致二分分支永远进不去（实测踩到：bge-m3 的 400 直接被当作致命错误抛出）。
BATCH_400_MARKERS = ("HTTP 400", "400 Client Error")


def _is_400(exc):
    text = str(exc)
    return any(marker in text for marker in BATCH_400_MARKERS)


def _fingerprint(records):
    """指纹绑定「文档集合 + 切分参数 + 分词方案 + 向量模型」，防止续建时张冠李戴。"""
    h = hashlib.sha256()
    h.update(OLLAMA_EMBED_MODEL.encode("utf-8"))
    h.update(f"|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{tokenizer_name()}".encode("utf-8"))
    for record in records:
        h.update(record["id"].encode("utf-8"))
        h.update(f"#{len(record['text'])}".encode("utf-8"))
    return h.hexdigest()


def _embed_once(texts):
    """单次请求 + 400 原地重试。非 400 错误（服务不可用等）立刻抛出。"""
    for attempt in range(1, EMBED_400_RETRIES + 1):
        try:
            return embed_texts(texts)
        except RuntimeError as exc:
            if not _is_400(exc) or attempt == EMBED_400_RETRIES:
                raise
            wait = EMBED_400_BACKOFF * attempt
            print(
                f"  [第 {attempt} 次重试（最多 {EMBED_400_RETRIES - 1} 次）] {len(texts)} 条请求返回 400，"
                f"{wait:.1f}s 后重发",
                flush=True,
            )
            time.sleep(wait)


def _embed_batch(texts):
    """取一批向量。400 先原地重试；仍然 400 才二分（真正的批过大上限）。"""
    try:
        vectors = _embed_once(texts)
    except RuntimeError as exc:
        # 只有「批太大」这类 400 才值得拆分；服务不可用要立刻抛出，交给进度文件保住成果
        if len(texts) <= 1 or not _is_400(exc):
            raise
        mid = len(texts) // 2
        return _embed_batch(texts[:mid]) + _embed_batch(texts[mid:])
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding 返回数量 {len(vectors)} 与输入文本块数量 {len(texts)} 不一致。"
        )
    return vectors


def _read_progress(fingerprint, n_records):
    """读取可续建的进度；指纹或条数不匹配一律忽略，当作全新构建。"""
    if not (PROGRESS_PATH.exists() and PART_PATH.exists()):
        return None
    try:
        progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if progress.get("fingerprint") != fingerprint or progress.get("n_records") != n_records:
        return None
    return progress


def _write_progress(fingerprint, n_records, done, dim):
    payload = {
        "fingerprint": fingerprint,
        "n_records": n_records,
        "done": done,
        "dim": dim,
        "embedding_model": OLLAMA_EMBED_MODEL,
        "tokenizer": tokenizer_name(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, PROGRESS_PATH)  # 原子替换，避免读到半截 JSON


def _build_embeddings(records, fingerprint):
    """流式写向量矩阵，支持断点续建。返回 (矩阵形状, 是否续建)。"""
    n_records = len(records)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    matrix = None
    done = 0
    progress = _read_progress(fingerprint, n_records)
    if progress:
        matrix = np.load(PART_PATH, mmap_mode="r+")
        if matrix.shape[0] != n_records:
            matrix = None  # 尺寸对不上，重建
        else:
            done = int(progress.get("done", 0))
            print(
                f"发现未完成的建库进度：已完成 {done}/{n_records} 条向量，继续。",
                flush=True,
            )

    total_batches = max(1, (n_records - done + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE)
    started = time.perf_counter()
    batch_no = 0
    while done < n_records:
        batch = records[done:done + EMBED_BATCH_SIZE]
        vectors = np.asarray(_embed_batch([r["text"] for r in batch]), dtype=np.float32)
        if vectors.ndim != 2:
            raise RuntimeError("Embedding 返回结果不是二维数组，无法写入向量矩阵。")

        if matrix is None:
            matrix = np.lib.format.open_memmap(
                PART_PATH, mode="w+", dtype=np.float32,
                shape=(n_records, vectors.shape[1]),
            )
        elif vectors.shape[1] != matrix.shape[1]:
            raise RuntimeError(
                f"向量维度不一致：已有 {matrix.shape[1]} 维，本批 {vectors.shape[1]} 维。"
                "请检查是否中途更换了 Embedding 模型。"
            )

        matrix[done:done + len(batch)] = vectors
        done += len(batch)
        batch_no += 1

        # 每批都落盘：写一个几百字节的 JSON 很便宜，丢一批 256 条的向量则要重跑 4 秒
        matrix.flush()
        _write_progress(fingerprint, n_records, done, int(matrix.shape[1]))

        if batch_no % PRINT_EVERY == 0 or done >= n_records:
            elapsed = time.perf_counter() - started
            speed = done / elapsed if elapsed else 0.0
            eta = (n_records - done) / speed if speed else 0.0
            print(
                f"向量进度 {done}/{n_records} ({done / n_records:.1%}) | "
                f"{batch_no}/{total_batches} 批 | {elapsed:.0f}s 已用 | 预计剩余 {eta:.0f}s",
                flush=True,
            )

    if matrix is not None:
        matrix.flush()
        dim = int(matrix.shape[1])
        # Windows 下必须先显式关掉 memmap 句柄，否则 os.replace 会报 WinError 32
        handle = getattr(matrix, "_mmap", None)
        del matrix
        if handle is not None:
            handle.close()
        gc.collect()
    else:  # n_records == 0，前面已经拦住了，这里只是兜底
        dim = 0

    os.replace(PART_PATH, EMBEDDINGS_PATH)
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
    return dim


def main():
    print(
        "[deprecated] 这是旧链入口（index.json + embeddings.npy）。"
        "新链：python scripts/ingest.py --dir data 建库，python -m src.ask \"问题\" 问答",
        file=sys.stderr,
    )
    docs = load_documents()
    if not docs:
        raise SystemExit("data/ 中没有可读取的 md、txt 或 pdf 文件。")

    records = []
    for doc in docs:
        chunks = split_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"{doc['source']}::chunk-{i:04d}",
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk,
            })

    print(f"读取 {len(docs)} 份文档，切分得到 {len(records)} 个文本块。", flush=True)
    print(
        f"正在使用 Embedding 模型 {OLLAMA_EMBED_MODEL} 生成向量"
        f"（批大小 {EMBED_BATCH_SIZE}）……",
        flush=True,
    )

    fingerprint = _fingerprint(records)
    dim = _build_embeddings(records, fingerprint)
    print(f"向量已保存：{EMBEDDINGS_PATH}（{len(records)} × {dim}，float32）", flush=True)

    # BM25 倒排表：分词只在这里做一次，查询时只扫命中词项的 posting list。
    print(f"正在用 {tokenizer_name()} 建立 BM25 倒排索引……", flush=True)
    bm25 = build_bm25(records, BM25_K1, BM25_B)
    print(
        f"BM25 词项数={len(bm25['postings'])}，"
        f"平均文档长度={sum(bm25['doc_len']) / max(len(records), 1):.1f} 词",
        flush=True,
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": OLLAMA_EMBED_MODEL,
        "embedding_dim": dim,
        "n_records": len(records),
        "embeddings_file": EMBEDDINGS_PATH.name,
        "embeddings_dtype": "float32",
        "embeddings_normalized": False,  # 查询时在 retrieve 里做行归一化
        "tokenizer": tokenizer_name(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "bm25": bm25,
        "records": records,  # 注意：不再包含 embedding
    }
    INDEX_PATH.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    size_mb = INDEX_PATH.stat().st_size / 1048576
    print(f"索引已保存：{INDEX_PATH}（{size_mb:.1f} MB，不含向量）", flush=True)
    print("提醒：更换 Embedding 模型或分词方案后必须重建索引。", flush=True)


if __name__ == "__main__":
    main()
