import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
# 允许用环境变量把数据/存储指到别处：便于小批量试跑而不污染正式索引
DATA_DIR = Path(os.getenv("RAG_DATA_DIR") or ROOT_DIR / "data")
STORAGE_DIR = Path(os.getenv("RAG_STORAGE_DIR") or ROOT_DIR / "storage")
INDEX_PATH = STORAGE_DIR / "index.json"
# 向量单独存 float32 二进制：JSON 里存十进制文本会膨胀 3.3 倍，且必须整文件解析
EMBEDDINGS_PATH = STORAGE_DIR / "embeddings.npy"
QUESTIONS_PATH = ROOT_DIR / "experiments" / "questions.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:9b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
# 换 embedding 模型必须重建索引（python -m src.build_index），否则维度不符会直接报错。
# 2026-11-02 从 nomic-embed-text 换成 bge-m3：中文问英文跨语言检索的 gold 排名
# p50 158→22、p90 15394→145（全量 81k 实测，见 experiments/probes/full-rank-*.json）。

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
# 单次 embedding 请求的文本条数。实测 bge-m3 在 224~320 条/批都能跑，但约 30% 的请求
# 会偶发 400（runner 重启），build_index 里已改成「400 先重试、再二分」；
# batch=256 时约 24 ms/条，全量 81,027 条约 40 分钟。
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))
TOP_K = int(os.getenv("TOP_K", "4"))

# BM25（Okapi）参数：k1 控制词频饱和，b 控制文档长度归一化强度
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# 混合检索：RRF 常数 + 加权融合时的 BM25 权重（hybrid_score 模式）
RRF_K = int(os.getenv("RRF_K", "60"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "4"))
