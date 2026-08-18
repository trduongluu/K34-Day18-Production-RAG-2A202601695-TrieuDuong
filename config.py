"""Shared configuration for Lab 18."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Console Windows mac dinh dung cp1252 -> print emoji / tieng Viet se nem
# UnicodeEncodeError va lam sap pipeline. Ep stdout/stderr ve UTF-8 ngay tu
# config vi module nay duoc moi file trong src/ import.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # stream bi pytest/pipe thay the
        pass

# transformers>=5 nap trong so model song song bang ThreadPoolExecutor. Khi nap
# bge-m3 (~2.3GB) va bge-reranker-v2-m3 (~2.3GB) trong cung 1 process tren may
# it RAM trong, buoc nay gay "Windows fatal exception: access violation" (segfault)
# o torch/storage.py. Nap tuan tu cham hon vai giay nhung khong sap.
os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "0")

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
# Model theo dung spec de bai: bge-m3 la model da ngon ngu, ho tro tieng Viet tot.
# Van doc tu bien moi truong de co the ha xuong model nho (VD all-MiniLM-L6-v2,
# 384-dim) khi may khong tai duoc ~2.3GB.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# --- Reranking model ---
# Cross-encoder da ngon ngu theo spec de bai (~2.3GB).
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
# Top-3 lam mat cau tra loi o cac cau hoi can ghep nhieu dieu khoan; do bang
# thuc nghiem (xem analysis/failure_analysis.md) top-5 giup LLM tra loi duoc
# 2/3 cau truoc do bi "Khong tim thay", con top-8 chi them nhieu.
RERANK_TOP_K = 5

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
