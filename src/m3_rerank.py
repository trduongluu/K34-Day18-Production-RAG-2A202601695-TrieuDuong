from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K, RERANK_MODEL


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """Cross-encoder reranker.

    Khác với bi-encoder (dense retrieval) mã hoá query và document *riêng biệt*
    rồi so cosine, cross-encoder đưa cặp (query, document) qua cùng một
    transformer nên attention nhìn thấy cả hai — chính xác hơn nhiều, nhưng đắt
    hơn nhiều. Vì vậy pipeline chuẩn là: retrieval rẻ lấy top-20 → rerank đắt
    thu về top-3.
    """

    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding:
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]
        scores = model.predict(pairs)

        # model.predict trả scalar khi chỉ có 1 cặp → chuẩn hoá thành list.
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)

        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker
            self._model = Ranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        try:
            from flashrank import RerankRequest
            model = self._load_model()
            passages = [{"id": i, "text": d.get("text", "")} for i, d in enumerate(documents)]
            results = model.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:
            print(f"  ⚠️  Flashrank rerank failed: {e}")
            return []

        return [
            RerankResult(
                text=r["text"],
                original_score=float(documents[r["id"]].get("score", 0.0)),
                rerank_score=float(r["score"]),
                metadata=documents[r["id"]].get("metadata", {}),
                rank=i,
            )
            for i, r in enumerate(results[:top_k])
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
