from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words.

    underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép"). BM25 ở dưới tokenize bằng
    split(" ") nên nếu giữ "_" thì document có token "nghỉ_phép" còn query lại
    tách thành "nghỉ" + "phép" → không bao giờ khớp. Vì vậy phải replace("_", " "):
    ta chỉ giữ lại *lợi ích chuẩn hoá ranh giới từ* của underthesea, không giữ
    ký hiệu nối.
    """
    if not text:
        return text
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception as e:  # underthesea chưa cài / lỗi model → degrade gracefully
        print(f"  ⚠️  Vietnamese segmentation failed ({e}); dùng raw text.")
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk.get("text", "")).lower().split()
            for chunk in chunks
        ]
        # BM25Okapi vỡ nếu corpus rỗng → chỉ build khi có dữ liệu.
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_indices:
            if scores[i] <= 0:      # score 0 = không share token nào với query
                continue
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        if not chunks:
            return

        # Tạo lại collection từ đầu để lần chạy sau không lẫn vector cũ.
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True, batch_size=16)

        points = [
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={**chunks[i].get("metadata", {}), "text": texts[i]},
            )
            for i, vector in enumerate(vectors)
        ]

        # Upsert theo batch — payload quá lớn trong 1 request dễ timeout.
        for start in range(0, len(points), 128):
            self.client.upsert(collection, points[start:start + 128])

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # qdrant-client hiện đại dùng query_points(), KHÔNG phải search().
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(collection, query=query_vector, limit=top_k)
        except Exception as e:
            print(f"  ⚠️  Dense search failed: {e}")
            return []

        return [
            SearchResult(
                text=point.payload.get("text", ""),
                score=float(point.score),
                metadata=point.payload,
                method="dense",
            )
            for point in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank).

    RRF chỉ dùng *thứ hạng*, không dùng điểm số thô — nên nó ghép được BM25
    (điểm không chặn trên) với dense cosine (điểm 0..1) mà không cần normalize.
    Hằng số k=60 làm phẳng đóng góp của các hạng đầu, giúp 1 hệ thống đơn lẻ
    không áp đảo kết quả hợp nhất.
    """
    rrf_scores: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda e: e["score"], reverse=True)

    return [
        SearchResult(
            text=entry["result"].text,
            score=entry["score"],
            metadata=entry["result"].metadata,
            method="hybrid",
        )
        for entry in ranked[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
