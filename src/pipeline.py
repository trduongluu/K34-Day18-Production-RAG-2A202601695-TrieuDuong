from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

# (source, parent_id) → text của parent chunk. Được build ở bước chunking và dùng
# lại ở run_query để làm "small-to-big retrieval": match trên child 256 chars cho
# chính xác, nhưng đưa cho LLM cả parent 2048 chars cho đủ ngữ cảnh.
_PARENT_INDEX: dict[tuple[str, str], str] = {}

ANSWER_SYSTEM_PROMPT = (
    "Bạn là trợ lý tra cứu chính sách nội bộ. Quy tắc:\n"
    "1. Chỉ dùng thông tin có trong context, không dùng kiến thức ngoài.\n"
    "2. Được phép suy luận và tính toán số học từ dữ kiện trong context "
    "(ví dụ: tính 85% của mức lương, cộng ngày phép theo thâm niên).\n"
    "3. Nếu context có nhiều phiên bản của cùng một chính sách, trả lời theo "
    "phiên bản mới nhất/hiện hành và nói rõ phiên bản cũ đã bị thay thế.\n"
    "4. Trả lời trực tiếp vào câu hỏi, nêu rõ con số và đơn vị, ngắn gọn 1-3 câu.\n"
    "5. Chỉ trả lời 'Không tìm thấy.' khi context thực sự không chứa dữ kiện nào "
    "liên quan — đừng từ chối chỉ vì phải ghép thông tin từ nhiều đoạn."
)


def _expand_to_parents(contexts: list[str], metadatas: list[dict]) -> list[str]:
    """Small-to-big: đổi child chunk sang parent chunk tương ứng (nếu tìm được).

    Child 256 chars thường bị cắt ngang câu chứa đáp án — đó chính là lý do
    pipeline trả 'Không tìm thấy' dù retrieval đã lấy đúng tài liệu. Trả về
    parent giúp LLM nhìn thấy trọn vẹn điều khoản, đồng thời khử trùng lặp khi
    nhiều child cùng trỏ về một parent.
    """
    expanded: list[str] = []
    for text, metadata in zip(contexts, metadatas):
        key = (metadata.get("source", ""), metadata.get("parent_id", ""))
        parent_text = _PARENT_INDEX.get(key)
        chosen = parent_text if parent_text else text
        if chosen not in expanded:
            expanded.append(chosen)
    return expanded


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    _PARENT_INDEX.clear()
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        source = doc["metadata"].get("source", "")
        # parent_id chỉ duy nhất TRONG một document ("parent_0" lặp ở mọi file),
        # nên khoá của parent index phải là cặp (source, parent_id).
        for parent in parents:
            _PARENT_INDEX[(source, parent.metadata["parent_id"])] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    if reranked:
        contexts = _expand_to_parents([r.text for r in reranked], [r.metadata for r in reranked])
    else:
        contexts = _expand_to_parents([r.text for r in results[:3]], [r.metadata for r in results[:3]])

    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0, messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
