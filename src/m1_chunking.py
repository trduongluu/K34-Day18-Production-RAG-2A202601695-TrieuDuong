from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Lazy encoder cho semantic chunking ──────────────────

_SEMANTIC_ENCODER = None
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\n")
_HEADER_RE = re.compile(r"^#{1,3}\s+.+$")


def _get_semantic_encoder():
    """Load encoder 1 lần duy nhất cho cả process (model load rất tốn thời gian)."""
    global _SEMANTIC_ENCODER
    if _SEMANTIC_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_ENCODER


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Thuật toán: encode từng câu bằng all-MiniLM-L6-v2, rồi đi tuần tự so cosine
    similarity giữa câu i-1 và câu i:
      - sim >= threshold → hai câu vẫn cùng chủ đề, gộp vào chunk hiện tại;
      - sim <  threshold → gặp "semantic boundary", mở chunk mới.
    Nhờ vậy ranh giới chunk rơi đúng chỗ đổi ý thay vì cắt cứng theo độ dài.
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0],
                      metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    embeddings = _get_semantic_encoder().encode(sentences)

    def cosine_sim(a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append([sentences[i]])       # đổi chủ đề → chunk mới
        else:
            groups[-1].append(sentences[i])     # cùng chủ đề → gộp tiếp

    return [
        Chunk(text=" ".join(group),
              metadata={**metadata, "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _make_parent(text: str, metadata: dict, index: int) -> Chunk:
    """Tạo 1 parent chunk với id ổn định dạng `parent_<n>`."""
    pid = f"parent_{index}"
    return Chunk(
        text=text.strip(),
        metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                  "chunk_index": index, "strategy": "hierarchical"},
    )


def _split_by_size(text: str, max_size: int) -> list[str]:
    """Cắt text thành các mảnh <= max_size, ưu tiên ranh giới câu rồi mới tới từ."""
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n+", text) if u and u.strip()]
    pieces: list[str] = []
    current = ""

    for unit in units:
        # Câu đơn dài hơn cả max_size → buộc phải cắt theo từ để không vượt ngưỡng.
        while len(unit) > max_size:
            head = unit[:max_size]
            if " " in head:
                head, remainder = head.rsplit(" ", 1)
                unit = remainder + unit[max_size:]
            else:
                unit = unit[max_size:]
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.append(head.strip())

        if current and len(current) + len(unit) + 1 > max_size:
            pieces.append(current.strip())
            current = unit
        else:
            current = f"{current} {unit}" if current else unit

    if current.strip():
        pieces.append(current.strip())
    return [p for p in pieces if p]


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Ý tưởng: index child nhỏ (256 chars) để embedding "đặc" và match chính xác;
    nhưng khi đưa cho LLM thì có thể trả về parent lớn (2048 chars) để không mất
    ngữ cảnh xung quanh. Mỗi child giữ `parent_id` trỏ ngược lên parent của nó.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    # 1) Gộp paragraphs thành parent chunks (mỗi parent <= parent_size chars)
    parents: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > parent_size:
            parents.append(_make_parent(current, metadata, len(parents)))
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        parents.append(_make_parent(current, metadata, len(parents)))

    # 2) Mỗi parent → cắt nhỏ thành children (mỗi child <= child_size chars)
    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        for j, piece in enumerate(_split_by_size(parent.text, child_size)):
            children.append(Chunk(
                text=piece,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid,
                          "child_index": j, "strategy": "hierarchical"},
                parent_id=pid,
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Ranh giới chunk chính là ranh giới heading (`#`, `##`, `###`) do tác giả tài
    liệu đặt ra, nên mỗi chunk là một section hoàn chỉnh và luôn mang theo tiêu
    đề của nó — tiêu đề đó là tín hiệu rất mạnh cho cả BM25 lẫn dense retrieval.
    """
    metadata = metadata or {}
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        body = buffer.strip()
        full = f"{current_header}\n\n{body}".strip() if current_header else body
        if full:
            chunks.append(Chunk(
                text=full,
                metadata={**metadata, "section": current_header.lstrip("# ").strip(),
                          "strategy": "structure", "chunk_index": len(chunks)},
            ))
        buffer = ""

    for part in parts:
        if part and _HEADER_RE.match(part.strip()) and part.strip().startswith("#"):
            flush()
            current_header = part.strip()
        else:
            buffer += part or ""
    flush()

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
