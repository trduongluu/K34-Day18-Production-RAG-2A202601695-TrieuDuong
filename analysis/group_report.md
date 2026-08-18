# Group Report — Lab 18: Production RAG

**Nhóm:** Bài tập cá nhân — ASSIGNMENT.md quy định "Bài tập **cá nhân** — implement toàn bộ 5 modules", nên không có phân công nhóm. Template được giữ nguyên cấu trúc và điền theo góc nhìn một người thực hiện toàn bộ.
**Ngày:** 2026-08-18
**Model:** `BAAI/bge-m3` (1024-dim) + `BAAI/bge-reranker-v2-m3` — đúng spec đề bài

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Lưu Nhân Triệu Dương (2A202601695) | M1: Chunking | ☑ | 13/13 |
| Lưu Nhân Triệu Dương (2A202601695) | M2: Hybrid Search | ☑ | 5/5 |
| Lưu Nhân Triệu Dương (2A202601695) | M3: Reranking | ☑ | 5/5 |
| Lưu Nhân Triệu Dương (2A202601695) | M4: Evaluation | ☑ | 4/4 |
| Lưu Nhân Triệu Dương (2A202601695) | M5: Enrichment | ☑ | 10/10 |
| | **Tổng** | | **37/37** |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8183 | **0.8375** | +0.0192 |
| Answer Relevancy | 0.6705 | **0.8356** | +0.1651 |
| Context Precision | 0.9250 | **0.9583** | +0.0333 |
| Context Recall | 0.9000 | **0.9333** | +0.0333 |

Cả 4 metric đều ≥ 0.83. Nguồn: `reports/naive_baseline_report.json`, `reports/ragas_report.json`.

## Key Findings

1. **Biggest improvement — Answer Relevancy +0.165 (0.671 → 0.836).**
   Với bge-m3, baseline đã rất mạnh ở phần retrieval (precision 0.925, recall 0.900), nên phần production cải thiện được dồn gần hết vào chất lượng *câu trả lời*. Nói cách khác: khi retrieval đã tốt, giá trị của hybrid search + reranking + enrichment thể hiện ở chỗ context **sạch và mạch lạc hơn**, giúp LLM chịu ghép nhiều tài liệu thay vì bỏ cuộc.

2. **Model embedding là biến số lớn nhất — lớn hơn cả pipeline.**
   Chạy cùng một code với `all-MiniLM-L6-v2` (English-centric) thay vì `bge-m3`, baseline tụt từ 0.925 → 0.621 context_precision và 0.900 → 0.583 context_recall. Đổi *một dòng config* có tác động lớn hơn toàn bộ phần kỹ thuật chunking + hybrid + rerank cộng lại. Với corpus tiếng Việt, model đa ngôn ngữ là điều kiện cần, không phải tuỳ chọn.

3. **Biggest challenge — pipeline index child chunk rồi lại trả về child chunk.**
   Vòng production đầu chỉ đạt 0.633 / 0.610 / 0.679 / 0.692 với 9/20 câu bị trả "Không tìm thấy" — kể cả câu có `context_precision = 1.0`. Nguyên nhân: child chunk 256 ký tự cắt ngang câu chứa đáp án. Sửa bằng `_expand_to_parents()` (small-to-big), `RERANK_TOP_K` 3→5 và viết lại prompt. **Điểm đáng sợ: 37/37 unit test vẫn xanh suốt thời gian pipeline sai** — chỉ metric mới phát hiện được lỗi *nối module*.

4. **Surprise finding — reranking chiếm 97% latency của khâu retrieval.**
   `bge-reranker-v2-m3` (XLM-R large, 568M params) mất **11.3 giây** để rerank 20 tài liệu trên CPU, so với 4.5 ms của BM25 và 324.6 ms của dense search. RRF thì gần như miễn phí (0.063 ms). Với cấu hình đúng spec chạy CPU-only, hệ thống **không dùng được cho production thời gian thực** — cần GPU, hoặc giảm số ứng viên rerank, hoặc chấp nhận reranker nhỏ hơn.

5. **Faithfulness thấp ≠ trả lời sai.**
   Trong bottom-5, chỉ **1 câu là failure thật** (thiếu context về khoảng lương Senior). Bốn câu còn lại đều trả lời **đúng** nhưng bị faithfulness phạt vì lập luận nhiều bước (VD: 15.000.000 × 2% × 5/30 = 50.000 VNĐ — kết quả không nằm nguyên văn trong tài liệu nào). RAGAS chấm *mức độ neo vào context*, không chấm *tính đúng đắn* — đây là giới hạn phải nêu rõ khi báo cáo.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** 0.818/0.671/0.925/0.900 → 0.838/0.836/0.958/0.933; mạnh nhất là answer_relevancy (+0.165). Cả 4 metric ≥ 0.83.
2. **Biggest win — module nào, tại sao:** M1 hierarchical chunking, nhưng **chỉ khi dùng trọn vẹn cả hai nửa** (match trên child, trả về parent). Một hàm 12 dòng đã kéo cả 4 metric lên ~0.15–0.22 ở lần chạy MiniLM.
3. **Case study — Error Tree walkthrough:** câu *"Lương thử việc Junior cao nhất?"* — `context_recall = 1.0` ngay từ đầu ở cả 3 cấu hình, tức retrieval **chưa bao giờ** là vấn đề; lỗi nằm ở generation (multi-hop). Diagnostic Tree ngăn ta sửa nhầm chỗ.
4. **Next optimization nếu có thêm 1 giờ:** (a) prompt "nêu dữ kiện nguồn trước, kết luận sau" — sửa được 4/5 câu trong bottom-5, (b) query decomposition cho câu hỏi đa chủ đề, (c) đưa reranker lên GPU hoặc giảm ứng viên rerank để latency về mức dùng được.

## Bằng chứng & tái lập

```bash
docker compose up -d
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python main.py            # baseline + production + bảng so sánh
python check_lab.py       # 37/37 tests, 0 TODO
```

| File | Nội dung |
|---|---|
| `reports/ragas_report.json` | Kết quả production (bge-m3 + bge-reranker) — bản nộp chính |
| `reports/naive_baseline_report.json` | Baseline (bge-m3, dense-only) |
| `reports/ragas_report_minilm.json` | Ablation: cùng pipeline với model nhỏ |
| `reports/ragas_report_v1_no_parent_expansion.json` | Ablation: trước khi có parent expansion |
| `reports/latency_breakdown.json` | Latency từng bước (bge) |
| `reports/latency_breakdown_minilm.json` | Latency từng bước (model nhỏ) |
