# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Bài tập cá nhân (ASSIGNMENT.md quy định cá nhân — không chia nhóm)
**Thành viên:** Lưu Nhân Triệu Dương (MSSV 2A202601695) → thực hiện toàn bộ M1 · M2 · M3 · M4 · M5
**Ngày chạy:** 2026-08-18 · **Test set:** `test_set.json` (20 câu, 6 dạng)
**Model:** `BAAI/bge-m3` (1024-dim) + `BAAI/bge-reranker-v2-m3` — đúng spec đề bài

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8183 | **0.8375** | +0.0192 |
| Answer Relevancy | 0.6705 | **0.8356** | **+0.1651** |
| Context Precision | 0.9250 | **0.9583** | +0.0333 |
| Context Recall | 0.9000 | **0.9333** | +0.0333 |

- **Naive baseline** = paragraph chunking (57 chunks) + dense-only top-3, không rerank, không enrichment.
- **Production** = hierarchical chunking (104 child / 26 parent) + enrichment 1 call/chunk + hybrid BM25/dense + RRF + cross-encoder rerank top-5 + small-to-big parent expansion.
- File gốc: `reports/naive_baseline_report.json`, `reports/ragas_report.json`.

**Cả 4 metric đều ≥ 0.83.** Điểm cần chú ý: baseline với bge-m3 đã rất mạnh ở phần *retrieval* (context_precision 0.925, context_recall 0.900), nên phần cải thiện của production dồn gần hết vào **answer_relevancy (+0.165)** — tức là chất lượng *câu trả lời*, không phải chất lượng *tìm kiếm*.

---

## Ablation 1: Model embedding quyết định bao nhiêu?

Lab được chạy hai lần với hai bộ model khác nhau trên **cùng một pipeline, cùng một code**:

| Cấu hình | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|---|---|---|---|---|
| Baseline + `all-MiniLM-L6-v2` (384-dim, English-centric) | 0.5548 | 0.4238 | 0.6208 | 0.5833 |
| Baseline + `bge-m3` (1024-dim, đa ngôn ngữ) | **0.8183** | **0.6705** | **0.9250** | **0.9000** |
| Production + MiniLM + `ms-marco-TinyBERT-L-2-v2` | 0.8500 | 0.7872 | 0.7992 | 0.9083 |
| Production + `bge-m3` + `bge-reranker-v2-m3` | 0.8375 | **0.8356** | **0.9583** | **0.9333** |

(Kết quả lần chạy MiniLM được giữ lại ở `reports/ragas_report_minilm.json` và `reports/naive_baseline_report_minilm.json`.)

**Kết luận rút ra:**

1. **Chỉ đổi embedding model, baseline nhảy +0.30 context_precision và +0.32 context_recall.** Đây là thay đổi *một dòng config* nhưng tác động lớn hơn toàn bộ phần kỹ thuật chunking/hybrid/rerank cộng lại. Với corpus tiếng Việt, dùng model English-centric là tự bắn vào chân — không kỹ thuật retrieval nào bù lại được việc embedding không hiểu ngôn ngữ.
2. **Nhưng kỹ thuật pipeline vẫn quan trọng, chỉ là ở chỗ khác.** Với MiniLM yếu, pipeline production phải "cứu" phần retrieval (recall 0.583 → 0.908). Với bge-m3 mạnh, retrieval đã tốt sẵn nên pipeline chuyển sang cải thiện phần generation (answer_relevancy 0.671 → 0.836).
3. **Faithfulness của production (0.8375) hơi thấp hơn lần chạy MiniLM (0.8500).** Không phải nghịch lý: với context tốt hơn, LLM trả lời được nhiều câu khó hơn thay vì nói "Không tìm thấy" — mà càng suy luận nhiều bước thì càng dễ bị RAGAS chấm là "claim không suy ra trực tiếp từ context" (xem failure #3 và #5 bên dưới, cả hai đều trả lời **đúng** nhưng faithfulness thấp). Từ chối trả lời là cách rẻ tiền để giữ faithfulness cao.

---

## Ablation 2: Ghép module đúng cách quan trọng ngang implement đúng

Vòng production **đầu tiên** (chạy trên MiniLM) chỉ đạt 0.633 / 0.610 / 0.679 / 0.692, với **9/20 câu bị LLM trả lời "Không tìm thấy."** — kể cả những câu mà `context_precision = 1.0`, tức **retrieval đã lấy đúng tài liệu**. Đây là bằng chứng loại trừ trực tiếp: lỗi nằm ở *sau* retrieval.

| Cấu hình (đều trên MiniLM) | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|---|---|---|---|---|
| v1 — rerank top-3, trả về **child chunk**, prompt gốc | 0.6333 | 0.6095 | 0.6792 | 0.6917 |
| v2 — rerank top-5, **parent expansion**, prompt có quy tắc | **0.8500** | **0.7872** | **0.7992** | **0.9083** |

(v1 lưu ở `reports/ragas_report_v1_no_parent_expansion.json`.)

**Ba nguyên nhân và ba thay đổi:**

1. **Child chunk 256 ký tự cắt ngang câu chứa đáp án.** → `_expand_to_parents()` trong `src/pipeline.py`: match trên child cho chính xác, nhưng đưa cho LLM cả **parent 2048 ký tự**. Đây mới là trọn vẹn kỹ thuật hierarchical chunking mà M1 dạy — v1 index child rồi lại *trả về* child, tức bỏ phí nửa sau của kỹ thuật.
2. **`RERANK_TOP_K = 3` quá hẹp** với câu hỏi cần ghép nhiều điều khoản. → nâng lên 5 (đo thực nghiệm: top-5 khôi phục 2/3 câu fail, top-8 chỉ thêm nhiễu).
3. **Prompt gốc đẩy model về phía từ chối** (`"Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"`) và cấm luôn việc tính toán. → `ANSWER_SYSTEM_PROMPT`: cho phép suy luận/tính toán từ context, ưu tiên phiên bản chính sách mới nhất, chỉ từ chối khi context thực sự trống.

**Điểm đáng sợ nhất:** trong suốt thời gian pipeline sai như vậy, **toàn bộ 37 unit test vẫn xanh**. Unit test xác nhận từng module đúng; chỉ metric tách theo khâu (generation vs retrieval) mới lộ ra rằng các module bị *nối* sai.

---

## Bottom-5 Failures (lần chạy chuẩn với bge)

### #1 — Nhân viên Senior 9 năm thâm niên: bao nhiêu ngày phép và lương khoảng nào?

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên (9÷3) = 18 ngày phép. Lương Senior (P3–P4): 20–35 triệu VNĐ/tháng.
- **Got:** "…được 18 ngày phép năm (15 + 3). Về lương, không có thông tin cụ thể trong context để xác định mức lương của nhân viên Senior."
- **Worst metric:** answer_relevancy = 0.000 (faithfulness 0.667, context_precision 1.0, context_recall **0.5**)
- **Error Tree:**
  - Output sai? → **Đúng một nửa.** Phần ngày phép chính xác tuyệt đối, kể cả phép chia 9÷3.
  - Context đúng? → **Thiếu.** `context_recall = 0.5`: context có tài liệu nghỉ phép nhưng **không có** bảng lương Senior. `context_precision = 1.0` cho thấy những gì lấy về đều liên quan — vấn đề là *thiếu*, không phải *nhiễu*.
  - Query OK? → Query hỏi **hai chủ đề** trong một câu; retrieval bị kéo về phía "nghỉ phép" mạnh hơn phía "lương".
  - → **Root cause: single-vector retrieval không phủ được truy vấn đa chủ đề.** Một embedding duy nhất phải đại diện cho hai chủ đề tách biệt, kết quả là nó nằm giữa và nghiêng về chủ đề có tín hiệu mạnh hơn.
- **Suggested fix:** **Query decomposition** — tách câu hỏi đa chủ đề thành sub-query ("ngày phép của Senior 9 năm thâm niên", "khoảng lương Senior"), retrieve riêng rồi hợp nhất context. Đây là failure duy nhất trong bottom-5 mà pipeline **thực sự trả lời thiếu**; 4 câu còn lại đều trả lời đúng.
- **Lưu ý về metric:** answer_relevancy bị chấm 0.000 vì câu trả lời chứa mệnh đề từ chối — RAGAS phạt rất nặng phần "không thể xác định", kể cả khi nửa còn lại của câu trả lời hoàn toàn đúng.

### #2 — Tạm ứng 15 triệu, thanh toán sau 20 ngày, bị phạt bao nhiêu?

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; 2%/tháng trên 15.000.000 = 300.000 VNĐ/tháng → pro-rata ≈ 50.000 VNĐ cho 5 ngày.
- **Got:** "…quá hạn 5 ngày. Phí phạt 2%/tháng… Tính cho 5 ngày (1/6 tháng): 15.000.000 × 2% × (5/30) = **50.000 VNĐ**." → **đúng hoàn toàn, kể cả bước pro-rata.**
- **Worst metric:** faithfulness = 0.250 (answer_relevancy 0.817, context_precision 1.0, context_recall 0.667)
- **Error Tree:**
  - Output sai? → **Không.** Đây là câu ở lần chạy MiniLM bị sai (dừng ở 300.000 VNĐ, quên pro-rata) và giờ đã đúng.
  - Context đúng? → `context_precision = 1.0`, `context_recall = 0.667`.
  - Query OK? → Có.
  - → **Root cause: giới hạn của chính metric, không phải của pipeline.** Faithfulness kiểm tra từng claim có *suy ra trực tiếp* từ context không. Các bước tính trung gian ("5 ngày = 1/6 tháng", "15.000.000 × 2% × 5/30") không xuất hiện nguyên văn ở đâu cả nên bị chấm là không faithful.
- **Suggested fix:** Với các câu numeric, tách phần *trích dẫn điều khoản* khỏi phần *tính toán* trong output (nêu điều khoản gốc trước, rồi mới trình bày phép tính) để RAGAS quy chiếu được claim về context. Nếu chấm bằng người thì câu này phải được tính là đúng.

### #3 — Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất 20.000.000 VNĐ/tháng → lương thử việc = 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** "Lương thử việc của nhân viên Junior (mức cao nhất 20.000.000 VNĐ) là **17.000.000 VNĐ** (85% của 20.000.000 VNĐ)." → **đúng.**
- **Worst metric:** faithfulness = 0.333 (answer_relevancy 0.806, **context_precision 1.0, context_recall 1.0**)
- **Error Tree:**
  - Output sai? → **Không.** Ở lần chạy MiniLM câu này trả "Không tìm thấy"; với bge-m3 nó ghép được hai tài liệu.
  - Context đúng? → **Hoàn hảo** — cả precision lẫn recall = 1.0. Cả `bang_luong_2024.md` (bậc Junior) và `thu_viec.md` (85%) đều có mặt.
  - Query OK? → Có.
  - → **Root cause: multi-hop reasoning bị faithfulness phạt.** Kết quả 17.000.000 không nằm trong bất kỳ tài liệu nào — nó là *tích* của hai dữ kiện ở hai tài liệu khác nhau.
- **Suggested fix:** Giống #2 — yêu cầu model nêu rõ từng dữ kiện nguồn trước khi kết luận. Đây là câu minh hoạ rõ nhất rằng **faithfulness thấp không đồng nghĩa với sai**.

### #4 — Nghỉ phép không lương 20 ngày cần ai phê duyệt?

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16–30 ngày cần Giám đốc điều hành (CEO). Lưu ý thêm: nghỉ trên 14 ngày không lương thì nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** "Nghỉ phép không lương 20 ngày cần phê duyệt của **Giám đốc điều hành (CEO)**." → **đúng phần được hỏi**, thiếu lưu ý về bảo hiểm.
- **Worst metric:** faithfulness = 0.500 (answer_relevancy 0.825, context_precision 1.0, context_recall 1.0)
- **Error Tree:**
  - Output sai? → **Không**, nhưng **thiếu**.
  - Context đúng? → **Đầy đủ** (recall 1.0) — chi tiết bảo hiểm có trong context nhưng không được đưa vào câu trả lời.
  - Query OK? → Có.
  - → **Root cause: prompt yêu cầu "ngắn gọn 1-3 câu"** nên model cắt bỏ thông tin phụ trợ mà ground truth lại coi là một phần của đáp án.
- **Suggested fix:** Nới ràng buộc độ dài và thêm chỉ dẫn "nếu context có điều kiện/ngoại lệ kèm theo, hãy nêu luôn". Đây là đánh đổi có ý thức giữa *ngắn gọn* và *đầy đủ* — không có lựa chọn nào đúng tuyệt đối, phụ thuộc vào yêu cầu sản phẩm.

### #5 — Tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng, hoàn trả bao nhiêu?

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết làm việc tối thiểu 1 năm; nghỉ sau 8 tháng là trước hạn → hoàn trả 100% chi phí = 25.000.000 VNĐ.
- **Got:** "Phải hoàn trả **100% chi phí** đào tạo, tức là **25.000.000 VNĐ**." → **đúng số.**
- **Worst metric:** faithfulness = 0.500 (answer_relevancy 0.903, context_precision 1.0, context_recall 1.0)
- **Error Tree:**
  - Output sai? → **Không.**
  - Context đúng? → **Đầy đủ** (precision và recall đều 1.0).
  - Query OK? → Có.
  - → **Root cause: model bỏ qua bước lập luận trung gian.** Nó nhảy thẳng tới "100%" mà không nói rõ *vì sao* (8 tháng < cam kết 12 tháng), nên claim "100%" không được neo vào điều kiện trong context.
- **Suggested fix:** Buộc model nêu điều kiện trước rồi mới ra kết luận ("cam kết là 12 tháng, nghỉ ở tháng thứ 8 → trước hạn → hoàn 100%"). Vừa tăng faithfulness vừa giúp người dùng kiểm chứng được câu trả lời.

**Nhận xét chung về bottom-5:** chỉ có **#1 là failure thật** (thiếu context về khoảng lương). Bốn câu còn lại đều trả lời **đúng nội dung** nhưng bị faithfulness phạt vì lập luận nhiều bước hoặc trả lời quá gọn. Đây là giới hạn quan trọng của RAGAS cần nêu khi bảo vệ: nó chấm *mức độ neo vào context*, không chấm *tính đúng đắn*.

---

## Case Study (presentation)

**Question chọn phân tích:** *"Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?"* — vì cùng một câu hỏi này đi qua **cả ba trạng thái** của pipeline và cho ba kết quả khác nhau.

| Cấu hình | Câu trả lời | context_recall | faithfulness |
|---|---|---|---|
| Production v1 (MiniLM, child chunk, top-3) | "Không tìm thấy." | 1.0 | 0.000 |
| Production v2 (MiniLM, parent, top-5) | "Không tìm thấy." | 1.0 | 0.000 |
| Production (bge-m3 + bge-reranker) | "17.000.000 VNĐ (85% của 20.000.000)" ✅ | 1.0 | 0.333 |

**Error Tree walkthrough:**

1. **Output đúng?** → Không, ở cả hai vòng MiniLM. Model từ chối trả lời.
2. **Context đúng?** → **Có, ngay từ đầu** — `context_recall = 1.0` ở cả ba lần. Cả bảng lương lẫn quy định 85% đều đã nằm trong context. **Đây là bước loại trừ quan trọng nhất: retrieval chưa bao giờ là vấn đề của câu hỏi này.**
3. **Query rewrite OK?** → Query rõ ràng, không cần rewrite.
4. **Fix ở bước nào?** → Ở **generation**. Câu hỏi đòi multi-hop (tra bậc lương ở tài liệu A × 85% ở tài liệu B). Chỉ khi context đủ sạch và đủ mạch lạc (bge-m3 + rerank tốt hơn) thì model mới chịu ghép hai tài liệu thay vì bỏ cuộc.

**Bài học rút ra:** nếu chỉ nhìn "câu trả lời sai" mà không có metric tách theo khâu, phản xạ tự nhiên là đi tune retrieval — trong khi `context_recall = 1.0` đã nói rõ retrieval hoàn hảo ngay từ đầu. Đây chính là giá trị của Diagnostic Tree: **nó ngăn ta sửa nhầm chỗ.**

**Nếu có thêm 1 giờ, sẽ optimize:**

1. **Prompt "nêu dữ kiện nguồn trước, kết luận sau"** (~15 phút) — sửa được faithfulness của cả 4 câu #2–#5 trong bottom-5, vì cả 4 đều đúng mà bị phạt do thiếu bước lập luận trung gian. Đây là fix có tỉ lệ lợi ích/công sức cao nhất.
2. **Query decomposition** cho câu hỏi đa chủ đề (~25 phút) — sửa failure #1, failure thật duy nhất.
3. **Version-aware filtering** bằng metadata từ M5 (~20 phút) — không còn nằm trong bottom-5 nữa (bge-m3 đã xử lý tốt các câu version), nhưng vẫn đáng làm cho corpus lớn hơn.

---

## Latency Breakdown

Đo trên Windows 11, **CPU-only** (không GPU), Qdrant local trong Docker.

### Chi phí một lần dựng pipeline (index time)

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| M1 — Load 26 docs + hierarchical chunking | 0.0s | Thuần CPU, không gọi model |
| M5 — Enrichment 104 chunks | 276.7s | **1 API call/chunk**; nếu dùng 4 kỹ thuật riêng lẻ sẽ là 416 calls (~4× thời gian và chi phí) |
| M2 — Index BM25 + dense (104 chunks, bge-m3) | 61.3s | Gần như toàn bộ là encode embedding trên CPU |
| M3 — Khởi tạo reranker | 0.0s | Lazy load; model thật được nạp ở query đầu tiên |

### Chi phí mỗi truy vấn (query time) — so sánh hai bộ model

| Bước | `bge-m3` + `bge-reranker-v2-m3` (spec) | `MiniLM` + `TinyBERT-L-2` | Tỉ lệ |
|------|--------------------------------------|---------------------------|-------|
| BM25 search (top-20) | **4.5 ms** | 1.7 ms | 2.6× |
| Dense search (top-20) | **324.6 ms** | 52.6 ms | 6.2× |
| RRF fusion | **0.063 ms** | 0.074 ms | ~1× |
| Cross-encoder rerank (20 → 5) | **11 259.7 ms** (min 10 794 / max 12 553) | 51.2 ms | **220×** |
| **Tổng retrieval + rerank** | **~11.6 s** | ~105 ms | 110× |
| LLM generation (gpt-4o-mini) | ~1–3 s | ~1–3 s | — |
| RAGAS eval 20 câu × 4 metrics | ~50 s | ~50 s | — |

Số liệu thô: `reports/latency_breakdown.json` (bge) và `reports/latency_breakdown_minilm.json`.

**Nhận xét:**

1. **RRF gần như miễn phí** (0.063 ms) và không phụ thuộc model — chi phí hợp nhất hai hệ thống retrieval nhỏ hơn chính hai hệ thống đó hàng nghìn lần. Không có lý do gì để *không* dùng hybrid search.
2. **BM25 rẻ hơn dense ~72 lần** (4.5 ms vs 324.6 ms) vì nó chỉ tra bảng tần suất từ, còn dense phải chạy forward pass của encoder 1024-dim để nhúng query. BM25 cũng là thành phần *duy nhất* trong pipeline không cần GPU và không đắt lên khi model to ra.
3. **Reranking là cái giá thật của chất lượng: 11.3 giây cho 20 tài liệu.** `bge-reranker-v2-m3` là XLM-RoBERTa-large (~568M tham số) và phải chạy forward pass cho **từng cặp** (query, document) — 20 forward pass của một model large trên CPU. So với TinyBERT-L-2 (2 layer) thì chậm hơn **220 lần**.
4. **Kết luận vận hành:** với cấu hình đúng spec chạy trên CPU, reranking chiếm **97% latency** của toàn khâu retrieval và làm người dùng phải chờ hơn 11 giây — **không dùng được cho production thời gian thực.** Ba hướng xử lý: (a) chạy reranker trên GPU, (b) giảm số ứng viên đưa vào rerank (top-20 → top-10), hoặc (c) dùng reranker nhỏ hơn và chấp nhận mất một phần precision. Chính vì vậy pattern chuẩn **retrieval rẻ lọc → rerank đắt trên ít ứng viên** không phải là tối ưu hoá cho vui mà là điều kiện để hệ thống chạy được.

---

## Ghi chú môi trường

Ba vấn đề môi trường đã phải xử lý để chạy được đúng spec (chi tiết ở `analysis/reflections/reflection_TrieuDuong.md`):

| Vấn đề | Cách xử lý |
|---|---|
| `ragas` không import được (`langchain_community.chat_models.vertexai` đã bị gỡ ở langchain-community 0.4) | Tạo `.venv` riêng với đúng bộ pin `requirements.txt` (`ragas 0.1.22` + `langchain-community 0.2.19`) |
| `UnicodeEncodeError` cp1252 khi print emoji/tiếng Việt trên console Windows | Ép `sys.stdout`/`sys.stderr` về UTF-8 trong `config.py` |
| Segfault (`access violation`) khi nạp bge-m3 + bge-reranker cùng process | `HF_ENABLE_PARALLEL_LOADING=0` trong `config.py` — `transformers>=5` nạp trọng số song song bằng ThreadPoolExecutor và crash khi RAM trống thấp |
