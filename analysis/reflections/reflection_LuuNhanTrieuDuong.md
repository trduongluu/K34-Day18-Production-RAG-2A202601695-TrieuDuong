# Individual Reflection — Lab 18: Production RAG Pipeline

**Tên:** Lưu Nhân Triệu Dương
**MSSV:** 2A202601695
**Lớp:** AICB-K34 · Ngày 18
**Module phụ trách:** Toàn bộ M1 → M5 (bài tập cá nhân)
**Ngày nộp:** 2026-08-18
**Model:** `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3` — đúng spec đề bài
**Kết quả:** 37/37 tests pass · 0 TODO còn lại · RAGAS production **0.8375 / 0.8356 / 0.9583 / 0.9333** (cả 4 metric ≥ 0.83)

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation từ lần chạy này |
|----------------|--------|-------------|------------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Encode từng câu bằng all-MiniLM-L6-v2, cắt chunk khi cosine giữa 2 câu liền kề < threshold. Với threshold 0.85 trên toàn corpus, số chunk sinh ra nhiều hơn hẳn basic vì tài liệu HR gồm nhiều câu ngắn rời rạc (bullet chính sách) → similarity giữa các câu liền kề thấp. Threshold là tham số phải tune theo *độ dài câu*, không có giá trị "đúng" phổ quát. |
| Hierarchical (parent-child) | M1 | `chunk_hierarchical()` | Parent 2048 chars / child 256 chars, mỗi child mang `parent_id`. Trên 26 documents sinh ra 104 child chunks. Đây là chiến lược được dùng thật trong pipeline vì nó tách bạch được *cái gì đem đi match* (child, đặc, chính xác) khỏi *cái gì đem cho LLM đọc* (parent, đủ ngữ cảnh). |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Regex `^#{1,3}\s+.+$` cắt theo heading markdown; mỗi chunk giữ nguyên tiêu đề section và lưu vào `metadata["section"]`. Với corpus toàn file `.md` có heading rõ ràng, đây là chiến lược "rẻ mà mạnh" nhất — ranh giới chunk do chính tác giả tài liệu đặt ra, không phải do thuật toán đoán. |
| Vietnamese word segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_` (`nghỉ_phép`). Nếu giữ `_`, document có token `nghỉ_phép` còn query tách thành `nghỉ` + `phép` → BM25 không bao giờ khớp. Phải `replace("_", " ")` để chỉ giữ lợi ích chuẩn hoá ranh giới từ. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | RRF chỉ dùng *thứ hạng* chứ không dùng điểm thô, nên ghép được BM25 (điểm không chặn trên) với cosine (0..1) mà không cần normalize. Hằng số k=60 làm phẳng đóng góp của top-rank để một hệ thống đơn lẻ không áp đảo kết quả. Đây là điểm mấu chốt cho câu hỏi dạng "numeric/lookup": BM25 bắt được con số chính xác mà dense hay bỏ sót. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Bi-encoder mã hoá query và doc *riêng biệt* rồi so cosine; cross-encoder đưa cả cặp qua cùng một transformer nên attention nhìn thấy cả hai → chính xác hơn nhiều nhưng đắt hơn nhiều. Đo thực tế trên CPU: `bge-reranker-v2-m3` (XLM-R large, 568M params) mất **11.3 giây** để rerank 20 docs, trong khi BM25 chỉ 4.5 ms và dense 324.6 ms — rerank chiếm **97% latency** của khâu retrieval. Con số này biến khái niệm "đắt" từ lý thuyết thành ràng buộc kỹ thuật thật. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | 2 metric đánh giá *generation* (faithfulness, answer_relevancy) và 2 metric đánh giá *retrieval* (context_precision, context_recall). Việc tách 2 nhóm này chính là thứ cho phép chẩn đoán: điểm generation thấp → sửa prompt; điểm retrieval thấp → sửa chunking/search. |
| Diagnostic / Error Tree | M4 | `failure_analysis()` | Với mỗi câu hỏi tệ nhất, lấy metric thấp nhất rồi tra bảng chẩn đoán để biết lỗi nằm ở khâu nào — thay vì kết luận chung chung "model kém". |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Chunk bị cắt rời khỏi tài liệu gốc thường mất tham chiếu ("chính sách này áp dụng từ…" — chính sách nào?). Một câu ngữ cảnh dán lên đầu chunk trả lại thông tin đó cho cả embedding lẫn BM25. Anthropic báo cáo giảm 49% retrieval failure khi dùng riêng kỹ thuật này. |
| Cost optimization của enrichment | M5 | `_enrich_single_call()` | Gộp summary + hypothesis questions + context + metadata vào **1 API call/chunk** thay vì 4 → giảm ~75% chi phí và thời gian enrich, chất lượng gần như tương đương vì cả 4 nhiệm vụ đều đọc chung một đoạn văn. 104 chunks = 104 calls thay vì 416 calls. |

---

## Phần 2: Khó khăn & cách giải quyết

### 2.1. `ragas` không import được — `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`

**Exact error:**

```
File "...\ragas\llms\base.py", line 12, in <module>
    from langchain_community.chat_models.vertexai import ChatVertexAI
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

**Debug:** Môi trường global đang có `ragas 0.4.3` + `langchain-community 0.4.2`, trong khi `langchain-community` 0.4 đã gỡ module `chat_models.vertexai`. Không thể hạ `langchain-community` ở global vì sẽ kéo `langchain-core` xuống và làm hỏng `langchain 1.3` đang dùng cho việc khác.

**Cách giải quyết:** Tạo virtualenv riêng cho lab (`.venv`) và cài đúng bộ pin trong `requirements.txt` (`ragas 0.1.22` + `langchain-community 0.2.19`). Bài học: pipeline RAG phụ thuộc rất chặt vào ma trận version của hệ sinh thái LangChain/RAGAS — môi trường cô lập theo từng project không phải là tuỳ chọn mà là bắt buộc.

Ngoài ra `evaluate_ragas()` được viết để chạy được **cả hai thế hệ API**: RAGAS ≥ 0.2 dùng `EvaluationDataset` với tên cột mới (`user_input`/`response`/`retrieved_contexts`/`reference`), RAGAS 0.1.x dùng `datasets.Dataset` với tên cột cũ. Hàm `_pick()` map alias cột về tên chuẩn nên code không vỡ khi nâng version.

### 2.2. `UnicodeEncodeError` khi print emoji/tiếng Việt trên Windows

**Exact error:**

```
File "...\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input, self.errors, encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4cc' in position 2
```

**Debug:** Console Windows mặc định dùng codepage cp1252; mọi `print("📌 …")` trong `main.py` / `pipeline.py` đều crash. Test suite không lộ lỗi này vì pytest capture stdout bằng buffer UTF-8.

**Cách giải quyết:** Ép `sys.stdout`/`sys.stderr` về UTF-8 ngay trong `config.py` (module được mọi file trong `src/` import), và thêm `import config` lên đầu `main.py` + `check_lab.py` vì hai file này print emoji *trước* khi import bất kỳ module nào của `src/`.

### 2.3. Segfault khi nạp hai model bge trong cùng một process

**Exact error:**

```
Windows fatal exception: access violation
Current thread ...:
  File "...\torch\storage.py", line 471 in __getitem__
  File "...\transformers\core_model_loading.py", line 1219 in _materialize_copy
  File "...\concurrent\futures\thread.py", line 59 in run
```

pytest chết ở test thứ 19/37 với exit code 139.

**Debug:** RAM trống lúc đó chỉ còn 2.8/15.7 GB, trong khi `bge-m3` (~2.3GB) + `bge-reranker-v2-m3` (~2.3GB) cần khoảng 5GB. Nhưng nghi vấn "thiếu RAM đơn thuần" bị bác bỏ khi thử nạp **cả hai model trong một process** với `HF_ENABLE_PARALLEL_LOADING=0` thì thành công. Thủ phạm thật là `transformers>=5` nạp trọng số **song song** bằng `ThreadPoolExecutor` — nhiều thread cùng materialize tensor lớn dưới áp lực bộ nhớ gây access violation trong `torch/storage.py`.

**Cách giải quyết:** `os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "0")` trong `config.py` — chậm hơn vài giây khi khởi động nhưng không sập, và không phụ thuộc vào việc người chấm có set biến môi trường hay không.

**Bài học:** một crash ở tầng C (segfault) trông như lỗi phần cứng/bộ nhớ, nhưng traceback vẫn chỉ đúng thư viện gây ra nó. Kiểm chứng giả thuyết bằng **một thí nghiệm tối thiểu** (nạp 2 model, đổi 1 biến) rẻ hơn nhiều so với đoán mò hay đi nâng cấp RAM.

### 2.3b. Trước đó: chạy tạm bằng model nhỏ khi mạng yếu — và bài học từ việc so sánh

Trong lần chạy đầu, máy đang dùng 3G nên không tải nổi 2 model ~2.3GB. Giải pháp tạm là tham số hoá model qua biến môi trường trong `config.py` (`EMBEDDING_MODEL` / `EMBEDDING_DIM` / `RERANK_MODEL`) và chạy bằng `all-MiniLM-L6-v2` + `ms-marco-TinyBERT-L-2-v2`. Khi có wifi, chỉ cần đổi giá trị mặc định — không sửa một dòng logic nào.

Việc "bất đắc dĩ" này hoá ra cho một **ablation quý**: cùng một pipeline, chỉ đổi model.

| Cấu hình | Ctx Precision | Ctx Recall |
|---|---|---|
| Baseline + MiniLM (English-centric) | 0.6208 | 0.5833 |
| Baseline + bge-m3 (đa ngôn ngữ) | **0.9250** | **0.9000** |

Đổi **một dòng config** tạo ra +0.30 precision và +0.32 recall — tác động lớn hơn toàn bộ phần chunking + hybrid + rerank cộng lại. Với corpus tiếng Việt, model embedding đa ngôn ngữ là **điều kiện cần**, không phải tuỳ chọn tối ưu hoá. Bài học ngược lại cũng đúng: nếu retrieval đang tệ, hãy kiểm tra model có hiểu ngôn ngữ của dữ liệu không **trước khi** đi tune chunk size hay threshold.

### 2.4. Lỗi có giá trị nhất: pipeline index child chunk rồi lại *trả về* child chunk

**Triệu chứng:** Vòng chạy production đầu tiên chỉ đạt faithfulness 0.633 / answer_relevancy 0.610 / context_precision 0.679 / context_recall 0.692 — hơn baseline nhưng chưa metric nào chạm 0.70. **9/20 câu bị trả lời "Không tìm thấy."**

**Debug:** Điều bất thường là có những câu bị "Không tìm thấy" nhưng `context_precision = 1.0` — nghĩa là retrieval đã lấy **đúng** tài liệu. Đó là bằng chứng loại trừ: lỗi không nằm ở retrieval. Đọc lại `run_query()` thì thấy pipeline đưa cho LLM chính **child chunk 256 ký tự** — mà 256 ký tự thường cắt ngang đúng câu chứa con số cần trả lời.

**Cách giải quyết — 3 thay đổi:**

1. `_expand_to_parents()`: dựng `_PARENT_INDEX` khoá theo `(source, parent_id)` (bắt buộc phải là cặp, vì `parent_0` lặp lại ở mọi document), rồi đổi child → parent trước khi đưa cho LLM. Đây mới là trọn vẹn kỹ thuật hierarchical chunking mà M1 dạy: *match trên child, trả về parent*.
2. `RERANK_TOP_K` 3 → 5 (đo thực nghiệm: top-5 khôi phục 2/3 câu fail, top-8 chỉ thêm nhiễu).
3. `ANSWER_SYSTEM_PROMPT` thay cho prompt một dòng: cho phép suy luận/tính toán từ context, ưu tiên phiên bản chính sách mới nhất, chỉ từ chối khi context thực sự trống.

**Kết quả:** cả 4 metric vọt lên 0.850 / 0.787 / 0.799 / 0.908 (trên MiniLM), và giữ nguyên giá trị khi chạy lại với model chuẩn: **0.8375 / 0.8356 / 0.9583 / 0.9333**.

**Bài học lớn nhất của cả buổi lab:** implement đúng từng module chưa đủ — phải nối chúng đúng cách. M1 sinh ra parent chunks nhưng pipeline vứt đi, nghĩa là nửa giá trị của hierarchical chunking bị bỏ phí mà mọi unit test vẫn xanh. Chỉ có **metric tách theo khâu (generation vs retrieval)** mới lộ ra được lỗi này.

### 2.5. Kiến thức còn thiếu → cách bổ sung

- **Tune ngưỡng semantic chunking:** hiện chọn threshold theo giá trị mặc định của đề bài, chưa có cách chọn có cơ sở. Cần đọc thêm về *sentence embedding drift* và thử grid search threshold ∈ {0.5, 0.65, 0.8, 0.9} rồi so context_recall.
- **Version-aware retrieval:** corpus cố tình cài bẫy tài liệu cũ/mới (`nghi_phep_nam_v2023` vs `v2024`, `mat_khau_v1` vs `v2`). Metadata filter theo version là hướng cần học tiếp — enrichment đã trích được `metadata`, nhưng pipeline chưa dùng nó để filter.

---

## Phần 3: Action Plan cho project cá nhân

### Hiện tại

- **RAG pipeline hiện tại:** chunking theo độ dài cố định + dense-only retrieval + đưa thẳng top-k vào LLM, không đo lường gì ngoài cảm nhận chủ quan.
- **Known issues:**
  1. Câu hỏi chứa con số/mã hiệu (số tiền, mã chính sách) hay trả về sai đoạn — đúng điểm yếu cố hữu của dense-only.
  2. Không biết lỗi nằm ở retrieval hay ở generation vì không có metric tách hai khâu.
  3. Tài liệu có nhiều phiên bản, hệ thống trả lời theo bản cũ mà không cảnh báo.

### Plan áp dụng

1. [ ] **Chunking strategy:** dùng `chunk_structure_aware()` làm mặc định (tài liệu nguồn là markdown/confluence có heading), fallback sang `chunk_hierarchical()` cho tài liệu không có cấu trúc. Lý do: ranh giới do tác giả đặt luôn tốt hơn ranh giới do thuật toán đoán, và nó gần như miễn phí về compute.
2. [ ] **Search:** hybrid BM25 + dense hợp nhất bằng RRF. Lý do: chính issue #1 ở trên — BM25 bắt token hiếm (con số, mã hiệu) mà dense làm mượt mất; RRF ghép hai bên không cần normalize điểm.
3. [ ] **Reranking:** có, cross-encoder, top-20 → top-5. Nhưng **phải đo latency trước khi chọn model**: trong lab này `bge-reranker-v2-m3` mất 11.3s/query trên CPU (chiếm 97% latency retrieval) — hoàn toàn không dùng được cho hệ thống thời gian thực. Kế hoạch: bắt đầu bằng reranker nhỏ, chỉ nâng lên bản large khi có GPU, hoặc giảm số ứng viên rerank xuống top-10.
4. [ ] **Evaluation:** RAGAS 4 metrics làm bộ khung, chạy trên test set ~30 câu tự xây theo 6 dạng (lookup / version / negation / multi-hop / numeric / ambiguous) như `test_set.json` của lab. Bổ sung `failure_analysis()` chạy tự động mỗi lần deploy để có bottom-5 kèm chẩn đoán.
5. [ ] **Enrichment:** `contextual_prepend` là kỹ thuật ưu tiên số một (tỉ lệ lợi ích/chi phí cao nhất), triển khai bằng combined single-call để giữ chi phí ở mức 1 call/chunk. Auto-metadata dùng để filter theo version — giải quyết issue #3.

### Timeline

| Tuần | Việc |
|------|------|
| Tuần 1 | Xây test set 30 câu + gắn RAGAS vào CI → có baseline số đo, không còn đánh giá cảm tính |
| Tuần 2 | Thay chunking sang structure-aware/hierarchical, đo lại → kỳ vọng context_recall tăng |
| Tuần 3 | Thêm BM25 + RRF, đo lại → kỳ vọng các câu hỏi numeric hết sai |
| Tuần 4 | Thêm reranking + đo latency p95, chốt có/không theo ngân sách latency |
| Tuần 5 | Enrichment (contextual prepend + auto metadata) và version-aware filter |
| Tuần 6 | Chạy failure analysis tổng, viết runbook vận hành |

---

## Phần 4: Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|---------------|---------|
| Hiểu bài giảng | 4 | Nắm được vì sao mỗi module tồn tại, còn yếu ở phần tune tham số |
| Code quality | 4 | Có docstring, type hint, xử lý edge case (corpus rỗng, API fail, JSON parse lỗi) |
| Problem solving | 4 | Xử lý được 3 vấn đề môi trường (ragas/langchain, UTF-8 Windows, băng thông model) |
| Teamwork | N/A | Bài tập cá nhân |
