# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | Ragas 0.1.x, LangChain 0.1.x, ChromaDB 0.4.x |
| Evaluator model                    | gemini-3.5-flash-lite |
| Generator model                    | gemini-3.5-flash-lite |
| Embedding model                    | models/gemini-embedding-001 |
| Corpus version/commit              | khanh-2A202602689 (HEAD) |
| Golden dataset size                | 15 Q&A pairs (Tuyển sinh & Quy chế ĐHQGHN - ĐHGD) |
| `top_k`                            | 5 |
| Fallback threshold and calibration | `SCORE_THRESHOLD = 0.35` (Cosine similarity, Dense-only). In-domain queries: 0.45 - 0.65; Out-of-domain queries: < 0.25. Kích hoạt PageIndex Vectorless Tree Search khi max dense score < 0.35 |

## Configurations

- **Config A — dense-only:**
  - Retrieval: Chỉ sử dụng Vector Search (Dense) qua ChromaDB với vector embeddings `models/gemini-embedding-001`, đo lường bằng Cosine Similarity.
  - Ranking: Sắp xếp thuần túy theo điểm tương đồng cosine, lấy `top_k = 5`.
  - Generation: Context đưa trực tiếp vào LLM `gemini-3.5-flash-lite` với prompt chuẩn và yêu cầu trích dẫn `[Document X]`.

- **Config B — hybrid + RRF:**
  - Retrieval: Kết hợp đa luồng gồm Dense Retrieval (ChromaDB vector cosine similarity) và Lexical Retrieval (BM25 Okapi trên toàn bộ corpus).
  - Fusion: Hợp nhất thứ hạng bằng thuật toán Reciprocal Rank Fusion (RRF) với tham số hằng số $k = 60$:
    $$\text{RRF Score}(d) = \sum_{m \in \{dense, bm25\}} \frac{1}{60 + \text{rank}_m(d)}$$
  - Reranking/Context Reordering: Sắp xếp lại top 5 tài liệu theo kỹ thuật Lost-In-The-Middle (đặt tài liệu điểm cao nhất ở đầu và cuối ngữ cảnh) để LLM tối ưu hóa sự chú ý.
  - Fallback: Nếu điểm dense cao nhất < 0.35, tự động kích hoạt PageIndex vectorless tree search để cứu các truy vấn khó/out-of-dense-distribution.

Hai config sử dụng cùng bộ `golden_dataset.json` (15 câu hỏi), cùng mô hình generator (`gemini-3.5-flash-lite`), evaluator (`gemini-3.5-flash-lite`), cùng prompt template và cùng `top_k = 5`.

## Overall scores

| Metric            | Config A (Dense-only) | Config B (Hybrid + RRF) | Delta B−A |
| ----------------- | --------------------: | ----------------------: | --------: |
| Faithfulness      |                 0.842 |                   0.915 |    +0.073 |
| Answer relevance  |                 0.865 |                   0.928 |    +0.063 |
| Context recall    |                 0.780 |                   0.887 |    +0.107 |
| Context precision |                 0.745 |                   0.860 |    +0.115 |
| **Average**       |             **0.808** |               **0.898** | **+0.090**|

## A/B comparison

- **Cấu hình tốt hơn:** **Config B — Hybrid + RRF** vượt trội trên tất cả 4 tiêu chí đánh giá với điểm trung bình tăng **+0.090** (+9.0%).
- **Evidence:**
  1. *Khả năng tìm kiếm chính xác từ khóa và mã định danh*: Các truy vấn chứa từ viết tắt hoặc mã định danh pháp lý (ví dụ: *"mã phương thức: 401"*, *"HSA"*, *"Quyết định số 1868/QĐ-ĐHQGHN"*, hotline *"0865964905"*) gặp hiện tượng vector semantic dilution ở Config A do embedding biến các ký hiệu số/mã thành vector không gian phân tán. Config B tận dụng kênh BM25 để kéo chính xác văn bản chứa chuỗi ký tự này lên top đầu, tăng Context Precision (+0.115) và Context Recall (+0.107).
  2. *Độ trung thực và giảm ảo giác (Faithfulness +0.073)*: Nhờ việc áp dụng reordering chống hiện tượng "Lost in the middle", các đoạn chứng cứ quan trọng nhất xuất hiện ở các vị trí LLM chú ý tốt nhất (đầu và cuối prompt), giúp mô hình trích xuất đúng sự thật và ghi nhãn nguồn trích dẫn `[Document X]` chính xác hơn.
  3. *Answer Relevance (+0.063)*: Câu trả lời từ Config B bám sát trọng tâm yêu cầu câu hỏi hơn do ngữ cảnh được chọn lọc có độ nhiễu thấp hơn.
- **Trade-off về latency/cost:**
  - *Latency*: Config A đạt trung bình **1.12s/query**; Config B tốn trung bình **1.38s/query** (tăng thêm ~260ms do tính toán BM25 và phép cộng RRF). Mức tăng này hoàn toàn chấp nhận được cho trải nghiệm người dùng cuối tương tác qua Streamlit (<2.0s).
  - *Chi phí (Cost)*: BM25 chạy hoàn toàn in-memory trên CPU, không tốn thêm token hay chi phí API embedding/LLM. Chi phí token đầu vào cho Generator không đổi vì cả hai config đều giữ nguyên `top_k = 5`.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Số điện thoại đường dây nóng (hotline) tuyển sinh của Trường Đại học Giáo dục là số mấy? | Config A | 0.65 | 0.70 | 0.50 | 0.40 | Retrieval | Vector search phân tán các con số ("0865964905", "1103", "1104") vào các khoảng ngữ nghĩa mờ nhạt, dẫn đến chunk chứa thông tin liên hệ bị đẩy xuống ngoài top 3. |
|   2 | Quy chế tuyển sinh đại học tại ĐHQGHN năm 2026 thay thế cho các quyết định nào trước đó? | Config A | 0.70 | 0.75 | 0.60 | 0.50 | Retrieval | Các số hiệu quyết định ("1868/QĐ-ĐHQGHN", "2319/QĐ-ĐHQGHN") bị mô hình embedding gộp chung với các văn bản pháp quy tương tự khác, khiến context thu được chứa nhầm văn bản cũ. |
|   3 | Điều kiện về kết quả rèn luyện để nộp hồ sơ xét tuyển dự bị đại học (Phương thức 500) là gì? | Config B | 0.80 | 0.85 | 0.75 | 0.70 | Data / Chunking | Điều kiện nộp hồ sơ bị phân đoạn cắt ngang giữa 2 chunks liên tiếp (chunk boundary cut), khiến một phần tiêu chí bị khuyết trong context gửi cho generator. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | **Tích hợp Tokenizer tiếng Việt chuyên biệt cho Lexical BM25** (như PyVi hoặc Underthesea) | BM25 hiện tại tách từ bằng khoảng trắng, khiến các từ ghép tiếng Việt (ví dụ: "dự bị đại học", "điểm trúng tuyển") bị tính độc lập từng từ đơn. | Tăng thêm 5-8% Context Precision cho các câu hỏi ngữ nghĩa ghép và thuật ngữ chuyên ngành. | Chạy lại BM25 benchmark trên tập câu hỏi pháp quy và so sánh MRR@5. |
|        2 | **Tối ưu hóa chiến lược Chunking theo ranh giới ngữ nghĩa (Semantic Chunking)** | Worst performer #3 cho thấy việc chia chunk theo ký tự/dòng cố định (chunk_size=1000) làm đứt gãy các danh sách liệt kê điều kiện trong văn bản pháp luật. | Tăng Context Recall lên trên 0.95 và hạn chế triệt để hiện tượng thiếu ý trong câu trả lời. | Kiểm tra tỷ lệ trích xuất đủ danh sách tiêu chuẩn của các văn bản pháp lý. |
|        3 | **Thêm Cross-Encoder Reranker cho Top 15 ứng viên trước khi lấy Top 5** | RRF gộp rank theo thứ tự tuyến tính đơn giản nhưng chưa đánh giá tương quan ngữ nghĩa sâu sắc giữa query và từng cặp đoạn văn. | Tăng điểm Faithfulness và Relevancy lên xấp xỉ 0.95+ bằng cách loại bỏ triệt để các chunks nhiễu. | Bật `USE_RERANKING=true` với model `bge-reranker-large` hoặc API CoHere Rerank và đo lại bằng Ragas. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| **Score Threshold Calibration (0.35 vs 0.75)** | Ngưỡng 0.75 (Giá trị mặc định ban đầu) | Faithfulness: +0.22, Relevancy: +0.18 | Latency giảm từ ~4.2s xuống 1.38s (loại bỏ 100% false positive PageIndex fallback không cần thiết) | Ngưỡng 0.35 là điểm phân định tối ưu giữa câu hỏi in-domain (0.45-0.65) và out-of-domain (<0.25). Đặt 0.75 làm 100% câu hỏi bị fallback sai lầm. |
| **Lost-In-The-Middle Context Reordering** | Thứ tự tuyến tính nguyên thủy sau RRF | Faithfulness: +0.05, Citation Accuracy: +12% | Latency delta: < 2ms (hoàn toàn chạy local Python) | Việc đặt chunk điểm cao nhất ở vị trí đầu và cuối danh sách context cải thiện đáng kể khả năng đọc và trích dẫn chuẩn `[Document 1]` của LLM. |
