# Task 5–7: Retrieval

## Cách tích hợp

Task 4 phải triển khai `embed_texts()` và `get_collection()`, index các chunk
theo `MODULE_CONTRACTS.md`, và dùng collection có cosine distance. Task 5 dùng
chung embedding của Task 4 với `embed_texts([query], is_query=True)`; Task 6 đọc chính các chunk đã index bằng
`get_collection().get()`, không chunk lại tài liệu.

```python
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf

query = "Các phương thức xét tuyển năm 2026 là gì?"
dense = semantic_search(query, top_k=10)
sparse = lexical_search(query, top_k=10)
hybrid = rerank_rrf([dense, sparse], top_k=5)

# Task 9 dùng điểm dense gốc để quyết định fallback, không dùng điểm RRF.
best_dense_score = dense[0]["score"] if dense else 0.0
```

Task 5 trả `1 - cosine_distance`, có thể âm. Task 6 dùng BM25Okapi, chuẩn hóa
Unicode NFC và chữ thường, tách dấu câu, giữ dấu tiếng Việt. Đây là tokenizer
đơn giản, chưa tách từ ghép tiếng Việt. Chỉ chunk chứa ít nhất một token câu
hỏi mới được trả về; giữ cả điểm bằng 0 hoặc âm vì corpus nhỏ có thể có IDF
không dương. Điểm BM25 không phải xác suất hoặc cosine similarity.

Task 6 đọc snapshot Chroma mỗi lần tìm kiếm và cache index theo token corpus.
Nội dung thay đổi dù số chunk giữ nguyên cũng làm index được xây lại; metadata
luôn lấy từ snapshot hiện tại. Cách này phù hợp corpus nhỏ của bài lab, nhưng
vẫn có chi phí đọc/tokenize toàn bộ corpus mỗi query. Với corpus lớn cần cơ chế
version/invalidation và nạp theo batch.

Task 7 cộng `1 / (k + rank)` theo ID, rank bắt đầu từ 1. Mỗi ID chỉ đóng góp
một lần trong mỗi danh sách, giữ vị trí gốc của lần xuất hiện đầu tiên. Khi
bằng điểm, giữ thứ tự gặp đầu tiên. Không thay đổi score của dense đầu vào.

Query rỗng, `top_k <= 0` hoặc corpus rỗng trả `[]`. Lỗi provider được truyền
lên caller để pipeline xử lý, không giả thành kết quả tìm kiếm rỗng.

## Kiểm thử offline

```powershell
python -m pytest tests/test_retrieval.py tests/test_rrf.py -q
python -m pytest tests/test_contracts.py -q -k "semantic_search or lexical_search or rrf or public_function_signatures"
```

Các test dùng collection/embedding giả và BM25 thật, không gọi API hoặc tải
model. Có thể gán `src.task6_lexical_search.CORPUS` bằng danh sách chunk để thử
BM25 offline; gán lại `None` để đọc Chroma, gán `[]` để thử corpus rỗng.

Chạy thật sau khi Task 4 hoàn thành:

```powershell
python -m src.task4_chunking_indexing
python -m src.task5_semantic_search
python -m src.task6_lexical_search
```

Các lệnh search trên dùng `test query` trong entry point; để thử câu hỏi tuyển
sinh thực tế, dùng đoạn Python tích hợp ở trên.

## Cấu hình đã kiểm tra trên corpus hiện có

Dense search đã chạy với `EMBEDDING_PROVIDER=gemini`,
`EMBEDDING_MODEL=gemini-embedding-001`, vector 768 chiều và index 517 chunks.
Điền `GEMINI_API_KEY` trong `.env` local; không commit key hoặc `chroma_db/`.
Người clone repo cần chạy Task 4 để tạo index tương ứng trước khi chạy thật.
Không dùng BAAI/bge-m3 (1024 chiều) để query index Gemini 768 chiều.

Kiểm tra ngày 25/09/2026: Task 5–7 đạt 29 test; toàn bộ suite đạt 49 test.
Test tự động dùng dữ liệu giả cho retrieval; kiểm tra API thật được ghi riêng
trong báo cáo cá nhân của Bình. Kết quả này không thay thế evaluation A/B.
