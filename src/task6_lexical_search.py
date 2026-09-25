"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


import re
import unicodedata
from functools import lru_cache

from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import get_collection


# None: đọc snapshot từ Chroma. Gán list chunks để chạy offline/kiểm thử.
CORPUS: list[dict] | None = None


def tokenize(text: str) -> list[str]:
    """Chuẩn hóa Unicode, chữ thường, tách dấu câu và giữ dấu tiếng Việt."""
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold())


def load_indexed_corpus() -> list[dict]:
    """Đọc đúng ID, nội dung và nguồn mà dense search đang sử dụng."""
    response = get_collection().get(include=["documents", "metadatas"])
    items = []
    for item_id, content, metadata in zip(
        response["ids"], response["documents"], response["metadatas"]
    ):
        meta = dict(metadata)
        meta.setdefault("url", None)
        items.append({"id": item_id, "content": content, "metadata": meta})
    return items


@lru_cache(maxsize=1)
def _cached_index(tokenized: tuple[tuple[str, ...], ...]) -> BM25Okapi | None:
    if not any(tokenized):
        return None
    return BM25Okapi(tokenized)


def build_bm25_index(corpus: list[dict]):
    """Tái sử dụng index nếu token corpus không đổi; corpus không có từ trả None."""
    tokenized = tuple(tuple(tokenize(item["content"])) for item in corpus)
    return _cached_index(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Xếp hạng các chunk có từ khớp; điểm BM25 có thể bằng 0 hoặc âm.

    Mặc định đọc lại snapshot Chroma mỗi lần để nhận thay đổi sau re-index.
    Index BM25 được cache theo token corpus, không theo số lượng tài liệu.
    """
    tokens = tokenize(query)
    if not tokens or top_k <= 0:
        return []

    source = CORPUS if CORPUS is not None else load_indexed_corpus()
    # Giữ lần xuất hiện đầu tiên, cùng quy ước với RRF.
    unique = {}
    for item in source:
        unique.setdefault(item["id"], item)
    corpus = list(unique.values())
    bm25 = build_bm25_index(corpus)
    if bm25 is None:
        return []

    scores = bm25.get_scores(tokens)
    query_terms = set(tokens)
    candidates = [
        index for index, frequencies in enumerate(bm25.doc_freqs)
        if query_terms.intersection(frequencies)
    ]
    indices = sorted(candidates, key=lambda index: float(scores[index]), reverse=True)
    return [
        {
            "id": corpus[index]["id"],
            "content": corpus[index]["content"],
            "score": float(scores[index]),
            "metadata": corpus[index]["metadata"],
            "retrieval_method": "bm25",
        }
        for index in indices[:top_k]
    ]


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
