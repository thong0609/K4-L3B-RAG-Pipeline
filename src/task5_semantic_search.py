"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả cosine score gốc (có thể âm), không chuẩn hóa hoặc cắt về 0.

    Collection của Task 4 phải dùng cosine distance. Lỗi provider được truyền
    lên caller; query rỗng, top_k <= 0 hoặc collection rỗng trả [].
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    query_vector = embed_texts([query])[0]
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    results = {}
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        score = 1.0 - float(distance)
        if item_id in results and results[item_id]["score"] >= score:
            continue
        meta = dict(metadata)
        meta.setdefault("url", None)
        results[item_id] = {
            "id": item_id,
            "content": content,
            "score": score,
            "metadata": meta,
            "retrieval_method": "dense",
        }
    return sorted(results.values(), key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
