"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .contracts import SearchResult, validate_search_results

load_dotenv()

logger = logging.getLogger(__name__)

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_cache.json"


def _get_client() -> Any:
    """Khởi tạo PageIndexClient nếu có API key."""
    if not PAGEINDEX_API_KEY:
        return None
    try:
        from pageindex import PageIndexClient

        return PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except Exception as exc:
        logger.warning("Không thể khởi tạo PageIndexClient: %s", exc)
        return None


def upload_documents() -> None:
    """Upload tai lieu va luu document IDs de tai su dung."""
    client = _get_client()
    if not client:
        print("[PageIndex] PAGEINDEX_API_KEY is not configured. Skipping upload.")
        return

    mapping: dict[str, str] = {}
    if CACHE_FILE.exists():
        try:
            mapping = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            mapping = {}

    md_files = list(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print("[PageIndex] No Markdown files found in data/standardized/.")
        return

    for path in md_files:
        rel_key = str(path.relative_to(STANDARDIZED_DIR))
        if rel_key in mapping:
            continue

        try:
            print(f"[PageIndex] Uploading: {rel_key}...")
            resp = client.submit_document(file_path=str(path))
            doc_id = resp.get("doc_id") or resp.get("id")
            if doc_id:
                mapping[rel_key] = doc_id
        except Exception as exc:
            logger.warning("[PageIndex] Error uploading %s: %s", rel_key, exc)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PageIndex] Uploaded and cached {len(mapping)} documents.")


_QUERY_CACHE: dict[str, list[dict]] = {}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult với query cache và rate limit protection."""
    clean_query = query.strip()
    if not clean_query:
        return []

    # Kiểm tra cache trước để tránh gọi lặp lại API
    if clean_query in _QUERY_CACHE:
        return _QUERY_CACHE[clean_query][:top_k]

    client = _get_client()
    if not client or not CACHE_FILE.exists():
        return []

    try:
        mapping: dict[str, str] = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

    import time

    results: list[dict] = []
    # Giới hạn truy vấn tối đa 3 tài liệu liên quan thay vì quét toàn bộ 11 tài liệu cùng lúc
    target_docs = list(mapping.items())[:3]

    for rel_path, doc_id in target_docs:
        if len(results) >= top_k:
            break
        try:
            resp = client.submit_query(doc_id=doc_id, query=clean_query)
            retrieval_id = resp.get("retrieval_id")
            if not retrieval_id:
                continue

            # PageIndex xử lý bất đồng bộ, đợi trạng thái hoàn thành
            ret_data = {}
            for _ in range(6):
                time.sleep(0.5)
                ret_data = client.get_retrieval(retrieval_id=retrieval_id)
                if ret_data.get("status") == "completed":
                    break

            nodes = (
                ret_data.get("retrieved_nodes")
                or ret_data.get("nodes")
                or ret_data.get("results")
                or []
            )

            doc_type = "legal" if "legal" in rel_path.lower() else "news"
            source_name = Path(rel_path).name
            title = source_name.replace("-", " ").replace(".md", "").title()

            for idx, node in enumerate(nodes):
                contents = []
                for sub in node.get("relevant_contents", []):
                    if isinstance(sub, list):
                        for item in sub:
                            if isinstance(item, dict) and "relevant_content" in item:
                                contents.append(item["relevant_content"])
                    elif isinstance(sub, dict) and "relevant_content" in sub:
                        contents.append(sub["relevant_content"])

                content = (
                    "\n".join(contents).strip()
                    if contents
                    else node.get("text", node.get("title", "")).strip()
                )
                if not content:
                    continue

                raw_score = node.get("score")
                score = (
                    float(raw_score)
                    if isinstance(raw_score, (int, float))
                    else max(0.1, 1.0 - (len(results) * 0.1))
                )

                item: SearchResult = {
                    "id": f"pageindex-{doc_id}-{idx}",
                    "content": content,
                    "score": round(score, 4),
                    "metadata": {
                        "source": source_name,
                        "title": title,
                        "doc_type": doc_type,
                        "url": None,
                        "chunk_index": idx,
                    },
                    "retrieval_method": "pageindex",
                }
                results.append(item)
                if len(results) >= top_k:
                    break
        except Exception as exc:
            err_msg = str(exc).lower()
            if "429" in err_msg or "too many" in err_msg or "rate" in err_msg:
                logger.warning("[PageIndex] Đạt giới hạn rate limit: %s. Tạm dừng truy vấn cloud.", exc)
                break
            logger.warning("[PageIndex] Lỗi truy vấn doc %s: %s", doc_id, exc)

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]

    try:
        validate_search_results(results, top_k=top_k, expected_method="pageindex")
    except Exception:
        results = []

    _QUERY_CACHE[clean_query] = results
    return results


if __name__ == "__main__":
    upload_documents()
    