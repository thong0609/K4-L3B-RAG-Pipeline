"""
Task 4 — Chunking, embedding và indexing.

1. Đọc Markdown trong data/standardized/ (YAML front matter do Task 3 ghi).
2. Chunk bằng RecursiveCharacterTextSplitter, ưu tiên cắt ở ranh giới
   Chương/Điều (văn bản pháp quy) và heading Markdown (bài báo).
3. Embed bằng một provider duy nhất theo EMBEDDING_PROVIDER trong .env.
4. Upsert vào ChromaDB (cosine). ID ổn định -> chạy lại không tạo dữ liệu trùng;
   chunk không còn trong corpus bị xoá khỏi collection.

Task 5 phải dùng chung embed_texts(); với query gọi embed_texts([query], is_query=True).
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = ROOT / "data" / "standardized"
CHROMA_DIR = ROOT / "chroma_db"

load_dotenv(ROOT / ".env")

# Tiếng Việt ~4-5 ký tự/token -> 1000 ký tự ≈ 200-250 token: đủ chứa trọn một
# khoản trong quy chế hoặc một đoạn bảng điểm chuẩn, vẫn đủ nhỏ để retrieval chính xác.
# Overlap 150 ký tự giữ ngữ cảnh khi một điều khoản bị cắt ngang.
# CHUNK_SIZE là độ dài tối đa của cả chunk, tính cả header ngữ cảnh.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "recursive"
# Phần "[tiêu đề › mục]" gắn đầu chunk được giới hạn trong ngân sách này,
# nên tổng độ dài chunk (header + text) không vượt CHUNK_SIZE.
HEADER_BUDGET = 200
SEPARATORS = [
    "\nChương ", "\nĐiều ",            # cấu trúc văn bản quy phạm
    "\n## ", "\n### ", "\n#### ",      # heading Markdown của bài báo
    "\n\n", "\n", ". ", " ", "",
]
# Dòng được coi là "tiêu đề mục" để gắn ngữ cảnh cho chunk.
SECTION_PATTERN = re.compile(
    r"^(#{1,4} .+|Chương [IVXLC\d]+.*|Điều \d+\..*|[IVX]+\. .+)$", re.MULTILINE
)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").strip().lower()
_DEFAULT_MODELS = {
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
    "sentence_transformers": "BAAI/bge-m3",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or _DEFAULT_MODELS.get(EMBEDDING_PROVIDER, "")
# Gemini/OpenAI cho phép chọn số chiều; bge-m3 cố định 1024.
EMBEDDING_DIM = 1024 if EMBEDDING_PROVIDER == "sentence_transformers" else 768
EMBED_BATCH_SIZE = 100
# Free tier Gemini tính mỗi text trong batch là 1 request: 100 request/phút/model.
GEMINI_EMBED_PER_MINUTE = int(os.getenv("GEMINI_EMBED_PER_MINUTE", "100"))

COLLECTION_NAME = "rag_documents"


import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
COLLECTION_NAME = "rag_documents"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo vector embeddings theo provider cấu hình trong .env."""
    if not texts:
        return []

    provider = EMBEDDING_PROVIDER.strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    # Nếu chọn gemini hoặc sentence_transformers chưa cài đặt nhưng có Gemini key
    if provider == "gemini" or (gemini_key and provider == "sentence_transformers"):
        from google import genai

        import time

        client = genai.Client(api_key=gemini_key)
        all_embeddings: list[list[float]] = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(5):
                try:
                    resp = client.models.embed_content(
                        model="models/gemini-embedding-001",
                        contents=batch,
                    )
                    for emb in resp.embeddings:
                        all_embeddings.append(list(emb.values))
                    time.sleep(1.0)
                    break
                except Exception as exc:
                    err_text = str(exc)
                    if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
                        print(f"[Gemini Embed] Rate limit 429, waiting 42s (attempt {attempt+1}/5)...")
                        time.sleep(42)
                    else:
                        raise
        return all_embeddings

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL or "text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in resp.data]

    # Mặc định thử sentence_transformers nếu được cài
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        return model.encode(texts).tolist()
    except Exception as exc:
        raise RuntimeError(
            f"Không thể embed bằng provider '{provider}'. Hãy đặt GEMINI_API_KEY trong .env để dùng Gemini embeddings: {exc}"
        )


_PROVIDERS = {
    "gemini": _embed_gemini,
    "openai": _embed_openai,
    "sentence_transformers": _embed_sentence_transformers,
}


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed danh sách text (đã L2-normalize). is_query=True khi embed câu hỏi."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER not in _PROVIDERS:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}")
    vectors = _PROVIDERS[EMBEDDING_PROVIDER](texts, is_query)
    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: {len(vectors)} != {len(texts)}")
    return _normalize(vectors)


# ---------------------------------------------------------------- vector store

def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        doc_type = "legal" if "legal" in path.parts else "news"
        title = path.stem.replace("-", " ").title()
        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": path.read_text(encoding="utf-8"),
                "metadata": {
                    "source": path.name,
                    "title": title,
                    "doc_type": doc_type,
                    "url": None,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        for index, text in enumerate(splitter.split_text(document["content"])):
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    collection = get_collection()
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    print("1. Loading documents from data/standardized/...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("2. Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("3. Generating embeddings...")
    embedded_chunks = embed_chunks(chunks)

    print("4. Indexing into ChromaDB...")
    index_to_vectorstore(embedded_chunks)
    print(f"Successfully indexed {len(embedded_chunks)} chunks to ChromaDB at {CHROMA_DIR}!")


if __name__ == "__main__":
    run_pipeline()
