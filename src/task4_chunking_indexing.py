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


# ---------------------------------------------------------------- embedding

def _normalize(vectors: list[list[float]]) -> list[list[float]]:
    normalized = []
    for vector in vectors:
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        normalized.append([value / norm for value in vector])
    return normalized


def _with_retry(call, retries: int = 6):
    """Retry khi bị rate limit / lỗi mạng tạm thời; ưu tiên thời gian chờ server gợi ý."""
    for attempt in range(retries):
        try:
            return call()
        except Exception as error:
            message = str(error).lower()
            transient = any(s in message for s in ("429", "rate", "quota", "503", "timeout", "unavailable"))
            if not transient or attempt == retries - 1:
                raise
            suggested = re.search(r"retry in ([\d.]+)s", message)
            delay = float(suggested.group(1)) + 2 if suggested else min(60, 5 * 2 ** attempt)
            print(f"  rate limited, retrying in {delay:.0f}s ({attempt + 1}/{retries - 1})")
            time.sleep(delay)


def _embed_gemini(texts: list[str], is_query: bool) -> list[list[float]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
        output_dimensionality=EMBEDDING_DIM,
    )
    batch_size = min(EMBED_BATCH_SIZE, GEMINI_EMBED_PER_MINUTE)
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        started = time.monotonic()
        response = _with_retry(
            lambda: client.models.embed_content(model=EMBEDDING_MODEL, contents=batch, config=config)
        )
        vectors.extend(embedding.values for embedding in response.embeddings)
        if start + batch_size < len(texts):
            # Giãn cách batch để không vượt quota/phút.
            wait = 60 * len(batch) / GEMINI_EMBED_PER_MINUTE - (time.monotonic() - started)
            print(f"  embedded {len(vectors)}/{len(texts)}, waiting {max(wait, 0):.0f}s for quota")
            time.sleep(max(wait, 0))
    return vectors


def _embed_openai(texts: list[str], is_query: bool) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI()
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        response = _with_retry(
            lambda: client.embeddings.create(model=EMBEDDING_MODEL, input=batch, dimensions=EMBEDDING_DIM)
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


_st_model = None


def _embed_sentence_transformers(texts: list[str], is_query: bool) -> list[list[float]]:
    global _st_model
    from sentence_transformers import SentenceTransformer

    if _st_model is None:
        _st_model = SentenceTransformer(EMBEDDING_MODEL)
    return _st_model.encode(texts, batch_size=16).tolist()


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
