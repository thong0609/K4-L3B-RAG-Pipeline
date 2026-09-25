"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh."""


import logging
import os

from dotenv import load_dotenv

from .contracts import GenerationResult, validate_generation_result
from .task9_retrieval_pipeline import retrieve

load_dotenv()

logger = logging.getLogger(__name__)

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Bạn là trợ lý AI trả lời câu hỏi tuyển sinh dựa trên tài liệu được cung cấp.
QUY TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên các thông tin có trong phần Ngữ cảnh (Context) bên dưới.
2. Mỗi khẳng định, số liệu hoặc sự kiện phải có trích dẫn nguồn tương ứng dạng [Document X].
3. Nếu ngữ cảnh không có thông tin hoặc không đủ bằng chứng để trả lời chính xác, bạn PHẢI trả lời đúng câu:
"Tôi không thể xác minh thông tin này từ nguồn hiện có."
Tuyệt đối không bịa đặt hoặc tự suy diễn ngoài tài liệu."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (chống lost-in-the-middle)."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Không rõ tiêu đề")
        source = meta.get("source", "Không rõ nguồn")
        content = chunk.get("content", "").strip()
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = LLM_PROVIDER.strip()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY chưa được cấu hình.")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = LLM_MODEL or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or ""

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY chưa được cấu hình.")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = LLM_MODEL or "gemini-2.5-flash"
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=TEMPERATURE,
        )
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=config,
        )
        return response.text or ""

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY chưa được cấu hình.")
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        model = LLM_MODEL or "claude-3-5-sonnet-20241022"
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
            max_tokens=1024,
        )
        return response.content[0].text or ""

    raise ValueError(f"LLM_PROVIDER không hợp lệ: {provider}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}"

    try:
        raw_answer = call_llm(SYSTEM_PROMPT, user_message)
        answer = raw_answer.strip() if raw_answer and raw_answer.strip() else "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    except Exception as exc:
        logger.warning("Lỗi khi gọi LLM (%s): %s", LLM_PROVIDER, exc)
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    method = chunks[0].get("retrieval_method", "hybrid")
    retrieval_source = method if method in {"hybrid", "pageindex"} else "hybrid"

    result: GenerationResult = {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }
    validate_generation_result(result)
    return result


if __name__ == "__main__":
    print(generate_with_citation("Chỉ tiêu tuyển sinh năm 2026?"))

