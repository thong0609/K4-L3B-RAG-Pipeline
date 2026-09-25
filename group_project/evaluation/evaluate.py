import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from dotenv import load_dotenv

# Tạm thời comment dòng import này vì file generation chưa hoàn thiện
from src.task10_generation import generate_with_citation

# def mock_generate_with_citation(query: str, retrieval_method: str):
#     """
#     Hàm giả lập để code chạy không bị lỗi trong lúc chờ các bạn khác code xong.
#     Sau này sẽ xóa hàm này và dùng hàm thật từ src.task10_generation.
#     """
#     return {
#         "answer": f"Đây là câu trả lời giả lập cho câu hỏi: {query}",
#         "sources": [
#             {
#                 "content": "Đây là nội dung tài liệu giả lập. ĐHQGHN tuyển sinh bằng 4 phương thức.",
#                 "score": 0.9,
#                 "id": "chunk_1",
#                 "metadata": {}
#             }
#         ],
#         "retrieval_source": retrieval_method
#     }

def load_golden_dataset(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation(dataset_path: str, retrieval_method: str):
    """
    Chạy đánh giá cho một phương thức tìm kiếm (dense hoặc hybrid).
    """
    print(f"[EVAL] Collecting answers for: {retrieval_method.upper()}...")
    os.environ["USE_RERANKING"] = "true" if retrieval_method.lower() == "hybrid" else "false"
    data = load_golden_dataset(dataset_path)
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for idx, item in enumerate(data, 1):
        q = item["question"]
        gt = item.get("expected_answer") or item.get("answer")
        print(f"  - [{idx}/{len(data)}] Processing question {idx}...")
        # 1. Gọi hàm sinh câu trả lời thật từ task 10
        result = generate_with_citation(q)
        
        generated_answer = result["answer"]
        # Rút trích list các đoạn văn bản thô từ sources
        retrieved_contexts = [src["content"] for src in result.get("sources", [])]
        
        questions.append(q)
        answers.append(generated_answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(gt)
        
    # Tạo Dataset đúng định dạng HuggingFace mà ragas yêu cầu
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    print(f"[EVAL] Evaluating with LLM-as-a-judge for {retrieval_method.upper()}...")
    # 2. Chạy evaluate với 4 metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    from ragas.run_config import RunConfig

    # Giới hạn 2 luồng đồng thời và tăng timeout để tránh nghẽn API Gemini
    run_config = RunConfig(
        timeout=300,
        max_workers=2,
        max_wait=120,
        max_retries=10,
    )

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if gemini_key and not openai_key:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        judge_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=gemini_key)
        judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=gemini_key)
        eval_results = evaluate(
            eval_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
        )
    else:
        eval_results = evaluate(eval_dataset, metrics=metrics, run_config=run_config)
    
    return eval_results

if __name__ == "__main__":
    load_dotenv()
    
    dataset_file = "group_project/evaluation/golden_dataset.json"
    
    # 1. Đánh giá phương thức Dense (Chỉ Vector)
    dense_results = run_evaluation(dataset_file, "dense")
    
    # 2. Đánh giá phương thức Hybrid (Vector + BM25)
    hybrid_results = run_evaluation(dataset_file, "hybrid")
    
    # 3. In kết quả ra màn hình
    print("\n" + "="*50)
    print("EVALUATION RESULTS (A/B TESTING)")
    print("="*50)
    
    print("\n[A] DENSE-ONLY:")
    print(dense_results)
    
    print("\n[B] HYBRID + RRF:")
    print(hybrid_results)
    
    print("\n>>> Copy the scores above into RESULT.md <<<")
