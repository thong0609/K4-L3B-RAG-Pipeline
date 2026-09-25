import streamlit as st
from dotenv import load_dotenv
try:
    from src.task10_generation import generate_with_citation
except ImportError:
    from task10_generation import generate_with_citation

load_dotenv()

st.set_page_config(
    page_title="Chatbot Tuyển sinh ĐHQGHN",
    page_icon="🎓",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("⚙️ Cài đặt")
    st.caption("Tìm kiếm văn bản tuyển sinh ĐHQGHN")
    top_k = st.slider("Số lượng tài liệu tham khảo (top_k)", 1, 10, 3)

st.title("🎓 Chatbot Tuyển sinh ĐHQGHN 2026")
st.caption("Trợ lý AI trả lời câu hỏi tuyển sinh có trích dẫn nguồn (ĐHQGHN)")

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 Xem tài liệu tham khảo (Nguồn: {message['retrieval_source']})"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**Tài liệu {idx+1}: {src['metadata']['title']}** (Điểm: `{src['score']:.2f}`)")
                    st.info(src["content"])

query = st.chat_input("Nhập câu hỏi của bạn về tuyển sinh ĐHQGHN...")

if query:
    # 1. Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Sinh câu trả lời với mock data
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tổng hợp..."):
            result = generate_with_citation(query, top_k=top_k)
            
        answer = result["answer"]
        sources = result["sources"]
        retrieval_source = result["retrieval_source"]
        
        st.markdown(answer)
        
        # Hiển thị sources ngay lúc vừa sinh xong
        if sources:
            with st.expander(f"📚 Xem tài liệu tham khảo (Nguồn: {retrieval_source})"):
                for idx, src in enumerate(sources):
                    st.markdown(f"**Tài liệu {idx+1}: {src['metadata']['title']}** (Điểm: `{src['score']:.2f}`)")
                    st.info(src["content"])

    # 3. Lưu lại vào session state
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source
    })
