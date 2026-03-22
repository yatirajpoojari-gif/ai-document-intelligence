import streamlit as st
import uuid
from app.document_loader import load_documents
from app.rag_pipeline import build_vector_store, generate_answer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Document Chat", layout="wide")

# =========================
# 🔥 CUSTOM CSS (IMPORTANT)
# =========================
st.markdown("""
<style>
.bottom-bar {
    position: fixed;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    width: 70%;
    background: white;
    border-radius: 30px;
    padding: 10px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    z-index: 999;
}

.upload-btn {
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("🤖 AI Document Chat")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# 🔥 FLOATING INPUT BAR
# =========================
st.markdown('<div class="bottom-bar">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 8])

with col1:
    uploaded_files = st.file_uploader(
        "➕",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

with col2:
    user_input = st.chat_input("Ask anything")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PROCESS FILES
# =========================
if uploaded_files:
    with st.spinner("Processing..."):
        all_docs = []

        for file in uploaded_files:
            path = f"temp_{uuid.uuid4().hex}.pdf"

            with open(path, "wb") as f:
                f.write(file.read())

            docs = load_documents(path)

            for d in docs:
                d.metadata["source"] = file.name

            all_docs.extend(docs)

        st.session_state["vectorstore"] = build_vector_store(all_docs)

    st.toast("✅ Uploaded")

# =========================
# CHAT LOGIC
# =========================
if user_input and "vectorstore" in st.session_state:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Thinking..."):
        answer = generate_answer(
            user_input,
            st.session_state["vectorstore"]
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()