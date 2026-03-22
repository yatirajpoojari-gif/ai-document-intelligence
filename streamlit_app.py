import streamlit as st
import uuid
from app.document_loader import load_documents
from app.rag_pipeline import build_vector_store, generate_answer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 AI Document Intelligence")
st.sidebar.markdown("""
Upload documents, extract insights, and ask questions.

Built for:
- Exporters
- Businesses
- Finance teams
""")
st.sidebar.info("Powered by AI 🚀")

# =========================
# MAIN HEADER
# =========================
st.title("📄 AI Document Intelligence System")
st.markdown("### Extract insights from your documents instantly")

# =========================
# SESSION STATE INIT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# 📂 UPLOAD SECTION
# =========================
st.markdown("### 📂 Add / Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):

        all_documents = []

        for file in uploaded_files:
            file_path = f"temp_{uuid.uuid4().hex}.pdf"

            with open(file_path, "wb") as f:
                f.write(file.read())

            docs = load_documents(file_path)

            for doc in docs:
                doc.metadata["source"] = file.name

            all_documents.extend(docs)

            # 🔥 Track current document
            st.session_state["current_doc"] = file.name

        vectorstore = build_vector_store(all_documents)
        st.session_state["vectorstore"] = vectorstore

    st.success("✅ Document(s) uploaded successfully!")

# =========================
# CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# CHAT INPUT
# =========================
if "vectorstore" in st.session_state:
    st.markdown("### 💬 Ask Questions")

    user_input = st.chat_input("Ask something about the document...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("Thinking..."):
            answer = generate_answer(
                user_input,
                st.session_state["vectorstore"]
            )

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })