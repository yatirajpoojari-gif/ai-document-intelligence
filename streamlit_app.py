import streamlit as st
from app.document_loader import load_documents
from app.rag_pipeline import build_vector_store, generate_answer

# Page config
st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)

# Sidebar
st.sidebar.title("📊 AI Document Intelligence")
st.sidebar.markdown("""
Upload invoices, extract data, and ask questions.

Built for:
- Exporters
- Businesses
- Finance teams
""")
st.sidebar.info("Powered by AI 🚀")

# Main UI
st.title("📄 AI Document Intelligence System")
st.markdown("### Extract insights from your documents instantly")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload multiple PDFs
uploaded_files = st.file_uploader(
    "📂 Upload your PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Process files
if uploaded_files:
    with st.spinner("Processing documents..."):

        all_documents = []

       import uuid

for file in uploaded_files:
    file_path = f"temp_{uuid.uuid4().hex}.pdf"

    with open(file_path, "wb") as f:
        f.write(file.read())

    docs = load_documents(file_path)

    # Add source info
    for doc in docs:
        doc.metadata["source"] = file.name

    all_documents.extend(docs)

            # Add source metadata
            
            for doc in docs:
                doc.metadata["source"] = file.name

            all_documents.extend(docs)

        vectorstore = build_vector_store(all_documents)
        st.session_state["vectorstore"] = vectorstore

    st.success(f"✅ {len(uploaded_files)} documents processed!")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if "vectorstore" in st.session_state:
    user_input = st.chat_input("Ask something about the document...")

    if user_input:
        # Show user message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response
        with st.spinner("Thinking..."):
            answer = generate_answer(user_input, st.session_state["vectorstore"])

        # Show AI response
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})