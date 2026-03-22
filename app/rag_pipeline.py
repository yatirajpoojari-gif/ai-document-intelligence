import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI


# =========================
# BUILD VECTOR STORE
# =========================
def build_vector_store(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(docs, embeddings)

    return vectorstore


# =========================
# DOCUMENT TYPE DETECTION
# =========================
def detect_document_type(context):

    prompt = f"""
Classify the document type from the context.

Possible types:
- invoice
- resume
- report
- contract
- other

Return ONLY one word.

Context:
{context}
"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = llm.invoke(prompt)

    return response.content.strip().lower()


# =========================
# GENERATE ANSWER
# =========================
def generate_answer(query, vectorstore):

    # 🔥 Retrieve relevant docs
    docs = vectorstore.similarity_search(query, k=12)

    # Extract sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # Build context
    context = "\n".join([doc.page_content for doc in docs])

    print("\n--- Retrieved Context ---\n")
    print(context)

    # 🔥 Detect document type
    doc_type = detect_document_type(context)

    print("Detected Document Type:", doc_type)

    # =========================
    # SMART PROMPT BASED ON TYPE
    # =========================

    if doc_type == "invoice":

        prompt = f"""
Extract the following details in JSON:

{{
  "invoice_number": "...",
  "date": "...",
  "total_amount": "...",
  "vendor_name": "..."
}}

Rules:
- Do not guess
- If missing, return null

Context:
{context}
"""

    elif doc_type == "resume":

        prompt = f"""
Extract key details from resume in JSON:

{{
  "name": "...",
  "skills": "...",
  "experience": "...",
  "education": "..."
}}

Context:
{context}
"""

    else:

        prompt = f"""
Answer the question based on context.

Rules:
- Do NOT guess
- Be accurate

Context:
{context}

Question:
{query}
"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = llm.invoke(prompt)

    answer = response.content

    # Add sources
    if sources:
        answer += f"\n\n📄 Source: {', '.join(sources)}"

    return answer