from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import os
import shutil
import re


# =========================
# BUILD VECTOR STORE
# =========================
def build_vector_store(documents):

    # 🔥 Delete old DB (important)
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="vector_db"
    )

    return vectorstore


# =========================
# GENERATE ANSWER
# =========================
def generate_answer(query, vectorstore):

    # 🔥 Retrieve docs
    docs = vectorstore.similarity_search(query, k=12)

    # Extract sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # Build context
    context = "\n".join([doc.page_content for doc in docs])

    print("\n--- Retrieved Context ---\n")
    print(context)

    # =========================
    # STRUCTURED EXTRACTION
    # =========================
    if "invoice" in query.lower():

        prompt = f"""
You are an intelligent document extraction system.

Extract structured data from the document.

Return output in JSON format like this:

{{
  "invoice_number": "...",
  "date": "...",
  "total_amount": "...",
  "vendor_name": "..."
}}

Rules:
- Extract only if present
- Do not guess
- If missing, return null

Context:
{context}

Question:
{query}
"""

    else:
        # Normal Q&A
        prompt = f"""
You are an intelligent document assistant.

Answer the question using ONLY the context.

Rules:
- Do NOT guess
- If answer not found, say "Not found in document"

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

    # Add source
    if sources:
        answer += f"\n\n📄 Source: {', '.join(sources)}"

    return answer