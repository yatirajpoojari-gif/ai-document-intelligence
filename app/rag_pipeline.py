from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI


def build_vector_store(documents):
    """
    Convert documents into embeddings and store in vector database
    """

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ In-memory vector DB (no crash)
    vectorstore = Chroma.from_documents(
        docs,
        embeddings
    )

    return vectorstore


def generate_answer(query, vectorstore):
    import re
    import os

    # 🔥 Step 1: Retrieve documents
    docs = vectorstore.similarity_search(query, k=12)

    # 🔥 Step 2: Extract sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # 🔥 Step 3: Build context
    context = "\n".join([doc.page_content for doc in docs])

    print("\n--- Retrieved Context ---\n")
    print(context)

    # =========================
    # 🔥 STRUCTURED EXTRACTION
    # =========================

    # ✅ Invoice Number
    if "invoice number" in query.lower():
        match = re.search(r"INV-\d{4}-\d{3}", context)
        if match:
            return f"{match.group(0)}\n\n📄 Source: {', '.join(sources)}"

    # ✅ Resume Name
    if "name" in query.lower():
        match = re.search(r"[A-Z][a-z]+\s[A-Z][a-z]+", context)
        if match:
            return f"{match.group(0)}\n\n📄 Source: {', '.join(sources)}"

    # ✅ Skills (basic extraction)
    if "skills" in query.lower():
        return f"Skills found in document:\n{context[:500]}...\n\n📄 Source: {', '.join(sources)}"

    # ✅ Experience
    if "experience" in query.lower():
        return f"Experience details:\n{context[:500]}...\n\n📄 Source: {', '.join(sources)}"

    # =========================
    # 🤖 LLM FALLBACK
    # =========================

    prompt = f"""
You are an intelligent document assistant.

Answer the question based on the context.

Rules:
- Answer ONLY from the context
- Be specific (not generic)
- If it's a resume, say it's a resume
- Keep answer short and clear

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

    if sources:
        answer += f"\n\n📄 Source: {', '.join(sources)}"

    return answer