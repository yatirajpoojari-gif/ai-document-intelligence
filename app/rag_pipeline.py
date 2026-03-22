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

    # ✅ FIX: No persist_directory (in-memory DB)
    vectorstore = Chroma.from_documents(
        docs,
        embeddings
    )

    return vectorstore


def generate_answer(query, vectorstore):
    import re
    import os

    # 🔥 Retrieve relevant docs
    docs = vectorstore.similarity_search(query, k=12)

    # 🔥 Extract sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # 🔥 Optional filtering (only for invoices)
    filtered_docs = []
    for doc in docs:
        text = doc.page_content.lower()
        if "invoice" in text and "inv" in text:
            filtered_docs.append(doc)

    if "invoice" in query.lower() and len(filtered_docs) > 0:
        docs = filtered_docs

    # 🔥 Build context
    context = "\n".join([doc.page_content for doc in docs])

    print("\n--- Retrieved Context ---\n")
    print(context)

    # 🔥 Structured extraction

    if "invoice number" in query.lower():
        match = re.search(r"INV-\d{4}-\d{3}", context)
        if match:
            result = match.group(0)
            return f"{result}\n\n📄 Source: {', '.join(sources)}"

    if any(word in query.lower() for word in ["transit", "shipment", "delivery", "lead time"]):
        match = re.search(
            r"Estimated Transit Time:\s*\d+\s*Days",
            context,
            re.IGNORECASE
        )
        if match:
            result = match.group(0)
            return f"{result}\n\n📄 Source: {', '.join(sources)}"

    # 🔥 LLM fallback

    prompt = f"""
You are an intelligent document assistant.

Answer the question based on the context.

Rules:
- Try to find the best possible answer from the context
- If exact answer is not present, give the closest relevant answer
- Do NOT say "Not found" unless absolutely nothing is relevant
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