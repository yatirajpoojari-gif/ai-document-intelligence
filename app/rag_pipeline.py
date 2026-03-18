from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI



def build_vector_store(documents):
    """
    Convert documents into embeddings and store in vector database
    """

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store vectors
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="vector_db"
    )

    return vectorstore


def search_documents(query):
    """
    Search the vector database for relevant document chunks
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="vector_db",
        embedding_function=embeddings
    )

    results = vectorstore.similarity_search(query, k=3)

    return results

def generate_answer(query, vectorstore):
    import re
    import os

    # 🔥 Step 1: Retrieve documents
    docs = vectorstore.similarity_search(query, k=8)

    # 🔥 Step 2: Extract sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # 🔥 Step 3: Filter docs (optional logic)
    filtered_docs = []
    for doc in docs:
        text = doc.page_content.lower()
        if "invoice" in text and "inv" in text:
            filtered_docs.append(doc)

    if len(filtered_docs) > 0:
        docs = filtered_docs

    # 🔥 Step 4: Build context
    context = "\n".join([doc.page_content for doc in docs])

    print("\n--- Retrieved Context ---\n")
    print(context)

    # 🔥 Step 5: Structured extraction

    if "invoice number" in query.lower():
        match = re.search(r"INV-\d{4}-\d{3}", context)
        if match:
            result = match.group(0)
            return f"{result}\n\n📄 Source: {', '.join(sources)}"

    if any(word in query.lower() for word in ["transit", "shipment", "delivery", "lead time"]):
        match = re.search(r"Estimated Transit Time:\s*\d+\s*Days", context, re.IGNORECASE)
        if match:
            result = match.group(0)
            return f"{result}\n\n📄 Source: {', '.join(sources)}"

    # 🔥 Step 6: LLM fallback
    prompt = f"""
You are an intelligent document assistant.

Extract the exact answer from the context.

Context:
{context}

Question:
{query}

If the answer exists, return it EXACTLY.
If not, say "Not found".
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