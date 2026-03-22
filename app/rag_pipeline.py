import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI


def build_vector_store(documents):
    """
    Convert documents into embeddings and store in vector database
    """

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


def generate_answer(query, vectorstore):
    # 🔥 Retrieve documents
    docs = vectorstore.similarity_search(query, k=12)

    # 🔥 Filter by current document
    current_doc = st.session_state.get("current_doc", None)

    if current_doc:
        docs = [
            doc for doc in docs
            if doc.metadata.get("source") == current_doc
        ]

    # 🔥 Extract sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # 🔥 Build context
    context = "\n".join([doc.page_content for doc in docs])

    print("\n--- Retrieved Context ---\n")
    print(context)

    # 🔥 Dynamic LLM prompt (NO HARDCODING)
    prompt = f"""
You are an intelligent document extraction assistant.

Your job is to extract structured information from the document.

Instructions:
- Identify document type (resume, invoice, report, etc.)
- Extract key fields accordingly
- Keep output clean and structured

If it's a RESUME, extract:
- Name
- Skills
- Experience
- Education

If it's an INVOICE, extract:
- Invoice Number
- Date
- Total Amount
- Buyer Name

If it's something else:
- Provide a summary

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