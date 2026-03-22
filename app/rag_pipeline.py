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

    # In-memory vector store (no persistence → avoids crash)
    vectorstore = Chroma.from_documents(
        docs,
        embeddings
    )

    return vectorstore


def generate_answer(query, vectorstore):
    import os

    # 🔥 Step 1: Retrieve relevant documents
    docs = vectorstore.similarity_search(query, k=12)

    # 🔥 Step 2: Extract sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # 🔥 Step 3: Build context
    context = "\n".join([doc.page_content for doc in docs])

    print("\n--- Retrieved Context ---\n")
    print(context)

    # =========================
    # 🤖 DYNAMIC EXTRACTION (NO HARDCODING)
    # =========================

    prompt = f"""
You are an intelligent document extraction assistant.

Your job is to extract structured information from the document.

Instructions:
- Identify what type of document this is (resume, invoice, report, etc.)
- Extract key fields depending on document type
- Keep output structured and clean

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
- Provide a clear summary

Keep the response concise and well formatted.

Context:
{context}

Question:
{query}
"""

    # 🔥 Step 4: LLM call
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = llm.invoke(prompt)

    answer = response.content

    # 🔥 Step 5: Add source info
    if sources:
        answer += f"\n\n📄 Source: {', '.join(sources)}"

    return answer