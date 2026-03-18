<<<<<<< HEAD
from document_loader import load_documents
from rag_pipeline import build_vector_store, generate_answer

def main():
    file_path = "data/sample.pdf"
    documents = load_documents(file_path)

    vectorstore = build_vector_store(documents)

    while True:
        query = input("Ask a question about the document: ")

        if query.lower() == "exit":
            break

        answer = generate_answer(query, vectorstore)

        print("\nAI Answer:\n")
        print(answer)

if __name__ == "__main__":
=======
from document_loader import load_documents
from rag_pipeline import build_vector_store, generate_answer

def main():
    file_path = "data/sample.pdf"
    documents = load_documents(file_path)

    vectorstore = build_vector_store(documents)

    while True:
        query = input("Ask a question about the document: ")

        if query.lower() == "exit":
            break

        answer = generate_answer(query, vectorstore)

        print("\nAI Answer:\n")
        print(answer)

if __name__ == "__main__":
>>>>>>> 554fa7eeabe33b61e0e30ada32bb427573b0f83a
    main()