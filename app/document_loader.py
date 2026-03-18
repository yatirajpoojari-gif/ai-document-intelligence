from langchain_community.document_loaders import PyPDFLoader


def load_documents(file_path):
    """
    Load a PDF document and return extracted pages
    """

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents