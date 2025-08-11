""" Truy vấn cụ thể collection nào thì viết vào đây """

from app.rag._utils import get_files_in_directory, load_file_as_markdown
from app.rag.collections import COLLECTIONS
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_file(path, text_splitter):
    """ docx, pdf, txt"""
    file_name = path.split("/")[-1]
    markdown_text = load_file_as_markdown(path)
    chunks = text_splitter.split_text(markdown_text)
    chunk_contents = ["Nguồn: " + file_name + '\n' + chunk.replace("\n", " ") for chunk in chunks if len(chunk) >= 40]
    return chunk_contents

def load_documents_from_path(path):
    chunk_contents = []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=512, chunk_overlap=126
            )

    # Xử lý file .docx, .txt và .pdf
    if path.endswith(".docx") or path.endswith(".pdf") or path.endswith(".txt"):
        print(f"Processing file: {path}")
        chunk_contents = process_file(path, text_splitter)
    else:
        pass

    return chunk_contents

def _load_and_index_data(vector_store, path):
    vector_store.recreate_collection()

    print(f"Indexing data from {path}...")
    files = get_files_in_directory(path)
    
    for path in files:
        chunk_contents = load_documents_from_path(path)
        vector_store.insert_data(["content"], chunk_contents)

def index_stock_collection():
    _load_and_index_data(COLLECTIONS["stock"], "data")
