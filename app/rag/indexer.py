""" Truy vấn cụ thể collection nào thì viết vào đây """

from app.db.qdrant import VectorStore
from app.rag._utils import load_documents_from_path, get_files_in_directory

def load_and_index_data(vector_store, path):
    print(f"Indexing data from {path}...")
    files = get_files_in_directory(path)
    
    for path in files:
        chunk_contents = load_documents_from_path(path)
        vector_store.insert_data(["content"], chunk_contents)

