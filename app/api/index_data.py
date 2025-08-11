from app.db.qdrant import VectorStore
from utils import load_documents_from_path, get_files_in_directory

def load_and_index_data(vector_store, path):
    # self.create_collection()

    print(f"Indexing data from {path}...")
    files = get_files_in_directory(path)
    
    for path in files:
        chunk_contents = load_documents_from_path(path)
        vector_store.insert_data(["content"], chunk_contents)


if __name__ == "__main__":
    vector_store = VectorStore()
    load_and_index_data(vector_store, "data")