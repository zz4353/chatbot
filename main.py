from app.db.qdrant import VectorStore
from app.rag.indexer import load_and_index_data

if __name__ == "__main__":
    vector_store = VectorStore()
    load_and_index_data(vector_store, "data")
    print(vector_store.hybrid_search("chứng khoán"))