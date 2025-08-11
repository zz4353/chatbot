from app.db.qdrant import VectorStore

if __name__ == "__main__":
    vector_store = VectorStore()
    print(vector_store.hybrid_search("chứng khoán"))