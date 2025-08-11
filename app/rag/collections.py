from app.db.qdrant import VectorStore

COLLECTIONS = {
    "stock": VectorStore(collection_name="stock_collection"),
}