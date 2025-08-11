""" Truy vấn cụ thể collection nào thì viết vào đây """
from langchain.schema import Document
from app.rag.collections import COLLECTIONS

def _search(vector_store, query, top_k=3, threshold=0.3):
    docs = []
    results = vector_store.hybrid_search(query, top_k, threshold)
    for point in results:
        docs.append(Document(
            page_content=point.payload["content"],
            metadata={"score": point.score, "id": point.id}
        ))

    return docs

def search_stock_collection(query):
    return _search(COLLECTIONS["stock"], query)
