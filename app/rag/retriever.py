from langchain.schema import Document
from app.rag.collections import COLLECTIONS

def _search(vector_store, query, top_k=5, threshold=0.3):
    results = vector_store.search(query, top_k, threshold)

    docs = []
    for point in results:
        docs.append(Document(
            page_content=point.payload["content"],
            metadata={"score": point.score, "id": point.id}
        ))

    return docs

def search_stock_collection(query, top_k=5, threshold=0.2):
    return _search(COLLECTIONS["stock"], query, top_k, threshold)
