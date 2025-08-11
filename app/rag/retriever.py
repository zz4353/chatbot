from langchain.schema import Document
from app.rag.collections import COLLECTIONS
from app.rag.reranker import rerank

def _search(vector_store, query, top_k=3, threshold=0.3):
    results = vector_store.hybrid_search(query, top_k * 2 + 1, threshold)
    results = rerank(query, results, [point.payload["content"] for point in results], top_k)

    docs = []
    for point in results:
        docs.append(Document(
            page_content=point.payload["content"],
            metadata={"score": point.score, "id": point.id}
        ))

    return docs

def search_stock_collection(query, top_k=3, threshold=0.3):
    return _search(COLLECTIONS["stock"], query, top_k, threshold)
