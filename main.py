from app.rag.indexer import index_stock_collection
from app.rag.retriever import search_stock_collection
from app.rag.collections import COLLECTIONS

if __name__ == "__main__":
    index_stock_collection()
    print(search_stock_collection("chứng khoán"))
    print()
    print(COLLECTIONS["stock"].hybrid_search("chứng khoán"))

