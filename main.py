from app.rag.indexer import index_stock_collection
from app.rag.retriever import search_stock_collection

if __name__ == "__main__":
    index_stock_collection()
    print(search_stock_collection("chứng khoán"))
