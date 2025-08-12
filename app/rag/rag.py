from app.rag.retriever import search_stock_collection
from app.llm.chat import ask_rag

def ask_stock_rag(prompt, chat_history=[]):
    documents = search_stock_collection(prompt) 
    return ask_rag(prompt, documents, chat_history)