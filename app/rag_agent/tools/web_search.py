import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import Tool
from langchain.schema import Document
from langchain_tavily import TavilySearch

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def web_search(question, top_k=5):
    documents = []
    try:
        search = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=top_k)
        docs = search.invoke({"query": question})
        for doc in docs['results']:
            documents.append(Document(page_content=doc["content"], metadata={"title": doc["title"], "source": doc["url"]}))
        return documents
    except Exception as error:
        print(error)

    return documents

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Câu hỏi truy vấn để tìm kiếm trên web")
    top_k: int = Field(5, description="Số lượng kết quả tối đa cần truy xuất")

web_search_tool = Tool.from_function(
    web_search,
    name="web_search",
    description="Thực hiện tìm kiếm trên web.",
    input_schema=WebSearchInput
)
