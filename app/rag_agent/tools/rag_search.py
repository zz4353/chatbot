from pydantic import BaseModel, Field
from langchain.tools import Tool
from app.rag.retriever import search_stock_collection

class StockInput(BaseModel):
    Query: str = Field(..., description="Câu hỏi truy vấn liên quan đến chứng khoán")
    top_k: int = Field(5, description="Số lượng tài liệu liên quan tối đa cần truy xuất")
    threshold: float = Field(0.3, description="Ngưỡng độ tin cậy cho kết quả.")

stock_rag_search_tool = Tool.from_function(
    func=search_stock_collection,
    name="search_stock_data",
    description="Thực hiện tìm kiếm trong tập tài liệu chứng khoán.",
    input_schema=StockInput
)
