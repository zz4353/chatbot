from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_agent.agent import run_agentic_rag

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    return {"reply": run_agentic_rag(req.message)}
