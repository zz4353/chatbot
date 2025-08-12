import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

load_dotenv()
LLM_TYPE = os.getenv("LLM_TYPE", "openai")

def init_openai_chat(temperature):
    return ChatOpenAI(
        model=os.getenv("CHAT_MODEL"),
        streaming=True,
        temperature=temperature,
    )

def init_ollama_chat(temperature):
    return OllamaLLM(
        model=os.getenv("CHAT_MODEL"),
        streaming=True,
        temperature=temperature,
    )

MAP_LLM_TYPE_TO_CHAT_MODEL = {
    "openai": init_openai_chat,
    "ollama": init_ollama_chat,
}

def get_llm(temperature=0):
    if LLM_TYPE not in MAP_LLM_TYPE_TO_CHAT_MODEL:
        raise Exception(
            "LLM type not found. Please set LLM_TYPE to one of: "
            + ", ".join(MAP_LLM_TYPE_TO_CHAT_MODEL.keys())
            + "."
        )

    return MAP_LLM_TYPE_TO_CHAT_MODEL[LLM_TYPE](temperature=temperature)