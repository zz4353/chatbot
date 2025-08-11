import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM 

load_dotenv()

model = OllamaLLM(model=os.getenv("OLLAMA_MODEL"), base_url=os.getenv("OLLAMA_BASE_URL"), temperature=0, reasoning=False)


def ask_ollama(prompt):
    return model.invoke(prompt)

def ask_ollama_stream(prompt):
    for chunk in model.stream(prompt):
        yield chunk

def ask_llm(prompt):
    return ask_ollama(prompt)


def ask_llm_stream(prompt):
    return ask_ollama_stream(prompt)


