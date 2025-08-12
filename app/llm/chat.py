import os
from app.llm.llm_integrations import get_llm
from app.llm._utils import render_prompt

def ask_llm(prompt):
    response = get_llm().invoke(prompt)
    if isinstance(response, str):
        return response
    return response.content

def ask_rag(prompt, documents, chat_history=[]):
    prompt = render_prompt(os.path.join(os.path.realpath(__file__),"..", "prompts", "prompt.txt"), documents, prompt, chat_history)
    return ask_llm(prompt)