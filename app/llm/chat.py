from app.llm.llm_integrations import get_llm

def ask_llm(prompt):
    llm = get_llm()
    return llm.invoke(prompt).content
