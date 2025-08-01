from langchain_ollama import OllamaLLM 


OLLAMA_BASE_URL = "http://localhost:11434"
model = OllamaLLM(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL, temperature=0, reasoning=False)


def ask_ollama(prompt):
    return model.invoke(prompt)

def ask_ollama_stream(prompt):
    for chunk in model.stream(prompt):
        yield chunk

def ask_llm(prompt):
    return ask_ollama(prompt)


def ask_llm_stream(prompt):
    return ask_ollama_stream(prompt)




if __name__ == "__main__":
    question = "Giới thiệu ngắn về mô hình ngôn ngữ lớn là gì?"
    for chunk in ask_llm(question):
        print(chunk, end='')

    # answer = ask_llm(question)
    # print(answer)

