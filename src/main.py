from index_data import VectorStore
from chat import ask_llm_stream


def main():
    vector_store = VectorStore()
    vector_store.load_and_index_data(path="data")
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        context = vector_store.search(query=question, top_k=1)

        question += "\n"
        for text, score in context:
            question += str(text) + "\n"

        print(f"Response:")
        for chunk in ask_llm_stream(question):
            print(chunk, end='')

if __name__ == "__main__":
    main()