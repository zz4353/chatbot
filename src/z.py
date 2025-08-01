import pandas as pd
from index_data import Vector_Store
from chat import ask_llm
from utils import render_prompt


def main():
    vector_store = Vector_Store()
    vector_store.index_data(path="data")

    # đọc file xlxs
    df = pd.read_excel("data/test_question.xlsx")

    # duyệt từng dòng
    answers = []
    for _, row in df.iterrows():
        question = row['inputs']
        docs = vector_store.search(query=question, top_k=5)
        docs = [doc for doc in docs if doc.metadata["score"] >= 0.1]
        prompt = render_prompt("templates/prompt.txt", docs, question=question)
        answer = ask_llm(prompt)
        answer = answer.replace("\n", " ")
        answers.append(answer)

        print("=====================================")
        print(f"Question: {question}")
        print(answer)

    df['answer'] = answers 

    # ghi vào file csv
    df.to_csv("answers.csv", index=False)

if __name__ == "__main__":
    main()

