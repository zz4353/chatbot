from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from app.llm.llm_integrations import get_llm
from app.rag_agent.tools.rag_search import stock_rag_search_tool
from app.rag_agent.tools.web_search import web_search_tool


load_dotenv()
tools = [stock_rag_search_tool, web_search_tool]

agent = initialize_agent(
    tools=tools,
    llm=get_llm(),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "system_message": "Bạn là một trợ lý chuyên trả lời về tài chính và chứng khoán Việt Nam. Luôn trả lời ngắn gọn, chính xác, có dẫn nguồn nếu có."
    },
)


def chatbot_agentic_rag():
    print("Agentic RAG Chatbot is running! Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Chatbot session ended.")
            break
        try:
            response = agent.run(user_query)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")