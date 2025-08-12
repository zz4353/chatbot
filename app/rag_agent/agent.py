import os
from langchain.agents import initialize_agent, AgentType
from app.llm.llm_integrations import get_llm
from app.rag_agent.tools.rag_search import stock_rag_search_tool
from app.rag_agent.tools.web_search import web_search_tool
from app.rag_agent._utils import render_prompt

tools = [stock_rag_search_tool, web_search_tool]

agent = initialize_agent(
    tools=tools,
    llm=get_llm(),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

def ask_agentic_rag(prompt):
    try:
        prompt = render_prompt(os.path.join(os.path.realpath(__file__), "../prompts/agent_prompt.txt"), prompt)
        response = agent.invoke({"input": prompt})
        return response["output"]
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Xin lỗi, đã xảy ra lỗi trong quá trình xử lý yêu cầu của bạn."

