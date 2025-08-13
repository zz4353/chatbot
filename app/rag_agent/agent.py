import os
from langchain.agents import initialize_agent, AgentType
from app.llm.llm_integrations import get_llm
from app.rag_agent.tools.rag_search import stock_rag_search_tool
from app.rag_agent.tools.web_search import web_search_tool
from app.rag_agent._utils import load_prompt_sections


def build_zero_shot_agent():
    tools = [stock_rag_search_tool, web_search_tool]
    prompt = load_prompt_sections(os.path.join(os.path.dirname(__file__), "prompts", "prompt_template.txt"))

    agent = initialize_agent(
        tools=tools,
        llm=get_llm(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={
            'prefix': prompt["PREFIX"],
            'format_instructions': prompt["FORMAT_INSTRUCTIONS"],
            'suffix': prompt["SUFFIX"],
        },
        max_iterations=(len(tools) + 4)
    )
    return agent

zero_shot_agent = build_zero_shot_agent()

def run_agentic_rag(prompt):
    try:
        response = zero_shot_agent.invoke({"input": prompt})
        return response["output"]
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Xin lỗi, đã xảy ra lỗi trong quá trình xử lý yêu cầu của bạn."
    

