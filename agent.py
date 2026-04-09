import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.prompts import PromptTemplate

from tools import get_all_tools

load_dotenv()

REACT_PROMPT_TEMPLATE = """
You are an AI assistant built by Pranjal.

You can use tools to answer questions.

Available tools:
{tools}

Tool names:
{tool_names}

Instructions:
- Think step by step
- Use tools only when needed
- Keep answers clear

Format:

Question: {input}

Thought: what should I do?
Action: one of [{tool_names}]
Action Input: input for the tool
Observation: result from tool

... (repeat if needed)

Final Answer: give final answer

{agent_scratchpad}
"""


def create_agent_executor():

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing")

    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.3,
        groq_api_key=groq_api_key,
    )

    tools = get_all_tools()

    prompt = PromptTemplate(
        input_variables=[
            "tools",
            "tool_names",
            "input",
            "agent_scratchpad",
        ],
        template=REACT_PROMPT_TEMPLATE,
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=5,
        handle_parsing_errors=True
    )

    return executor


def run_agent(executor, query):
    try:
        result = executor.invoke({"input": query})

        return {
            "output": result.get("output", "No output"),
            "intermediate_steps": result.get("intermediate_steps", []),
            "error": None,
        }

    except Exception as e:
        return {
            "output": f"❌ Error: {str(e)}",
            "intermediate_steps": [],
            "error": str(e),
        }
