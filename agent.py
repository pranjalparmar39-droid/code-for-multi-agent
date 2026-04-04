"""
agent.py
--------
ReAct agent using LangChain + Groq (clean & simplified)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from tools import get_all_tools

# Load env variables
load_dotenv()


# ─────────────────────────────────────────────
# PROMPT (SIMPLIFIED + HUMAN-LIKE)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# CREATE AGENT
# ─────────────────────────────────────────────
def create_agent_executor():

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing")

    # LLM
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "moonshotai/kimi-k2-instruct-0905"), #new
        temperature=0.3,
        groq_api_key=groq_api_key,
    )

    # Tools
    tools = get_all_tools()

    # Prompt
    prompt = PromptTemplate(
        input_variables=[
            "tools",
            "tool_names",
            "chat_history",
            "input",
            "agent_scratchpad",
        ],
        template=REACT_PROMPT_TEMPLATE,
    )

    # Agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Executor (simplified)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=5
    )

    return executor


# ─────────────────────────────────────────────
# RUN AGENT
# ─────────────────────────────────────────────
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