"""
agent.py
--------
Core ReAct agent using LangChain + Groq (Llama3).
Includes ConversationBufferMemory for multi-turn context.
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from tools import get_all_tools

load_dotenv()


# ─────────────────────────────────────────────
# SYSTEM PROMPT  (ReAct style)
# ─────────────────────────────────────────────
REACT_PROMPT_TEMPLATE = """You are a helpful, intelligent research assistant with access to multiple tools.
You help users find information, perform calculations, search the web, look up Wikipedia, and read PDF documents.
Always reason carefully before acting, and provide detailed, accurate final answers.

You have access to the following tools:

{tools}

Use the following format STRICTLY — do not deviate:

Question: the input question you must answer
Thought: think step by step about what to do
Action: the action to take, must be exactly one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def create_agent_executor() -> AgentExecutor:
    """Build and return the ReAct AgentExecutor with memory."""

    # 1. LLM — Groq Llama3 (fast & free)
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.2,
        groq_api_key=os.getenv("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"),
    )

    # 2. Tools
    tools = get_all_tools()

    # 3. Prompt
    prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
        template=REACT_PROMPT_TEMPLATE,
    )

    # 4. ReAct Agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # 5. Memory — keeps full conversation history in session
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,   # Plain text for ReAct prompt
    )

    # 6. AgentExecutor — runs the Thought → Action → Observation loop
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,            # Prints reasoning to terminal
        handle_parsing_errors=True,
        max_iterations=8,        # Safety cap
        max_execution_time=60,   # Timeout in seconds
    )

    return executor


def run_agent(executor: AgentExecutor, user_query: str) -> dict:
    """
    Run the agent on a user query.
    Returns a dict with 'output' and 'intermediate_steps'.
    """
    try:
        result = executor.invoke({"input": user_query})
        return {
            "output": result.get("output", "No response generated."),
            "intermediate_steps": result.get("intermediate_steps", []),
            "error": None,
        }
    except Exception as e:
        return {
            "output": f" Agent encountered an error: {str(e)}",
            "intermediate_steps": [],
            "error": str(e),
        }
