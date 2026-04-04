"""
app.py
------
Streamlit chat interface for the Multi-Agent Research Assistant.
Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import wikipedia
from agent import create_agent_executor, run_agent


load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant Made By PRANJAL",
   
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM UI Styling


st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Chat message styling */
    .user-msg {
        background: linear-gradient(135deg, #1e3a5f, #1a2d4a);
        border-left: 3px solid #4da6ff;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e8f4ff;
    }
    .assistant-msg {
        background: linear-gradient(135deg, #1a2a1a, #1e3a1e);
        border-left: 3px solid #4dff88;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e8ffe8;
    }

    /* Tool use expander */
    .tool-step {
        background: #1c1c2e;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 10px;
        margin: 4px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85em;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #161b22; }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #1c2333;
        color: white;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  Pranjal's AI Research Assistant")
    st.markdown("---")
    
    
    #_____________ API Keys___________________________
    # st.markdown("### 🔑 API Keys")
    # groq_key = st.text_input(
    #     "Groq API Key",
    #     type="password",
    #     value=os.getenv("GROQ_API_KEY", ""),
    #     placeholder="gsk_...",
    # )
    # tavily_key = st.text_input(
    #     "Tavily API Key",
    #     type="password",
    #     value=os.getenv("TAVILY_API_KEY", ""),
    #     placeholder="tvly-...",
    # )

    # if groq_key:
    #     os.environ["GROQ_API_KEY"] = groq_key
    # if tavily_key:
    #     os.environ["TAVILY_API_KEY"] = tavily_key

#_______________________________________________________________
   
    st.markdown("---")
    st.markdown("### 🛠️ Available Tools")
    tools_info = {
        "🌐 Web Search": "Tavily — real-time internet search",
        "📖 Wikipedia": "Encyclopedic knowledge lookup",
        "🧮 Calculator": "Math expressions & formulas",
        "📄 PDF Reader": "Extract text from PDF files",
    }
    for tool, desc in tools_info.items():
        st.markdown(f"**{tool}**  \n{desc}")

    st.markdown("---")
    st.markdown("### 📄 Upload PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF to analyze", type=["pdf"])
    pdf_path = None
    if uploaded_pdf:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name
        st.success(f"PDF loaded: `{uploaded_pdf.name}`")
        st.info(f"Ask the agent to read it using path:\n`{pdf_path}`")

    st.markdown("---")
    # if st.button("🗑️ Clear Conversation", use_container_width=True):
    #     st.session_state.messages = []
    #     st.session_state.agent_executor = None
    #     st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    examples = [
        "What is the latest news about AI today?",
        "Explain Transformers architecture",
        "What is  sqrt(144)?",
        
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.pending_query = ex


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
def get_agent():
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = create_agent_executor()
    return st.session_state.agent_executor

# if "agent_executor" not in st.session_state:
#     st.session_state.agent_executor = None

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# # ─────────────────────────────────────────────
# # INITIALIZE AGENT 
# # ─────────────────────────────────────────────
if st.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.session_state.pop("agent_executor", None)
    st.rerun()

# ─────────────────────────────────────────────
# MAIN CHAT UI
# ─────────────────────────────────────────────
st.markdown("#   AI Research Assistant")
st.markdown("Powered by **Groq (Llama 3)** · **LangChain ReAct** · 4 Intelligent Tools")
st.markdown("---")

# Render existing messages
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    steps = msg.get("steps", [])

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            # Show tool use steps
            if steps:
                with st.expander(f"🔍 Agent Reasoning ({len(steps)} step(s))", expanded=False):
                    for i, (action, observation) in enumerate(steps, 1):
                        tool_name = getattr(action, "tool", "unknown")
                        tool_input = getattr(action, "tool_input", "")
                        st.markdown(f"""
**Step {i} — Tool: `{tool_name}`**
- **Input:** `{tool_input}`
- **Result:** {str(observation)[:500]}{"..." if len(str(observation)) > 500 else ""}
""")

# Handle example button click
if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    user_input = st.chat_input("Ask anything...")

# Process input
if user_input:
    # Show user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get agent response
    agent = get_agent()
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking ..."):
            result = run_agent(agent, user_input)

        output = result["output"]
        steps = result["intermediate_steps"]

        st.markdown(output)

        if steps:
            with st.expander(f"🔍 Agent Reasoning ({len(steps)} step(s))", expanded=True):
                for i, (action, observation) in enumerate(steps, 1):
                    tool_name = getattr(action, "tool", "unknown")
                    tool_input = getattr(action, "tool_input", "")

                    # Tool badge color
                    color_map = {
                        "web_search": "🌐",
                        "wikipedia_search": "📖",
                        "calculator": "🧮",
                        "pdf_reader": "📄",
                    }
                    icon = color_map.get(tool_name, "🔧")

                    st.markdown(f"""
**{icon} Step {i} — `{tool_name}`**
- **Input:** `{tool_input}`
- **Result:** {str(observation)[:600]}{"..." if len(str(observation)) > 600 else ""}
---""")

    st.session_state.messages.append({
        "role": "assistant",
        "content": output,
        "steps": steps,
    })
