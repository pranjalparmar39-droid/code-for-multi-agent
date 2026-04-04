"""
tools.py
--------
Defines the 4 tools available to the ReAct agent:
  1. Tavily Web Search
  2. Wikipedia Search
  3. Calculator
  4. PDF Reader
"""

import os
import math
from langchain_core.tools import Tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults

# ─────────────────────────────────────────────
# 1. TAVILY WEB SEARCH
# ─────────────────────────────────────────────
def get_tavily_tool() -> Tool:
    """Real-time web search using Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")

    def tavily_search(query: str) -> str:
        try:
            searcher = TavilySearchResults(max_results=3)
            result = searcher.invoke(query)
            if isinstance(result, list):
                return "\n\n".join([
                    f"**{r.get('title', 'Result')}**\n{r.get('content', '')}\nURL: {r.get('url', '')}"
                    for r in result
                ])
            return str(result)
        except Exception as e:
            return f"Web search error: {str(e)}"

    return Tool(
        name="web_search",
        func=tavily_search,
        description=(
            "Use this tool to search the internet for current, real-time information. "
            "Input should be a clear search query. "
            "Use this for recent events, news, facts you are unsure about, or anything time-sensitive."
        ),
    )


# ─────────────────────────────────────────────
# 2. WIKIPEDIA SEARCH
# ─────────────────────────────────────────────
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a summary."""
    try:
        import wikipedia as wiki
        wiki.set_lang("en")
        results = wiki.search(query, results=3)
        if not results:
            return "No Wikipedia results found for your query."
        # Try to get the best matching page
        for title in results:
            try:
                page = wiki.page(title, auto_suggest=False)
                summary = wiki.summary(title, sentences=5, auto_suggest=False)
                return f"**{page.title}**\n\n{summary}\n\nSource: {page.url}"
            except wiki.DisambiguationError as e:
                try:
                    page = wiki.page(e.options[0], auto_suggest=False)
                    summary = wiki.summary(e.options[0], sentences=5, auto_suggest=False)
                    return f"**{page.title}**\n\n{summary}\n\nSource: {page.url}"
                except Exception:
                    continue
            except Exception:
                continue
        return "Could not retrieve a Wikipedia article for this query."
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"


def get_wikipedia_tool() -> Tool:
    return Tool(
        name="wikipedia_search",
        func=wikipedia_search,
        description=(
            "Use this tool to look up factual, encyclopedic information from Wikipedia. "
            "Good for concepts, historical events, biographies, science topics, and definitions. "
            "Input should be a topic name or search phrase."
        ),
    )


# ─────────────────────────────────────────────
# 3. CALCULATOR
# ─────────────────────────────────────────────
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, log, sin, cos, tan, pi, e
    """
    try:
        # Build a safe namespace with math functions
        safe_globals = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "pow": pow,
            "factorial": math.factorial,
            "floor": math.floor,
            "ceil": math.ceil,
        }
        result = eval(expression.strip(), safe_globals)  # noqa: S307
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Calculation error: {str(e)}. Please provide a valid mathematical expression."


def get_calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        func=calculate,
        description=(
            "Use this tool to perform mathematical calculations. "
            "Input should be a valid mathematical expression as a string, e.g. '2 ** 10', 'sqrt(144)', '(15 * 3) / 2'. "
            "Supports: +, -, *, /, **, sqrt, log, log10, sin, cos, tan, pi, e, factorial, floor, ceil."
        ),
    )


# ─────────────────────────────────────────────
# 4. PDF READER
# ─────────────────────────────────────────────
def read_pdf(file_path: str) -> str:
    """Load and extract text from a PDF file."""
    try:
        file_path = file_path.strip().strip('"').strip("'")
        if not os.path.exists(file_path):
            return f"File not found: '{file_path}'. Please provide a valid path to a PDF file."
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        if not pages:
            return "The PDF appears to be empty or could not be read."
        # Return first ~3000 chars to stay within token limits
        full_text = "\n\n".join([p.page_content for p in pages])
        truncated = full_text[:3000]
        note = "\n\n[...Content truncated for length...]" if len(full_text) > 3000 else ""
        return f"PDF Content ({len(pages)} pages):\n\n{truncated}{note}"
    except Exception as e:
        return f"PDF read error: {str(e)}"


def get_pdf_tool() -> Tool:
    return Tool(
        name="pdf_reader",
        func=read_pdf,
        description=(
            "Use this tool to read and extract text content from a PDF file. "
            "Input should be the full file path to the PDF, e.g. '/path/to/document.pdf'. "
            "Use this when the user uploads or references a PDF document they want analyzed."
        ),
    )


# ─────────────────────────────────────────────
# COLLECT ALL TOOLS
# ─────────────────────────────────────────────
def get_all_tools() -> list:
    return [
        get_tavily_tool(),
        get_wikipedia_tool(),
        get_calculator_tool(),
        get_pdf_tool(),
    ]
