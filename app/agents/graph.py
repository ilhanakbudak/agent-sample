"""A minimal ReAct agent built explicitly with LangGraph.

The graph automates the loop you'd otherwise run by hand:

    START -> agent -> (tool_calls?) --yes--> tools -> agent -> ...
                          |
                          no
                          v
                         END

State is the conversation: a list of messages. The `add_messages` reducer makes
every node's return value APPEND to that list instead of replacing it.

For a one-line shortcut, `langgraph.prebuilt.create_react_agent(llm, tools)`
builds essentially this same graph — but we wire it explicitly here so the
mechanics (nodes, conditional edges, the loop) are visible and customisable.
"""
from __future__ import annotations

import logging
from typing import Annotated, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agents.tools import TOOLS, make_search_tool
from app.llm import get_llm

logger = logging.getLogger(__name__)

# The agent's POLICY. Tools are capabilities; this prompt governs behaviour —
# specifically, that domain questions must be grounded in the document store
# rather than answered from the model's own training.
DEFAULT_SYSTEM_PROMPT = (
    "You are a retrieval-grounded assistant. You have NO reliable knowledge of "
    "your own about any subject matter, so you must NOT answer informational "
    "questions from memory.\n\n"
    "Rules:\n"
    "1. For ANY request for information, a definition, an explanation, a fact, or "
    "a comparison (e.g. 'what is X', 'how does Y work', 'compare A and B'), you "
    "MUST call `search_documents` FIRST, then answer using ONLY the returned "
    "passages and cite their [chunk_id] markers.\n"
    "2. If the retrieved passages do not contain the answer, reply that you don't "
    "know based on the provided documents. Do NOT fall back on your own knowledge.\n"
    "3. Use `calculator` for arithmetic and `word_count` for word counts.\n"
    "4. Only for greetings or small talk may you respond without any tool."
)


class AgentState(TypedDict):
    """The shared state flowing through the graph: just the conversation."""
    messages: Annotated[list, add_messages]


def _should_continue(state: AgentState) -> str:
    """Conditional edge: did the model ask for a tool, or is it done?"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def build_agent(
    retriever=None,
    model: BaseChatModel | None = None,
    system_prompt: str | None = None,
):
    """Compile a ReAct agent.

    retriever:     if given, the agent gains a `search_documents` tool (agentic RAG).
    model:         inject a model for testing; defaults to get_llm() bound to tools.
    system_prompt: the agent's behaviour policy; defaults to DEFAULT_SYSTEM_PROMPT.
    """
    tools = list(TOOLS)
    if retriever is not None:
        tools.append(make_search_tool(retriever))
    tools_by_name = {t.name: t for t in tools}

    if model is None:
        model = get_llm().bind_tools(tools)

    system = SystemMessage(content=system_prompt or DEFAULT_SYSTEM_PROMPT)

    # --- node 1: the model decides (system prompt prepended every turn) ---
    def agent_node(state: AgentState) -> dict:
        return {"messages": [model.invoke([system] + state["messages"])]}

    # --- node 2: we execute whatever the model requested (the dispatch glue) ---
    def tool_node(state: AgentState) -> dict:
        outputs = []
        for call in state["messages"][-1].tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = f"ERROR: unknown tool '{call['name']}'"
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:  # never let a tool crash the loop
                    result = f"ERROR: {exc}"
            outputs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": outputs}

    # --- wire the graph ---
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")  # loop back after running tools

    compiled = graph.compile()
    logger.info("Agent compiled with tools: %s", list(tools_by_name))
    return compiled