"""Tools the agent can call.

A "tool" is a plain Python function the LLM may choose to invoke. The model
never executes code itself — it emits a structured request (e.g. call
`calculator` with expression="12*9") and OUR code runs the function and feeds
the result back.

The @tool decorator turns a function + its type hints + its docstring into a
schema the model reads. That docstring is the model's ONLY instruction on when
and how to use the tool, so treat it as prompt engineering.
"""
from __future__ import annotations

import ast
import logging
import operator
from typing import Callable

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Whitelisted operators for a *safe* arithmetic evaluator (never use eval()).
_OPERATORS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the exact result.

    Use this for ALL arithmetic instead of computing it yourself. Supports
    + - * / ** % and parentheses. Example: "4825 * (391 + 2)".
    """
    try:
        result = _eval_node(ast.parse(expression, mode="eval").body)
        return str(result)
    except Exception as exc:  # return the error so the model can recover
        logger.warning("calculator failed on %r: %s", expression, exc)
        return f"ERROR: could not evaluate '{expression}' ({exc})"


@tool
def word_count(text: str) -> str:
    """Count the number of words in the given text."""
    return str(len(text.split()))


# Registry the agent loop uses to dispatch a tool call by name.
TOOLS = [calculator, word_count]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def make_search_tool(retriever):
    """Wrap a RAG retriever as a tool the agent can choose to call.

    This is what turns a fixed RAG pipeline into *agentic* RAG: the model
    decides whether a question needs a document search at all, instead of
    every query being forced through retrieval.
    """

    @tool
    def search_documents(query: str) -> str:
        """Search the indexed knowledge base for passages relevant to a query.
        Use this for any question about the document collection's contents."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(
            f"[{d.metadata.get('chunk_id', '?')}] {d.page_content}" for d in docs
        )

    return search_documents