"""Minimal MCP server exposing this project's tools over the Model Context Protocol.

MCP lets tools live in a standalone server that ANY MCP-compatible client
(Claude Desktop, IDEs, other agents) can connect to and call — instead of the
tools being locked inside one app.

Run it:
    python -m app.mcp_server                      # stdio transport (local subprocess)

For a remote server, call mcp.run(transport="streamable-http") instead.

FastMCP turns a decorated function + its docstring into an MCP tool, exactly
like @tool did for LangChain — the docstring is again the model's interface.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("genai-starter-tools")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the exact result.
    Supports + - * / ** % and parentheses, e.g. "4825 * (391 + 2)"."""
    from app.agents.tools import calculator as _calc  # reuse the safe evaluator
    return _calc.invoke({"expression": expression})


@mcp.tool()
def word_count(text: str) -> str:
    """Count the number of words in the given text."""
    return str(len(text.split()))


@mcp.tool()
def search_documents(query: str) -> str:
    """Search the indexed knowledge base for passages relevant to a query.

    Requires that the index has been built first (python -m app.ingest ...).
    The retriever is loaded lazily so the server starts even with no index.
    """
    from app.retrieval.embeddings import get_embedder
    from app.retrieval.vector_store import load_retriever

    retriever = load_retriever(get_embedder())
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        f"[{d.metadata.get('chunk_id', '?')}] {d.page_content}" for d in docs
    )


if __name__ == "__main__":
    mcp.run()  # stdio by default