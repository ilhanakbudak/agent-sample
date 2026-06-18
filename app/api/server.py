"""FastAPI application — serves the RAG agent over HTTP.

Best practices demonstrated here (call them out in the interview):
  * lifespan: load the expensive embedder/agent ONCE at startup, not per request
  * dependency injection (Depends): hand endpoints a ready agent, fail clearly if absent
  * pydantic schemas: automatic request validation + auto OpenAPI docs at /docs
  * a single exception handler mapping our typed AppError -> the right HTTP status
  * non-blocking: run the synchronous, network-bound agent call off the event loop
  * separation: /ingest (build the index) is distinct from /query (use it)
"""
from __future__ import annotations

import asyncio
import logging
import re
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from app.agents.graph import build_agent
from app.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.exceptions import AppError, ConfigError, VectorStoreError
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_documents
from app.logging_config import setup_logging
from app.retrieval.embeddings import get_embedder
from app.retrieval.vector_store import build_index, load_retriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown. Load heavy resources once; degrade gracefully if absent."""
    setup_logging()
    app.state.embedder = None
    app.state.agent = None
    try:
        app.state.embedder = get_embedder()  # expensive: load the model once
    except Exception as exc:
        logger.warning("Embedder not loaded at startup: %s", exc)

    if app.state.embedder is not None:
        try:
            retriever = load_retriever(app.state.embedder)
            app.state.agent = build_agent(retriever=retriever)
            logger.info("Existing index found; agent is ready.")
        except Exception as exc:
            logger.warning("No index yet (%s). POST /ingest to build one.", exc)

    yield
    logger.info("Shutting down.")


app = FastAPI(title="GenAI RAG Agent", version="0.1.0", lifespan=lifespan)


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Map every typed domain error to its HTTP status in one place."""
    logger.error("%s: %s", type(exc).__name__, exc.message)
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": exc.message, "detail": exc.detail},
    )


def get_agent(request: Request):
    """DI: provide the ready agent, or fail with a clear error if not built."""
    if request.app.state.agent is None:
        raise VectorStoreError("No index built yet.", detail="POST /ingest first.")
    return request.app.state.agent


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", index_ready=request.app.state.agent is not None)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request) -> IngestResponse:
    if request.app.state.embedder is None:
        raise ConfigError("Embedder unavailable.", detail="Check embedding config/keys.")
    # load+chunk are light; build_index (embedding) is the heavy part.
    chunks = chunk_documents(load_documents(req.source))
    await asyncio.to_thread(build_index, chunks, request.app.state.embedder)
    retriever = load_retriever(request.app.state.embedder)
    request.app.state.agent = build_agent(retriever=retriever)
    return IngestResponse(chunks_indexed=len(chunks), source=req.source)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, agent=Depends(get_agent)) -> QueryResponse:
    # agent.invoke is synchronous and network-bound; run it off the event loop
    # so one slow request doesn't block the whole server.
    out = await asyncio.to_thread(agent.invoke, {"messages": [HumanMessage(req.question)]})
    final = out["messages"][-1].content
    tools_used = [
        c["name"]
        for m in out["messages"]
        if getattr(m, "tool_calls", None)
        for c in m.tool_calls
    ]
    sources = re.findall(r"\[([^\]]+)\]", final)  # chunk_ids the model cited
    return QueryResponse(answer=final, sources=sources, tools_used=tools_used)