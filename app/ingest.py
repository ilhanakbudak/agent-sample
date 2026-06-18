"""Ingestion entrypoint — build the vector index from documents.

Run this ONCE when your documents change:

    python -m app.ingest sample_docs

The query/agent path then just opens the persisted store with load_retriever();
it never rebuilds. Separating ingestion (expensive, occasional) from serving
(cheap, per-request) is the core architectural split of a RAG system.
"""
from __future__ import annotations

import logging
import sys

from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_documents
from app.logging_config import setup_logging
from app.retrieval.embeddings import get_embedder
from app.retrieval.vector_store import build_index

logger = logging.getLogger(__name__)


def main() -> int:
    setup_logging()
    source = sys.argv[1] if len(sys.argv) > 1 else "sample_docs"
    chunks = chunk_documents(load_documents(source))
    build_index(chunks, get_embedder())  # reset=True wipes for a clean rebuild
    logger.info("Ingestion complete from '%s'.", source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())