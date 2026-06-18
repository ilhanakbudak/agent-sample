from app.logging_config import setup_logging
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.embeddings import get_embedder
from app.retrieval.vector_store import build_index, load_retriever

setup_logging()
embedder = get_embedder()                      # real model, via config
chunks = chunk_documents(load_documents("sample_docs"), 300, 50)
build_index(chunks, embedder)

retriever = load_retriever(embedder, k=2)
for h in retriever.invoke("how do we measure if two texts mean the same thing?"):
    print(h.metadata["chunk_id"], "->", h.page_content[:60])