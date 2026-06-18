# Retrieval-Augmented Generation

RAG combines a retriever with a generator. The retriever finds relevant text
from a corpus, and the generator (an LLM) produces an answer grounded in that
text. This reduces hallucination and lets the model use private or recent data
it was never trained on.

## Embeddings

An embedding is a vector that captures the meaning of a piece of text. Texts
with similar meaning have vectors that are close together. We measure closeness
with cosine similarity. Embeddings are what make semantic search possible.

## Vector Stores

A vector store indexes embeddings and answers nearest-neighbour queries quickly.
Chroma is simple and persistent. FAISS is fast and runs in memory. The right
choice depends on scale, persistence needs, and deployment constraints.
