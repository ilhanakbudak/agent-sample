from app.logging_config import setup_logging
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.embeddings import get_embedder
from app.retrieval.vector_store import build_index, load_retriever
from app.llm import get_llm
from app.rag import answer_question

setup_logging()
embedder = get_embedder()
build_index(chunk_documents(load_documents("sample_docs")), embedder)
retriever = load_retriever(embedder)
llm = get_llm()

print(answer_question("When would I use FAISS instead of Chroma?", retriever, llm))

#print(answer_question("What is the capital of France?", retriever, llm))