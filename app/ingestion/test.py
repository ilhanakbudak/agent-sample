from app.logging_config import setup_logging
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents

setup_logging()
docs = load_documents("sample_docs")

for size in (150, 800):
    chunks = chunk_documents(docs, chunk_size=size, chunk_overlap=30)
    print(f"\nsize={size} -> {len(chunks)} chunks")
    print("first chunk:", repr(chunks[0].page_content[:80]))