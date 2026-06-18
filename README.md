# GenAI RAG Agent — Starter

A modular, provider-agnostic **Retrieval-Augmented Generation (RAG)** system with an
**agentic** layer, served over a **REST API** and exposed via the **Model Context
Protocol (MCP)**. Built in Python with clean separation of concerns, typed
configuration, centralized logging, and typed error handling.

---

## Overview

This system answers questions grounded in a private document collection. It ingests
documents, retrieves the most relevant passages with semantic search, and uses an LLM
to generate answers **cited back to their sources**. An agent layer decides, per
question, whether to search the documents, do arithmetic, count words, or answer
directly. The same capabilities are exposed three ways: a Python API, a FastAPI HTTP
service, and an MCP server.

## Key features

- **Ingestion** — `.txt` / `.md` / `.pdf` loading, recursive chunking with overlap, per-chunk metadata (`source`, `page`, `chunk_id`).
- **Retrieval** — embeddings + a persistent Chroma vector store; idempotent indexing (no duplicate/stale chunks).
- **Grounded generation** — answers use *only* retrieved context, cite `chunk_id` markers, and refuse ("I don't know based on the provided documents") when the answer isn't present.
- **Agentic RAG** — a LangGraph ReAct agent routes between tools (`search_documents`, `calculator`, `word_count`) and is steered by a system prompt that forces retrieval-grounding for factual questions.
- **Provider-agnostic** — OpenAI / Anthropic / Ollama LLMs; OpenAI / local `sentence-transformers` embeddings. Runs cloud or **fully offline**.
- **Interfaces** — FastAPI (`/health`, `/ingest`, `/query`) and an MCP stdio server.
- **Engineering hygiene** — `pydantic-settings` config, centralized logging, a typed exception hierarchy mapped to HTTP status codes.

## Architecture (high level)

```
INDEXING (run once, when documents change)
  documents ──▶ loader ──▶ chunker ──▶ embedder ──▶ Chroma (./.vectorstore)

QUERY (per request)
  question ──▶ Agent (LangGraph ReAct)
                 │ decides which tool(s)
                 ├─▶ search_documents ─▶ retriever ─▶ Chroma ─▶ chunks ─┐
                 ├─▶ calculator                                         │
                 ├─▶ word_count                                         │
                 └─▶ (answer directly)                                  │
                          ▼                                             │
                        LLM ◀──── grounded prompt + retrieved context ◀─┘
                          ▼
                    answer + cited sources
```

See **[Architecture](docs/ARCHITECTURE.md)** for component responsibilities and design rationale.

## Project structure

```
genai-starter/
├── app/
│   ├── config.py            # typed settings (pydantic-settings)
│   ├── logging_config.py    # centralized logging
│   ├── exceptions.py        # typed error hierarchy (carry http_status)
│   ├── llm.py               # provider-agnostic chat LLM factory
│   ├── rag.py               # fixed RAG pipeline (retrieve -> grounded answer)
│   ├── ingest.py            # ingestion entrypoint (build the index)
│   ├── main.py              # skeleton smoke test
│   ├── ingestion/
│   │   ├── loader.py        # txt/md/pdf -> Documents (+ metadata)
│   │   └── chunker.py       # split into overlapping chunks (+ chunk_id)
│   ├── retrieval/
│   │   ├── embeddings.py    # cloud or local embedder (provider-agnostic)
│   │   └── vector_store.py  # Chroma build_index / load_retriever
│   ├── agents/
│   │   ├── tools.py         # calculator, word_count, retriever-as-tool
│   │   └── graph.py         # LangGraph ReAct agent + system prompt
│   ├── api/
│   │   ├── schemas.py       # pydantic request/response models
│   │   └── server.py        # FastAPI app (lifespan, DI, error handler)
│   └── mcp_server.py        # MCP stdio server exposing the tools
├── docs/
│   ├── OPERATOR_HANDBOOK.md # step-by-step: make every piece functional
│   └── ARCHITECTURE.md      # components + design decisions + rationale
├── sample_docs/             # example corpus
├── requirements.txt
├── .env.example
└── README.md
```

## Tech stack and why

| Layer | Choice | Why |
|---|---|---|
| LLM | OpenAI `gpt-4o-mini` (default); Anthropic / Ollama | Cheap, capable, swappable; Ollama gives a fully offline fallback |
| Embeddings | OpenAI `text-embedding-3-small`; local `all-MiniLM-L6-v2` | Cloud quality by default, local model for offline / no-key runs |
| Vector store | Chroma | Persists to disk, stores text+metadata, easy metadata filtering, minimal setup |
| Chunking | `RecursiveCharacterTextSplitter` | Splits on natural boundaries (paragraph→line→sentence→word) |
| Agent framework | LangGraph (+ LangChain core) | 2026 standard for stateful, cyclic tool-using agents |
| API | FastAPI + uvicorn | Async, pydantic validation, auto OpenAPI docs |
| Config | pydantic-settings | Type-safe config from `.env`, fails loudly on bad values |
| Tool sharing | MCP (`mcp` SDK) | Open standard; tools become reusable by any MCP client |

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env            # fill in keys (or use the offline profile below)
python -m app.ingest sample_docs
uvicorn app.api.server:app --port 8000
# then open http://localhost:8000/docs
```

## Configuration

All config is read from `.env` (see `.env.example`). Key variables:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | model id for the chosen provider |
| `OPENAI_API_KEY` | — | required if using OpenAI |
| `EMBEDDING_PROVIDER` | `openai` | `openai` \| `sentence-transformers` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | embedding model id |
| `VECTOR_STORE_DIR` | `./.vectorstore` | Chroma persistence directory |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | chunking parameters |
| `TOP_K` | `4` | passages retrieved per query |
| `LOG_LEVEL` | `INFO` | logging verbosity |

**Offline profile** (no keys, nothing leaves the machine):
`EMBEDDING_PROVIDER=sentence-transformers`, `LLM_PROVIDER=ollama`.

## Usage

```bash
# 1) Build the index (run when documents change)
python -m app.ingest sample_docs

# 2a) Ask via HTTP
uvicorn app.api.server:app --port 8000
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"question":"What is an embedding?"}'

# 2b) Ask via the MCP server (launched by an MCP client; see handbook)
python -m app.mcp_server
```

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status, index_ready}` |
| POST | `/ingest` | `{source}` | `{chunks_indexed, source}` |
| POST | `/query` | `{question}` | `{answer, sources, tools_used}` |

Interactive docs: `http://localhost:8000/docs`.

## Data sent to external services

Be explicit about what leaves the machine (this depends on configured providers):

- **`EMBEDDING_PROVIDER=openai`** — document text (at ingest time) and your query text (at query time) are sent to OpenAI's embeddings API.
- **`LLM_PROVIDER=openai` / `anthropic`** — your question plus the retrieved document passages are sent to that provider's chat API to generate the answer.
- **`EMBEDDING_PROVIDER=sentence-transformers` + `LLM_PROVIDER=ollama`** — **nothing leaves the machine**; embeddings and generation run locally.

API keys are read from `.env` and must **never** be committed (`.env` is git-ignored).

## Documentation

- 📐 [Architecture](docs/ARCHITECTURE.md) — components, data flow, and design rationale.
- 📖 [Operator Handbook](docs/OPERATOR_HANDBOOK.md) — step-by-step setup and troubleshooting.

## Limitations, assumptions, and improvements

**Assumptions**
- Source documents contain extractable text (scanned PDFs would need OCR).
- A single document corpus per vector store directory.
- On Lightning AI, one Studio = one shared environment (no per-project venv).

**Limitations**
- Retrieval is dense-only (no hybrid/keyword or reranking yet).
- The agent forces retrieval for any factual question, so it behaves close to fixed RAG for in-domain queries (a deliberate grounding trade-off).
- No authentication or rate limiting on the API.
- Chunking is character-based, not token-based.

**Improvements (next steps)**
- Hybrid retrieval (dense + BM25) and a cross-encoder reranker.
- Token-aware chunking; per-source metadata filtering at query time.
- Streaming responses (SSE) for token-by-token output.
- Auth, rate limiting, request-id middleware; background-task ingestion.
- Evaluation harness (e.g., Ragas) for retrieval/answer quality.
