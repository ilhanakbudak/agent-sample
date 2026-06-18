# Architecture

This document explains the components, the data flow, and — most importantly — **why**
each design decision was made. It is the companion to the README's high-level diagram.

See the [README](../README.md) for an overview and quickstart, or the
[Operator Handbook](OPERATOR_HANDBOOK.md) for step-by-step setup and operation.

---

## Design principles

1. **Separation of concerns** — ingestion, retrieval, generation, agent, and serving are
   independent modules with narrow interfaces, so any one can be swapped or tested alone.
2. **Provider-agnostic** — model/provider choices live behind factories (`get_llm`,
   `get_embedder`) and are selected by config, never hard-coded.
3. **Fail loud, fail typed** — every stage raises a specific exception carrying an HTTP
   status, so failures are diagnosable and map cleanly to API responses.
4. **Separate the expensive from the frequent** — indexing (expensive, occasional) is
   decoupled from querying (cheap, per-request).

---

## Components and responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Single typed source of runtime configuration (`pydantic-settings`). |
| `logging_config.py` | One `setup_logging()`; modules log via `getLogger(__name__)`. |
| `exceptions.py` | Typed error hierarchy; each error carries `http_status`. |
| `ingestion/loader.py` | Parse `.txt/.md/.pdf` into `Document`s with `source`/`page` metadata. |
| `ingestion/chunker.py` | Split into overlapping chunks; assign stable `chunk_id`. |
| `retrieval/embeddings.py` | Build a cloud or local embedder from config. |
| `retrieval/vector_store.py` | `build_index` (idempotent) and `load_retriever` over Chroma. |
| `llm.py` | Build a chat model (OpenAI/Anthropic/Ollama) from config. |
| `rag.py` | Fixed pipeline: retrieve → grounded prompt → cited answer. |
| `agents/tools.py` | `calculator`, `word_count`, and a retriever-as-tool factory. |
| `agents/graph.py` | LangGraph ReAct agent + grounding system prompt. |
| `api/server.py` | FastAPI app: lifespan, DI, exception handler, endpoints. |
| `mcp_server.py` | Exposes the tools over MCP (stdio). |

---

## Data flow

**Indexing (one-time):**
`documents → loader → chunker → embedder → Chroma (persisted to ./.vectorstore)`

**Query, fixed RAG (`rag.py`):**
`question → retriever → top-k chunks → grounded prompt → LLM → answer + sources`

**Query, agentic (`agents/graph.py`):**
`question → agent → (decides) → [search_documents | calculator | word_count | direct]`
`→ (tool results loop back) → LLM → grounded, cited answer`

---

## Key decisions and rationale

**Chunking: 800 chars / 120 overlap, recursive splitter.**
Large chunks dilute the embedding and waste context tokens; tiny chunks sever facts from
context. ~800 with ~15% overlap balances retrieval precision against context continuity.
The recursive splitter breaks on natural boundaries (paragraph→line→sentence→word).
*Trade-off:* character-based, not token-based — simpler, slightly less precise on budget.

**Embeddings: cloud default, local fallback. Same model both sides.**
Cloud embeddings are higher quality; the local `all-MiniLM-L6-v2` enables fully offline,
no-key operation (and exam resilience). The index and query path **must** use the same
embedder — different models produce incompatible vector spaces.

**Vector store: Chroma over FAISS.**
Chroma persists to disk, stores text+metadata beside vectors, and filters by metadata with
near-zero setup — ideal for a real, restart-surviving knowledge base. FAISS is faster and
lighter but treats persistence and metadata as manual concerns; it is the right pick when
raw speed/scale dominates. We chose convenience and persistence.

**Idempotent indexing.**
Chroma appends by default, so naive re-indexing accumulates duplicate/stale chunks (and
orphaned segment folders). `build_index(reset=True)` wipes the store directory for a clean
rebuild, and stable `chunk_id`s make re-adds upsert rather than duplicate.

**Grounded generation + refusal.**
The prompt instructs the model to answer only from retrieved context, cite `chunk_id`
markers, and refuse when the answer is absent. `temperature=0` favors faithful extraction
over creativity. This is what prevents confident hallucination.

**Agentic RAG with a forcing system prompt.**
Tools are capabilities; the system prompt is the policy. A permissive prompt let a small
model answer factual questions from memory and skip retrieval. The prompt now states the
agent has no reliable internal knowledge and **must** search for any factual question.
*Trade-off:* in-domain questions always retrieve (close to fixed RAG), while math, counts,
and small talk still route correctly or answer directly.

**Typed exceptions → HTTP status.**
Each domain error carries `http_status`; one FastAPI handler maps them centrally
(`IngestionError→422`, `RetrievalError/VectorStoreError→500`, `EmbeddingError/ToolError→502`),
removing per-route try/except and keeping responses consistent.

**API: lifespan + DI + non-blocking.**
The embedder and agent load once at startup (lifespan), are injected via `Depends`, and the
synchronous, network-bound `agent.invoke` runs via `asyncio.to_thread` so a slow request
never blocks the event loop.

**MCP exposure.**
Wrapping the same tools in an MCP server makes them reusable by any MCP host, decoupling
tool definitions from any single agent.

---

## Extension points

- Swap providers via `.env` (no code change).
- Add a tool: define it in `agents/tools.py`; the agent and MCP server pick it up.
- Add hybrid retrieval / reranking inside `retrieval/`.
- Add endpoints or middleware (auth, rate limiting, streaming) in `api/`.
