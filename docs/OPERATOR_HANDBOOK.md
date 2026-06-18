# Operator Handbook

A step-by-step runbook to bring the system up **one piece at a time**, verifying each
stage before moving on. Each step lists the command, the expected result, and what it
proves. If a step fails, see **Troubleshooting** at the bottom.

See the [README](../README.md) for an overview and quickstart, and
[Architecture](ARCHITECTURE.md) for component responsibilities and design decisions.

---

## Prerequisites

- Python 3.10+
- One LLM access path: an OpenAI/Anthropic API key, **or** a local Ollama install.
- ~200 MB free disk if using the local embedding model.

---

## Step 0 — Environment

**Lightning AI Studio / Google Colab** (single shared environment — `venv` is blocked):
```bash
pip install -r requirements.txt
```

**Local machine** (isolate first):
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

*Verifies:* dependencies install into a reusable environment.

---

## Step 1 — Configure

```bash
cp .env.example .env
```
Edit `.env`. Choose one profile:

- **Cloud:** `EMBEDDING_PROVIDER=openai`, `LLM_PROVIDER=openai`, `OPENAI_API_KEY=sk-...`
- **Offline:** `EMBEDDING_PROVIDER=sentence-transformers`, `LLM_PROVIDER=ollama` (Ollama running)
- **Mixed (no key, cheap):** `EMBEDDING_PROVIDER=sentence-transformers`, `LLM_PROVIDER=openai`

*Verifies:* configuration is in place and secrets stay out of source control.

---

## Step 2 — Smoke-test the skeleton

```bash
python -m app.main
```
*Expected:* `Skeleton is alive.` plus the configured provider/model lines.
*Proves:* config loads, logging works, the package imports cleanly.

---

## Step 3 — Build the index (ingestion)

```bash
python -m app.ingest sample_docs
```
*Expected:* logs ending in `Indexed N chunk(s) into Chroma at './.vectorstore'`.
*Proves:* loading → chunking → embedding → persistence works end to end.
*Note:* re-running wipes and rebuilds cleanly (no duplicate/stale chunks). Run this only when documents change.

---

## Step 4 — Query via the RAG pipeline (no agent)

```python
from app.logging_config import setup_logging
from app.retrieval.embeddings import get_embedder
from app.retrieval.vector_store import load_retriever
from app.llm import get_llm
from app.rag import answer_question

setup_logging()
emb = get_embedder()
retriever = load_retriever(emb)
print(answer_question("What is an embedding?", retriever, get_llm()))
```
*Expected:* a `RagAnswer` whose answer is grounded in the docs, with `sources` chunk_ids.
*Proves:* retrieval + grounded generation + citations work.
*Check the refusal path:* ask "What is the capital of France?" → it must reply *"I don't know based on the provided documents."*

---

## Step 5 — Run the agent (agentic RAG)

```python
from langchain_core.messages import HumanMessage
from app.logging_config import setup_logging
from app.retrieval.embeddings import get_embedder
from app.retrieval.vector_store import load_retriever
from app.agents.graph import build_agent

setup_logging()
agent = build_agent(retriever=load_retriever(get_embedder()))

def ask(q):
    out = agent.invoke({"messages": [HumanMessage(q)]})
    used = [c["name"] for m in out["messages"]
            if getattr(m, "tool_calls", None) for c in m.tool_calls]
    print(q, "->", used or "none", "|", out["messages"][-1].content[:120])

ask("What is an embedding?")        # -> search_documents (grounded)
ask("What is 348 * 17?")            # -> calculator
ask("Hello")                        # -> none
```
*Proves:* the agent routes correctly and grounds factual answers in the corpus.

---

## Step 6 — Run the HTTP API

```bash
uvicorn app.api.server:app --port 8000
```
In another terminal:
```bash
curl localhost:8000/health
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"question":"What is an embedding?"}'
```
*Expected:* `/health` → `{"status":"ok","index_ready":true}`; `/query` → JSON answer with sources.
Open `http://localhost:8000/docs` for the interactive UI.
*Proves:* the system is served correctly with validation, DI, and error mapping.

> If `index_ready` is `false`, call `POST /ingest {"source":"sample_docs"}` (or run Step 3) first.

---

## Step 7 — Run the MCP server

The server speaks **stdio** and is launched by a client; run alone it waits silently
(that is correct). Verify with the bundled client pattern:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["-m", "app.mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])
            print(await session.call_tool("calculator", {"expression": "21*2"}))

asyncio.run(main())
```
*Expected:* `tools: ['calculator', 'word_count', 'search_documents']` and a result of `42`.
*Proves:* tools are reusable by any MCP-compatible host.

---

## Routine operations

- **Documents changed?** Re-run Step 3 (`python -m app.ingest <path>`), then restart the API.
- **Switch LLM/embeddings?** Edit `.env` only — no code changes.
- **Go fully offline?** Set the offline profile (Step 1) and ensure Ollama is running.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Venv creation is not allowed` | Lightning Studio allows one env | Skip venv; `pip install` into the existing env (Step 0) |
| `Cannot uninstall PyJWT ... RECORD file not found` | system-managed package | `pip install mcp --ignore-installed PyJWT` |
| HF Hub "unauthenticated requests" warning | no HF token | harmless; optionally set `HF_TOKEN` for faster downloads |
| Answers ignore the documents / aren't grounded | weak/missing agent system prompt | ensure `DEFAULT_SYSTEM_PROMPT` is active in `app/agents/graph.py` |
| Duplicate/stale chunks; many store folders | non-idempotent indexing | fixed via `reset=True`; `rm -rf .vectorstore` once, then re-ingest |
| `/query` returns 500 "No index built yet" | no index | run Step 3 or `POST /ingest` |
| `/ingest` returns 422 | bad `source` path or unreadable file | check the path; scanned PDFs need OCR |
| `ConfigError: ... API_KEY is not set` | missing key for chosen provider | set the key in `.env` or switch provider |
| MCP server "does nothing" when run alone | stdio server waits for a client | normal — launch it from a client (Step 7) |
