from langchain_core.messages import HumanMessage
from app.logging_config import setup_logging
from app.retrieval.embeddings import get_embedder
from app.retrieval.vector_store import load_retriever
from app.agents.graph import build_agent

setup_logging()
retriever = load_retriever(get_embedder())   # loads the existing store
agent = build_agent(retriever=retriever)

def ask(q):
    out = agent.invoke({"messages": [HumanMessage(q)]})
    used = [c["name"] for m in out["messages"]
            if getattr(m, "tool_calls", None) for c in m.tool_calls]
    print(f"\nQ: {q}\n  tools used: {used or 'none'}\n  A: {out['messages'][-1].content[:300]}")

# --- the battery: each line tests a different routing decision ---
ask("What is an embedding?")                       # in-domain  -> search_documents, grounded + cite
ask("When would I use FAISS instead of Chroma?")   # in-domain  -> search_documents
ask("What is the capital of France?")              # OUT-of-domain -> search, find nothing, REFUSE (not 'Paris')
ask("What is 348 * 17?")                            # math       -> calculator
ask("How many words are in: the cat sat on the mat?")  # count  -> word_count
ask("Hello, who are you?")                         # smalltalk  -> none