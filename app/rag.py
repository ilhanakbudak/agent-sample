"""RAG pipeline.

Composes the pieces into a question-answering function:

    question -> retrieve top-k chunks -> build grounded prompt -> LLM -> answer

The prompt is the control system: it forces the model to use ONLY the retrieved
context, to refuse ("I don't know...") when the context lacks the answer, and to
cite the [chunk_id] markers it relied on. That discipline is what keeps a RAG
system honest instead of fluently wrong.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from app.exceptions import GenerationError

logger = logging.getLogger(__name__)

_REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question using ONLY the "
    "context provided below. If the answer is not contained in the context, "
    f'reply exactly: "{_REFUSAL}". Do not rely on outside knowledge. '
    "Cite the sources you used with their [chunk_id] markers.\n\n"
    "Context:\n{context}"
)


@dataclass
class RagAnswer:
    """Structured result: the text answer plus the chunk_ids it drew from."""
    answer: str
    sources: list[str] = field(default_factory=list)


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks into a labelled context block for the prompt."""
    return "\n\n".join(
        f"[{d.metadata.get('chunk_id', 'unknown')}]\n{d.page_content}" for d in docs
    )


def answer_question(
    question: str,
    retriever: VectorStoreRetriever,
    llm: BaseChatModel,
) -> RagAnswer:
    """Run retrieval + grounded generation for a single question."""
    docs = retriever.invoke(question)
    if not docs:
        logger.info("No chunks retrieved; returning refusal.")
        return RagAnswer(answer=_REFUSAL, sources=[])

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    chain = prompt | llm  # LCEL: pipe the prompt into the model

    try:
        response = chain.invoke({"context": format_context(docs), "question": question})
    except Exception as exc:
        raise GenerationError("LLM failed to generate an answer", detail=str(exc)) from exc

    answer = getattr(response, "content", str(response))
    # De-dupe while preserving order: clean provenance even if a chunk repeats.
    sources = list(dict.fromkeys(d.metadata.get("chunk_id", "unknown") for d in docs))
    logger.info("Generated answer from %d source chunk(s)", len(docs))
    return RagAnswer(answer=answer, sources=sources)