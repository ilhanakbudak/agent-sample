# The Practical Handbook of Retrieval-Augmented Generation

## 1. Introduction

Retrieval-Augmented Generation (RAG) is an architecture that combines a retrieval system
with a generative language model. Instead of relying solely on the parametric knowledge
baked into a model's weights during training, a RAG system fetches relevant information
from an external knowledge source at query time and supplies it to the model as context.
This lets the model answer questions about private data, recent events, or any corpus it
was never trained on, while grounding its responses in verifiable source material.

The motivation for RAG comes from three persistent weaknesses of large language models.
First, models have a fixed knowledge cutoff and cannot know anything that happened after
training. Second, they have no access to private or proprietary data unless that data was
in the training set. Third, when a model lacks knowledge, it tends to hallucinate, that
is, to generate fluent and confident text that is factually wrong. Retraining or
fine-tuning a model for every new document is prohibitively expensive and slow. RAG
sidesteps all three problems by injecting the right information at inference time.

## 2. The RAG Pipeline

A RAG system has two phases. The indexing phase happens offline, usually once, and is
repeated only when the underlying documents change. The querying phase happens online, on
every user request.

During indexing, raw documents are loaded and parsed into text, the text is split into
smaller chunks, each chunk is converted into an embedding vector, and the vectors are
stored in a vector database along with the original text and metadata.

During querying, the user's question is converted into an embedding using the same model,
the vector database is searched for the chunks whose embeddings are most similar to the
question, those chunks are inserted into a prompt as context, and the language model
generates an answer grounded in that context. A well-built system also returns citations
that point back to the source chunks.

## 3. Document Ingestion and Parsing

Ingestion is the process of turning raw files into clean text. Real-world corpora are
messy: PDFs may be scanned images with no extractable text, HTML is full of navigation and
boilerplate, and Office documents have complex internal structures. The ingestion layer
must handle multiple formats and fail gracefully when a document cannot be parsed.

For plain text and Markdown, parsing is trivial. For PDFs, libraries such as pypdf or
pdfplumber extract text page by page; pdfplumber additionally extracts tables. When a PDF
is a scanned image, optical character recognition with a tool like Tesseract is required
to recover the text. A robust ingestion stage attaches metadata to every piece of text,
recording at minimum the source filename and, where applicable, the page number. This
metadata is essential later for citations and for filtering.

A common mistake is to treat ingestion as an afterthought. In practice, the quality of the
extracted text sets a ceiling on the quality of the entire system. Garbage in, garbage
out applies with full force: no amount of clever retrieval or prompting can recover
information that was lost or mangled during parsing.

## 4. Chunking Strategies

Chunking is the process of splitting documents into smaller passages. It is the single
most consequential design choice in a RAG system, and it is governed by a fundamental
tension.

Embeddings represent a bounded span of meaning. If a chunk is too large, its embedding
becomes an average of many distinct ideas, and that averaged vector retrieves poorly
because it matches everything weakly and nothing strongly. If a chunk is too small, it
loses the surrounding context that gives a fact its meaning; a sentence that says "it runs
entirely in memory" is useless if the subject of "it" was split into a different chunk.

There are several chunking strategies. Fixed-size chunking splits text every N characters
or tokens, which is simple but cuts across sentence and word boundaries. Recursive
character splitting improves on this by trying a hierarchy of separators in priority
order: paragraph breaks first, then line breaks, then sentence boundaries, then words.
This keeps chunks aligned with the natural structure of the text. Structure-aware chunking
uses the document's own headings and sections as boundaries, which works well for
well-formatted Markdown or HTML. Semantic chunking uses embeddings to detect topic shifts
and splits where the meaning changes, which is the most sophisticated but also the most
expensive approach.

Chunk overlap is a related parameter. By repeating a small window of text, typically ten
to twenty percent of the chunk size, across consecutive chunks, you ensure that a sentence
straddling a boundary survives intact in at least one chunk. Typical configurations use
chunks of five hundred to one thousand characters or tokens, with an overlap of fifty to
two hundred. The right values depend on the documents and should be tuned empirically.

A subtle but important point is the unit of measurement. Splitting by character count is
simple, but language models think in tokens, not characters. For precise control over how
much context you send to the model, and therefore over cost and over staying within the
context window, token-based chunking is preferable. The trade-off is a small amount of
additional complexity.

## 5. Embeddings

An embedding is a dense vector of floating-point numbers that captures the semantic meaning
of a piece of text. The defining property of a good embedding model is that texts with
similar meaning produce vectors that are close together in the vector space, while
unrelated texts produce vectors that are far apart. This is what allows semantic search to
match a query to a passage even when they share no words in common.

Closeness is measured with a distance or similarity metric. Cosine similarity, the cosine
of the angle between two vectors, is the most common; it ranges from minus one for opposite
meanings to one for identical direction. Euclidean distance and dot product are also used.
The choice of metric must match how the embedding model was trained.

Embedding models produce vectors of a fixed dimensionality. Larger dimensions can capture
more nuance but cost more to store and compare. Cloud models such as OpenAI's
text-embedding-3-small produce vectors of around fifteen hundred dimensions and offer high
quality. Local models such as all-MiniLM-L6-v2 produce smaller vectors of a few hundred
dimensions, run entirely on your own hardware with no API calls, and are a good choice when
data must not leave the machine or when no API key is available.

One inviolable rule governs embeddings: the same model must be used to index the documents
and to embed the queries. Vectors from two different models live in incompatible spaces,
and mixing them silently destroys retrieval quality without raising any error.

## 6. Vector Stores

A vector store, or vector database, is a system that indexes embeddings and answers
nearest-neighbor queries quickly. Given a query vector, it returns the stored vectors most
similar to it. At small scale a brute-force comparison against every vector is feasible,
but at large scale this is too slow, so vector stores use approximate nearest neighbor
algorithms that trade a tiny amount of accuracy for a large gain in speed.

Several vector stores are in common use. Chroma is a lightweight, developer-friendly store
that persists to disk, holds the original text and metadata alongside the vectors, and
supports metadata filtering with minimal setup. FAISS, from Meta, is a high-performance
library that is extremely fast and memory-efficient but treats persistence and metadata as
manual concerns. Qdrant is a production-grade database with rich filtering and a network
API. The pgvector extension adds vector search to PostgreSQL, which is attractive when you
already run Postgres. Pinecone and Weaviate are managed, cloud-native services.

The choice among them is a real engineering decision. For a small project that needs
persistence and easy metadata filtering, Chroma is an excellent default. For raw speed on a
large in-memory index, FAISS is hard to beat. For a production system with heavy filtering
and horizontal scaling needs, a dedicated database like Qdrant or a managed service is
appropriate. Being able to articulate this trade-off is a sign of practical experience.

A practical hazard with persistent stores is non-idempotent indexing. Because the store
writes to disk, naively re-running an indexing script appends to the existing collection
rather than replacing it, which silently accumulates duplicate and stale chunks across
runs. The fix is to make indexing idempotent: either wipe the store before rebuilding, or
use stable identifiers so that re-adding a chunk updates it in place rather than creating a
copy.

## 7. Retrieval Strategies

The simplest retrieval strategy is dense retrieval: embed the query and return the top-k
most similar chunks by vector similarity. This is fast and captures semantic meaning, but
it can miss exact keyword matches, such as product codes or rare proper nouns, that do not
embed distinctively.

Sparse retrieval, exemplified by the BM25 algorithm, ranks documents by keyword overlap and
excels exactly where dense retrieval struggles. Hybrid retrieval combines dense and sparse
scores to get the best of both, and it is increasingly the default in serious systems.

Several techniques improve retrieval further. Maximal marginal relevance reduces redundancy
by penalizing chunks that are too similar to ones already selected, which increases the
diversity of the context. Reranking uses a more powerful but slower cross-encoder model to
re-score an initial set of candidates, dramatically improving precision at the cost of some
latency. Multi-query retrieval generates several paraphrases of the question and merges
their results to improve recall. Hypothetical document embeddings, or HyDE, generate a
hypothetical answer first and retrieve against that, which can help for vague queries.

The number of chunks retrieved, k, is itself a tuning parameter. Too few and the answer may
lack necessary information; too many and the prompt fills with marginally relevant text
that dilutes the signal, wastes tokens, and can push relevant content into the middle of a
long context where models attend to it poorly. A value between four and eight is a common
starting point.

## 8. Prompt Construction and Grounding

Once the relevant chunks are retrieved, they are assembled into a prompt along with the
user's question and instructions for the model. This prompt is the control system of the
RAG pipeline, and careful design here is what separates a trustworthy system from one that
hallucinates.

Three disciplines matter. First, ground the model: instruct it to answer using only the
provided context and to explicitly say it does not know when the answer is not present. The
refusal escape hatch is what prevents the model from filling gaps with its training data.
Second, require citations: label each context block with its source identifier and tell the
model to cite the sources it used, so every claim is traceable. Third, control
determinism: set the sampling temperature to zero so the model extracts faithfully rather
than inventing creative variations.

When the retrieved context exceeds the model's context window, more advanced prompt
construction is needed. Map-reduce processes each chunk separately and then combines the
partial answers. Refine builds an answer incrementally, updating it as each new chunk is
considered. Both are slower and more expensive than simply stuffing all context into one
prompt, so they are used only when the context is genuinely too large.

## 9. Generation and Citations

The generation step calls the language model with the constructed prompt and returns its
output. A well-designed system distinguishes between the sources it retrieved, which is the
provenance of the context it provided, and the sources the model actually cited in its
answer, which is what the answer rests on. Returning both gives the user transparency and
gives developers a debugging signal.

The choice of generation model involves trade-offs among quality, speed, and cost. Smaller
models are cheaper and faster but follow instructions less reliably, which matters because
grounding depends on the model obeying the instruction to use only the provided context.
Larger models are more faithful but slower and more expensive. The provider is also a
decision: cloud APIs offer top quality, while locally hosted models via tools like Ollama
keep all data on the machine and remove per-call costs.

## 10. Evaluation

Evaluating a RAG system is harder than evaluating a classifier because the output is open
text. Evaluation should cover both the retrieval stage and the generation stage.

For retrieval, context precision measures what fraction of retrieved chunks are relevant,
and context recall measures what fraction of the relevant chunks were retrieved. For
generation, faithfulness measures whether the answer is supported by the retrieved context,
that is, whether it is free of hallucination, and answer relevance measures whether the
answer actually addresses the question. Frameworks such as Ragas and DeepEval automate these
metrics, often using a language model as a judge to score the outputs.

A practical evaluation workflow starts with a small, curated set of question-and-answer
pairs that represent real usage, runs the system against them, and tracks the metrics over
time as the system changes. Without such a harness, every change is a guess, and
regressions go unnoticed until users complain.

## 11. Agentic RAG and Advanced Patterns

A fixed RAG pipeline runs the same retrieve-then-generate sequence for every query. Agentic
RAG adds a decision-making layer: an agent equipped with tools decides, for each question,
whether to retrieve at all, which tool to use, and whether to retrieve again after seeing
intermediate results. This avoids wasting a retrieval call on a greeting and allows the
system to combine retrieval with other tools such as calculators or web search.

The core loop of an agent is the reason-act-observe cycle, often called ReAct. The model
reasons about what to do, acts by calling a tool, observes the result, and repeats until it
can answer. Crucially, the agent's behavior is governed not just by its available tools but
by its system prompt, which encodes its policy. A common failure is giving an agent a
retrieval tool but no instruction to prefer it; a capable model will then answer factual
questions from its own memory and bypass the documents entirely. The fix is a forceful
system prompt that strips the model's discretion and requires retrieval for factual
questions.

Several advanced patterns build on this foundation. Query routing sends different kinds of
questions to different data sources or tools. Self-RAG has the model critique and revise
its own retrieval and answers. Corrective RAG detects when retrieved context is irrelevant
and falls back to an alternative source such as web search. Multi-agent systems decompose a
complex task across several specialized agents coordinated by a supervisor.

## 12. Production Concerns

Moving a RAG prototype into production raises a new set of concerns. Latency must be managed,
often by caching embeddings and frequent query results, and by streaming the answer token by
token so the user sees output immediately. Cost must be controlled, since every embedding
and every generation call has a price; caching and choosing right-sized models help.
Security and privacy require care about what data is sent to external services and about
protecting against prompt injection, where malicious text in a retrieved document tries to
hijack the model's instructions. Personally identifiable information may need to be redacted
before indexing. Finally, observability through logging, tracing, and metrics is essential
for diagnosing failures in a system with many moving parts.

## 13. Common Failure Modes

Several failure modes recur across RAG systems. Retrieval returns topically correct but
detail-poor chunks, usually fixed by enlarging chunks or adding overlap. The model answers
from its own knowledge instead of the documents, fixed by strengthening the grounding
prompt. Duplicate or stale chunks pollute retrieval, fixed by idempotent indexing. The
answer is correct but uncited, fixed by labeling context and instructing the model to cite.
Latency is too high, addressed by caching, smaller models, and streaming. Recognizing these
patterns quickly, and knowing the corresponding fix, is the mark of an engineer who has
actually operated a RAG system rather than only read about one.
