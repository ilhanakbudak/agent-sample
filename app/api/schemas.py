"""API request/response schemas.

Pydantic models are the API's contract. FastAPI uses them to (1) validate
incoming requests automatically — returning a 422 with details on bad input
without you writing a line of validation — and (2) document every endpoint in
the auto-generated OpenAPI schema at /docs.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source: str = Field(
        default="sample_docs",
        description="Path to a file or directory of documents to index.",
    )


class IngestResponse(BaseModel):
    chunks_indexed: int
    source: str


class QueryRequest(BaseModel):
    # `...` makes the field required; min_length rejects empty strings -> 422.
    question: str = Field(..., min_length=1, description="The user's question.")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    index_ready: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None