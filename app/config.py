"""Centralised, type-safe configuration.

All runtime config lives here and is read from environment variables / a .env
file exactly once. Import `settings` anywhere instead of calling os.getenv()
scattered around the codebase.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from the environment or a `.env` file.

    pydantic-settings validates types for free: if LLM_TEMPERATURE isn't a
    float, the app fails loudly at startup instead of mysteriously later.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # tolerate unrelated env vars
    )

    # --- Provider selection (keep the app provider-agnostic) ---
    llm_provider: str = "openai"          # "openai" | "anthropic" | "ollama"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # --- Secrets (never hard-code; supplied via .env) ---
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # --- Embeddings (cloud default, local fallback) ---
    embedding_provider: str = "openai"    # "openai" | "sentence-transformers"
    embedding_model: str = "text-embedding-3-small"

    # --- Retrieval ---
    vector_store_dir: str = "./.vectorstore"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # --- Ops ---
    log_level: str = "INFO"


# Single shared instance imported across the app.
settings = Settings()
