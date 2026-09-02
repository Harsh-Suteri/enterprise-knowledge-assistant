"""Central configuration.

Every tunable lives here so experiments are reproducible: change a value,
re-run the eval, compare the numbers. Nothing else in the codebase should
hard-code a chunk size or a model name.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- paths ---
    data_dir: Path = ROOT / "data"
    chroma_dir: Path = ROOT / ".chroma"

    # --- chunking ---
    # 800 chars ~= 200 tokens. Big enough to hold a complete idea, small
    # enough that one chunk isn't half the context window.
    chunk_size: int = 800
    # Overlap stops a sentence that straddles a boundary from being lost
    # to both chunks. ~15% is the usual starting point.
    chunk_overlap: int = 120

    # --- embeddings ---
    # Local model: no API key, no per-call cost, runs on CPU.
    # 384 dimensions, good enough for semantic search over documents.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- retrieval ---
    top_k: int = 5

    # --- generation (optional; leave key unset to run retrieval-only) ---
    openai_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"

    collection_name: str = "enterprise_docs"


settings = Settings()
