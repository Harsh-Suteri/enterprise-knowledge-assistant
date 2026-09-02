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
    # 300 was chosen by measurement, not by feel. See eval/run_eval.py:
    # at k=1 over the sample corpus, 300 hits 100% with p50 11ms, while 800
    # also hits 100% but at 28ms, and 200 drops to 87% because answers start
    # splitting across chunk boundaries. Re-run the sweep on your own corpus
    # before trusting this number.
    chunk_size: int = 300
    # Overlap stops a sentence that straddles a boundary from being lost
    # to both chunks. ~15% of chunk size.
    chunk_overlap: int = 45

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
