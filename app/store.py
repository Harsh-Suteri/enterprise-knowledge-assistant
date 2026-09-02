"""Embed chunks and persist them in a vector store.

Chroma is the default because it persists to disk with no server to run.
The `VectorStore` wrapper keeps the rest of the app from importing Chroma
directly, so swapping in Qdrant or FAISS later touches only this file.
"""

from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ingest import Chunk


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    """Load the embedding model once and reuse it.

    Loading weights takes seconds; doing it per request would dominate
    latency. lru_cache makes this a process-wide singleton.
    """
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts.

    Batching matters: encoding 100 texts in one call is far faster than
    100 separate calls, because the model runs them as a single tensor.
    """
    vectors = _embedder().encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # lets cosine similarity use a dot product
    )
    return vectors.tolist()


class VectorStore:
    def __init__(self) -> None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> int:
        """Insert chunks. Re-ingesting the same file overwrites, not duplicates.

        Chroma upserts on id collision, and Chunk.id is derived from
        (source, chunk_index) — so ingestion is idempotent.
        """
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embed([c.text for c in chunks]),
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Return the k nearest chunks to the query."""
        k = k or settings.top_k
        result = self._collection.query(
            query_embeddings=embed([query]),
            n_results=k,
        )
        # Chroma nests results one level per query; we only ever send one.
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        return [
            {
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                # cosine distance -> similarity, so higher is better
                "score": round(1 - dist, 4),
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(settings.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
