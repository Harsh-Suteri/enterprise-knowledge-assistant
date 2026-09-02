"""End-to-end check with no server and no API key.

Run this first. If it passes, ingestion, embedding, storage and retrieval
all work — which is the whole RAG core. The API is just a wrapper on top.

    py scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import ingest_dir
from app.store import VectorStore


def main() -> int:
    print("1. Ingesting data/ ...")
    started = time.perf_counter()
    chunks = ingest_dir()
    print(
        f"   -> {len(chunks)} chunks in {(time.perf_counter() - started) * 1000:.0f}ms"
    )
    if not chunks:
        print("   !! data/ is empty. Drop a PDF, .txt, .md or .docx in there.")
        return 1

    print("2. Embedding + indexing (first run downloads the model, ~90MB) ...")
    started = time.perf_counter()
    store = VectorStore()
    added = store.add(chunks)
    print(
        f"   -> indexed {added} chunks in {(time.perf_counter() - started) * 1000:.0f}ms"
    )

    questions = [
        "How many days do I have to request a refund?",
        "What is the parental leave policy?",
        "Who approves expense claims over 50,000?",
    ]

    print("3. Retrieval:")
    latencies = []
    for q in questions:
        started = time.perf_counter()
        hits = store.search(q, k=3)
        ms = (time.perf_counter() - started) * 1000
        latencies.append(ms)
        print(f"\n   Q: {q}  ({ms:.0f}ms)")
        for i, h in enumerate(hits, 1):
            preview = " ".join(h["text"].split())[:110]
            print(f"     [{i}] score={h['score']:.3f} {h['source']} :: {preview}...")

    print(f"\n   mean retrieval latency: {sum(latencies) / len(latencies):.0f}ms")
    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
