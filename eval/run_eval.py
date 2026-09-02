"""Retrieval evaluation harness.

Answers one question: does the retriever actually put the answer in front of
the model? Everything else in a RAG system is downstream of that — a perfect
prompt cannot rescue a chunk that was never retrieved.

Metrics
-------
hit@k   fraction of questions where SOME retrieved chunk contains the expected
        answer text. This is the ceiling on answer accuracy.
MRR     mean reciprocal rank of the first correct chunk. hit@k says "was it
        there at all"; MRR says "how near the top" — which matters because
        passages further down get less attention from the model.
p50/p95 retrieval latency.

Usage
-----
    py eval/run_eval.py                     # evaluate current settings
    py eval/run_eval.py --sweep 300 400 800 # compare chunk sizes
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.ingest import ingest_dir
from app.store import VectorStore

QUESTIONS = Path(__file__).parent / "questions.jsonl"


def load_questions() -> list[dict]:
    with QUESTIONS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(k: int = 5) -> dict:
    """Re-index from scratch with current settings, then score retrieval."""
    store = VectorStore()
    store.reset()
    chunks = ingest_dir()
    store.add(chunks)

    questions = load_questions()
    hits, reciprocal_ranks, latencies, misses = 0, [], [], []

    for item in questions:
        started = time.perf_counter()
        results = store.search(item["question"], k=k)
        latencies.append((time.perf_counter() - started) * 1000)

        # Rank of the first chunk containing the expected answer text.
        # Substring matching is crude but unambiguous — no LLM in the loop
        # means the metric can't drift between runs.
        rank = next(
            (
                i
                for i, r in enumerate(results, 1)
                if item["expect"].lower() in r["text"].lower()
            ),
            None,
        )
        if rank:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append(item)

    n = len(questions)
    return {
        "chunk_size": config.settings.chunk_size,
        "chunk_overlap": config.settings.chunk_overlap,
        "chunks_indexed": len(chunks),
        "questions": n,
        "hit_rate": hits / n,
        "mrr": statistics.mean(reciprocal_ranks),
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "misses": misses,
    }


def print_report(r: dict, verbose: bool = True) -> None:
    print(
        f"  chunk_size={r['chunk_size']:<5} overlap={r['chunk_overlap']:<4} "
        f"chunks={r['chunks_indexed']:<4} "
        f"hit@k={r['hit_rate']:.0%}  MRR={r['mrr']:.3f}  "
        f"p50={r['p50_ms']:.0f}ms  p95={r['p95_ms']:.0f}ms"
    )
    if verbose and r["misses"]:
        print(f"    misses ({len(r['misses'])}):")
        for m in r["misses"]:
            print(f"      - [{m['section']}] {m['question']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5, help="passages retrieved per query")
    parser.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        metavar="SIZE",
        help="chunk sizes to compare, e.g. --sweep 300 400 800",
    )
    args = parser.parse_args()

    if not args.sweep:
        print(f"\nRetrieval eval (k={args.k})\n")
        print_report(evaluate(k=args.k))
        return 0

    print(f"\nChunk size sweep (k={args.k})\n")
    results = []
    for size in args.sweep:
        # Overlap tracks chunk size at ~15%, otherwise a small chunk with a
        # large fixed overlap becomes mostly duplicated text.
        config.settings.chunk_size = size
        config.settings.chunk_overlap = max(40, int(size * 0.15))
        r = evaluate(k=args.k)
        results.append(r)
        print_report(r, verbose=False)

    best = max(results, key=lambda r: (r["hit_rate"], r["mrr"]))
    print(
        f"\nBest: chunk_size={best['chunk_size']} "
        f"(hit@k={best['hit_rate']:.0%}, MRR={best['mrr']:.3f})"
    )
    if best["misses"]:
        print("\nStill missing at best setting:")
        for m in best["misses"]:
            print(f"  - [{m['section']}] {m['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
