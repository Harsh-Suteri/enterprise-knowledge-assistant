# Enterprise Knowledge Assistant

Retrieval-Augmented Generation over your own documents, with cited answers.

Drop PDFs, Word files or Markdown into `data/`, ingest them, and ask questions.
Answers are grounded in the retrieved passages and cite which passage they came
from — if the documents don't contain the answer, the system says so rather than
inventing one.

---

## Status

**Stage 1 complete** — ingestion, chunking, embedding, vector storage, retrieval
and a FastAPI service. Runs end to end with no API key (retrieval-only mode).

Not yet built: agent layer, evaluation harness, Docker, CI. See [Roadmap](#roadmap).

---

## Quickstart

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
py scripts/smoke_test.py
```

The first run downloads the embedding model (~90 MB). After that it's local and offline.

To run the API:

```bash
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs for the interactive API.

```bash
# index everything in data/
curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d "{}"

# ask a question
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" ^
  -d "{\"question\": \"How many days do I have to request a refund?\"}"
```

Without an `OPENAI_API_KEY` the `/query` endpoint returns the retrieved passages
and `answer: null`. Add a key to `.env` to get a synthesised, cited answer.

---

## How it works

```
data/*.pdf ──► ingest.py ──► store.py ──► retrieve.py ──► main.py
              load + chunk   embed +      search +        FastAPI
                             persist      generate
```

**Chunking** (`app/ingest.py`) — `RecursiveCharacterTextSplitter` tries paragraph
breaks first, then lines, then sentences, then words, only falling back to a hard
character cut. That keeps chunks semantically whole. Default 800 chars with 120
overlap; overlap stops a sentence straddling a boundary from being lost to both sides.

**Embedding** (`app/store.py`) — `all-MiniLM-L6-v2`, 384 dimensions, runs on CPU with
no API cost. Vectors are L2-normalised so cosine similarity reduces to a dot product.
The model is loaded once per process via `lru_cache`; loading it per request would
dominate latency.

**Storage** — Chroma with an HNSW index, persisted to `.chroma/`. Chunk IDs are
`{source}::{index}`, so re-ingesting a file upserts instead of duplicating.

**Retrieval** (`app/retrieve.py`) — top-k cosine search, passages numbered and passed
to the LLM with a system prompt that forbids outside knowledge and requires citations.
`temperature=0` so the same context gives the same answer.

---

## Measured performance

From `scripts/smoke_test.py` on the sample document (5 chunks, CPU only):

| Metric | Value |
|---|---|
| Ingest + embed + index | 7,831 ms (cold, includes model load) |
| Mean retrieval latency | **14 ms** |
| Chunks from sample doc | 5 |

> These are honest numbers from a tiny corpus. They are not benchmarks. Replace them
> with real ones once the eval harness (Stage 2) runs against a proper question set.

---

## Known issues

**Chunks are too coarse.** The sample document produces only 5 chunks at
`chunk_size=800`, which bundles unrelated sections together. Asking about parental
leave returns the chunk starting at "Leave Policy / Annual Leave" — the right region,
but not a tight match. Retrieval precision should improve at `chunk_size=300–400`
for structured documents like handbooks.

This is exactly what the evaluation harness is for: change the number, re-run,
compare. Don't guess.

---

## Roadmap

- [x] Ingestion, chunking, embedding, vector store, retrieval, FastAPI
- [ ] **Evaluation harness** — question set with expected sources; measure retrieval
      hit-rate and answer faithfulness. Tune `chunk_size` against real numbers.
- [ ] **Reranking** — cross-encoder over the top-20 to sharpen the top-5
- [ ] **LangGraph agent** — multi-step retrieval with tool calling
- [ ] **Docker + docker-compose** — one command to run the whole thing
- [ ] **GitHub Actions CI** — lint and test on every push
- [ ] Streamlit UI

---

## Configuration

All tunables live in `app/config.py` and can be overridden in `.env`
(copy `.env.example`).

| Setting | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | 800 | Lower for structured docs |
| `CHUNK_OVERLAP` | 120 | ~15% of chunk size |
| `TOP_K` | 5 | Passages sent to the LLM |
| `OPENAI_API_KEY` | unset | Omit to run retrieval-only |

---

## Layout

```
app/
  config.py     settings, all tunables in one place
  ingest.py     file -> text -> chunks
  store.py      embed + persist + search
  retrieve.py   retrieval + grounded generation
  main.py       FastAPI service
scripts/
  smoke_test.py end-to-end check, no server needed
data/           put your documents here (gitignored)
eval/           evaluation harness (Stage 2)
```
