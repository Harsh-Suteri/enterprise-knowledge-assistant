# Enterprise Knowledge Assistant

Retrieval-Augmented Generation over your own documents, with cited answers,
a measured retrieval pipeline, and a LangGraph agent for multi-step questions.

Drop PDFs, Word files or Markdown into `data/`, index them, and ask questions.
Answers are grounded in retrieved passages and cite their source — if the
documents don't contain the answer, the system says so rather than inventing one.

**Runs with no API key.** Without `OPENAI_API_KEY` it operates in retrieval-only
mode: you get the ranked passages, just not a synthesised answer.

---

## Measured results

Retrieval quality is measured, not asserted. `eval/run_eval.py` scores 15
questions against the sample corpus and reports hit-rate, MRR and latency.

**Chunk size sweep at k=1** (the hard setting — only the single best chunk counts):

| chunk_size | chunks | hit@1 | MRR | p50 | p95 |
|---|---|---|---|---|---|
| 200 | 26 | 87% | 0.867 | 11 ms | 13 ms |
| **300** | **14** | **100%** | **1.000** | **11 ms** | **14 ms** |
| 400 | 10 | 93% | 0.933 | 12 ms | 13 ms |
| 600 | 7 | 100% | 1.000 | 23 ms | 25 ms |
| 800 | 5 | 100% | 1.000 | 28 ms | 34 ms |

**300 is the default** — same accuracy as 800 at **2.5× lower latency**. Below
300, answers start splitting across chunk boundaries and hit-rate drops.

Reproduce:

```bash
py eval/run_eval.py -k 1 --sweep 200 300 400 600 800
```

> **Caveat, stated plainly:** this is one document and 15 questions. The numbers
> are real but the corpus is small — at k=5 every chunk size scores 100%, because
> retrieving 5 of 5 chunks trivially contains the answer. That's why the sweep
> above runs at k=1. Re-run it on your own corpus before trusting the default.

---

## Quickstart

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

py scripts/smoke_test.py          # end-to-end check, no server
py eval/run_eval.py -k 1          # retrieval metrics
pytest tests -q                   # unit tests
```

First run downloads the embedding model (~90 MB), then works offline.

**API:**

```bash
uvicorn app.main:app --reload
```

http://127.0.0.1:8000/docs for interactive docs.

**UI:**

```bash
streamlit run ui/streamlit_app.py
```

**Docker:**

```bash
docker compose up --build
```

---

## Architecture

```
data/*.pdf
    │
    ▼
ingest.py ──► store.py ──► retrieve.py ──► main.py ──► ui/
load+chunk    embed+       search+          FastAPI     Streamlit
              persist      generate            │
                                               ▼
                                          agent.py
                                     LangGraph, multi-step
```

### Chunking — `app/ingest.py`
`RecursiveCharacterTextSplitter` tries paragraph breaks first, then lines, then
sentences, then words, falling back to a hard character cut only when nothing
else fits. That keeps most chunks semantically whole, which is what makes the
embedding meaningful. 300 chars with 45 overlap; overlap prevents a sentence
straddling a boundary from being lost to both sides.

### Embedding — `app/store.py`
`all-MiniLM-L6-v2`, 384 dimensions, CPU-only, no API cost. Vectors are
L2-normalised so cosine similarity reduces to a dot product. The model loads
once per process via `lru_cache` — loading per request would dominate latency.
Encoding is batched (32 at a time) because one tensor beats N round trips.

### Storage
Chroma with an HNSW index, persisted to `.chroma/`. Chunk IDs are
`{source}::{chunk_index}`, so re-ingesting a file **upserts rather than
duplicates** — ingestion is idempotent.

### Retrieval — `app/retrieve.py`
Top-k cosine search. Passages are numbered and passed to the model with a system
prompt that forbids outside knowledge and requires citations. `temperature=0`,
so identical context yields identical answers — a precondition for evaluating
anything.

### Agent — `app/agent.py`
Plain RAG does exactly one retrieval. That fails on questions needing more than
one lookup ("compare the notice period for managers with junior staff"). The
LangGraph agent loops — search, observe, decide again — and stops when it has
enough or hits `MAX_STEPS`, so a confused model can't spin forever.

```
model ──► tool calls? ──yes──► search_documents ──┐
            │                                      │
            no                                     │
            ▼                              (back to model)
          answer
```

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + chunk count |
| `POST /ingest` | Index `data/`, or one file via `{"filename": "x.pdf"}` |
| `POST /query` | Single-shot RAG. Returns answer + cited passages + latency |
| `POST /agent` | Multi-step agent. Returns answer + the searches it ran |

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days do I have to request a refund?"}'
```

---

## Evaluation

`eval/questions.jsonl` holds question / expected-answer pairs. Scoring uses
substring matching rather than an LLM judge — crude, but it can't drift between
runs, so a regression is a real regression.

| Metric | Meaning |
|---|---|
| `hit@k` | Fraction of questions where some retrieved chunk contains the answer. The ceiling on answer accuracy. |
| `MRR` | Mean reciprocal rank of the first correct chunk. Passages further down get less model attention. |
| `p50` / `p95` | Retrieval latency. |

CI fails the build if `hit@1` drops below 90%. Retrieval quality is a build
artifact, not a vibe.

---

## Testing & CI

- `pytest tests -q` — 9 tests covering chunking, ID uniqueness, size limits and
  overlap preservation. No network or model needed.
- `.github/workflows/ci.yml` — lint (ruff), format check, unit tests, retrieval
  eval with a regression gate, and a Docker build, on every push.

---

## Docker

Multi-stage build: wheels compile in a builder image, the runtime ships without
build toolchains. Runs as a non-root user. The embedding model is baked into the
image so first request isn't a 90 MB download. `data/` mounts read-only; the
Chroma index lives in a named volume so it survives rebuilds.

---

## Configuration

All tunables in `app/config.py`, overridable via `.env` (copy `.env.example`).

| Setting | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | 300 | Chosen by measurement — see the sweep above |
| `CHUNK_OVERLAP` | 45 | ~15% of chunk size |
| `TOP_K` | 5 | Passages sent to the model |
| `OPENAI_API_KEY` | unset | Omit for retrieval-only mode |
| `LLM_MODEL` | gpt-4o-mini | |

---

## Layout

```
app/
  config.py       all tunables in one place
  ingest.py       file -> text -> chunks
  store.py        embed + persist + search
  retrieve.py     retrieval + grounded generation
  agent.py        LangGraph multi-step agent
  main.py         FastAPI service
eval/
  questions.jsonl question set with expected answers
  run_eval.py     hit-rate, MRR, latency, chunk-size sweep
scripts/
  smoke_test.py   end-to-end check, no server
tests/            unit tests
ui/               Streamlit front-end
data/             your documents (gitignored)
```

---

## Known limitations

- **Small eval corpus.** One document, 15 questions. Enough to tune chunk size,
  not enough to claim general performance.
- **No reranking.** A cross-encoder over the top-20 would likely sharpen the
  top-5 on a larger corpus. Not needed at this scale — adding it now would be
  optimising something that already scores 100%.
- **Substring-based scoring.** Cheap and stable, but it can't tell a correct
  answer from a chunk that merely contains the right string.
- **Single vector store.** Chroma only. The `VectorStore` wrapper isolates it,
  so swapping in Qdrant or pgvector touches one file.
