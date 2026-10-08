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

## Local inference

Generation runs against either a hosted OpenAI model or a local one served by
[Ollama](https://ollama.com). Ollama implements the OpenAI wire format, so the
same client class drives both and switching is configuration rather than a
second code path:

```bash
ollama pull llama3.2:3b
LLM_BACKEND=ollama uvicorn app.main:app
```

`GET /health` reports which backend is live:

```json
{"status": "ok", "chunks_indexed": 14, "llm_backend": "ollama:llama3.2:3b"}
```

**Why bother.** No key, no per-call cost, and document text never leaves the
machine — which is the difference between a demo and something you can point at
a confidential corpus.

**The tradeoff, measured.** On CPU with `llama3.2:3b` (Q4_K_M), a single-shot
`/query` takes about **17 s** end to end, against roughly 1-2 s for
`gpt-4o-mini`. Local is free and private; it is not fast.

Grounding survives the smaller model. Asked something outside the corpus, the
3B model still returns `"I don't have enough information to answer that."` with
`grounded: false` rather than inventing an answer — the citation discipline is
in the prompt and the retrieval contract, not in model size.

**`/agent` is deliberately not wired to the local backend.** Multi-step tool
calling is unreliable at 3B, and a confused agent loop is worse than a clear
501. Single-shot RAG is the path that works locally.

From inside a container, point `OLLAMA_BASE_URL` at
`http://host.docker.internal:11434/v1` — `localhost` there is the container
itself, not the host running Ollama.

---

## Fine-tuning the grounding contract — `finetune/`

Grounding survives at 3B because the model is big enough to follow the system
prompt. It does not survive at 135M: that model invents answers instead of
refusing. So the question is whether the behaviour can be **trained in** rather
than prompted in — which matters, because in RAG the facts come from retrieval
and what the generator owes you is knowing when it has nothing.

`finetune/` LoRA-tunes `SmolLM2-135M-Instruct` on that contract: 4.88M
trainable parameters (3.5%), two epochs, **76 minutes on 4 CPU cores, no GPU**.
It trains on three unrelated policy documents and is evaluated on
`sample_hr_policy.md`, which is **absent from training** — so the result
measures the behaviour generalising, not memorisation.

| metric | base | + LoRA |
|---|---|---|
| refusal accuracy (12 unanswerable) | 0% | **100%** |
| citation rate | 7% | **67%** |
| precision when it answers | 47% | **70%** |
| false refusal rate | 0% | **33%** (worse) |
| p50 latency | 4724 ms | **1889 ms** |

Before, asked to name a CEO the handbook never mentions, it produced
"**John Smith**". After, it refuses — every time.

**The regression is real and reported, not buried:** a third of the training
set was refusal examples, and the model learned "refuse" as a slightly too
cheap default, so it now declines five questions it could have answered. Full
analysis, including a case where substring scoring marks a correct answer
wrong, is in [`finetune/README.md`](finetune/README.md).

> QLoRA would be the natural choice and is what most job specs ask for. It
> needs `bitsandbytes`, which needs CUDA; this was trained on an i5-10310U with
> no NVIDIA GPU, so fp32 LoRA on a small model is the honest version of the
> experiment on this hardware.

---

## Configuration

All tunables in `app/config.py`, overridable via `.env` (copy `.env.example`).

| Setting | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | 300 | Chosen by measurement — see the sweep above |
| `CHUNK_OVERLAP` | 45 | ~15% of chunk size |
| `TOP_K` | 5 | Passages sent to the model |
| `OPENAI_API_KEY` | unset | Omit for retrieval-only mode |
| `LLM_MODEL` | gpt-4o-mini | Used when `LLM_BACKEND=openai` |
| `LLM_BACKEND` | openai | `openai` or `ollama` — see Local inference |
| `OLLAMA_BASE_URL` | http://localhost:11434/v1 | `host.docker.internal` from a container |
| `OLLAMA_MODEL` | llama3.2:3b | Any model Ollama has pulled |

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
  unanswerable_questions.jsonl  questions the corpus cannot answer
  run_eval.py     hit-rate, MRR, latency, chunk-size sweep
finetune/
  corpus/         training-only documents (never evaluated on)
  facts/          question/answer/gold triples per document
  build_dataset.py  corpus -> grounded + refusal training examples
  train_lora.py     LoRA fine-tune, CPU, no GPU required
  eval_generation.py  answer/citation/refusal metrics
  compare.py        before/after table from two result files
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
