"""FastAPI service.

Three endpoints: ingest documents, ask a question, check health.
The store is created once at startup and reused — building it per request
would reload the embedding model every time.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.ingest import ingest_dir, ingest_file
from app.retrieve import answer_question
from app.store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("eka")

state: dict[str, VectorStore] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading embedding model and vector store")
    state["store"] = VectorStore()
    log.info(f"ready, {state['store'].count()} chunks indexed")
    yield
    state.clear()


app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="RAG over your own documents, with cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, examples=["What is the refund policy?"])
    k: int | None = Field(default=None, ge=1, le=20)


class IngestRequest(BaseModel):
    filename: str | None = Field(
        default=None,
        description="Single file inside data/. Omit to ingest the whole directory.",
    )


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": state["store"].count()}


@app.post("/ingest")
def ingest(req: IngestRequest):
    store = state["store"]
    started = time.perf_counter()

    if req.filename:
        path = settings.data_dir / Path(req.filename).name  # strip any path traversal
        if not path.exists():
            raise HTTPException(404, f"{req.filename} not found in data/")
        chunks = ingest_file(path)
    else:
        chunks = ingest_dir()

    if not chunks:
        raise HTTPException(400, "No supported documents found in data/")

    added = store.add(chunks)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    log.info(f"ingested {added} chunks in {elapsed_ms}ms")
    return {
        "chunks_added": added,
        "total_indexed": store.count(),
        "elapsed_ms": elapsed_ms,
    }


@app.post("/agent")
def agent(req: QueryRequest):
    """Multi-step answering: the model runs as many searches as it needs.

    Slower and costlier than /query — use it for questions that require
    combining facts from different parts of the corpus.
    """
    store = state["store"]
    if store.count() == 0:
        raise HTTPException(409, "Nothing indexed yet. POST /ingest first.")
    if not settings.openai_api_key:
        raise HTTPException(
            501, "Agent path requires OPENAI_API_KEY. Use /query instead."
        )

    from app.agent import ask

    started = time.perf_counter()
    result = ask(req.question, store)
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    log.info(f"agent searches={result['searches']} {result['elapsed_ms']}ms")
    return result


@app.post("/query")
def query(req: QueryRequest):
    store = state["store"]
    if store.count() == 0:
        raise HTTPException(409, "Nothing indexed yet. POST /ingest first.")

    started = time.perf_counter()
    result = answer_question(req.question, store, k=req.k)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    log.info(
        f'query="{req.question[:60]}" passages={len(result.passages)} {elapsed_ms}ms'
    )

    return {
        "question": result.question,
        "answer": result.answer,
        "grounded": result.grounded,
        "passages": result.passages,
        "elapsed_ms": elapsed_ms,
    }
