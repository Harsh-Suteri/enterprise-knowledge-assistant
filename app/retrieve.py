"""Retrieval and answer generation.

Kept separate from the vector store so the "how do we find context" logic
and the "how do we store vectors" logic can change independently.

Generation has three states, in order of cost: no backend configured at all
(the API returns retrieved passages with no synthesised answer, so retrieval
can be built and demoed before spending anything), a local model served by
Ollama, or a hosted OpenAI model. Ollama implements the OpenAI wire format,
so one client class covers both and the choice is configuration, not a branch
in the request path.
"""

from dataclasses import dataclass

from app.config import settings
from app.store import VectorStore

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.

Rules:
- If the context does not contain the answer, say exactly: "I don't have enough information to answer that."
- Cite the passage numbers you used, like [1] or [2][3].
- Do not use knowledge from outside the context.
- Be concise."""


@dataclass
class Answer:
    question: str
    answer: str | None
    passages: list[dict]
    grounded: bool


def _format_context(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {p['source']})\n{p['text']}"
        for i, p in enumerate(passages)
    )


def backend_name() -> str | None:
    """Which generation backend is configured, or None for retrieval-only."""
    if settings.llm_backend == "ollama":
        return f"ollama:{settings.ollama_model}"
    if settings.openai_api_key:
        return f"openai:{settings.llm_model}"
    return None


def _client():
    """Build the LLM client, or return None if generation is unavailable.

    Ollama does not check the API key but the OpenAI client requires one to
    be set, hence the placeholder.
    """
    from openai import OpenAI

    if settings.llm_backend == "ollama":
        client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
        return client, settings.ollama_model
    if settings.openai_api_key:
        return OpenAI(api_key=settings.openai_api_key), settings.llm_model
    return None


def answer_question(question: str, store: VectorStore, k: int | None = None) -> Answer:
    passages = store.search(question, k=k)

    if not passages:
        return Answer(
            question, "I don't have enough information to answer that.", [], False
        )

    configured = _client()
    if configured is None:
        # Retrieval-only mode: no backend configured, so return the evidence
        # without a synthesised answer rather than failing.
        return Answer(question, None, passages, False)

    client, model = configured
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # deterministic: same context should give the same answer
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{_format_context(passages)}\n\nQuestion: {question}",
            },
        ],
    )
    text = response.choices[0].message.content.strip()
    grounded = "i don't have enough information" not in text.lower()
    return Answer(question, text, passages, grounded)
