"""Retrieval and answer generation.

Kept separate from the vector store so the "how do we find context" logic
and the "how do we store vectors" logic can change independently.

If no OpenAI key is configured the API still works — it returns the
retrieved passages without a generated answer. That means you can build,
test and demo the retrieval half before spending anything on tokens.
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


def answer_question(question: str, store: VectorStore, k: int | None = None) -> Answer:
    passages = store.search(question, k=k)

    if not passages:
        return Answer(question, "I don't have enough information to answer that.", [], False)

    if not settings.openai_api_key:
        # Retrieval-only mode: no key configured, so return evidence without
        # a synthesised answer rather than failing.
        return Answer(question, None, passages, False)

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
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
