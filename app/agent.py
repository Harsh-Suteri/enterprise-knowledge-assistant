"""LangGraph agent with tool calling.

Plain RAG does exactly one retrieval and answers from it. That fails on
questions that need more than one lookup — "compare the notice period for
managers with the one for junior staff" needs two searches, and a
single-shot retriever will usually return chunks for only one of them.

This agent loops: the model decides whether to search, sees the result,
and decides again. It stops when it has enough to answer, or when it hits
MAX_STEPS so a confused model can't spin forever.

    model ──► tools? ──yes──► search_documents ──► back to model
                │
                no
                ▼
              answer

Requires OPENAI_API_KEY. Without one, use the plain path in retrieve.py.
"""

from typing import Annotated, TypedDict

from app.config import settings
from app.store import VectorStore

MAX_STEPS = 6

AGENT_SYSTEM_PROMPT = """You answer questions about the user's documents.

You have one tool: search_documents. Use it before answering — you have no
knowledge of these documents otherwise.

- Break a multi-part question into separate searches rather than one vague search.
- If the first search returns nothing useful, try different wording before giving up.
- Answer ONLY from what the searches return. Cite sources by filename.
- If the documents don't contain the answer, say so plainly."""


class AgentState(TypedDict):
    """State passed between graph nodes.

    `add_messages` appends rather than overwrites, so each node adds its turn
    to the conversation instead of replacing history.
    """

    messages: Annotated[list, "add_messages"]


def build_agent(store: VectorStore):
    """Construct the agent graph. Imports are local so the module stays
    importable (and testable) without langgraph installed."""
    from langchain_core.messages import SystemMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    @tool
    def search_documents(query: str) -> str:
        """Search the indexed documents. Returns the most relevant passages.

        Args:
            query: A specific, focused search phrase. Prefer several narrow
                searches over one broad one.
        """
        passages = store.search(query, k=settings.top_k)
        if not passages:
            return "No matching passages found."
        return "\n\n".join(
            f"[{i}] (source: {p['source']}, score: {p['score']})\n{p['text']}"
            for i, p in enumerate(passages, 1)
        )

    tools = [search_documents]

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    ).bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def call_model(state: State):
        # Prepend the system prompt on every call; it is not part of state,
        # so it can't be crowded out as the conversation grows.
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    def should_continue(state: State) -> str:
        last = state["messages"][-1]
        # Hard stop: each tool call adds 2 messages, so this caps the loop
        # regardless of what the model wants to do next.
        if len(state["messages"]) > MAX_STEPS * 2:
            return END
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")  # after a tool runs, let the model decide again

    return graph.compile()


def ask(question: str, store: VectorStore) -> dict:
    """Run the agent once and return the answer plus a trace of its searches."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the agent path.")

    from langchain_core.messages import HumanMessage

    agent = build_agent(store)
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    searches = [
        call["args"].get("query")
        for msg in result["messages"]
        for call in (getattr(msg, "tool_calls", None) or [])
    ]
    return {
        "question": question,
        "answer": result["messages"][-1].content,
        "searches": searches,
        "steps": len(result["messages"]),
    }
