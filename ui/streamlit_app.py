"""Streamlit front-end.

Talks to the FastAPI service over HTTP rather than importing the app
directly — so the UI and the API can be deployed and scaled separately,
and the UI stays a thin client with no model loading of its own.

    streamlit run ui/streamlit_app.py
"""

import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📄")
st.title("Enterprise Knowledge Assistant")
st.caption("Ask questions about your documents. Every answer cites its sources.")

with st.sidebar:
    st.header("Index")
    try:
        health = requests.get(f"{API}/health", timeout=5).json()
        st.metric("Chunks indexed", health["chunks_indexed"])
    except requests.RequestException:
        st.error(f"API not reachable at {API}")
        st.stop()

    if st.button("Ingest data/", use_container_width=True):
        with st.spinner("Indexing..."):
            r = requests.post(f"{API}/ingest", json={}, timeout=300)
        if r.ok:
            d = r.json()
            st.success(f"{d['chunks_added']} chunks in {d['elapsed_ms']}ms")
            st.rerun()
        else:
            st.error(r.json().get("detail", "Ingest failed"))

    k = st.slider("Passages to retrieve", 1, 10, 5)

question = st.text_input(
    "Question", placeholder="How many days do I have to request a refund?"
)

if question:
    with st.spinner("Searching..."):
        r = requests.post(
            f"{API}/query", json={"question": question, "k": k}, timeout=120
        )

    if not r.ok:
        st.error(r.json().get("detail", "Query failed"))
        st.stop()

    data = r.json()

    if data["answer"]:
        st.markdown("### Answer")
        st.write(data["answer"])
        if not data["grounded"]:
            st.warning("The documents don't appear to contain this answer.")
    else:
        st.info(
            "Retrieval-only mode — no `OPENAI_API_KEY` configured. "
            "The passages below are what would be sent to the model."
        )

    st.caption(f"{len(data['passages'])} passages · {data['elapsed_ms']}ms")

    st.markdown("### Sources")
    for i, p in enumerate(data["passages"], 1):
        with st.expander(f"[{i}] {p['source']}  ·  score {p['score']}"):
            st.write(p["text"])
