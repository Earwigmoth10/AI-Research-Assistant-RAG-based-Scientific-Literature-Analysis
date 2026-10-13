"""Streamlit web UI:  streamlit run app.py"""

import tempfile
from pathlib import Path

import streamlit as st

from rag_assistant.config import Settings
from rag_assistant.pipeline import ResearchAssistant

st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")
st.title("🔬 AI Research Assistant")
st.caption("Retrieval-Augmented Generation for scientific literature: grounded answers with citations.")


@st.cache_resource
def get_assistant(embedder: str, llm: str) -> ResearchAssistant:
    settings = Settings()
    settings.embedder, settings.llm_backend = embedder, llm
    return ResearchAssistant(settings)


with st.sidebar:
    st.header("Settings")
    embedder = st.selectbox("Embedder", ["tfidf", "sentence-transformers"])
    llm = st.selectbox("Answer generator", ["extractive", "anthropic", "openai"])
    mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "bm25"])
    top_k = st.slider("Passages to retrieve", 1, 10, 5)

    try:
        assistant = get_assistant(embedder, llm)
    except Exception as exc:  # missing API key / optional dependency
        st.error(str(exc))
        st.stop()

    st.header("Library")
    uploads = st.file_uploader("Upload papers (PDF, TXT, MD)", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploads and st.button("Index uploaded papers", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            for up in uploads:
                (Path(tmp) / up.name).write_bytes(up.getbuffer())
            with st.spinner("Indexing..."):
                summary = assistant.ingest([tmp])
        st.success(f"Added {summary['new_chunks']} chunks from {summary['files']} file(s).")

    stats = assistant.stats()
    st.metric("Papers", stats["n_sources"])
    st.metric("Chunks", stats["n_chunks"])
    for source in stats["sources"]:
        st.write(f"• {source}")

question = st.chat_input("Ask a question about your papers...")
if question:
    if stats["n_chunks"] == 0:
        st.warning("Index some papers first (sidebar).")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching and reasoning..."):
                answer = assistant.ask(question, top_k=top_k, mode=mode)
            st.write(answer.text)
            st.subheader("Sources")
            for i, r in enumerate(answer.sources, start=1):
                with st.expander(f"[{i}] {r.chunk.source} · page {r.chunk.page} · score {r.score:.4f}"):
                    st.write(r.chunk.text)
