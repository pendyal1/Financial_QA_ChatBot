from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finrag.answer import answer_question  # noqa: E402
from finrag.config import FAISS_INDEX_PATH, INDEX_METADATA_PATH  # noqa: E402
from finrag.query import analyze_query, evidence_for_question  # noqa: E402
from finrag.remote_qwen import DEFAULT_QWEN_ENDPOINT, answer_with_remote_qwen  # noqa: E402


st.set_page_config(page_title="FinRAG", page_icon="F", layout="wide")
st.title("FinRAG")
st.caption("Ask questions about downloaded SEC 10-K filings.")

if not FAISS_INDEX_PATH.exists() or not INDEX_METADATA_PATH.exists():
    st.error(
        "The FAISS index is missing. Run the download, chunking, and indexing commands from the README first."
    )
    st.stop()

question = st.text_input(
    "Question",
    value="What risks did Apple report related to supply chains?",
)
top_k = st.slider("Retrieved chunks", min_value=3, max_value=10, value=5)
backend = st.selectbox(
    "Answer backend",
    options=["Colab GPU Qwen endpoint", "Debug extractive fallback"],
    index=0 if DEFAULT_QWEN_ENDPOINT else 1,
)
endpoint = st.text_input(
    "Colab Qwen endpoint",
    value=DEFAULT_QWEN_ENDPOINT,
    help="Public URL for the Qwen server running in Colab, for example https://name.trycloudflare.com",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving evidence and generating answer..."):
        try:
            if backend == "Colab GPU Qwen endpoint":
                if not endpoint.strip():
                    st.error("Paste the Colab Qwen endpoint URL before asking.")
                    st.stop()
                response = answer_with_remote_qwen(
                    question=question,
                    endpoint=endpoint,
                    top_k=top_k,
                    max_new_tokens=350,
                )
            else:
                response = answer_question(question, top_k=top_k)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    st.subheader("Answer")
    st.write(response.answer)

    col1, col2 = st.columns(2)
    col1.metric("Confidence Score", f"{response.verification.confidence_score:.2f}")
    col2.metric("Hallucination Risk", response.verification.hallucination_risk)

    if response.verification.notes:
        st.warning(" ".join(response.verification.notes))

    st.subheader("Retrieved Sources")
    intent = analyze_query(question)
    for result in response.retrieved:
        if intent.tickers and result.ticker not in intent.tickers:
            st.error(f"Unexpected cross-company retrieval: {result.chunk_id}")
        with st.expander(f"{result.chunk_id} | {result.source} | score={result.score:.3f}"):
            st.write(evidence_for_question(question, result.text))
            st.link_button("Open SEC Filing", result.source_url)
