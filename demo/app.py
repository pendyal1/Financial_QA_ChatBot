from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finrag.answer import answer_question, build_response_from_retrieved  # noqa: E402
from finrag.config import FAISS_INDEX_PATH, INDEX_METADATA_PATH  # noqa: E402
from finrag.query import analyze_query, evidence_for_question  # noqa: E402
from finrag.remote_qwen import (  # noqa: E402
    DEFAULT_QWEN_ENDPOINT,
    answer_with_remote_qwen,
    answer_with_remote_qwen_retrieved,
)
from finrag.uploaded_filing import (  # noqa: E402
    SUPPORTED_UPLOAD_TYPES,
    build_uploaded_filing_index,
)


st.set_page_config(page_title="FinRAG", page_icon="F", layout="wide")
st.title("FinRAG")
st.caption("Ask questions about the indexed SEC corpus or a single uploaded SEC filing.")


@st.cache_resource(show_spinner=False)
def load_uploaded_index(filename: str, file_bytes: bytes):
    return build_uploaded_filing_index(filename=filename, file_bytes=file_bytes)

question = st.text_input(
    "Question",
    value="What risks did Apple report related to supply chains?",
)
document_source = st.selectbox(
    "Document source",
    options=["Indexed SEC corpus", "Uploaded SEC filing"],
    index=0,
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
uploaded_file = None
uploaded_index = None
expected_tickers: list[str] | None = None

if document_source == "Uploaded SEC filing":
    uploaded_file = st.file_uploader(
        "Upload SEC filing",
        type=SUPPORTED_UPLOAD_TYPES,
        help="Upload an SEC filing in HTML, XML, TXT, or PDF format.",
    )
    if uploaded_file is not None:
        try:
            uploaded_bytes = uploaded_file.getvalue()
            uploaded_index = load_uploaded_index(uploaded_file.name, uploaded_bytes)
            expected_tickers = [uploaded_index.metadata.ticker]
            st.info(
                f"Loaded {uploaded_index.metadata.filename} as "
                f"{uploaded_index.metadata.company} ({uploaded_index.metadata.ticker}), "
                f"{uploaded_index.metadata.form}, {len(uploaded_index.chunks)} chunks."
            )
        except Exception as exc:
            st.error(f"Could not parse uploaded filing: {exc}")
            st.stop()
else:
    if not FAISS_INDEX_PATH.exists() or not INDEX_METADATA_PATH.exists():
        st.error(
            "The FAISS index is missing. Run the download, chunking, and indexing commands from the README first."
        )
        st.stop()

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving evidence and generating answer..."):
        try:
            if document_source == "Uploaded SEC filing":
                if uploaded_index is None:
                    st.error("Upload a filing before asking a question.")
                    st.stop()
                retrieved = uploaded_index.search(question, top_k=top_k)
                if backend == "Colab GPU Qwen endpoint":
                    if not endpoint.strip():
                        st.error("Paste the Colab Qwen endpoint URL before asking.")
                        st.stop()
                    response = answer_with_remote_qwen_retrieved(
                        question=question,
                        retrieved=retrieved,
                        endpoint=endpoint,
                        max_new_tokens=350,
                        expected_tickers=expected_tickers,
                    )
                else:
                    response = build_response_from_retrieved(
                        question=question,
                        retrieved=retrieved,
                        expected_tickers=expected_tickers,
                    )
            else:
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
        if document_source == "Indexed SEC corpus" and intent.tickers and result.ticker not in intent.tickers:
            st.error(f"Unexpected cross-company retrieval: {result.chunk_id}")
        with st.expander(f"{result.chunk_id} | {result.source} | score={result.score:.3f}"):
            st.write(evidence_for_question(question, result.text))
            if result.source_url:
                st.link_button("Open SEC Filing", result.source_url)
