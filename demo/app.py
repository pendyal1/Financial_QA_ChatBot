from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finrag.answer import answer_question  # noqa: E402
from finrag.config import DEFAULT_SEC_USER_AGENT  # noqa: E402
from finrag.query import analyze_query, evidence_for_question  # noqa: E402
from finrag.remote_qwen import DEFAULT_QWEN_ENDPOINT, answer_with_remote_qwen  # noqa: E402


st.set_page_config(page_title="FinRAG", page_icon="F", layout="wide")
st.title("FinRAG")
st.caption(
    "Ask questions about public-company SEC filings. Retrieval uses the official SEC EDGAR submissions and companyfacts APIs on demand."
)

with st.sidebar:
    st.header("Retrieval")
    sec_user_agent = st.text_input(
        "SEC User-Agent",
        value=DEFAULT_SEC_USER_AGENT,
        help="SEC asks automated tools to identify the app and contact email, e.g. FinRAG adi@example.com.",
    )
    if "example.com" in sec_user_agent:
        st.warning("Replace the placeholder email before making repeated SEC API requests.")

question = st.text_input(
    "Question",
    value="What risks did Apple report related to supply chains?",
)
top_k = st.slider("Retrieved chunks", min_value=3, max_value=10, value=5)
backend = st.selectbox(
    "Answer backend",
    options=["LoRA Qwen endpoint", "Debug extractive fallback"],
    index=0 if DEFAULT_QWEN_ENDPOINT else 1,
)
endpoint = st.text_input(
    "LoRA Qwen endpoint",
    value=DEFAULT_QWEN_ENDPOINT,
    help="Public URL for finrag.qwen_server running on a GPU machine, for example https://name.ngrok-free.app",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving SEC evidence and generating answer..."):
        try:
            if backend == "LoRA Qwen endpoint":
                if not endpoint.strip():
                    st.error("Paste the LoRA Qwen endpoint URL before asking.")
                    st.stop()
                response = answer_with_remote_qwen(
                    question=question,
                    endpoint=endpoint,
                    top_k=top_k,
                    max_new_tokens=350,
                    user_agent=sec_user_agent,
                )
            else:
                response = answer_question(question, top_k=top_k, user_agent=sec_user_agent)
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
            if result.source_url:
                st.link_button("Open SEC Filing", result.source_url)
