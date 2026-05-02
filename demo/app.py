from __future__ import annotations

import sys
import traceback
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finrag.hallucination import detect_hallucinations  # noqa: E402
from finrag.remote_qwen import DEFAULT_QWEN_ENDPOINT, endpoint_generate  # noqa: E402
from finrag.answer import build_context, extractive_answer, is_low_content_answer  # noqa: E402
from finrag.sec_live import live_retrieve  # noqa: E402
from finrag.rerank import rerank  # noqa: E402


st.set_page_config(page_title="FinRAG", page_icon="F", layout="wide")
st.title("FinRAG — Financial Q&A with Live SEC Retrieval")
st.caption(
    "Ask any question about a public company. Mention the company name or ticker. "
    "Documents are fetched live from SEC EDGAR."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved chunks", min_value=3, max_value=15, value=8)
    use_reranker = st.toggle("Rerank results", value=False,
                             help="Uses BAAI/bge-reranker-v2-m3 to improve passage ordering.")
    use_hd = st.toggle("Hallucination detection", value=False,
                       help="Checks each claim in the answer against retrieved evidence. "
                            "Loads a 450 MB NLI model on first use.")
    backend = st.selectbox(
        "Answer backend",
        options=["Colab GPU Qwen endpoint", "Extractive fallback (no GPU needed)"],
    )
    endpoint = st.text_input(
        "Colab Qwen endpoint URL",
        value=DEFAULT_QWEN_ENDPOINT,
        help="Paste the ngrok URL from Colab after starting the Qwen server.",
    )

# ── Main input ────────────────────────────────────────────────────────────────
question = st.text_input(
    "Your question",
    placeholder="e.g. What risks did Apple report related to supply chains?",
)

example_questions = [
    "What was Microsoft's revenue in FY2024?",
    "What cybersecurity risks does Nvidia describe?",
    "What were Amazon's operating expenses?",
    "How did Tesla describe its competition risks?",
]
st.caption("Examples: " + " · ".join(f"*{q}*" for q in example_questions))

if st.button("Ask", type="primary") and question.strip():
    # ── Step 1: Live retrieval ─────────────────────────────────────────────
    with st.status("Retrieving from SEC EDGAR...", expanded=True) as status:
        try:
            st.write("Resolving company and fetching latest 10-K...")
            results = live_retrieve(question, top_k=top_k if not use_reranker else top_k * 2)

            if use_reranker:
                st.write("Reranking passages...")
                results = rerank(question, results, top_k=top_k)

            status.update(label=f"Retrieved {len(results)} passages", state="complete")
        except ValueError as e:
            st.error(str(e))
            st.info("Tip: mention the company name or ticker in your question, e.g. 'What was Apple's revenue?'")
            st.stop()
        except Exception as e:
            st.error(f"Retrieval failed: {type(e).__name__}: {e}")
            st.code(traceback.format_exc())
            st.stop()

    # ── Step 2: Answer generation ──────────────────────────────────────────
    with st.spinner("Generating answer..."):
        try:
            context = build_context(results, question=question)

            if backend == "Colab GPU Qwen endpoint":
                if not endpoint.strip():
                    st.error("Paste the Colab Qwen endpoint URL in the sidebar before asking.")
                    st.stop()
                answer = endpoint_generate(
                    endpoint=endpoint,
                    question=question,
                    retrieved=results,
                    max_new_tokens=350,
                )
            else:
                answer = extractive_answer(question, results)

            if is_low_content_answer(answer):
                answer = extractive_answer(question, results)

        except Exception as e:
            st.error(f"Answer generation failed: {e}")
            st.stop()

    # ── Step 3: Hallucination detection ───────────────────────────────────
    report = None
    if use_hd:
        with st.spinner("Checking for hallucinations..."):
            try:
                report = detect_hallucinations(answer, results)
            except Exception:
                pass

    # ── Display results ────────────────────────────────────────────────────
    st.subheader("Answer")
    st.write(answer)

    if report is not None:
        col1, col2, col3, col4 = st.columns(4)
        risk_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(report.overall_risk, "gray")
        col1.metric("Hallucination Risk", report.overall_risk)
        col2.metric("Confidence Score", f"{report.confidence_score:.2f}")
        col3.metric("Grounded Claims", report.grounded_count)
        col4.metric("Unsupported Claims", report.unsupported_count)

        if report.overall_risk == "High":
            st.warning("One or more claims could not be verified against the retrieved evidence. Review carefully.")

        with st.expander("Claim-level breakdown"):
            for cv in report.claims:
                icon = {"grounded": "✓", "partial": "~", "unsupported": "✗"}.get(cv.label, "?")
                color = {"grounded": "green", "partial": "orange", "unsupported": "red"}.get(cv.label, "gray")
                st.markdown(f":{color}[{icon} **{cv.label.upper()}** (conf: {cv.confidence:.2f})] {cv.claim.text}")
                if cv.evidence_snippet:
                    st.caption(f"Evidence: {cv.evidence_snippet[:150]}...")

    st.subheader("Retrieved Sources")
    company = results[0].company if results else ""
    st.caption(f"Source: {company} — live from SEC EDGAR")

    for result in results:
        with st.expander(f"{result.chunk_id} | score={result.score:.3f}"):
            st.write(result.text)
            st.link_button("Open SEC Filing", result.source_url)
