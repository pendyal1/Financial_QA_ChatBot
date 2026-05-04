from __future__ import annotations

import os
from typing import Any

import requests

from finrag.answer import build_context, extractive_answer, is_low_content_answer
from finrag.hallucination_detection import extract_citations, verify_answer
from finrag.models import Claim, ClaimVerification, HallucinationReport, RAGResponse, RetrievalResult
from finrag.query import analyze_query


DEFAULT_QWEN_ENDPOINT = os.getenv("COLAB_QWEN_ENDPOINT", "").rstrip("/")


def endpoint_generate(
    endpoint: str,
    question: str,
    retrieved: list[RetrievalResult],
    max_new_tokens: int = 350,
    timeout: int = 180,
) -> str:
    if not endpoint:
        raise ValueError("Missing Colab Qwen endpoint URL.")

    payload: dict[str, Any] = {
        "question": question,
        "context": build_context(retrieved, question=question),
        "allowed_citations": [result.chunk_id for result in retrieved],
        "max_new_tokens": max_new_tokens,
    }
    response = requests.post(
        f"{endpoint.rstrip('/')}/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["answer"]).strip()


def endpoint_hallucinate(
    endpoint: str,
    answer: str,
    retrieved: list[RetrievalResult],
    timeout: int = 120,
) -> HallucinationReport:
    """Call the /hallucinate endpoint on the Colab server and return a HallucinationReport."""
    if not endpoint:
        raise ValueError("Missing Colab Qwen endpoint URL.")

    payload: dict[str, Any] = {
        "answer": answer,
        "passages": [
            {
                "chunk_id": r.chunk_id,
                "score": r.score,
                "ticker": r.ticker,
                "company": r.company,
                "source": r.source,
                "source_url": r.source_url,
                "text": r.text,
            }
            for r in retrieved
        ],
    }
    response = requests.post(
        f"{endpoint.rstrip('/')}/hallucinate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    claims = [
        ClaimVerification(
            claim=Claim(
                text=c["claim_text"],
                sentence_index=c["sentence_index"],
                is_numerical=c["is_numerical"],
            ),
            label=c["label"],
            confidence=c["confidence"],
            supporting_chunk_ids=c["supporting_chunk_ids"],
            evidence_snippet=c["evidence_snippet"],
        )
        for c in data["claims"]
    ]
    return HallucinationReport(
        answer=answer,
        claims=claims,
        overall_risk=data["overall_risk"],
        confidence_score=data["confidence_score"],
        grounded_count=data["grounded_count"],
        partial_count=data["partial_count"],
        unsupported_count=data["unsupported_count"],
    )


def answer_with_remote_qwen(
    question: str,
    endpoint: str = DEFAULT_QWEN_ENDPOINT,
    top_k: int = 5,
    max_new_tokens: int = 350,
) -> RAGResponse:
    from finrag.retrieve import Retriever  # lazy — avoids loading faiss+PyTorch at startup
    intent = analyze_query(question)
    retriever = Retriever()
    retrieved = retriever.search(question, top_k=top_k, allowed_tickers=intent.tickers or None)
    answer = endpoint_generate(
        endpoint=endpoint,
        question=question,
        retrieved=retrieved,
        max_new_tokens=max_new_tokens,
    )
    if is_low_content_answer(answer):
        answer = extractive_answer(question, retrieved)
    citations = extract_citations(answer)
    verification = verify_answer(answer, retrieved, expected_tickers=intent.tickers or None)
    return RAGResponse(
        question=question,
        answer=answer,
        citations=citations,
        retrieved=retrieved,
        verification=verification,
    )
