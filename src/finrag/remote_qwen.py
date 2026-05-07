from __future__ import annotations

import os
from typing import Any

import requests

from finrag.answer import build_context, extractive_answer, is_low_content_answer
from finrag.answer_formatting import format_model_answer
from finrag.hallucination_detection import extract_citations, verify_answer
from finrag.models import RAGResponse, RetrievalResult
from finrag.sec_live import retrieve_live_sec


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


def answer_with_remote_qwen(
    question: str,
    endpoint: str = DEFAULT_QWEN_ENDPOINT,
    top_k: int = 5,
    max_new_tokens: int = 350,
) -> RAGResponse:
    company, retrieved = retrieve_live_sec(question, top_k=top_k)
    return answer_with_remote_qwen_retrieved(
        question=question,
        retrieved=retrieved,
        endpoint=endpoint,
        max_new_tokens=max_new_tokens,
        expected_tickers=[company.ticker],
    )


def answer_with_remote_qwen_retrieved(
    question: str,
    retrieved: list[RetrievalResult],
    endpoint: str = DEFAULT_QWEN_ENDPOINT,
    max_new_tokens: int = 350,
    expected_tickers: list[str] | None = None,
) -> RAGResponse:
    answer = endpoint_generate(
        endpoint=endpoint,
        question=question,
        retrieved=retrieved,
        max_new_tokens=max_new_tokens,
    )
    if is_low_content_answer(answer):
        answer = extractive_answer(question, retrieved)
    answer = format_model_answer(answer, question=question)
    citations = extract_citations(answer)
    verification = verify_answer(answer, retrieved, expected_tickers=expected_tickers)
    return RAGResponse(
        question=question,
        answer=answer,
        citations=citations,
        retrieved=retrieved,
        verification=verification,
    )
