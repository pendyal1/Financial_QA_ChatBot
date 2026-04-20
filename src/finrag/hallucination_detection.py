from __future__ import annotations

import re
from collections.abc import Iterable

from finrag.models import RetrievalResult, VerificationResult

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def extract_citations(answer: str) -> list[str]:
    return sorted(set(re.findall(r"\[([A-Z0-9_.-]+-\d{4}-\d{2}-\d{2}-\d{4})\]", answer)))


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if token not in STOPWORDS
    }


def sentence_split(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def overlap_ratio(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set:
        return 0.0
    return len(left_set & right_set) / len(left_set)


def citation_ticker(citation: str) -> str:
    return citation.split("-", 1)[0]


def verify_answer(
    answer: str,
    retrieved: list[RetrievalResult],
    expected_tickers: list[str] | None = None,
) -> VerificationResult:
    cited = extract_citations(answer)
    retrieved_by_id = {result.chunk_id: result for result in retrieved}
    valid = [citation for citation in cited if citation in retrieved_by_id]
    missing = [citation for citation in cited if citation not in retrieved_by_id]
    wrong_company = [
        citation
        for citation in valid
        if expected_tickers and citation_ticker(citation) not in expected_tickers
    ]
    notes: list[str] = []

    if not cited:
        notes.append("No bracketed citations were found in the answer.")
    if missing:
        notes.append("Some citations were not present in the retrieved evidence.")
    if wrong_company:
        notes.append(
            "Some citations refer to a different company than the question requested: "
            + ", ".join(wrong_company)
        )

    answer_without_citations = re.sub(r"\[[^\]]+\]", "", answer)
    answer_token_count = len(content_tokens(answer_without_citations))
    if answer_token_count < 5:
        notes.append("The answer contains citations but no substantive supported claim.")

    evidence_tokens = content_tokens(" ".join(result.text for result in retrieved))
    sentence_scores = [
        overlap_ratio(content_tokens(sentence), evidence_tokens)
        for sentence in sentence_split(answer_without_citations)
    ]
    support_score = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
    citation_score = len(valid) / len(cited) if cited else 0.0
    if wrong_company and cited:
        citation_score *= max(0.0, 1.0 - (len(wrong_company) / len(cited)))
    retrieval_score = max((result.score for result in retrieved), default=0.0)
    retrieval_score = max(0.0, min(1.0, retrieval_score))

    confidence = (0.45 * citation_score) + (0.35 * support_score) + (0.20 * retrieval_score)
    confidence = round(max(0.0, min(1.0, confidence)), 2)
    if answer_token_count < 5:
        confidence = min(confidence, 0.25)

    if confidence >= 0.72 and not missing and not wrong_company:
        risk = "Low"
    elif confidence >= 0.45:
        risk = "Medium"
    else:
        risk = "High"

    if support_score < 0.25:
        notes.append("Low lexical overlap between answer claims and retrieved evidence.")

    return VerificationResult(
        confidence_score=confidence,
        hallucination_risk=risk,
        valid_citations=valid,
        missing_citations=missing,
        notes=notes,
    )
