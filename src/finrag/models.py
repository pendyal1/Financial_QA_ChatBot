from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    ticker: str
    company: str
    form: str
    filing_date: str
    report_date: str
    accession_no: str
    source_url: str
    source: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    ticker: str
    company: str
    source: str
    source_url: str
    text: str


@dataclass(frozen=True)
class VerificationResult:
    confidence_score: float
    hallucination_risk: str
    valid_citations: list[str]
    missing_citations: list[str]
    notes: list[str]


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    citations: list[str]
    retrieved: list[RetrievalResult]
    verification: VerificationResult
