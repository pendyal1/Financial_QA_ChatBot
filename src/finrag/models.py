from __future__ import annotations

from dataclasses import dataclass, field


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
    chunk_type: str = "text"  # "text" | "table" | "evidence"


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
    """Lightweight heuristic verification — used for fast citation checking."""
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


# ── NLI-based hallucination detection models ──────────────────────────────────

@dataclass
class Claim:
    """A single atomic claim extracted from an LLM answer."""
    text: str
    sentence_index: int
    is_numerical: bool = False  # True when the claim contains financial figures


@dataclass
class ClaimVerification:
    """NLI verification result for one atomic claim."""
    claim: Claim
    # "grounded" | "partial" | "unsupported"
    label: str
    # Probability that the claim is entailed by at least one retrieved passage (0-1)
    confidence: float
    supporting_chunk_ids: list[str] = field(default_factory=list)
    # Short excerpt from the best-matching evidence passage
    evidence_snippet: str = ""


@dataclass
class HallucinationReport:
    """
    Full NLI-based hallucination report for a generated answer.

    Interface for Person B (hallucination detection):
        from finrag.hallucination import detect_hallucinations
        report = detect_hallucinations(answer, retrieved_chunks)
    """
    answer: str
    claims: list[ClaimVerification]
    overall_risk: str           # "Low" | "Medium" | "High"
    confidence_score: float     # 0-1; higher = more grounded
    grounded_count: int
    partial_count: int
    unsupported_count: int
    # Legacy lexical check included for comparison / backward compat
    legacy_verification: VerificationResult | None = None
