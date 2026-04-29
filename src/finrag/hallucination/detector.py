"""
finrag.hallucination.detector
------------------------------
Orchestrates claim extraction + NLI verification into a HallucinationReport.

Also re-exports the fast lexical helpers (extract_citations, verify_answer)
so existing code that imports from finrag.hallucination_detection continues
to work unchanged.

Team interface (Person B)
-------------------------
    from finrag.hallucination import detect_hallucinations

    # After you have the RAG response:
    report = detect_hallucinations(response.answer, response.retrieved)
    print(report.overall_risk)          # "Low" / "Medium" / "High"
    for cv in report.claims:
        print(f"[{cv.label}] {cv.claim.text}")
"""
from __future__ import annotations

from finrag.hallucination.claim_extractor import extract_claims
from finrag.hallucination.nli_verifier import NLIVerifier
from finrag.hallucination_detection import extract_citations, verify_answer  # backward compat
from finrag.models import ClaimVerification, HallucinationReport, RetrievalResult

# Module-level singleton — loaded lazily on first call
_verifier: NLIVerifier | None = None


def _get_verifier() -> NLIVerifier:
    global _verifier
    if _verifier is None:
        _verifier = NLIVerifier()
    return _verifier


class HallucinationDetector:
    """
    Stateful detector: reuses the same NLI model across multiple calls.

    Prefer this class when processing many answers (e.g., evaluation loop).
    For one-off use, call the module-level detect_hallucinations() instead.

    Example
    -------
    detector = HallucinationDetector()
    for response in rag_responses:
        report = detector.detect(response.answer, response.retrieved)
    """

    def __init__(self, nli_model: str = "cross-encoder/nli-deberta-v3-small") -> None:
        self._verifier = NLIVerifier(nli_model)

    def detect(
        self,
        answer: str,
        retrieved: list[RetrievalResult],
        include_legacy: bool = True,
    ) -> HallucinationReport:
        return _run_detection(answer, retrieved, self._verifier, include_legacy)


def detect_hallucinations(
    answer: str,
    retrieved: list[RetrievalResult],
    include_legacy: bool = True,
) -> HallucinationReport:
    """
    Module-level convenience function — uses a shared NLI model singleton.

    Parameters
    ----------
    answer : str
        The LLM-generated answer to verify.
    retrieved : list[RetrievalResult]
        Passages returned by the retriever (source of truth).
    include_legacy : bool
        Also run the fast lexical check and attach it as
        report.legacy_verification for comparison.

    Returns
    -------
    HallucinationReport
    """
    return _run_detection(answer, retrieved, _get_verifier(), include_legacy)


# ── Core logic ─────────────────────────────────────────────────────────────────

def _run_detection(
    answer: str,
    retrieved: list[RetrievalResult],
    verifier: NLIVerifier,
    include_legacy: bool,
) -> HallucinationReport:
    claims = extract_claims(answer)
    verified: list[ClaimVerification] = verifier.verify_all(claims, retrieved)

    grounded = sum(1 for cv in verified if cv.label == "grounded")
    partial = sum(1 for cv in verified if cv.label == "partial")
    unsupported = sum(1 for cv in verified if cv.label == "unsupported")
    total = len(verified)

    if total == 0:
        confidence = 0.0
    else:
        # Weight: grounded=1.0, partial=0.5, unsupported=0.0
        confidence = round((grounded + 0.5 * partial) / total, 3)

    # Numerical claims raise the bar: any unsupported numerical claim → High risk
    numerical_claims = [cv for cv in verified if cv.claim.is_numerical]
    has_unsupported_number = any(cv.label == "unsupported" for cv in numerical_claims)

    if has_unsupported_number:
        risk = "High"
    elif confidence >= 0.72 and unsupported == 0:
        risk = "Low"
    elif confidence >= 0.45:
        risk = "Medium"
    else:
        risk = "High"

    legacy = verify_answer(answer, retrieved) if include_legacy else None

    return HallucinationReport(
        answer=answer,
        claims=verified,
        overall_risk=risk,
        confidence_score=confidence,
        grounded_count=grounded,
        partial_count=partial,
        unsupported_count=unsupported,
        legacy_verification=legacy,
    )


# Re-export for any code that does:
#   from finrag.hallucination.detector import extract_citations, verify_answer
__all__ = [
    "HallucinationDetector",
    "detect_hallucinations",
    "extract_citations",
    "verify_answer",
]
