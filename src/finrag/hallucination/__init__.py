"""
finrag.hallucination — NLI-based hallucination detection.

Quick start (Person B interface):
    from finrag.hallucination import detect_hallucinations
    report = detect_hallucinations(answer, retrieved_chunks)
    print(report.overall_risk)          # "Low" | "Medium" | "High"
    for cv in report.claims:
        print(cv.label, cv.claim.text)  # grounded / partial / unsupported

The detector falls back to lexical overlap scoring when the NLI model
cannot be loaded (no GPU, no internet), so it always returns a result.
"""
from finrag.hallucination.detector import HallucinationDetector, detect_hallucinations

__all__ = ["HallucinationDetector", "detect_hallucinations"]
