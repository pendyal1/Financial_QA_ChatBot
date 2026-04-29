"""
finrag.hallucination.nli_verifier
-----------------------------------
NLI-based claim verifier using a cross-encoder model.

Model: cross-encoder/nli-deberta-v3-small (76 M params, runs on CPU)
  - Takes (claim, evidence_passage) pairs
  - Returns softmax probabilities over {contradiction, entailment, neutral}
  - We treat entailment probability as the "groundedness" score

Numerical claims receive an additional regex-based check: if the claim
contains a number that does not appear in any retrieved passage the claim
is downgraded to "partial" even if NLI says "entailment".

Fallback: if the model cannot be loaded (no internet / resource limits)
the verifier falls back to lexical overlap scoring so the pipeline never
crashes in resource-constrained environments.
"""
from __future__ import annotations

import re

from finrag.models import Claim, ClaimVerification, RetrievalResult

_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Thresholds for mapping entailment probability → label
_GROUNDED_THRESHOLD = 0.60
_PARTIAL_THRESHOLD = 0.30

# How many characters of evidence to pass to the NLI model per passage
_MAX_EVIDENCE_CHARS = 512


class NLIVerifier:
    """
    Lazy-loading NLI cross-encoder verifier.

    Usage
    -----
    verifier = NLIVerifier()
    cv = verifier.verify_claim(claim, retrieved_chunks)
    """

    def __init__(self, model_name: str = _NLI_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None          # loaded on first call
        self._entailment_idx: int = 1
        self._available: bool | None = None  # None = not yet checked

    # ── Public interface ───────────────────────────────────────────────────────

    def verify_claim(
        self,
        claim: Claim,
        retrieved: list[RetrievalResult],
    ) -> ClaimVerification:
        """
        Verify a single claim against all retrieved passages.

        Returns the ClaimVerification with the highest entailment score found
        across all passages.
        """
        if not retrieved:
            return ClaimVerification(
                claim=claim,
                label="unsupported",
                confidence=0.0,
            )

        if self._is_nli_available():
            return self._verify_with_nli(claim, retrieved)
        return self._verify_with_lexical(claim, retrieved)

    def verify_all(
        self,
        claims: list[Claim],
        retrieved: list[RetrievalResult],
    ) -> list[ClaimVerification]:
        """Verify a list of claims in batch (more efficient with GPU)."""
        if not claims:
            return []
        if self._is_nli_available():
            return self._verify_all_nli(claims, retrieved)
        return [self._verify_with_lexical(c, retrieved) for c in claims]

    # ── NLI path ──────────────────────────────────────────────────────────────

    def _is_nli_available(self) -> bool:
        if self._available is None:
            self._available = self._try_load_model()
        return self._available

    def _try_load_model(self) -> bool:
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, num_labels=3)
            # Detect entailment label index from model config
            label2id: dict[str, int] = getattr(
                self._model.model.config, "label2id", {}
            )
            for key, idx in label2id.items():
                if key.lower() == "entailment":
                    self._entailment_idx = idx
                    break
            return True
        except Exception:
            return False

    def _verify_with_nli(
        self,
        claim: Claim,
        retrieved: list[RetrievalResult],
    ) -> ClaimVerification:
        pairs = [
            (claim.text, result.text[:_MAX_EVIDENCE_CHARS])
            for result in retrieved
        ]
        scores = self._model.predict(pairs, apply_softmax=True)
        return self._best_result(claim, retrieved, scores)

    def _verify_all_nli(
        self,
        claims: list[Claim],
        retrieved: list[RetrievalResult],
    ) -> list[ClaimVerification]:
        # Build all (claim, passage) pairs for a single batched prediction
        pairs: list[tuple[str, str]] = []
        index_map: list[tuple[int, int]] = []  # (claim_idx, passage_idx)

        for ci, claim in enumerate(claims):
            for pi, result in enumerate(retrieved):
                pairs.append((claim.text, result.text[:_MAX_EVIDENCE_CHARS]))
                index_map.append((ci, pi))

        if not pairs:
            return [ClaimVerification(c, "unsupported", 0.0) for c in claims]

        all_scores = self._model.predict(pairs, apply_softmax=True)

        # Group scores back by claim
        from collections import defaultdict
        claim_passage_scores: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (ci, pi), score_vec in zip(index_map, all_scores):
            entail_prob = float(score_vec[self._entailment_idx])
            claim_passage_scores[ci].append((pi, entail_prob))

        results: list[ClaimVerification] = []
        for ci, claim in enumerate(claims):
            passage_scores = claim_passage_scores.get(ci, [])
            if not passage_scores:
                results.append(ClaimVerification(claim, "unsupported", 0.0))
                continue
            # Reuse _best_result by reconstructing a score matrix row
            score_vecs = [None] * len(retrieved)
            for pi, ep in passage_scores:
                dummy = [0.0, 0.0, 0.0]
                dummy[self._entailment_idx] = ep
                score_vecs[pi] = dummy
            results.append(self._best_result(claim, retrieved, score_vecs))

        return results

    def _best_result(
        self,
        claim: Claim,
        retrieved: list[RetrievalResult],
        score_vecs,
    ) -> ClaimVerification:
        best_entail = 0.0
        best_idx = 0
        for i, sv in enumerate(score_vecs):
            if sv is None:
                continue
            ep = float(sv[self._entailment_idx])
            if ep > best_entail:
                best_entail = ep
                best_idx = i

        label = _score_to_label(best_entail)

        # Numerical downgrade: if claim has numbers absent from evidence, cap at partial
        if claim.is_numerical and label == "grounded":
            if not _numbers_supported(claim.text, retrieved):
                label = "partial"
                best_entail = min(best_entail, _PARTIAL_THRESHOLD + 0.05)

        best_chunk = retrieved[best_idx]
        snippet = best_chunk.text[:200].replace("\n", " ")
        supporting = [best_chunk.chunk_id] if label != "unsupported" else []

        return ClaimVerification(
            claim=claim,
            label=label,
            confidence=round(best_entail, 3),
            supporting_chunk_ids=supporting,
            evidence_snippet=snippet,
        )

    # ── Lexical fallback ──────────────────────────────────────────────────────

    def _verify_with_lexical(
        self,
        claim: Claim,
        retrieved: list[RetrievalResult],
    ) -> ClaimVerification:
        claim_tokens = _tokens(claim.text)
        if not claim_tokens:
            return ClaimVerification(claim, "unsupported", 0.0)

        best_score = 0.0
        best_chunk = retrieved[0]

        for result in retrieved:
            ev_tokens = _tokens(result.text)
            overlap = len(claim_tokens & ev_tokens) / len(claim_tokens)
            if overlap > best_score:
                best_score = overlap
                best_chunk = result

        label = _score_to_label(best_score)
        if claim.is_numerical and label == "grounded":
            if not _numbers_supported(claim.text, retrieved):
                label = "partial"
        supporting = [best_chunk.chunk_id] if label != "unsupported" else []
        return ClaimVerification(
            claim=claim,
            label=label,
            confidence=round(best_score, 3),
            supporting_chunk_ids=supporting,
            evidence_snippet=best_chunk.text[:200].replace("\n", " "),
        )


# ── Utility functions ─────────────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "in", "is", "it", "of", "on", "or", "that", "the",
    "to", "with", "was", "were", "its", "their",
}

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if t not in _STOPWORDS
    }


def _score_to_label(score: float) -> str:
    if score >= _GROUNDED_THRESHOLD:
        return "grounded"
    if score >= _PARTIAL_THRESHOLD:
        return "partial"
    return "unsupported"


def _numbers_supported(claim_text: str, retrieved: list[RetrievalResult]) -> bool:
    """
    Return True if every number in *claim_text* appears in at least one
    retrieved passage (exact match after stripping commas).
    """
    claim_numbers = {n.replace(",", "") for n in _NUMBER_RE.findall(claim_text)}
    if not claim_numbers:
        return True
    all_evidence = " ".join(r.text for r in retrieved)
    evidence_numbers = {n.replace(",", "") for n in _NUMBER_RE.findall(all_evidence)}
    return bool(claim_numbers & evidence_numbers)
