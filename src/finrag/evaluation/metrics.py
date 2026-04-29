"""
finrag.evaluation.metrics
--------------------------
Unified evaluation metrics for the ablation study.

Covers four measurement areas:
  1. Answer accuracy     — exact match, token F1
  2. Numerical accuracy  — extract numbers, compare within tolerance
  3. Hallucination       — precision/recall/F1 over claim labels, AUROC over scores
  4. Retrieval           — recall@k (whether gold passage was retrieved)

All functions are pure and stateless — pass raw strings or lists and get back
a scalar or dict.  No model loading happens here.
"""
from __future__ import annotations

import re
from collections import Counter

from finrag.models import ClaimVerification, HallucinationReport

# ─────────────────────────────────────────────────────────────────────────────
# 1. Answer accuracy
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lower-case, strip punctuation/commas from numbers, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)   # 1,234 → 1234
    text = text.replace("$", "").replace("%", " percent")
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> bool:
    """True if prediction and gold normalize to the same string."""
    return normalize(prediction) == normalize(gold)


def token_f1(prediction: str, gold: str) -> float:
    """
    Token-level F1 between prediction and gold after normalization.
    Standard metric from SQuAD / FinanceBench papers.
    """
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    overlap = 0
    for token in pred_tokens:
        if gold_counts.get(token, 0) > 0:
            overlap += 1
            gold_counts[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Numerical accuracy
# ─────────────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:\s*(?:billion|million|thousand|B|M|K))?", re.IGNORECASE)
_SCALE = {"billion": 1e9, "million": 1e6, "thousand": 1e3, "b": 1e9, "m": 1e6, "k": 1e3}


def _parse_number(token: str) -> float | None:
    """Parse a single numeric token (possibly with scale suffix) into a float."""
    token = token.strip().replace(",", "")
    match = re.match(r"([-+]?\d+\.?\d*)\s*([a-zA-Z]*)", token)
    if not match:
        return None
    value = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix in _SCALE:
        value *= _SCALE[suffix]
    return value


def extract_numbers(text: str) -> list[float]:
    """Return all numbers found in *text*, parsed to floats."""
    nums = []
    for tok in _NUM_RE.findall(text):
        v = _parse_number(tok)
        if v is not None:
            nums.append(v)
    return nums


def numerical_accuracy(prediction: str, gold: str, tolerance: float = 0.01) -> bool:
    """
    True if prediction contains a number within *tolerance* (relative) of
    the gold number.  Used for financial Q&A where exact string match fails
    due to rounding or unit differences.

    tolerance=0.01 means ±1%.
    """
    gold_nums = extract_numbers(gold)
    pred_nums = extract_numbers(prediction)
    if not gold_nums or not pred_nums:
        return False

    gold_val = gold_nums[0]
    if gold_val == 0:
        return any(abs(p) < 1e-6 for p in pred_nums)

    return any(abs(p - gold_val) / abs(gold_val) <= tolerance for p in pred_nums)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hallucination detection metrics
# ─────────────────────────────────────────────────────────────────────────────

def hallucination_metrics(
    reports: list[HallucinationReport],
    ground_truth_flags: list[bool],
) -> dict[str, float]:
    """
    Evaluate hallucination detection quality.

    Parameters
    ----------
    reports : list[HallucinationReport]
        One HallucinationReport per answer (from detect_hallucinations()).
    ground_truth_flags : list[bool]
        Parallel list: True = answer IS hallucinated, False = answer is grounded.

    Returns
    -------
    dict with keys:
        precision, recall, f1          — at threshold 0.5 on confidence_score
        auroc                          — AUROC of confidence_score as hallucination detector
        mean_confidence                — average confidence score
        pct_high_risk                  — % of answers classified "High" risk
        claim_accuracy                 — across all claims, fraction with label == "grounded"
    """
    if len(reports) != len(ground_truth_flags):
        raise ValueError("reports and ground_truth_flags must have the same length")

    scores = [r.confidence_score for r in reports]
    # High confidence → grounded; low confidence → hallucinated
    # For AUROC we need "probability of being hallucinated" = 1 - confidence
    hal_scores = [1.0 - s for s in scores]

    tp = fp = fn = tn = 0
    for score, is_hal in zip(hal_scores, ground_truth_flags):
        predicted_hal = score >= 0.5
        if predicted_hal and is_hal:
            tp += 1
        elif predicted_hal and not is_hal:
            fp += 1
        elif not predicted_hal and is_hal:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    auroc = _auroc(hal_scores, ground_truth_flags)

    all_claims = [cv for r in reports for cv in r.claims]
    claim_acc = (
        sum(1 for cv in all_claims if cv.label == "grounded") / len(all_claims)
        if all_claims
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "mean_confidence": round(sum(scores) / len(scores), 4),
        "pct_high_risk": round(
            sum(1 for r in reports if r.overall_risk == "High") / len(reports), 4
        ),
        "claim_accuracy": round(claim_acc, 4),
    }


def _auroc(scores: list[float], labels: list[bool]) -> float:
    """Compute AUROC via sklearn if available, else fall back to manual trapezoid."""
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except Exception:
        pass

    # Manual AUROC: count concordant pairs
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    concordant = 0
    for i, (si, li) in enumerate(zip(scores, labels)):
        for sj, lj in zip(scores, labels):
            if li and not lj:
                if si > sj:
                    concordant += 1
                elif si == sj:
                    concordant += 0.5

    return concordant / (n_pos * n_neg)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_recall_at_k(
    retrieved_ids: list[list[str]],
    gold_ids: list[list[str]],
) -> float:
    """
    Recall@K: fraction of questions where at least one gold chunk_id
    appears in the top-K retrieved chunks.

    Parameters
    ----------
    retrieved_ids : list[list[str]]
        For each question, the list of chunk_ids returned by the retriever.
    gold_ids : list[list[str]]
        For each question, the list of expected/gold chunk_ids.
    """
    if not retrieved_ids:
        return 0.0
    hits = sum(
        1 for ret, gold in zip(retrieved_ids, gold_ids)
        if any(g in ret for g in gold)
    )
    return round(hits / len(retrieved_ids), 4)
