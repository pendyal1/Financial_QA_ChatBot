from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finrag.config import DEFAULT_RERANKER_MODEL
from finrag.models import RetrievalResult


def reranker_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def load_reranker(model_name: str = DEFAULT_RERANKER_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = reranker_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def rerank_results(
    question: str,
    candidates: list[RetrievalResult],
    model_name: str = DEFAULT_RERANKER_MODEL,
    batch_size: int = 8,
) -> list[RetrievalResult]:
    if len(candidates) <= 1:
        return candidates

    tokenizer, model, device = load_reranker(model_name)
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            pairs = [[question, candidate.text[:2500]] for candidate in batch]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits.view(-1).float().detach().cpu().tolist()
            scores.extend(logits)

    reranked = [
        RetrievalResult(
            chunk_id=candidate.chunk_id,
            score=float(score),
            ticker=candidate.ticker,
            company=candidate.company,
            source=candidate.source,
            source_url=candidate.source_url,
            text=candidate.text,
        )
        for candidate, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked
