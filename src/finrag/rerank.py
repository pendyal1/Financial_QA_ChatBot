"""
finrag.rerank
--------------
Cross-encoder reranking of retrieved passages.

Uses BAAI/bge-reranker-v2-m3 to rescore (question, passage) pairs and
reorder results by relevance. Slot in after any retrieval step.

Usage
-----
    from finrag.rerank import rerank
    from finrag.sec_live import live_retrieve

    results = live_retrieve(question, top_k=15)   # fetch more than needed
    results = rerank(question, results, top_k=5)  # keep the best 5
"""
from __future__ import annotations

from finrag.models import RetrievalResult

_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_reranker_cache = None


def _get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_cache = CrossEncoder(_RERANKER_MODEL)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load reranker model {_RERANKER_MODEL}. "
                "Install sentence-transformers and ensure internet access."
            ) from exc
    return _reranker_cache


def rerank(
    question: str,
    results: list[RetrievalResult],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    Rerank retrieved passages using a cross-encoder.

    Fetches more passages than needed (e.g. top_k=15) then reranks to top_k=5
    for best quality. The reranker reads the question and each passage together,
    giving much more accurate relevance scores than embedding similarity alone.

    Parameters
    ----------
    question : str
        The original user question.
    results : list[RetrievalResult]
        Passages from FAISS or live retrieval to rerank.
    top_k : int
        How many to return after reranking.

    Returns
    -------
    list[RetrievalResult]
        Reranked and trimmed to top_k, with updated scores.
    """
    if not results:
        return results

    reranker = _get_reranker()
    pairs = [(question, r.text) for r in results]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    reranked: list[RetrievalResult] = []
    for result, score in ranked[:top_k]:
        reranked.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                score=round(float(score), 4),
                ticker=result.ticker,
                company=result.company,
                source=result.source,
                source_url=result.source_url,
                text=result.text,
            )
        )
    return reranked
