"""
finrag.live_retrieve
---------------------
Real-time retrieval from a live SEC EDGAR filing URL.

Teammates provide the HTML filing URL from data.sec.gov.
This module fetches it, parses it, finds the most relevant sections,
and returns RetrievalResult objects — the same format the rest of the
pipeline (Qwen server, hallucination detector) already expects.

Usage
-----
    from finrag.live_retrieve import retrieve_from_filing_url

    results = retrieve_from_filing_url(
        url="https://www.sec.gov/Archives/edgar/data/.../filing.htm",
        question="What was Boeing's revenue in FY2024?",
        ticker="BA",
        top_k=5,
    )
    # results is list[RetrievalResult] — pass directly to Qwen server or
    # hallucination detector, same as FAISS retrieval output.

Integration with teammates
--------------------------
    Their code calls SEC API → gets filing URL
    They pass that URL + the user's question to retrieve_from_filing_url()
    They get back RetrievalResult objects
    They POST to /generate with the context built from those results
"""
from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from finrag.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_SEC_USER_AGENT
from finrag.models import RetrievalResult

_CHUNK_SIZE = 800       # characters per chunk
_CHUNK_OVERLAP = 100    # overlap between chunks
_MIN_CHUNK = 80         # discard chunks shorter than this

_model_cache: SentenceTransformer | None = None


def _get_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(model_name)
    return _model_cache


# ── Fetching and parsing ──────────────────────────────────────────────────────

def fetch_and_parse(url: str, user_agent: str = DEFAULT_SEC_USER_AGENT) -> str:
    """Fetch an SEC filing HTML and return clean plain text."""
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return _html_to_text(resp.text)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, url: str, ticker: str) -> list[dict[str, Any]]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + _CHUNK_SIZE, len(text))
        chunk_text = text[start:end].strip()
        if len(chunk_text) >= _MIN_CHUNK:
            chunks.append({
                "chunk_id": f"{ticker}_live_{idx:04d}",
                "text": chunk_text,
                "start": start,
            })
            idx += 1
        start = end - _CHUNK_OVERLAP
    return chunks


# ── Semantic retrieval ────────────────────────────────────────────────────────

def _rank_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    model: SentenceTransformer,
) -> list[tuple[dict[str, Any], float]]:
    """Embed question + chunks, return top_k by cosine similarity."""
    texts = [c["text"] for c in chunks]
    q_emb = model.encode([question], normalize_embeddings=True)
    c_embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    scores = (c_embs @ q_emb.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices]


# ── Public interface ──────────────────────────────────────────────────────────

def retrieve_from_filing_url(
    url: str,
    question: str,
    ticker: str = "UNK",
    company: str = "",
    top_k: int = 5,
    user_agent: str = DEFAULT_SEC_USER_AGENT,
    filing_date: str = "",
) -> list[RetrievalResult]:
    """
    Fetch an SEC filing, find the most relevant sections, return RetrievalResult list.

    Parameters
    ----------
    url : str
        Direct URL to the HTML filing from SEC EDGAR.
    question : str
        The user's financial question.
    ticker : str
        Company ticker symbol (e.g. "BA" for Boeing).
    company : str
        Company name for display purposes.
    top_k : int
        Number of passages to return.
    user_agent : str
        SEC-required User-Agent header.
    filing_date : str
        Filing date string for metadata (e.g. "2024-02-15").

    Returns
    -------
    list[RetrievalResult]
        Same format as FAISS retrieval — drop-in replacement.
    """
    text = fetch_and_parse(url, user_agent)
    chunks = _chunk_text(text, url=url, ticker=ticker)

    if not chunks:
        return []

    model = _get_model()
    ranked = _rank_chunks(question, chunks, top_k=top_k, model=model)

    source_label = f"{company or ticker} 10-K {filing_date}".strip()

    return [
        RetrievalResult(
            chunk_id=chunk["chunk_id"],
            score=round(score, 4),
            ticker=ticker,
            company=company or ticker,
            source=source_label,
            source_url=url,
            text=chunk["text"],
        )
        for chunk, score in ranked
    ]


def build_context_string(results: list[RetrievalResult], max_chars: int = 3000) -> str:
    """
    Format RetrievalResult list into a context string for the Qwen server.
    Mirrors what finrag.answer.build_context does for FAISS results.
    """
    parts = []
    total = 0
    for r in results:
        passage = f"[{r.chunk_id}]\n{r.text}"
        if total + len(passage) > max_chars:
            break
        parts.append(passage)
        total += len(passage)
    return "\n\n---\n\n".join(parts)
