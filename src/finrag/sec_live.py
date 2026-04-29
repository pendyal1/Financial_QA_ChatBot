"""
finrag.sec_live
----------------
Live SEC EDGAR retrieval — resolves a company name or ticker from a question,
fetches the latest 10-K filing in real time, and returns ranked RetrievalResult
objects ready for the Qwen server and hallucination detector.

Also queries the SEC companyfacts API for numeric/financial-statement questions
to get precise structured figures alongside the filing text.

Usage
-----
    from finrag.sec_live import live_retrieve

    results = live_retrieve("What was Apple's revenue in FY2024?", top_k=5)
    # returns list[RetrievalResult] — same as FAISS retrieval output

Flow
----
    question → resolve ticker/CIK → fetch 10-K HTML → parse → chunk
             → (companyfacts if numerical) → embed → rank → top-k results
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from finrag.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_SEC_USER_AGENT
from finrag.models import RetrievalResult

# ── Constants ─────────────────────────────────────────────────────────────────

SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"
COMPANY_TICKERS_URL = f"{SEC_BASE}/files/company_tickers.json"

_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100
_MIN_CHUNK = 80
_MAX_HTML_CHARS = 1_500_000  # cap raw HTML before BeautifulSoup parses it (~1.5MB)
_MAX_DOC_CHARS = 120_000    # cap plain text after parsing
_MAX_CHUNKS_TO_EMBED = 200  # embed at most this many chunks

# Patterns that suggest the question needs numerical/financial-statement data
_NUMERICAL_QUESTION_RE = re.compile(
    r"\b(revenue|sales|income|profit|loss|earnings|eps|ebitda|margin|"
    r"cash flow|capex|debt|equity|assets|liabilities|dividend|shares|"
    r"how much|how many|what (?:was|is|were) the (?:total|net|gross))\b",
    re.IGNORECASE,
)

# Module-level caches
_ticker_map_cache: dict[str, str] | None = None  # ticker → CIK
_model_cache: SentenceTransformer | None = None


# ── SEC helpers ───────────────────────────────────────────────────────────────

def _headers(user_agent: str = DEFAULT_SEC_USER_AGENT) -> dict[str, str]:
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}


def _get_json(url: str, user_agent: str = DEFAULT_SEC_USER_AGENT) -> Any:
    resp = requests.get(url, headers=_headers(user_agent), timeout=45)
    resp.raise_for_status()
    return resp.json()


def _get_html(url: str, user_agent: str = DEFAULT_SEC_USER_AGENT) -> str:
    resp = requests.get(url, headers=_headers(user_agent), timeout=60)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


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


# ── Company resolution ────────────────────────────────────────────────────────

def _load_ticker_map(user_agent: str = DEFAULT_SEC_USER_AGENT) -> dict[str, str]:
    """Load and cache the SEC ticker → CIK mapping."""
    global _ticker_map_cache
    if _ticker_map_cache is None:
        payload = _get_json(COMPANY_TICKERS_URL, user_agent)
        _ticker_map_cache = {
            row["ticker"].upper(): str(row["cik_str"]).zfill(10)
            for row in payload.values()
        }
    return _ticker_map_cache


_name_to_ticker_cache: dict[str, str] | None = None
_ticker_to_name_cache: dict[str, str] | None = None


def _load_name_maps(user_agent: str = DEFAULT_SEC_USER_AGENT) -> tuple[dict[str, str], dict[str, str]]:
    global _name_to_ticker_cache, _ticker_to_name_cache
    if _name_to_ticker_cache is None:
        payload = _get_json(COMPANY_TICKERS_URL, user_agent)
        _name_to_ticker_cache = {}
        _ticker_to_name_cache = {}
        for row in payload.values():
            t = row["ticker"].upper()
            n = row.get("title", "").strip()
            if n:
                _name_to_ticker_cache[n.lower()] = t
                _ticker_to_name_cache[t] = n
    return _name_to_ticker_cache, _ticker_to_name_cache


def resolve_company(question: str, user_agent: str = DEFAULT_SEC_USER_AGENT) -> tuple[str, str, str]:
    """
    Extract company ticker and resolve to CIK from a question string.

    Returns (ticker, company_name, cik). Raises ValueError if no company found.

    Strategy:
      1. Look for explicit ticker symbols (e.g. AAPL, MSFT)
      2. Match known company names (e.g. "Apple", "Microsoft")
    """
    ticker_map = _load_ticker_map(user_agent)
    name_to_ticker, ticker_to_name = _load_name_maps(user_agent)

    # Step 1: look for uppercase ticker-like tokens (2-5 uppercase letters)
    candidates = re.findall(r"\b([A-Z]{1,5})\b", question)
    for candidate in candidates:
        if candidate in ticker_map:
            cik = ticker_map[candidate]
            name = ticker_to_name.get(candidate, candidate)
            return candidate, name, cik

    # Step 2: look for known company names in the question (case-insensitive)
    q_lower = question.lower()
    # Sort by length descending so longer names match first (e.g. "Johnson & Johnson" before "Johnson")
    for name_lower, ticker in sorted(name_to_ticker.items(), key=lambda x: -len(x[0])):
        if name_lower in q_lower:
            cik = ticker_map.get(ticker, "")
            if cik:
                return ticker, ticker_to_name.get(ticker, ticker), cik

    raise ValueError(
        f"Could not identify a public company in the question: '{question}'. "
        "Try mentioning the company name or ticker explicitly."
    )


# ── Filing fetch ──────────────────────────────────────────────────────────────

def _get_latest_10k_url(cik: str, user_agent: str = DEFAULT_SEC_USER_AGENT) -> tuple[str, str, str]:
    """
    Return (filing_url, filing_date, accession_no) for the latest 10-K.
    """
    submissions = _get_json(f"{SEC_DATA_BASE}/submissions/CIK{cik}.json", user_agent)
    recent = submissions["filings"]["recent"]

    for idx, form in enumerate(recent["form"]):
        if form == "10-K":
            accession_no = recent["accessionNumber"][idx]
            filing_date = recent["filingDate"][idx]
            primary_doc = recent["primaryDocument"][idx]
            accession_path = accession_no.replace("-", "")
            cik_int = str(int(cik))
            url = f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{accession_path}/{primary_doc}"
            return url, filing_date, accession_no

    raise ValueError(f"No 10-K filing found for CIK {cik}")


# ── companyfacts (numerical questions) ───────────────────────────────────────

def _fetch_companyfacts_summary(cik: str, user_agent: str = DEFAULT_SEC_USER_AGENT) -> str:
    """
    Fetch key financial facts from SEC XBRL companyfacts API and format as text.
    Used to supplement retrieval for numerical questions.
    """
    try:
        data = _get_json(f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{cik}.json", user_agent)
    except Exception:
        return ""

    facts = data.get("facts", {})
    us_gaap = facts.get("us-gaap", {})

    # Key metrics to extract
    METRICS = [
        ("Revenues", "Revenue"),
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue"),
        ("NetIncomeLoss", "Net Income"),
        ("GrossProfit", "Gross Profit"),
        ("OperatingIncomeLoss", "Operating Income"),
        ("EarningsPerShareBasic", "EPS Basic"),
        ("EarningsPerShareDiluted", "EPS Diluted"),
        ("Assets", "Total Assets"),
        ("LongTermDebt", "Long Term Debt"),
        ("CashAndCashEquivalentsAtCarryingValue", "Cash"),
        ("ResearchAndDevelopmentExpense", "R&D Expense"),
    ]

    lines = ["[SEC Structured Financial Data]"]
    for gaap_key, label in METRICS:
        if gaap_key not in us_gaap:
            continue
        units = us_gaap[gaap_key].get("units", {})
        values = units.get("USD", units.get("shares", []))
        if not values:
            continue
        # Get the most recent annual value
        annual = [v for v in values if v.get("form") == "10-K"]
        if not annual:
            continue
        latest = sorted(annual, key=lambda x: x.get("end", ""), reverse=True)[0]
        val = latest.get("val", "")
        end = latest.get("end", "")
        if isinstance(val, (int, float)) and val > 1_000_000:
            val_str = f"${val / 1_000_000_000:.2f}B" if val >= 1e9 else f"${val / 1_000_000:.1f}M"
        else:
            val_str = str(val)
        lines.append(f"{label} ({end}): {val_str}")

    return "\n".join(lines) if len(lines) > 1 else ""


# ── Chunking and embedding ────────────────────────────────────────────────────

def _chunk_text(text: str, ticker: str) -> list[dict[str, Any]]:
    text = text[:_MAX_DOC_CHARS]
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + _CHUNK_SIZE, len(text))
        chunk_text = text[start:end].strip()
        if len(chunk_text) >= _MIN_CHUNK:
            chunks.append({"chunk_id": f"{ticker}_live_{idx:04d}", "text": chunk_text})
            idx += 1
        start = end - _CHUNK_OVERLAP
    return chunks


def _get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    return _model_cache


def _rank_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[tuple[dict[str, Any], float]]:
    model = _get_model()
    chunks = chunks[:_MAX_CHUNKS_TO_EMBED]
    texts = [c["text"] for c in chunks]
    q_emb = model.encode([question], normalize_embeddings=True)
    c_embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    scores = (c_embs @ q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


# ── Public interface ──────────────────────────────────────────────────────────

def live_retrieve(
    question: str,
    top_k: int = 5,
    user_agent: str = DEFAULT_SEC_USER_AGENT,
    include_companyfacts: bool = True,
) -> list[RetrievalResult]:
    """
    Full live retrieval pipeline for a financial question.

    Resolves the company, fetches its latest 10-K from SEC EDGAR,
    finds the most relevant passages, and optionally prepends
    structured companyfacts for numerical questions.

    Parameters
    ----------
    question : str
        The user's financial question. Must mention a company name or ticker.
    top_k : int
        Number of text passages to return.
    user_agent : str
        SEC-required User-Agent string.
    include_companyfacts : bool
        If True and the question is numerical, prepend a structured
        financial facts chunk from the companyfacts API.

    Returns
    -------
    list[RetrievalResult]
        Drop-in replacement for FAISS retrieval results.
    """
    ticker, company, cik = resolve_company(question, user_agent)
    filing_url, filing_date, _ = _get_latest_10k_url(cik, user_agent)

    html = _get_html(filing_url, user_agent)
    html = html[:_MAX_HTML_CHARS]
    text = _html_to_text(html)
    del html
    chunks = _chunk_text(text, ticker)

    ranked = _rank_chunks(question, chunks, top_k=top_k)
    source_label = f"{company} 10-K {filing_date}"

    results: list[RetrievalResult] = [
        RetrievalResult(
            chunk_id=chunk["chunk_id"],
            score=round(score, 4),
            ticker=ticker,
            company=company,
            source=source_label,
            source_url=filing_url,
            text=chunk["text"],
        )
        for chunk, score in ranked
    ]

    # Prepend companyfacts chunk for numerical questions
    if include_companyfacts and _NUMERICAL_QUESTION_RE.search(question):
        facts_text = _fetch_companyfacts_summary(cik, user_agent)
        if facts_text:
            facts_result = RetrievalResult(
                chunk_id=f"{ticker}_companyfacts",
                score=1.0,
                ticker=ticker,
                company=company,
                source=f"{company} SEC XBRL Facts",
                source_url=f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{cik}.json",
                text=facts_text,
            )
            results = [facts_result] + results[:top_k - 1]

    return results
