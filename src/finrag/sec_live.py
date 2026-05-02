"""
finrag.sec_live
----------------
Live SEC EDGAR retrieval — resolves a company name or ticker from a question,
fetches the latest 10-K filing in real time, and returns ranked RetrievalResult
objects ready for the Qwen server and hallucination detector.

For numerical questions it also queries the SEC XBRL companyconcept API —
one small call per metric (~2-50 KB each) rather than the full companyfacts
JSON (~300 MB), which was the previous OOM culprit.

Usage
-----
    from finrag.sec_live import live_retrieve

    results = live_retrieve("What was Apple's revenue in FY2024?", top_k=5)
    # returns list[RetrievalResult] — same interface as FAISS retrieval

Flow
----
    question → resolve ticker/CIK → stream 10-K HTML (500 KB cap) → parse
             → chunk → embed → rank → top-k results
             → (+ XBRL structured facts if numerical question)
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
_MAX_HTML_BYTES = 500_000       # stream at most 500 KB — captures Risk Factors,
                                # MD&A, and Income Statement for most 10-Ks
_MAX_DOC_CHARS = 120_000        # cap plain text after HTML parsing
_MAX_CHUNKS_TO_EMBED = 200      # embed at most this many chunks

# Patterns that flag a question as needing structured financial data
_NUMERICAL_RE = re.compile(
    r"\b(revenue|sales|income|profit|loss|earnings|eps|ebitda|margin|"
    r"cash|capex|debt|equity|assets|liabilities|dividend|shares|"
    r"how much|how many|what (?:was|is|were) the (?:total|net|gross))\b",
    re.IGNORECASE,
)

# XBRL metrics: (display_label, [gaap_tags in priority order], unit_key)
# Multiple tags because companies use different GAAP concepts for the same line item.
_XBRL_METRICS: list[tuple[str, list[str], str]] = [
    ("Revenue", [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ], "USD"),
    ("Net Income", ["NetIncomeLoss"], "USD"),
    ("Gross Profit", ["GrossProfit"], "USD"),
    ("Operating Income", ["OperatingIncomeLoss"], "USD"),
    ("EPS (Basic)", ["EarningsPerShareBasic"], "USD/shares"),
    ("EPS (Diluted)", ["EarningsPerShareDiluted"], "USD/shares"),
    ("Total Assets", ["Assets"], "USD"),
    ("Long-Term Debt", ["LongTermDebt", "LongTermDebtNoncurrent"], "USD"),
    ("Cash & Equivalents", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ], "USD"),
    ("R&D Expense", ["ResearchAndDevelopmentExpense"], "USD"),
]

# Module-level caches (populated lazily)
_ticker_map_cache: dict[str, str] | None = None
_name_to_ticker_cache: dict[str, str] | None = None
_ticker_to_name_cache: dict[str, str] | None = None
_model_cache: SentenceTransformer | None = None


# ── SEC request helpers ───────────────────────────────────────────────────────

def _headers(user_agent: str = DEFAULT_SEC_USER_AGENT) -> dict[str, str]:
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}


def _get_json(url: str, user_agent: str = DEFAULT_SEC_USER_AGENT) -> Any:
    resp = requests.get(url, headers=_headers(user_agent), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _stream_html(url: str, user_agent: str = DEFAULT_SEC_USER_AGENT) -> str:
    """Stream filing HTML, stopping at _MAX_HTML_BYTES to avoid loading huge files."""
    resp = requests.get(url, headers=_headers(user_agent), timeout=60, stream=True)
    resp.raise_for_status()
    # Read encoding from headers BEFORE consuming the body stream.
    # (resp.apparent_encoding reads the full body — do NOT use it after streaming.)
    encoding = resp.encoding or "utf-8"
    raw_chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=65_536):
        raw_chunks.append(chunk)
        total += len(chunk)
        if total >= _MAX_HTML_BYTES:
            break
    resp.close()
    raw = b"".join(raw_chunks)[:_MAX_HTML_BYTES]
    return raw.decode(encoding, errors="replace")


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)[:_MAX_DOC_CHARS]


# ── Company resolution ────────────────────────────────────────────────────────

def _load_ticker_map(user_agent: str = DEFAULT_SEC_USER_AGENT) -> dict[str, str]:
    global _ticker_map_cache
    if _ticker_map_cache is None:
        payload = _get_json(COMPANY_TICKERS_URL, user_agent)
        _ticker_map_cache = {
            row["ticker"].upper(): str(row["cik_str"]).zfill(10)
            for row in payload.values()
        }
    return _ticker_map_cache


# Common legal suffixes to strip for loose name matching
_LEGAL_SUFFIX_RE = re.compile(
    r"\s*\b(?:inc|corp|corporation|ltd|llc|co|company|group|holdings|"
    r"plc|sa|ag|nv|bv|lp|trust|technologies|technology|systems|solutions"
    r"|enterprises|international|global|services|financial|bancorp|bancshares)"
    r"\.?\s*$",
    re.IGNORECASE,
)


def _short_name(full_name: str) -> str:
    """Return the company name stripped of legal suffixes, e.g. 'Apple Inc.' → 'apple'."""
    return _LEGAL_SUFFIX_RE.sub("", full_name).strip().lower()


def _load_name_maps(
    user_agent: str = DEFAULT_SEC_USER_AGENT,
) -> tuple[dict[str, str], dict[str, str]]:
    global _name_to_ticker_cache, _ticker_to_name_cache
    if _name_to_ticker_cache is None:
        payload = _get_json(COMPANY_TICKERS_URL, user_agent)
        _name_to_ticker_cache = {}
        _ticker_to_name_cache = {}
        for row in payload.values():
            t = row["ticker"].upper()
            n = row.get("title", "").strip()
            if not n:
                continue
            _ticker_to_name_cache[t] = n
            # Index both the full name and the suffix-stripped version
            _name_to_ticker_cache[n.lower()] = t
            short = _short_name(n)
            if short and short != n.lower():
                # Only add short name if it isn't already claimed by a longer match
                _name_to_ticker_cache.setdefault(short, t)
    return _name_to_ticker_cache, _ticker_to_name_cache


# Shortcut map for the most commonly queried companies.
# Checked before the full SEC name scan to avoid ambiguity from small companies
# that share common words (e.g. "Tesla Energy" conflicting with "Tesla Inc.").
_NAME_SHORTCUTS: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "goldman sachs": "GS",
    "berkshire": "BRK-B",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "walmart": "WMT",
    "exxon": "XOM",
    "visa": "V",
    "mastercard": "MA",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "intel": "INTC",
    "amd": "AMD",
    "qualcomm": "QCOM",
    "boeing": "BA",
    "disney": "DIS",
    "coca-cola": "KO",
    "coca cola": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "uber": "UBER",
    "airbnb": "ABNB",
    "palantir": "PLTR",
    "snowflake": "SNOW",
    "spotify": "SPOT",
    "paypal": "PYPL",
    "shopify": "SHOP",
}


def resolve_company(
    question: str, user_agent: str = DEFAULT_SEC_USER_AGENT
) -> tuple[str, str, str]:
    """
    Detect the company in a question and return (ticker, company_name, cik).
    Raises ValueError if no public company can be identified.

    Strategy:
      1. Check shortcut map for well-known companies (most reliable)
      2. Scan for uppercase ticker tokens (AAPL, MSFT, TSLA …)
      3. Full scan of SEC company name index
    """
    ticker_map = _load_ticker_map(user_agent)
    name_to_ticker, ticker_to_name = _load_name_maps(user_agent)
    q_lower = question.lower()

    # Step 1: shortcut map (longest match wins to avoid "meta" matching "metamaterials")
    for name_key in sorted(_NAME_SHORTCUTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name_key)}\b", q_lower):
            ticker = _NAME_SHORTCUTS[name_key]
            cik = ticker_map.get(ticker, "")
            if cik:
                return ticker, ticker_to_name.get(ticker, ticker), cik

    # Step 2: uppercase ticker tokens (AAPL, NVDA …)
    for candidate in re.findall(r"\b([A-Z]{2,5})\b", question):
        if candidate in ticker_map:
            cik = ticker_map[candidate]
            return candidate, ticker_to_name.get(candidate, candidate), cik

    # Step 3: full SEC name index scan (longest match first)
    for name_lower, ticker in sorted(name_to_ticker.items(), key=lambda x: -len(x[0])):
        if name_lower in q_lower:
            cik = ticker_map.get(ticker, "")
            if cik:
                return ticker, ticker_to_name.get(ticker, ticker), cik

    raise ValueError(
        f"Could not identify a public company in: '{question}'. "
        "Mention the company name or ticker explicitly (e.g. 'Apple' or 'AAPL')."
    )


# ── Filing fetch ──────────────────────────────────────────────────────────────

def _get_latest_10k_url(
    cik: str, user_agent: str = DEFAULT_SEC_USER_AGENT
) -> tuple[str, str, str]:
    """Return (filing_url, filing_date, accession_no) for the most recent 10-K."""
    submissions = _get_json(f"{SEC_DATA_BASE}/submissions/CIK{cik}.json", user_agent)
    recent = submissions["filings"]["recent"]
    for idx, form in enumerate(recent["form"]):
        if form == "10-K":
            accession_no = recent["accessionNumber"][idx]
            filing_date = recent["filingDate"][idx]
            primary_doc = recent["primaryDocument"][idx]
            accession_path = accession_no.replace("-", "")
            url = (
                f"{SEC_BASE}/Archives/edgar/data/{int(cik)}"
                f"/{accession_path}/{primary_doc}"
            )
            return url, filing_date, accession_no
    raise ValueError(f"No 10-K filing found for CIK {cik}")


# ── XBRL structured data (numerical questions) ───────────────────────────────

def _fetch_xbrl_metric(
    cik: str,
    gaap_tags: list[str],
    unit_key: str,
    user_agent: str,
) -> tuple[Any, str] | None:
    """
    Fetch a single financial metric via the companyconcept API.
    Returns (value, period_end_date) for the most recent annual 10-K filing,
    or None if unavailable.

    Uses ~2-50 KB per call vs the ~300 MB companyfacts JSON.
    """
    for tag in gaap_tags:
        try:
            url = (
                f"{SEC_DATA_BASE}/api/xbrl/companyconcept"
                f"/CIK{cik}/us-gaap/{tag}.json"
            )
            resp = requests.get(url, headers=_headers(user_agent), timeout=15)
            if resp.status_code != 200:
                continue
            units = resp.json().get("units", {})
            # EPS tags use "USD/shares"; everything else uses "USD"
            values = units.get(unit_key) or units.get("USD") or units.get("shares", [])
            annual = [v for v in values if v.get("form") == "10-K"]
            if not annual:
                continue
            latest = sorted(annual, key=lambda x: x.get("end", ""), reverse=True)[0]
            return latest.get("val"), latest.get("end", "")
        except Exception:
            continue
    return None


def _fetch_xbrl_summary(
    cik: str, user_agent: str = DEFAULT_SEC_USER_AGENT
) -> str:
    """
    Build a compact structured-data block for key financial metrics.
    Makes one small XBRL companyconcept call per metric (~2-50 KB each).
    """
    lines = ["[SEC Structured Financial Data]"]
    for label, gaap_tags, unit_key in _XBRL_METRICS:
        result = _fetch_xbrl_metric(cik, gaap_tags, unit_key, user_agent)
        if result is None:
            continue
        val, end = result
        if val is None:
            continue
        if isinstance(val, (int, float)) and abs(val) >= 1_000_000:
            val_str = (
                f"${val / 1_000_000_000:.2f}B"
                if abs(val) >= 1e9
                else f"${val / 1_000_000:.1f}M"
            )
        else:
            val_str = str(val)
        lines.append(f"{label} ({end}): {val_str}")
    return "\n".join(lines) if len(lines) > 1 else ""


# ── Chunking and embedding ────────────────────────────────────────────────────

def _chunk_text(text: str, ticker: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
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
    """Lazy-load the embedding model on first use."""
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
    c_embs = model.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    )
    scores = (c_embs @ q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


# ── Public interface ──────────────────────────────────────────────────────────

def live_retrieve(
    question: str,
    top_k: int = 5,
    user_agent: str = DEFAULT_SEC_USER_AGENT,
    include_xbrl: bool = True,
) -> list[RetrievalResult]:
    """
    Full live retrieval pipeline for a financial question.

    Resolves the company, streams the first 500 KB of its latest 10-K from
    SEC EDGAR (enough to cover Risk Factors, MD&A, and the Income Statement),
    finds the most relevant passages, and optionally prepends structured
    XBRL financial facts for numerical questions.

    Parameters
    ----------
    question : str
        User's financial question. Must mention a company name or ticker.
    top_k : int
        Number of text passages to return.
    user_agent : str
        SEC-required User-Agent string (email address recommended).
    include_xbrl : bool
        If True and the question appears numerical, prepend a compact block
        of structured financial metrics fetched via the XBRL companyconcept
        API (~2-50 KB per metric, not the full companyfacts JSON).

    Returns
    -------
    list[RetrievalResult]
        Drop-in replacement for FAISS retrieval results.
    """
    ticker, company, cik = resolve_company(question, user_agent)
    filing_url, filing_date, _ = _get_latest_10k_url(cik, user_agent)

    html = _stream_html(filing_url, user_agent)
    text = _html_to_text(html)
    del html

    chunks = _chunk_text(text, ticker)
    ranked = _rank_chunks(question, chunks, top_k=top_k)
    source_label = f"{company} 10-K ({filing_date})"

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

    if include_xbrl and _NUMERICAL_RE.search(question):
        xbrl_text = _fetch_xbrl_summary(cik, user_agent)
        if xbrl_text:
            xbrl_result = RetrievalResult(
                chunk_id=f"{ticker}_xbrl",
                score=1.0,
                ticker=ticker,
                company=company,
                source=f"{company} SEC XBRL Facts",
                source_url=f"{SEC_DATA_BASE}/api/xbrl/companyconcept/CIK{cik}/us-gaap/",
                text=xbrl_text,
            )
            results = [xbrl_result] + results[: top_k - 1]

    return results
