from __future__ import annotations

import re
from dataclasses import dataclass


COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "TSLA": ["tesla", "tsla"],
    "NVDA": ["nvidia", "nvda"],
    "AMZN": ["amazon", "amzn"],
}

# Financial abbreviation expansions — improves retrieval recall for jargon queries.
# Borrowed from the root retriever.py and extended.
ABBREVIATION_EXPANSIONS: dict[str, str] = {
    "EPS": "earnings per share EPS",
    "P/E": "price to earnings ratio P/E",
    "EBITDA": "earnings before interest taxes depreciation amortization EBITDA",
    "YoY": "year over year YoY",
    "QoQ": "quarter over quarter QoQ",
    "10-K": "annual report 10-K SEC filing",
    "10-Q": "quarterly report 10-Q SEC filing",
    "FCF": "free cash flow FCF",
    "ROE": "return on equity ROE",
    "ROA": "return on assets ROA",
    "CAPEX": "capital expenditure CAPEX",
    "COGS": "cost of goods sold COGS",
    "G&A": "general and administrative expenses",
    "R&D": "research and development",
    "SG&A": "selling general and administrative expenses SG&A",
    "NI": "net income NI",
    "OP": "operating profit OP",
    "GP": "gross profit GP",
    "GM": "gross margin GM",
    "TTM": "trailing twelve months TTM",
}

RISK_TERMS = {
    "risk",
    "risks",
    "risk-factor",
    "risk-factors",
    "cybersecurity",
    "competition",
    "competitive",
    "supply",
    "manufacturing",
    "production",
    "demand",
    "litigation",
    "regulatory",
    "macroeconomic",
}

RISK_EXPANSION = (
    "Item 1A Risk Factors material adverse effect business results operations "
    "financial condition competition cybersecurity regulatory litigation supply chain "
    "macroeconomic demand manufacturing production"
)


@dataclass(frozen=True)
class QueryIntent:
    tickers: list[str]
    is_risk_question: bool
    expanded_question: str
    sub_queries: list[str] = None  # populated by decompose_query for multi-part questions


def detect_tickers(question: str) -> list[str]:
    normalized = question.lower()
    detected: list[str] = []
    for ticker, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                detected.append(ticker)
                break
    return detected


def is_risk_question(question: str) -> bool:
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", question.lower()))
    return bool(tokens & RISK_TERMS)


# Conjunctions that signal a multi-part question worth decomposing.
_DECOMPOSE_CONJUNCTIONS = [" and how ", " and what ", " and when ", " and why ", " and where "]


def decompose_query(question: str) -> list[str]:
    """
    Split a multi-part question into sub-queries for independent retrieval.
    Results from each sub-query are merged (highest score wins per chunk).

    "What was Apple's revenue and how did margins change?"
    → ["What was Apple's revenue?", "how did margins change?"]

    Returns a single-element list if the question doesn't decompose.
    Adapted from teammate's original retriever.py.
    """
    q_lower = question.lower()
    for conj in _DECOMPOSE_CONJUNCTIONS:
        if conj in q_lower:
            idx = q_lower.index(conj)
            first = question[:idx].strip()
            second = question[idx + len(conj):].strip()
            if first and second:
                return [
                    first if first.endswith("?") else first + "?",
                    second if second.endswith("?") else second + "?",
                ]
    return [question]


def expand_abbreviations(question: str) -> str:
    """Replace financial abbreviations with their full forms for better retrieval."""
    for abbrev, expansion in ABBREVIATION_EXPANSIONS.items():
        # Word-boundary-safe replacement (abbrevs with / need special handling)
        pattern = re.escape(abbrev)
        question = re.sub(pattern, expansion, question)
    return question


def analyze_query(question: str) -> QueryIntent:
    risk_question = is_risk_question(question)
    expanded = expand_abbreviations(question)
    if risk_question:
        expanded = f"{expanded}\n{RISK_EXPANSION}"
    sub_queries = decompose_query(question)
    return QueryIntent(
        tickers=detect_tickers(question),
        is_risk_question=risk_question,
        expanded_question=expanded,
        sub_queries=sub_queries,
    )


def evidence_for_question(question: str, text: str) -> str:
    intent = analyze_query(question)
    if not intent.is_risk_question:
        return text
    match = re.search(r"item\s+1a\.?\s+.*?risk\s+factors|item\s+1a", text, flags=re.IGNORECASE)
    if match:
        return text[match.start() :]
    return text
