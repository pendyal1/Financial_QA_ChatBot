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


def analyze_query(question: str) -> QueryIntent:
    risk_question = is_risk_question(question)
    expanded = question
    if risk_question:
        expanded = f"{question}\n{RISK_EXPANSION}"
    return QueryIntent(
        tickers=detect_tickers(question),
        is_risk_question=risk_question,
        expanded_question=expanded,
    )


def evidence_for_question(question: str, text: str) -> str:
    intent = analyze_query(question)
    if not intent.is_risk_question:
        return text
    match = re.search(r"item\s+1a\.?\s+.*?risk\s+factors|item\s+1a", text, flags=re.IGNORECASE)
    if match:
        return text[match.start() :]
    return text
