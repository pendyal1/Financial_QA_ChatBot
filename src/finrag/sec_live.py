from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from finrag.chunk_documents import chunk_words
from finrag.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SEC_USER_AGENT,
    SEC_CACHE_DIR,
    ensure_data_dirs,
)
from finrag.download_sec_filings import (
    COMPANY_TICKERS_URL,
    SEC_BASE,
    SEC_DATA_BASE,
    get_json,
    get_text,
    html_to_text,
    safe_name,
)
from finrag.models import DocumentChunk, RetrievalResult
from finrag.query import analyze_query
from finrag.rerank import rerank_results
from finrag.retrieve import lexical_score, risk_score


COMMON_COMPANY_ALIASES = {
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "facebook": "META",
    "meta": "META",
    "netflix": "NFLX",
    "berkshire hathaway": "BRK-B",
}

COMPANYFACTS_KEYWORD_MAP = {
    "revenue": [("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"), ("us-gaap", "SalesRevenueNet")],
    "net sales": [("us-gaap", "SalesRevenueNet"), ("us-gaap", "Revenues")],
    "net income": [("us-gaap", "NetIncomeLoss")],
    "operating income": [("us-gaap", "OperatingIncomeLoss")],
    "operating expenses": [("us-gaap", "OperatingExpenses")],
    "gross profit": [("us-gaap", "GrossProfit")],
    "assets": [("us-gaap", "Assets")],
    "liabilities": [("us-gaap", "Liabilities")],
    "cash": [("us-gaap", "CashAndCashEquivalentsAtCarryingValue")],
    "cash flow": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
    "eps": [("us-gaap", "EarningsPerShareDiluted"), ("us-gaap", "EarningsPerShareBasic")],
    "shareholders equity": [("us-gaap", "StockholdersEquity")],
}


@dataclass(frozen=True)
class ResolvedCompany:
    ticker: str
    cik: str
    company: str


def normalize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(
        r"\b(corporation|corp|incorporated|inc|company|co|holdings|holding|group|plc|ltd|limited)\b",
        " ",
        value,
    )
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=1)
def sec_company_index(user_agent: str = DEFAULT_SEC_USER_AGENT) -> list[ResolvedCompany]:
    payload = get_json(COMPANY_TICKERS_URL, user_agent)
    entries: list[ResolvedCompany] = []
    for row in payload.values():
        entries.append(
            ResolvedCompany(
                ticker=str(row["ticker"]).upper(),
                cik=str(row["cik_str"]).zfill(10),
                company=str(row["title"]),
            )
        )
    return entries


def resolve_company(question: str, user_agent: str = DEFAULT_SEC_USER_AGENT) -> ResolvedCompany:
    question_lower = question.lower()
    for alias, ticker in COMMON_COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", question_lower):
            for entry in sec_company_index(user_agent):
                if entry.ticker == ticker:
                    return entry

    explicit_ticker_matches = re.findall(r"\b[A-Z]{1,5}(?:-[A-Z])?\b", question.upper())
    by_ticker = {entry.ticker: entry for entry in sec_company_index(user_agent)}
    for ticker in explicit_ticker_matches:
        if ticker in by_ticker:
            return by_ticker[ticker]

    normalized_question = normalize_name(question)
    best: ResolvedCompany | None = None
    best_len = 0
    for entry in sec_company_index(user_agent):
        full_name = normalize_name(entry.company)
        if full_name and full_name in normalized_question and len(full_name) > best_len:
            best = entry
            best_len = len(full_name)
            continue
        ticker_as_word = normalize_name(entry.ticker)
        if ticker_as_word and re.search(rf"\b{re.escape(ticker_as_word)}\b", normalized_question):
            return entry
    if best is not None:
        return best
    raise ValueError("Please mention a public company name or ticker in the question.")


def preferred_forms(question: str) -> list[str]:
    normalized = question.lower()
    if any(term in normalized for term in ["earnings release", "earnings call", "guidance", "press release"]):
        return ["8-K", "10-Q", "10-K"]
    if any(term in normalized for term in ["quarter", "q1", "q2", "q3", "q4", "three months", "nine months"]):
        return ["10-Q", "10-K", "8-K"]
    if any(term in normalized for term in ["risk", "risks", "competition", "cybersecurity", "litigation", "regulatory"]):
        return ["10-K", "10-Q"]
    return ["10-K", "10-Q", "8-K"]


def submission_cache_path(company: ResolvedCompany) -> Path:
    return SEC_CACHE_DIR / company.ticker / "submissions.json"


def companyfacts_cache_path(company: ResolvedCompany) -> Path:
    return SEC_CACHE_DIR / company.ticker / "companyfacts.json"


def load_company_submissions(company: ResolvedCompany, user_agent: str = DEFAULT_SEC_USER_AGENT) -> dict[str, Any]:
    ensure_data_dirs()
    cache_path = submission_cache_path(company)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = f"{SEC_DATA_BASE}/submissions/CIK{company.cik}.json"
    payload = get_json(url, user_agent)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def filing_records(submissions: dict[str, Any], forms: list[str], limit_per_form: int = 1) -> list[dict[str, str]]:
    recent = submissions["filings"]["recent"]
    counts = {form: 0 for form in forms}
    records: list[dict[str, str]] = []
    for idx, filing_form in enumerate(recent["form"]):
        if filing_form not in counts or counts[filing_form] >= limit_per_form:
            continue
        records.append(
            {
                "form": filing_form,
                "filing_date": recent["filingDate"][idx],
                "report_date": recent["reportDate"][idx],
                "accession_no": recent["accessionNumber"][idx],
                "primary_document": recent["primaryDocument"][idx],
            }
        )
        counts[filing_form] += 1
        if all(count >= limit_per_form for count in counts.values()):
            break
    return records


def filing_doc_url(company: ResolvedCompany, record: dict[str, str]) -> str:
    accession_path = record["accession_no"].replace("-", "")
    cik_int = str(int(company.cik))
    return f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{accession_path}/{record['primary_document']}"


def filing_cache_prefix(company: ResolvedCompany, record: dict[str, str]) -> Path:
    filename = safe_name(f"{company.ticker}_{record['filing_date']}_{record['form']}_{record['accession_no']}")
    return SEC_CACHE_DIR / company.ticker / filename


def fetch_filing_text(
    company: ResolvedCompany,
    record: dict[str, str],
    user_agent: str = DEFAULT_SEC_USER_AGENT,
) -> tuple[str, dict[str, Any]]:
    ensure_data_dirs()
    prefix = filing_cache_prefix(company, record)
    text_path = prefix.with_suffix(".txt")
    meta_path = prefix.with_suffix(".json")
    if text_path.exists() and meta_path.exists():
        return text_path.read_text(encoding="utf-8"), json.loads(meta_path.read_text(encoding="utf-8"))

    url = filing_doc_url(company, record)
    raw_html = get_text(url, user_agent)
    text = html_to_text(raw_html)
    metadata = {
        "doc_id": prefix.name,
        "ticker": company.ticker,
        "company": company.company,
        "cik": company.cik,
        "form": record["form"],
        "filing_date": record["filing_date"],
        "report_date": record.get("report_date", ""),
        "accession_no": record["accession_no"],
        "primary_document": record["primary_document"],
        "source_url": url,
        "source": f"{company.company} {record['form']} filed {record['filing_date']}",
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return text, metadata


def filing_chunks_for_question(
    company: ResolvedCompany,
    question: str,
    user_agent: str = DEFAULT_SEC_USER_AGENT,
    limit_per_form: int = 1,
    chunk_words_count: int = 450,
    overlap_words_count: int = 80,
) -> list[DocumentChunk]:
    submissions = load_company_submissions(company, user_agent)
    records = filing_records(submissions, preferred_forms(question), limit_per_form=limit_per_form)
    chunks: list[DocumentChunk] = []
    for record in records:
        text, metadata = fetch_filing_text(company, record, user_agent)
        for idx, chunk_text in enumerate(chunk_words(text, chunk_words_count, overlap_words_count), start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{company.ticker}-{metadata['filing_date']}-{idx:04d}",
                    doc_id=metadata["doc_id"],
                    ticker=company.ticker,
                    company=company.company,
                    form=metadata["form"],
                    filing_date=metadata["filing_date"],
                    report_date=metadata.get("report_date", ""),
                    accession_no=metadata["accession_no"],
                    source_url=metadata["source_url"],
                    source=metadata["source"],
                    text=chunk_text,
                )
            )
    if not chunks:
        raise ValueError(f"No relevant SEC filings found for {company.company}.")
    return chunks


def load_companyfacts(company: ResolvedCompany, user_agent: str = DEFAULT_SEC_USER_AGENT) -> dict[str, Any]:
    ensure_data_dirs()
    cache_path = companyfacts_cache_path(company)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{company.cik}.json"
    payload = get_json(url, user_agent)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def latest_fact_values(facts: dict[str, Any], taxonomy: str, concept: str) -> list[dict[str, Any]]:
    concept_data = facts.get("facts", {}).get(taxonomy, {}).get(concept, {})
    units = concept_data.get("units", {})
    all_facts: list[dict[str, Any]] = []
    for unit_name, values in units.items():
        for value in values:
            if value.get("form") not in {"10-K", "10-Q", "20-F", "40-F"}:
                continue
            enriched = dict(value)
            enriched["unit"] = unit_name
            enriched["taxonomy"] = taxonomy
            enriched["concept"] = concept
            all_facts.append(enriched)
    all_facts.sort(
        key=lambda item: (
            item.get("fy", 0),
            item.get("fp", ""),
            item.get("filed", ""),
            item.get("end", ""),
        ),
        reverse=True,
    )
    return all_facts


def number_keywords(question: str) -> list[tuple[str, list[tuple[str, str]]]]:
    normalized = question.lower()
    matches = []
    for keyword, concepts in COMPANYFACTS_KEYWORD_MAP.items():
        if keyword in normalized:
            matches.append((keyword, concepts))
    return matches


def companyfacts_chunks_for_question(
    company: ResolvedCompany,
    question: str,
    user_agent: str = DEFAULT_SEC_USER_AGENT,
) -> list[DocumentChunk]:
    matched = number_keywords(question)
    if not matched:
        return []
    facts = load_companyfacts(company, user_agent)
    chunks: list[DocumentChunk] = []
    chunk_idx = 1
    companyfacts_url = f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{company.cik}.json"
    seen_texts: set[str] = set()
    for keyword, concepts in matched:
        for taxonomy, concept in concepts:
            for fact in latest_fact_values(facts, taxonomy, concept)[:2]:
                end = fact.get("end", "")
                filed = fact.get("filed", "")
                form = fact.get("form", "")
                value = fact.get("val", "")
                unit = fact.get("unit", "")
                fiscal = f"FY{fact['fy']}" if fact.get("fy") else "the reported period"
                text = (
                    f"According to the SEC companyfacts API for {company.company}, {keyword} was {value} "
                    f"({unit}) for {fiscal}, period ending {end}, reported in a {form} filed on {filed}. "
                    f"This fact comes from taxonomy {taxonomy} concept {concept}."
                )
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                filing_date = filed or datetime.utcnow().date().isoformat()
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{company.ticker}-{filing_date}-{9000 + chunk_idx:04d}",
                        doc_id=f"{company.ticker}_companyfacts_{concept}",
                        ticker=company.ticker,
                        company=company.company,
                        form=form or "XBRL",
                        filing_date=filing_date,
                        report_date=end,
                        accession_no=fact.get("accn", "companyfacts"),
                        source_url=companyfacts_url,
                        source=f"{company.company} SEC companyfacts {concept}",
                        text=text,
                    )
                )
                chunk_idx += 1
    return chunks


class LiveSECRetriever:
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        user_agent: str = DEFAULT_SEC_USER_AGENT,
    ) -> tuple[ResolvedCompany, list[RetrievalResult]]:
        company = resolve_company(question, user_agent=user_agent)
        filing_chunks = filing_chunks_for_question(company, question, user_agent=user_agent)
        fact_chunks = companyfacts_chunks_for_question(company, question, user_agent=user_agent)
        all_chunks = fact_chunks + filing_chunks
        intent = analyze_query(question)
        query = self.model.encode(
            [intent.expanded_question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")[0]
        embeddings = self.model.encode(
            [chunk.text for chunk in all_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        dense_scores = np.dot(embeddings, query)
        overfetch = min(len(all_chunks), max(top_k * 8, 12))
        dense_order = np.argsort(-dense_scores)[:overfetch]
        dense_results: list[RetrievalResult] = []
        for idx in dense_order.tolist():
            chunk = all_chunks[idx]
            score = float(dense_scores[idx]) + (0.12 * lexical_score(question, chunk.text))
            if intent.is_risk_question:
                score += 0.28 * risk_score(chunk.text)
            dense_results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    ticker=chunk.ticker,
                    company=chunk.company,
                    source=chunk.source,
                    source_url=chunk.source_url,
                    text=chunk.text,
                )
            )
        reranked = rerank_results(question, dense_results)[:top_k]
        return company, reranked


@lru_cache(maxsize=1)
def get_live_retriever(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> LiveSECRetriever:
    return LiveSECRetriever(embedding_model=embedding_model)


def retrieve_live_sec(
    question: str,
    top_k: int = 5,
    user_agent: str = DEFAULT_SEC_USER_AGENT,
) -> tuple[ResolvedCompany, list[RetrievalResult]]:
    return get_live_retriever().retrieve(question=question, top_k=top_k, user_agent=user_agent)
