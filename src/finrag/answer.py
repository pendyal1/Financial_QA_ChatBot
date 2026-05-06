from __future__ import annotations

import argparse
import os
import re

from openai import OpenAI

from finrag.config import DEFAULT_OPENAI_MODEL
from finrag.hallucination_detection import extract_citations, verify_answer
from finrag.models import RAGResponse, RetrievalResult
from finrag.query import analyze_query, evidence_for_question
from finrag.sec_live import retrieve_live_sec


SYSTEM_PROMPT = """You are a financial analyst assistant for SEC filings.
Answer only from the supplied evidence.
Write in complete prose sentences — never use bullet points, numbered lists, or raw table data.
Start with the answer, not a citation list.
Every factual sentence must include at least one bracketed citation like [AAPL-2024-11-01-0007].
Do not cite sources that are not in the evidence.
If the evidence does not support an answer, say the available filings do not provide enough support."""

BOILERPLATE_PATTERNS = [
    r"\bwebsite\b",
    r"\bsocial media\b",
    r"\breports hub\b",
    r"\bcorporate social responsibility\b",
    r"\binvestor relations website\b",
    r"\bnot incorporated by reference\b",
    r"\bforward-looking statements\b",
    r"\bcause actual results and events to differ materially\b",
    r"\bfollowing summarizes factors\b",
    r"\bnot exhaustive\b",
    r"\bcomplete statement of all potential risks\b",
]

RISK_SIGNAL_PATTERNS = [
    r"\badversely affect\b",
    r"\badverse effect\b",
    r"\brisk\b",
    r"\bcompetition\b",
    r"\bcompetitive\b",
    r"\bcybersecurity\b",
    r"\bsecurity\b",
    r"\bAI\b",
    r"\bcloud\b",
    r"\bregulatory\b",
    r"\blitigation\b",
    r"\bmacroeconomic\b",
    r"\bsupply\b",
    r"\bdemand\b",
    r"\bmanufactur",
    r"\boperations\b",
    r"\bfinancial condition\b",
    r"\bresults of operations\b",
    r"\bharm our business\b",
    r"\bcould harm\b",
]

QUESTION_STOPWORDS = {
    "about",
    "aapl",
    "amazon",
    "amzn",
    "apple",
    "company",
    "describe",
    "did",
    "does",
    "filing",
    "filings",
    "highlight",
    "highlighted",
    "mention",
    "mentioned",
    "microsoft",
    "msft",
    "nvidia",
    "nvda",
    "related",
    "report",
    "reported",
    "reports",
    "risk",
    "risks",
    "say",
    "says",
    "tesla",
    "their",
    "tsla",
    "what",
}


def build_context(
    results: list[RetrievalResult],
    question: str | None = None,
    max_chars_per_chunk: int = 1800,
) -> str:
    blocks = []
    for result in results:
        text = evidence_for_question(question, result.text) if question else result.text
        text = text[:max_chars_per_chunk]
        blocks.append(
            f"Source ID: [{result.chunk_id}]\n"
            f"Company: {result.company} ({result.ticker})\n"
            f"Source: {result.source}\n"
            f"Evidence: {text}"
        )
    return "\n\n".join(blocks)


def is_boilerplate(sentence: str) -> bool:
    normalized = sentence.lower()
    return any(re.search(pattern, normalized) for pattern in BOILERPLATE_PATTERNS)


def risk_signal_score(sentence: str) -> float:
    score = 0.0
    for pattern in RISK_SIGNAL_PATTERNS:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            score += 1.0
    if re.search(r"\bdo not believe\b|\bnot materially affected\b", sentence, flags=re.IGNORECASE):
        score -= 2.0
    return score


def clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"^(Finally|Additionally|Also|However|In addition),?\s+", "", sentence)
    sentence = re.sub(r"^\d+\s+PART I\s+Item 1A\s+", "", sentence)
    if sentence.startswith("if "):
        sentence = "If " + sentence[3:]
    return sentence.strip()


def substantive_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if term not in QUESTION_STOPWORDS
    }


def remove_citations(text: str) -> str:
    return re.sub(r"\[[A-Z0-9_.-]+-\d{4}-\d{2}-\d{2}-\d{4}\]", "", text)


def is_low_content_answer(answer: str) -> bool:
    without_citations = remove_citations(answer)
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", without_citations)
    return len(words) < 8


def extractive_answer(question: str, results: list[RetrievalResult]) -> str:
    if not results:
        return "The available filings do not provide enough support to answer this question."

    intent = analyze_query(question)
    question_terms = substantive_terms(question)
    candidates: list[tuple[float, str, str]] = []
    for result in results:
        evidence_text = evidence_for_question(question, result.text)
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", evidence_text)]
        for sentence in sentences:
            if len(sentence) < 80:
                continue
            if len(sentence) > 1200:
                continue
            if is_boilerplate(sentence):
                continue
            terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", sentence.lower()))
            overlap = len(question_terms & terms)
            if intent.is_risk_question and question_terms and overlap == 0:
                continue
            score = overlap + result.score
            if intent.is_risk_question:
                score += 1.5 * risk_signal_score(sentence)
            candidates.append((score, sentence, result.chunk_id))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates or candidates[0][0] < 1.0:
        top = results[0]
        return (
            "The retrieved filing text is relevant, but it does not provide enough clear support "
            f"for a specific answer. [{top.chunk_id}]"
        )

    selected = []
    used_citations = set()
    used_sentences = set()
    for _, sentence, chunk_id in candidates:
        if chunk_id in used_citations:
            continue
        cleaned = clean_sentence(sentence)
        if len(cleaned) < 80:
            continue
        sentence_key = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
        if sentence_key in used_sentences:
            continue
        selected.append(f"{cleaned} [{chunk_id}]")
        used_citations.add(chunk_id)
        used_sentences.add(sentence_key)
        if len(selected) == 3:
            break
    if intent.is_risk_question and results:
        company = results[0].company.title()
        return f"{company} reports these risk areas: " + " ".join(selected)
    return " ".join(selected)


def llm_answer(question: str, results: list[RetrievalResult], model: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return extractive_answer(question, results)

    client = OpenAI()
    context = build_context(results, question=question)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nEvidence:\n{context}\n\nAnswer:",
            },
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def build_response_from_retrieved(
    question: str,
    retrieved: list[RetrievalResult],
    model: str = DEFAULT_OPENAI_MODEL,
    expected_tickers: list[str] | None = None,
) -> RAGResponse:
    answer = llm_answer(question, retrieved, model)
    if is_low_content_answer(answer):
        answer = extractive_answer(question, retrieved)
    citations = extract_citations(answer)
    verification = verify_answer(answer, retrieved, expected_tickers=expected_tickers)
    return RAGResponse(
        question=question,
        answer=answer,
        citations=citations,
        retrieved=retrieved,
        verification=verification,
    )


def answer_question(question: str, top_k: int = 5, model: str = DEFAULT_OPENAI_MODEL) -> RAGResponse:
    company, retrieved = retrieve_live_sec(question, top_k=top_k)
    return build_response_from_retrieved(
        question=question,
        retrieved=retrieved,
        model=model,
        expected_tickers=[company.ticker],
    )


def print_response(response: RAGResponse) -> None:
    print("\nAnswer:")
    print(response.answer)
    print("\nSources:")
    for result in response.retrieved:
        marker = "*" if result.chunk_id in response.citations else "-"
        print(f"{marker} {result.chunk_id} | score={result.score:.3f} | {result.source}")
        print(f"  {result.source_url}")
    print("\nHallucination Check:")
    print(f"Confidence Score: {response.verification.confidence_score:.2f}")
    print(f"Hallucination Risk: {response.verification.hallucination_risk}")
    for note in response.verification.notes:
        print(f"- {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a financial RAG question.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_response(answer_question(args.question, top_k=args.top_k, model=args.model))


if __name__ == "__main__":
    main()
