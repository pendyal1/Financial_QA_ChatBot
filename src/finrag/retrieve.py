from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from finrag.config import (
    DEFAULT_EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    INDEX_METADATA_PATH,
)
from finrag.models import RetrievalResult
from finrag.query import analyze_query, evidence_for_question


STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "for",
    "in",
    "of",
    "on",
    "report",
    "say",
    "says",
    "the",
    "their",
    "to",
    "what",
}

RISK_SECTION_PATTERNS = [
    r"item 1a\.? risk factors",
    r"\brisk factors\b",
    r"material adverse effect",
    r"adversely affect",
    r"adverse effect",
]


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if token not in STOPWORDS
    }


def lexical_score(question: str, text: str) -> float:
    question_tokens = tokens(question)
    if not question_tokens:
        return 0.0
    text_tokens = tokens(text)
    return len(question_tokens & text_tokens) / math.sqrt(len(question_tokens))


def risk_score(text: str) -> float:
    normalized = text.lower()
    score = 0.0
    for pattern in RISK_SECTION_PATTERNS:
        if re.search(pattern, normalized):
            score += 1.0
    return min(score / len(RISK_SECTION_PATTERNS), 1.0)


class Retriever:
    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = INDEX_METADATA_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")

        import faiss
        from sentence_transformers import SentenceTransformer

        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(embedding_model)

    def search(
        self,
        question: str,
        top_k: int = 5,
        allowed_tickers: list[str] | None = None,
    ) -> list[RetrievalResult]:
        intent = analyze_query(question)
        tickers = allowed_tickers if allowed_tickers is not None else intent.tickers
        query_text = intent.expanded_question
        query = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        overfetch = min(len(self.chunks), max(top_k * 30, 50))
        scores, indices = self.index.search(query, overfetch)

        candidates: list[tuple[float, RetrievalResult]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if tickers and chunk["ticker"] not in tickers:
                continue
            dense = float(score)
            rerank = dense + (0.12 * lexical_score(question, chunk["text"]))
            if intent.is_risk_question:
                rerank += 0.28 * risk_score(chunk["text"])
            result = RetrievalResult(
                chunk_id=chunk["chunk_id"],
                score=float(rerank),
                ticker=chunk["ticker"],
                company=chunk["company"],
                source=chunk["source"],
                source_url=chunk["source_url"],
                text=chunk["text"],
            )
            candidates.append((rerank, result))

        if len(candidates) < top_k and tickers:
            seen_ids = {result.chunk_id for _, result in candidates}
            for chunk in self.chunks:
                if chunk["ticker"] not in tickers or chunk["chunk_id"] in seen_ids:
                    continue
                rerank = 0.12 * lexical_score(question, chunk["text"])
                if intent.is_risk_question:
                    rerank += 0.28 * risk_score(chunk["text"])
                if rerank <= 0:
                    continue
                candidates.append(
                    (
                        rerank,
                        RetrievalResult(
                            chunk_id=chunk["chunk_id"],
                            score=float(rerank),
                            ticker=chunk["ticker"],
                            company=chunk["company"],
                            source=chunk["source"],
                            source_url=chunk["source_url"],
                            text=chunk["text"],
                        ),
                    )
                )

        candidates.sort(key=lambda item: item[0], reverse=True)
        results = [result for _, result in candidates[:top_k]]

        if not results and tickers:
            raise ValueError(f"No retrieved chunks matched ticker filter: {', '.join(tickers)}")
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve relevant financial chunks.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ticker", action="append", help="Restrict retrieval to one ticker. May repeat.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = Retriever(embedding_model=args.embedding_model)
    results = retriever.search(args.question, args.top_k, allowed_tickers=args.ticker)
    for idx, result in enumerate(results, start=1):
        preview = evidence_for_question(args.question, result.text)[:500].replace("\n", " ")
        print(f"\n{idx}. {result.chunk_id} score={result.score:.3f}")
        print(f"   {result.source}")
        print(f"   {preview}...")


if __name__ == "__main__":
    main()
