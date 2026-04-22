from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from finrag.chunk_documents import chunk_words, normalize_text
from finrag.config import DEFAULT_EMBEDDING_MODEL
from finrag.models import DocumentChunk, RetrievalResult
from finrag.query import analyze_query
from finrag.retrieve import lexical_score, risk_score


SUPPORTED_UPLOAD_TYPES = ["txt", "html", "htm", "xml", "xhtml", "pdf"]


@dataclass(frozen=True)
class UploadedFilingMetadata:
    filename: str
    doc_id: str
    ticker: str
    company: str
    form: str
    filing_date: str
    source: str


def infer_ticker(filename: str, text: str) -> str:
    stem = Path(filename).stem
    stem_tokens = re.split(r"[^A-Za-z]+", stem.upper())
    for token in stem_tokens:
        if not token:
            continue
        if token in {"SEC", "FORM", "FILED", "HTML", "HTM", "TXT", "XML", "PDF", "K", "Q"}:
            continue
        if 1 <= len(token) <= 5:
            return token
    for token in re.findall(r"\b[A-Z]{1,5}\b", text[:4000].upper()):
        if token not in {"SEC", "FORM", "FILED", "HTML", "HTM", "TXT", "XML", "PDF", "ITEM", "PART", "K", "Q"}:
            return token
    return "UPLD"


def infer_company(filename: str, text: str) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    stem = re.sub(r"\b\d{4}\b.*$", "", stem).strip()
    title_stem = re.sub(r"\s+", " ", stem).title()
    if re.fullmatch(r"[A-Za-z]{1,5}", title_stem):
        return title_stem.upper()
    if title_stem:
        return title_stem
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:120] if first_line else "Uploaded Filing"


def infer_form(filename: str, text: str) -> str:
    candidates = re.findall(r"\b(10-K|10-Q|8-K|20-F|6-K|S-1)\b", f"{filename} {text[:5000]}", flags=re.IGNORECASE)
    return candidates[0].upper() if candidates else "SEC Filing"


def parse_uploaded_filing(filename: str, file_bytes: bytes) -> tuple[str, UploadedFilingMetadata]:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_UPLOAD_TYPES:
        raise ValueError(f"Unsupported file type: .{suffix or 'unknown'}")

    if suffix == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        decoded = file_bytes.decode("utf-8", errors="ignore")
        if suffix in {"html", "htm", "xml", "xhtml"}:
            soup = BeautifulSoup(decoded, "html.parser")
            for tag in soup(["script", "style", "ix:header", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
        else:
            text = decoded

    text = normalize_text(text)
    if len(text) < 500:
        raise ValueError("The uploaded filing did not contain enough extractable text.")

    filing_date = date.today().isoformat()
    ticker = infer_ticker(filename, text)
    company = infer_company(filename, text)
    form = infer_form(filename, text)
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", Path(filename).stem).strip("_").lower() or "uploaded_filing"
    metadata = UploadedFilingMetadata(
        filename=filename,
        doc_id=f"upload-{safe_name}",
        ticker=ticker,
        company=company,
        form=form,
        filing_date=filing_date,
        source=f"{company} {form} uploaded file",
    )
    return text, metadata


def make_uploaded_chunks(
    filename: str,
    file_bytes: bytes,
    chunk_words_count: int = 450,
    overlap_words_count: int = 80,
) -> tuple[UploadedFilingMetadata, list[DocumentChunk]]:
    text, metadata = parse_uploaded_filing(filename, file_bytes)
    chunk_texts = chunk_words(text, chunk_words_count, overlap_words_count)
    chunks = [
        DocumentChunk(
            chunk_id=f"{metadata.ticker}-{metadata.filing_date}-{idx:04d}",
            doc_id=metadata.doc_id,
            ticker=metadata.ticker,
            company=metadata.company,
            form=metadata.form,
            filing_date=metadata.filing_date,
            report_date="",
            accession_no="uploaded",
            source_url="",
            source=metadata.source,
            text=chunk_text,
        )
        for idx, chunk_text in enumerate(chunk_texts, start=1)
    ]
    if not chunks:
        raise ValueError("The uploaded filing did not produce any chunks.")
    return metadata, chunks


class UploadedFilingIndex:
    def __init__(
        self,
        metadata: UploadedFilingMetadata,
        chunks: list[DocumentChunk],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.metadata = metadata
        self.chunks = chunks
        self.model = SentenceTransformer(embedding_model)
        self.embeddings = self.model.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def search(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        intent = analyze_query(question)
        query = self.model.encode(
            [intent.expanded_question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")[0]
        dense_scores = np.dot(self.embeddings, query)
        order = np.argsort(-dense_scores)[: max(top_k * 10, top_k)]

        candidates: list[tuple[float, RetrievalResult]] = []
        for idx in order.tolist():
            chunk = self.chunks[idx]
            rerank = float(dense_scores[idx]) + (0.12 * lexical_score(question, chunk.text))
            if intent.is_risk_question:
                rerank += 0.28 * risk_score(chunk.text)
            candidates.append(
                (
                    rerank,
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        score=float(rerank),
                        ticker=chunk.ticker,
                        company=chunk.company,
                        source=chunk.source,
                        source_url=chunk.source_url,
                        text=chunk.text,
                    ),
                )
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in candidates[:top_k]]


def build_uploaded_filing_index(
    filename: str,
    file_bytes: bytes,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> UploadedFilingIndex:
    metadata, chunks = make_uploaded_chunks(filename=filename, file_bytes=file_bytes)
    return UploadedFilingIndex(metadata=metadata, chunks=chunks, embedding_model=embedding_model)
