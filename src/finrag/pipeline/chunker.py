"""
finrag.pipeline.chunker
-----------------------
Table-aware chunker that produces DocumentChunk objects compatible with
the FAISS pipeline (build_index.py / retrieve.py).

Merges the best of two earlier implementations:
  - Table detection + sentence-boundary logic  (root chunker.py)
  - DocumentChunk model + metadata handling    (src/finrag/chunk_documents.py)
  - Evidence-aligned chunking for FinanceBench (root chunker.py)

Run standalone:
    python -m finrag.pipeline.chunker [--input-dir ...] [--output-path ...]
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from finrag.config import CHUNKS_PATH, RAW_DOCUMENTS_DIR, ensure_data_dirs
from finrag.models import DocumentChunk

# ── Tuneable constants ─────────────────────────────────────────────────────────
CHUNK_SIZE_CHARS = 1800     # target characters per chunk (~450 words)
CHUNK_OVERLAP_CHARS = 200   # character overlap between consecutive chunks
MIN_CHUNK_CHARS = 100       # discard chunks shorter than this

# Word-based fallback (used when metadata doesn't carry char counts)
CHUNK_WORDS = 450
OVERLAP_WORDS = 80


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(text: str, doc_id: str) -> list[dict]:
    """
    Split a financial document into chunks, keeping tables intact.

    Returns plain dicts (not DocumentChunk) so this function can also be
    called from root-level scripts that don't use the full finrag model.

    Strategy
    --------
    1. Detect table sections (lines dominated by | or numeric data).
    2. Keep each table as a single chunk (splitting a table mid-row is useless).
    3. Apply sliding-window chunking with sentence-boundary alignment to text.
    """
    all_chunks: list[dict] = []
    chunk_index = 0

    for segment_type, segment_text in _split_tables_from_text(text):
        if segment_type == "table":
            tc = _chunk_table(segment_text, doc_id, chunk_index)
            all_chunks.extend(tc)
            chunk_index += len(tc)
        else:
            for chunk in _chunk_sliding_window(segment_text, doc_id, chunk_index):
                all_chunks.append(chunk)
                chunk_index += 1

    return all_chunks


def chunk_financebench_evidence(financebench_records: list[dict]) -> list[dict]:
    """
    Use FinanceBench gold evidence passages directly as chunks.

    This guarantees the ground-truth passage is always in the index —
    important for measuring retrieval recall cleanly during evaluation.
    """
    seen: set[int] = set()
    chunks: list[dict] = []
    idx = 0

    for record in tqdm(financebench_records, desc="Evidence chunks"):
        for passage in record.get("evidence_passages", []):
            text = passage.get("text", "").strip()
            if not text or len(text) < MIN_CHUNK_CHARS:
                continue
            h = hash(text)
            if h in seen:
                continue
            seen.add(h)
            chunks.append({
                "chunk_id": f"fb_evidence_{idx:05d}",
                "doc_id": record.get("doc_name", "unknown"),
                "text": text,
                "page": passage.get("page"),
                "chunk_type": "evidence",
                "chunk_index": idx,
                "source_qa_id": record.get("id"),
            })
            idx += 1

    return chunks


def make_document_chunks(
    input_dir: Path,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[DocumentChunk]:
    """
    Load SEC filing text files from *input_dir*, chunk them, and return
    typed DocumentChunk objects for use with the FAISS pipeline.
    """
    chunks: list[DocumentChunk] = []
    for text_path in sorted(input_dir.glob("*.txt")):
        metadata = _load_metadata(text_path)
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        raw_chunks = chunk_document(text, metadata["doc_id"])

        for raw in raw_chunks:
            chunk_text = raw["text"]
            # Skip chunks that are too short after table-aware splitting
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{metadata['ticker']}-{metadata['filing_date']}-{raw['chunk_index']:04d}",
                    doc_id=metadata["doc_id"],
                    ticker=metadata["ticker"],
                    company=metadata["company"],
                    form=metadata["form"],
                    filing_date=metadata["filing_date"],
                    report_date=metadata.get("report_date", ""),
                    accession_no=metadata["accession_no"],
                    source_url=metadata["source_url"],
                    source=metadata["source"],
                    text=chunk_text,
                    chunk_type=raw.get("chunk_type", "text"),
                )
            )
    return chunks


def write_jsonl(chunks: list[DocumentChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_metadata(text_path: Path) -> dict:
    meta_path = text_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata for {text_path}: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _split_tables_from_text(text: str) -> list[tuple[str, str]]:
    """
    Partition document text into alternating ("text", ...) and ("table", ...)
    segments by detecting lines that look like table rows.
    """
    lines = text.split("\n")
    segments: list[tuple[str, str]] = []
    current_type = "text"
    current_lines: list[str] = []

    for line in lines:
        is_table = (
            line.count("|") >= 2
            or (
                bool(re.match(r"^[\d\s$%,.\-|]+$", line.strip()))
                and len(line.strip()) > 10
            )
        )

        if is_table and current_type == "text":
            if current_lines:
                segments.append(("text", "\n".join(current_lines)))
            current_lines = [line]
            current_type = "table"
        elif not is_table and current_type == "table":
            if current_lines:
                segments.append(("table", "\n".join(current_lines)))
            current_lines = [line]
            current_type = "text"
        else:
            current_lines.append(line)

    if current_lines:
        segments.append((current_type, "\n".join(current_lines)))

    return segments


def _find_sentence_boundary(text: str, pos: int) -> int:
    """Return the nearest sentence-end position at or before *pos*."""
    window = text[max(0, pos - 100): pos + 1]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in ".!?" and i + 1 < len(window) and window[i + 1] in " \n":
            return max(0, pos - 100) + i + 1
    for i in range(pos, max(0, pos - 50), -1):
        if i < len(text) and text[i] == " ":
            return i
    return pos


def _chunk_sliding_window(
    text: str,
    doc_id: str,
    index_start: int = 0,
) -> list[dict]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks: list[dict] = []
    start = 0
    ci = index_start

    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        if end >= len(text):
            chunk_text = text[start:]
        else:
            end = _find_sentence_boundary(text, end)
            chunk_text = text[start:end]

        chunk_text = chunk_text.strip()
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{ci}",
                "doc_id": doc_id,
                "text": chunk_text,
                "char_start": start,
                "char_end": start + len(chunk_text),
                "chunk_type": "text",
                "chunk_index": ci,
            })
            ci += 1

        start = end - CHUNK_OVERLAP_CHARS
        if start <= 0:
            break

    return chunks


def _chunk_table(table_text: str, doc_id: str, index_start: int) -> list[dict]:
    table_text = table_text.strip()
    if len(table_text) < MIN_CHUNK_CHARS:
        return []
    return [{
        "chunk_id": f"{doc_id}_table_{index_start}",
        "doc_id": doc_id,
        "text": table_text,
        "char_start": 0,
        "char_end": len(table_text),
        "chunk_type": "table",
        "chunk_index": index_start,
    }]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Table-aware chunker for SEC filings.")
    p.add_argument("--input-dir", type=Path, default=RAW_DOCUMENTS_DIR)
    p.add_argument("--output-path", type=Path, default=CHUNKS_PATH)
    p.add_argument("--chunk-words", type=int, default=CHUNK_WORDS)
    p.add_argument("--overlap-words", type=int, default=OVERLAP_WORDS)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()
    chunks = make_document_chunks(args.input_dir, args.chunk_words, args.overlap_words)
    write_jsonl(chunks, args.output_path)
    table_count = sum(1 for c in chunks if c.chunk_type == "table")
    print(f"Wrote {len(chunks)} chunks ({table_count} table, {len(chunks) - table_count} text) → {args.output_path}")


if __name__ == "__main__":
    main()
