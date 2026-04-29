"""
chunker.py  [DEPRECATED — use src/finrag/pipeline/chunker.py]
--------------------------------------------------------------
Original table-aware chunker (Person A).  The table detection, sentence
boundary, and evidence-aligned chunking logic has been merged into
finrag.pipeline.chunker, which also returns typed DocumentChunk objects
compatible with the FAISS pipeline.

Run the replacement instead:
    python -m finrag.pipeline.chunker [--input-dir ...] [--output-path ...]

This file is kept for reference only and will not receive updates.
"""

import os
import json
import re
from tqdm import tqdm


# ── Chunking Configuration ────────────────────────────────────────────────────
# These values are tunable — start here and adjust based on retrieval quality.

CHUNK_SIZE = 512        # Target characters per chunk (not tokens — simpler to reason about)
CHUNK_OVERLAP = 64      # Characters of overlap between consecutive chunks
                        # Overlap ensures context isn't lost at chunk boundaries
MIN_CHUNK_SIZE = 100    # Discard chunks shorter than this (usually headers/noise)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core Chunking Strategies
# ─────────────────────────────────────────────────────────────────────────────

def chunk_by_sliding_window(text, doc_id, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Basic sliding window chunking — splits text into overlapping windows.
    Tries to split at sentence boundaries to avoid cutting mid-sentence.

    This is the default strategy for plain text sections of financial documents.

    Args:
        text (str): document text
        doc_id (str): identifier for the source document
        chunk_size (int): target chunk size in characters
        overlap (int): overlap between consecutive chunks in characters

    Returns:
        list[dict]: list of chunk records
    """
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take whatever remains
            chunk_text = text[start:]
        else:
            # Try to end at a sentence boundary (. ! ?)
            boundary = _find_sentence_boundary(text, end)
            chunk_text = text[start:boundary]
            end = boundary

        chunk_text = chunk_text.strip()
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                "doc_id": doc_id,
                "text": chunk_text,
                "char_start": start,
                "char_end": start + len(chunk_text),
                "chunk_type": "text",
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        # Move forward with overlap
        start = end - overlap

    return chunks


def chunk_table(table_text, doc_id, chunk_index_start=0):
    """
    Handles table sections specially — keeps each table as a single chunk
    rather than splitting across rows, since splitting a table mid-row
    destroys its meaning.

    Args:
        table_text (str): table content as a string (pipe-separated or raw)
        doc_id (str): identifier for the source document
        chunk_index_start (int): starting index for chunk numbering

    Returns:
        list[dict]: single-element list containing the table chunk
    """
    table_text = table_text.strip()
    if len(table_text) < MIN_CHUNK_SIZE:
        return []

    return [{
        "chunk_id": f"{doc_id}_table_{chunk_index_start}",
        "doc_id": doc_id,
        "text": table_text,
        "char_start": 0,
        "char_end": len(table_text),
        "chunk_type": "table",
        "chunk_index": chunk_index_start,
    }]


def chunk_document(text, doc_id):
    """
    Main chunking function — intelligently splits a financial document
    by detecting table sections and handling them separately.

    Strategy:
        1. Detect table sections (lines with | separators)
        2. Keep tables intact as single chunks
        3. Apply sliding window to all other text sections

    Args:
        text (str): full document text
        doc_id (str): identifier for the source document

    Returns:
        list[dict]: all chunks from the document
    """
    all_chunks = []
    chunk_index = 0

    # Split document into table and text segments
    segments = _split_tables_from_text(text)

    for segment_type, segment_text in segments:
        if segment_type == "table":
            table_chunks = chunk_table(segment_text, doc_id, chunk_index)
            all_chunks.extend(table_chunks)
            chunk_index += len(table_chunks)
        else:
            text_chunks = chunk_by_sliding_window(segment_text, doc_id)
            # Re-index to be consistent across segments
            for chunk in text_chunks:
                chunk["chunk_id"] = f"{doc_id}_chunk_{chunk_index}"
                chunk["chunk_index"] = chunk_index
                chunk_index += 1
            all_chunks.extend(text_chunks)

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# 2. Evidence-Aligned Chunking (FinanceBench specific)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_financebench_evidence(financebench_records):
    """
    For FinanceBench, we have ground truth evidence passages.
    Rather than chunking blindly, we use the evidence passages directly
    as chunks — this guarantees the gold answer is always retrievable.

    This is important for evaluation: if your retriever can't find the
    gold evidence, your evaluation is measuring retrieval failure, not
    hallucination detection.

    Args:
        financebench_records (list[dict]): loaded FinanceBench Q&A records

    Returns:
        list[dict]: evidence passages as chunks, deduplicated
    """
    seen_texts = set()
    chunks = []
    chunk_index = 0

    for record in tqdm(financebench_records, desc="  Extracting evidence chunks"):
        for passage in record.get("evidence_passages", []):
            text = passage.get("text", "").strip()
            if not text or len(text) < MIN_CHUNK_SIZE:
                continue

            # Deduplicate — same passage may appear across multiple questions
            text_hash = hash(text)
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)

            chunks.append({
                "chunk_id": f"fb_evidence_{chunk_index}",
                "doc_id": record.get("doc_name", "unknown"),
                "text": text,
                "page": passage.get("page", None),
                "chunk_type": "evidence",
                "chunk_index": chunk_index,
                # Keep the QA pair ID so we can trace back during evaluation
                "source_qa_id": record.get("id"),
            })
            chunk_index += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _find_sentence_boundary(text, pos):
    """
    Finds the nearest sentence boundary (. ! ?) before or at pos.
    Falls back to the nearest whitespace if no sentence boundary found.
    Falls back to pos itself if nothing found.
    """
    # Search backward from pos for a sentence-ending punctuation
    search_window = text[max(0, pos - 100):pos + 1]
    for i in range(len(search_window) - 1, -1, -1):
        if search_window[i] in ".!?":
            # Make sure it's followed by whitespace (real sentence end)
            if i + 1 < len(search_window) and search_window[i + 1] in " \n":
                return max(0, pos - 100) + i + 1

    # Fallback: find nearest whitespace
    for i in range(pos, max(0, pos - 50), -1):
        if i < len(text) and text[i] == " ":
            return i

    return pos


def _split_tables_from_text(text):
    """
    Splits a document into alternating text and table segments.
    Detects tables by looking for lines with | separators or
    lines that are mostly numbers separated by whitespace.

    Returns:
        list[tuple]: list of (segment_type, segment_text) pairs
                     where segment_type is "text" or "table"
    """
    lines = text.split("\n")
    segments = []
    current_type = "text"
    current_lines = []

    for line in lines:
        # Heuristic: a table line has multiple | separators or
        # is a row of numbers/percentages
        is_table_line = (
            line.count("|") >= 2 or
            bool(re.match(r"^[\d\s\$\%\,\.\-\|]+$", line.strip()))
            and len(line.strip()) > 10
        )

        if is_table_line and current_type == "text":
            # Transition text → table
            if current_lines:
                segments.append(("text", "\n".join(current_lines)))
            current_lines = [line]
            current_type = "table"
        elif not is_table_line and current_type == "table":
            # Transition table → text
            if current_lines:
                segments.append(("table", "\n".join(current_lines)))
            current_lines = [line]
            current_type = "text"
        else:
            current_lines.append(line)

    # Don't forget the last segment
    if current_lines:
        segments.append((current_type, "\n".join(current_lines)))

    return segments


def print_chunk_stats(chunks):
    """Prints a summary of the chunks produced."""
    if not chunks:
        print("  No chunks produced.")
        return

    total = len(chunks)
    types = {}
    lengths = []

    for c in chunks:
        t = c.get("chunk_type", "unknown")
        types[t] = types.get(t, 0) + 1
        lengths.append(len(c["text"]))

    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    print(f"\n── Chunk Statistics ──────────────────────────────────")
    print(f"  Total chunks      : {total}")
    print(f"  By type           : {types}")
    print(f"  Avg length (chars): {avg_len:.0f}")
    print(f"  Min length (chars): {min_len}")
    print(f"  Max length (chars): {max_len}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Finance — Chunker")
    print("=" * 60)

    all_chunks = []

    # ── Part 1: Chunk raw documents from data/documents/ ──────────────────────
    doc_dir = "data/documents"
    if not os.path.exists(doc_dir):
        print(f"  ERROR: {doc_dir} not found. Run data_loader.py first.")
        exit(1)

    doc_files = [f for f in os.listdir(doc_dir) if f.endswith(".txt")]
    print(f"\n  Found {len(doc_files)} documents to chunk")

    for doc_file in tqdm(doc_files, desc="  Chunking documents"):
        doc_id = doc_file.replace(".txt", "")
        with open(os.path.join(doc_dir, doc_file), "r") as f:
            text = f.read()
        chunks = chunk_document(text, doc_id)
        all_chunks.extend(chunks)

    # ── Part 2: Add FinanceBench evidence chunks ───────────────────────────────
    fb_path = "data/financebench_qa.json"
    if os.path.exists(fb_path):
        print("\n  Adding FinanceBench evidence chunks...")
        with open(fb_path) as f:
            fb_records = json.load(f)
        evidence_chunks = chunk_financebench_evidence(fb_records)
        all_chunks.extend(evidence_chunks)
        print(f"  Added {len(evidence_chunks)} evidence chunks")
    else:
        print(f"\n  WARNING: {fb_path} not found — skipping evidence chunks")

    # ── Save all chunks ───────────────────────────────────────────────────────
    out_path = "data/chunks.json"
    with open(out_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print_chunk_stats(all_chunks)
    print(f"\n  Saved {len(all_chunks)} chunks → {out_path}")
    print("\n✓ Chunking complete.")
    print("  Next step: run embedder.py to embed chunks into ChromaDB.")
