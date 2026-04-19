from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from finrag.config import CHUNKS_PATH, RAW_DOCUMENTS_DIR, ensure_data_dirs
from finrag.models import DocumentChunk


def load_metadata(text_path: Path) -> dict:
    metadata_path = text_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file for {text_path}: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_words(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = normalize_text(text).split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_words - overlap_words)
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk) >= 250:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks


def make_chunks(input_dir: Path, chunk_words_count: int, overlap_words_count: int) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for text_path in sorted(input_dir.glob("*.txt")):
        metadata = load_metadata(text_path)
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        doc_chunks = chunk_words(text, chunk_words_count, overlap_words_count)

        for idx, chunk_text in enumerate(doc_chunks, start=1):
            chunk_id = f"{metadata['ticker']}-{metadata['filing_date']}-{idx:04d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
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
                )
            )
    return chunks


def write_jsonl(chunks: list[DocumentChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk downloaded SEC filings.")
    parser.add_argument("--input-dir", type=Path, default=RAW_DOCUMENTS_DIR)
    parser.add_argument("--output-path", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--chunk-words", type=int, default=450)
    parser.add_argument("--overlap-words", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    chunks = make_chunks(args.input_dir, args.chunk_words, args.overlap_words)
    for _ in tqdm(chunks, desc="Prepared chunks"):
        pass
    write_jsonl(chunks, args.output_path)
    print(f"Wrote {len(chunks)} chunks to {args.output_path}")


if __name__ == "__main__":
    main()
