from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCUMENTS_DIR = DATA_DIR / "raw_documents"
PROCESSED_CHUNKS_DIR = DATA_DIR / "processed_chunks"
INDEX_DIR = DATA_DIR / "index"
EVALUATION_DIR = DATA_DIR / "evaluation"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"

CHUNKS_PATH = PROCESSED_CHUNKS_DIR / "chunks.jsonl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
INDEX_METADATA_PATH = INDEX_DIR / "chunks_metadata.json"

DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT", "FinRAG academic research contact@example.com"
)


def ensure_data_dirs() -> None:
    for path in [
        RAW_DOCUMENTS_DIR,
        PROCESSED_CHUNKS_DIR,
        INDEX_DIR,
        EVALUATION_DIR,
        BENCHMARKS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
