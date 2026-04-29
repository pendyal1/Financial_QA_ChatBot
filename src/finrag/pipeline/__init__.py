"""
finrag.pipeline — data ingestion and indexing.

Typical run order (one-time setup):
    python -m finrag.pipeline.load_datasets      # download FinanceBench / FinQA / TAT-QA
    python -m finrag.download_sec_filings        # download SEC 10-K filings
    python -m finrag.pipeline.chunker            # chunk documents (table-aware)
    python -m finrag.build_index                 # embed chunks → FAISS index
"""
from finrag.pipeline.chunker import chunk_document, chunk_financebench_evidence
from finrag.pipeline.load_datasets import load_financebench, load_finqa, load_tatqa

__all__ = [
    "chunk_document",
    "chunk_financebench_evidence",
    "load_financebench",
    "load_finqa",
    "load_tatqa",
]
