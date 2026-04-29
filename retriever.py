"""
retriever.py  [DEPRECATED — use src/finrag/retrieve.py]
--------------------------------------------------------
Original ChromaDB retriever (Person A).  Unique features from this file have
been merged into the canonical FAISS-based retriever (src/finrag/retrieve.py):

  ✓ Financial abbreviation expansion (EPS, EBITDA, FCF, …) → query.py
  ✓ Multi-part query decomposition → query.py (analyze_query)
  ✓ Retrieval recall evaluation helper → evaluation/metrics.py

Usage (new):
    from finrag.retrieve import Retriever
    retriever = Retriever()
    results = retriever.search("What was Apple's gross margin in 2023?")

    # Or with abbreviation expansion and decomposition via query.py:
    from finrag.query import analyze_query
    intent = analyze_query("What was Apple's EPS in FY2023?")
    results = retriever.search(intent.expanded_question)

This file is kept for reference only and will not receive updates.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Must match embedder.py
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "financial_docs"
DEFAULT_TOP_K = 5                        # Number of passages to retrieve per query
MIN_SIMILARITY = 0.3                     # Discard results below this threshold
                                         # Lower = more permissive retrieval


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    A single retrieved passage with its metadata and similarity score.
    This is the interface between Person A's retriever and Person B's
    hallucination detector / answer generator.
    """
    chunk_id: str
    doc_id: str
    text: str
    score: float                          # Cosine similarity (0-1, higher = more relevant)
    chunk_type: str = "text"             # "text", "table", or "evidence"
    page: Optional[str] = None
    rank: int = 0                         # Position in ranked results (1 = most relevant)


@dataclass
class RetrievalResponse:
    """
    Full response from a retrieval query — returned to the rest of the pipeline.
    """
    query: str
    results: list = field(default_factory=list)   # list[RetrievalResult]
    num_results: int = 0
    retrieval_successful: bool = True

    def get_context_string(self, max_chars=3000):
        """
        Formats retrieved passages into a single context string for the LLM.
        Includes source citations so the answer generator can attribute claims.

        Args:
            max_chars (int): max total characters across all passages

        Returns:
            str: formatted context string ready to inject into LLM prompt
        """
        context_parts = []
        total_chars = 0

        for result in self.results:
            source_label = f"[Source: {result.doc_id}"
            if result.page:
                source_label += f", Page {result.page}"
            source_label += "]"

            passage = f"{source_label}\n{result.text}"

            if total_chars + len(passage) > max_chars:
                break

            context_parts.append(passage)
            total_chars += len(passage)

        return "\n\n---\n\n".join(context_parts)

    def get_source_docs(self):
        """Returns list of unique source document IDs in the results."""
        return list(dict.fromkeys(r.doc_id for r in self.results))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Query Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_query(query: str) -> str:
    """
    Cleans and optionally expands a user query before embedding.

    Financial questions often use abbreviations or colloquial phrasing.
    This function normalizes them to improve retrieval.

    Args:
        query (str): raw user question

    Returns:
        str: preprocessed query
    """
    query = query.strip()

    # Expand common financial abbreviations
    expansions = {
        "EPS": "earnings per share",
        "P/E": "price to earnings ratio",
        "EBITDA": "earnings before interest taxes depreciation amortization",
        "YoY": "year over year",
        "QoQ": "quarter over quarter",
        "10-K": "annual report 10-K SEC filing",
        "10-Q": "quarterly report 10-Q SEC filing",
        "FCF": "free cash flow",
        "ROE": "return on equity",
        "ROA": "return on assets",
    }

    for abbrev, expansion in expansions.items():
        # Case-sensitive replacement to avoid false matches
        query = query.replace(abbrev, f"{abbrev} ({expansion})")

    return query


def decompose_query(query: str) -> list:
    """
    Decomposes a complex multi-part question into sub-queries.
    Each sub-query is retrieved independently, then results are merged.

    This improves retrieval for questions like:
    "What was Apple's revenue in 2023 and how did margins change?"
    → ["Apple revenue 2023", "Apple margins change"]

    Simple heuristic version — can be upgraded to LLM-based decomposition later.

    Args:
        query (str): original user question

    Returns:
        list[str]: list of sub-queries (just the original if not decomposable)
    """
    # Split on common conjunctions indicating multi-part questions
    conjunctions = [" and how ", " and what ", " and when ", " and why "]
    for conj in conjunctions:
        if conj in query.lower():
            parts = query.lower().split(conj, 1)
            return [parts[0].strip() + "?", parts[1].strip() + "?"]

    # No decomposition needed
    return [query]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Retriever Class
# ─────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Main retriever class — wraps ChromaDB and the embedding model.

    This is the interface Person B and the answer generation module
    will import and use. Initialize once, call retrieve() many times.

    Example:
        retriever = Retriever()
        response = retriever.retrieve("What was Tesla's revenue in Q3 2023?")
        context = response.get_context_string()
        sources = response.get_source_docs()
    """

    def __init__(
        self,
        embedding_model=EMBEDDING_MODEL,
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
    ):
        print("  Initializing Retriever...")
        self.model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(collection_name)
        total = self.collection.count()
        print(f"  Connected to ChromaDB: {total} chunks indexed")

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = MIN_SIMILARITY,
        chunk_types: Optional[list] = None,
        decompose: bool = False,
    ) -> RetrievalResponse:
        """
        Retrieves the most relevant passages for a query.

        Args:
            query (str): user's financial question
            top_k (int): number of results to return
            min_similarity (float): minimum cosine similarity threshold
            chunk_types (list): filter by chunk type e.g. ["table"] for table-only
            decompose (bool): whether to decompose multi-part questions

        Returns:
            RetrievalResponse: ranked passages + metadata
        """
        # Preprocess query
        processed_query = preprocess_query(query)

        # Optionally decompose multi-part questions
        sub_queries = decompose_query(processed_query) if decompose else [processed_query]

        all_results = {}   # chunk_id → RetrievalResult (for deduplication)

        for sub_query in sub_queries:
            results = self._retrieve_single(sub_query, top_k, min_similarity, chunk_types)
            for result in results:
                # Keep the higher-scoring result if a chunk appears in multiple sub-queries
                if result.chunk_id not in all_results or result.score > all_results[result.chunk_id].score:
                    all_results[result.chunk_id] = result

        # Sort by score descending and re-rank
        ranked_results = sorted(all_results.values(), key=lambda r: r.score, reverse=True)[:top_k]
        for i, result in enumerate(ranked_results):
            result.rank = i + 1

        return RetrievalResponse(
            query=query,
            results=ranked_results,
            num_results=len(ranked_results),
            retrieval_successful=len(ranked_results) > 0,
        )

    def _retrieve_single(self, query, top_k, min_similarity, chunk_types):
        """
        Internal method — retrieves for a single query string.
        """
        # Build ChromaDB filter for chunk_type if specified
        where_filter = None
        if chunk_types:
            where_filter = {"chunk_type": {"$in": chunk_types}}

        # Embed the query
        query_embedding = self.model.encode([query]).tolist()

        # Query ChromaDB
        query_kwargs = {
            "query_embeddings": query_embedding,
            "n_results": min(top_k * 2, self.collection.count()),  # fetch extra, filter below
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        raw_results = self.collection.query(**query_kwargs)

        # Parse and filter results
        results = []
        docs = raw_results.get("documents", [[]])[0]
        metas = raw_results.get("metadatas", [[]])[0]
        ids = raw_results.get("ids", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            similarity = 1 - dist
            if similarity < min_similarity:
                continue

            results.append(RetrievalResult(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", "unknown"),
                text=doc,
                score=similarity,
                chunk_type=meta.get("chunk_type", "text"),
                page=meta.get("page"),
            ))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation Helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_retrieval(retriever, qa_records, top_k=5):
    """
    Evaluates retrieval recall on FinanceBench Q&A pairs.
    Measures: for what % of questions is the gold evidence passage retrieved?

    This is an important baseline metric — if retrieval recall is low,
    everything downstream (answer quality, hallucination detection) suffers.

    Args:
        retriever (Retriever): initialized retriever
        qa_records (list[dict]): FinanceBench Q&A pairs with evidence
        top_k (int): number of results to retrieve per question

    Returns:
        dict: recall metrics
    """
    print(f"\n── Retrieval Evaluation (top-{top_k}) ────────────────")
    total = 0
    hits = 0

    for record in tqdm(qa_records[:100], desc="  Evaluating"):  # Sample 100 for speed
        question = record.get("question", "")
        gold_passages = [p["text"] for p in record.get("evidence_passages", [])]
        if not gold_passages:
            continue

        response = retriever.retrieve(question, top_k=top_k)
        retrieved_texts = [r.text for r in response.results]

        # Hit = at least one gold passage found in retrieved results
        hit = any(
            any(gold[:100] in retrieved for retrieved in retrieved_texts)
            for gold in gold_passages
        )

        total += 1
        if hit:
            hits += 1

    recall = hits / total if total > 0 else 0
    print(f"  Recall@{top_k}: {recall:.3f} ({hits}/{total} questions)")
    return {"recall": recall, "hits": hits, "total": total}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Finance — Retriever")
    print("=" * 60)

    # Initialize retriever
    retriever = Retriever()

    # ── Test queries ──────────────────────────────────────────────────────────
    test_queries = [
        "What was Apple's total revenue in fiscal year 2022?",
        "How did Microsoft's cloud segment perform year over year?",
        "What are the main risk factors mentioned in Tesla's 10-K?",
        "What was Amazon's operating income margin?",
    ]

    print("\n── Test Queries ──────────────────────────────────────")
    for query in test_queries:
        print(f"\n  Q: {query}")
        response = retriever.retrieve(query, top_k=3)

        if not response.retrieval_successful:
            print("  No results found above similarity threshold.")
            continue

        for result in response.results:
            print(f"\n    Rank {result.rank} | Score: {result.score:.3f} | Doc: {result.doc_id}")
            print(f"    {result.text[:200]}...")

        print(f"\n  Context string preview (first 300 chars):")
        print(f"  {response.get_context_string()[:300]}...")

    # ── Evaluate on FinanceBench ──────────────────────────────────────────────
    fb_path = "data/financebench_qa.json"
    if os.path.exists(fb_path):
        with open(fb_path) as f:
            fb_records = json.load(f)
        evaluate_retrieval(retriever, fb_records, top_k=5)
    else:
        print(f"\n  Skipping evaluation — {fb_path} not found")

    print("\n✓ Retriever ready.")
    print("  Person B: import Retriever from retriever.py for hallucination detection.")
    print("  Person C: use data/finqa_qa.json and data/financebench_qa.json for fine-tuning.")
