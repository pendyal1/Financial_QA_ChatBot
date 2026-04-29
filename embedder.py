"""
embedder.py  [DEPRECATED — use src/finrag/build_index.py]
----------------------------------------------------------
Original ChromaDB embedder (Person A).  The project has standardized on FAISS
(src/finrag/build_index.py) which is already integrated with the retriever,
answer generation, demo, and evaluation pipeline.

The batch-embedding logic and verification query from this file are preserved
inside build_index.py.  ChromaDB is no longer used.

Run the replacement instead:
    python -m finrag.build_index [--chunks-path ...] [--index-path ...]

This file is kept for reference only and will not receive updates.
"""

import json
import os
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# ── Configuration ─────────────────────────────────────────────────────────────

# Fast, CPU-friendly embedding model — good starting point
# Upgrade to 'BAAI/bge-large-en-v1.5' for better retrieval quality (needs more RAM)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHROMA_DB_PATH = "./chroma_db"         # Local persistent storage
COLLECTION_NAME = "financial_docs"     # ChromaDB collection name
BATCH_SIZE = 64                        # Chunks per embedding batch
                                       # Lower if you run out of memory


# ─────────────────────────────────────────────────────────────────────────────
# 1. Embedding Model
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model(model_name=EMBEDDING_MODEL):
    """
    Loads the sentence transformer embedding model.

    Args:
        model_name (str): HuggingFace model identifier

    Returns:
        SentenceTransformer: loaded model
    """
    print(f"\n  Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. ChromaDB Setup
# ─────────────────────────────────────────────────────────────────────────────

def get_chroma_collection(db_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME):
    """
    Initializes a persistent ChromaDB client and returns the collection.
    Creates the collection if it doesn't exist.
    If the collection already exists and has data, asks before overwriting.

    Args:
        db_path (str): path to persist ChromaDB data
        collection_name (str): name of the collection

    Returns:
        chromadb.Collection: the collection to store/query embeddings
    """
    os.makedirs(db_path, exist_ok=True)

    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )

    # Check if collection already exists and has data
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        collection = client.get_collection(collection_name)
        count = collection.count()
        if count > 0:
            print(f"\n  WARNING: Collection '{collection_name}' already has {count} chunks.")
            response = input("  Delete and re-embed? (y/n): ").strip().lower()
            if response == "y":
                client.delete_collection(collection_name)
                print("  Deleted existing collection.")
            else:
                print("  Keeping existing collection. Exiting.")
                return collection

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for semantic search
    )
    print(f"  ChromaDB collection ready: '{collection_name}' at {db_path}")
    return collection


# ─────────────────────────────────────────────────────────────────────────────
# 3. Embedding & Storing
# ─────────────────────────────────────────────────────────────────────────────

def embed_and_store(chunks, model, collection, batch_size=BATCH_SIZE):
    """
    Embeds chunks in batches and stores them in ChromaDB.

    Each chunk is stored with:
        - id: unique chunk identifier
        - embedding: vector representation of the text
        - document: the raw text (for retrieval)
        - metadata: doc_id, chunk_type, page, etc.

    Args:
        chunks (list[dict]): chunks from chunker.py
        model (SentenceTransformer): embedding model
        collection (chromadb.Collection): target ChromaDB collection
        batch_size (int): number of chunks to embed at once
    """
    print(f"\n  Embedding {len(chunks)} chunks in batches of {batch_size}...")
    start_time = time.time()

    # Process in batches
    for i in tqdm(range(0, len(chunks), batch_size), desc="  Embedding"):
        batch = chunks[i:i + batch_size]

        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]

        # Build metadata dicts — ChromaDB only accepts str/int/float/bool values
        metadatas = []
        for c in batch:
            meta = {
                "doc_id": str(c.get("doc_id", "")),
                "chunk_type": str(c.get("chunk_type", "text")),
                "chunk_index": int(c.get("chunk_index", 0)),
                "char_start": int(c.get("char_start", 0)),
                "char_end": int(c.get("char_end", 0)),
            }
            # Optional fields — only add if present
            if c.get("page") is not None:
                meta["page"] = str(c["page"])
            if c.get("source_qa_id"):
                meta["source_qa_id"] = str(c["source_qa_id"])
            metadatas.append(meta)

        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Store in ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    elapsed = time.time() - start_time
    print(f"\n  Embedded {len(chunks)} chunks in {elapsed:.1f}s")
    print(f"  Average: {elapsed / len(chunks) * 1000:.1f}ms per chunk")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_collection(collection, model, test_query="What was Apple's revenue?"):
    """
    Quick sanity check — runs a test query and prints top results.
    Call this after embedding to verify everything works before moving
    to retriever.py.

    Args:
        collection (chromadb.Collection): the populated collection
        model (SentenceTransformer): embedding model
        test_query (str): a sample financial question
    """
    print(f"\n── Verification Query ────────────────────────────────")
    print(f"  Query: '{test_query}'")

    query_embedding = model.encode([test_query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        similarity = 1 - dist  # cosine distance → similarity
        print(f"\n  Result {rank} (similarity: {similarity:.3f})")
        print(f"    Doc ID    : {meta.get('doc_id', 'unknown')}")
        print(f"    Chunk type: {meta.get('chunk_type', 'unknown')}")
        print(f"    Text      : {doc[:200]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Finance — Embedder")
    print("=" * 60)

    # Load chunks
    chunks_path = "data/chunks.json"
    if not os.path.exists(chunks_path):
        print(f"  ERROR: {chunks_path} not found. Run chunker.py first.")
        exit(1)

    with open(chunks_path) as f:
        chunks = json.load(f)
    print(f"\n  Loaded {len(chunks)} chunks from {chunks_path}")

    # Load model
    model = load_embedding_model()

    # Setup ChromaDB
    collection = get_chroma_collection()

    # Embed and store
    embed_and_store(chunks, model, collection)

    # Verify
    verify_collection(collection, model)

    print(f"\n✓ Embedding complete.")
    print(f"  ChromaDB persisted at: {CHROMA_DB_PATH}")
    print(f"  Total chunks stored  : {collection.count()}")
    print("  Next step: run retriever.py to test semantic search.")
