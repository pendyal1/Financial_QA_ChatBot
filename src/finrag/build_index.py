from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from finrag.config import (
    CHUNKS_PATH,
    DEFAULT_EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    INDEX_METADATA_PATH,
    ensure_data_dirs,
)


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing chunks file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_faiss_index(chunks: list[dict], model_name: str, batch_size: int) -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[start : start + batch_size]
        embeddings.append(
            model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
    return np.vstack(embeddings).astype("float32")


def write_index(embeddings: np.ndarray, chunks: list[dict], index_path: Path, metadata_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index over financial chunks.")
    parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--index-path", type=Path, default=FAISS_INDEX_PATH)
    parser.add_argument("--metadata-path", type=Path, default=INDEX_METADATA_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    chunks = load_chunks(args.chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {args.chunks_path}")
    embeddings = build_faiss_index(chunks, args.embedding_model, args.batch_size)
    write_index(embeddings, chunks, args.index_path, args.metadata_path)
    print(f"Indexed {len(chunks)} chunks")
    print(f"FAISS index: {args.index_path}")
    print(f"Metadata: {args.metadata_path}")


if __name__ == "__main__":
    main()
