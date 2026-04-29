"""
data_loader.py  [DEPRECATED — use src/finrag/pipeline/load_datasets.py]
------------------------------------------------------------------------
Original data loading script (Person A).  The logic has been migrated to
finrag.pipeline.load_datasets, which integrates with the finrag config
system, adds TAT-QA support, and works with the FAISS pipeline.

Run the replacement instead:
    python -m finrag.pipeline.load_datasets --datasets financebench finqa tatqa

This file is kept for reference only and will not receive updates.
"""

import os
import json
import requests
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("data/documents", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FinanceBench
# ─────────────────────────────────────────────────────────────────────────────

def load_financebench():
    """
    Loads FinanceBench from HuggingFace.

    Each entry contains:
        - question: natural language financial question
        - answer: ground truth answer
        - evidence: list of source passages with page citations
        - doc_name: source document (e.g. 'APPLE_2022_10K')
        - domain: always 'finance'

    Returns:
        list[dict]: cleaned Q&A records
    """
    print("\n── Loading FinanceBench ──────────────────────────────")

    try:
        # Primary: load from HuggingFace
        dataset = load_dataset("PatronusAI/financebench", split="train")
        print(f"  Loaded {len(dataset)} FinanceBench records from HuggingFace")
    except Exception as e:
        print(f"  HuggingFace load failed ({e}), trying GitHub fallback...")
        dataset = _load_financebench_github()

    records = []
    doc_texts = {}

    for row in tqdm(dataset, desc="  Processing FinanceBench"):
        # Extract evidence passages for the retrieval corpus
        evidence_passages = []
        if "evidence" in row and row["evidence"]:
            for ev in row["evidence"]:
                passage = {
                    "text": ev.get("evidence_text", ""),
                    "page": ev.get("page_number", None),
                    "doc_name": row.get("doc_name", "unknown"),
                }
                evidence_passages.append(passage)

                # Accumulate document text keyed by doc name
                doc_name = row.get("doc_name", "unknown")
                if doc_name not in doc_texts:
                    doc_texts[doc_name] = []
                doc_texts[doc_name].append(ev.get("evidence_text", ""))

        record = {
            "id": row.get("financebench_id", f"fb_{len(records)}"),
            "source": "financebench",
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "evidence_passages": evidence_passages,
            "doc_name": row.get("doc_name", "unknown"),
            "question_type": row.get("question_type", "unknown"),
        }
        records.append(record)

    # Save Q&A pairs
    out_path = "data/financebench_qa.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved {len(records)} Q&A pairs → {out_path}")

    # Save document texts for retrieval corpus
    for doc_name, passages in doc_texts.items():
        doc_path = f"data/documents/{doc_name}.txt"
        with open(doc_path, "w") as f:
            f.write("\n\n".join(passages))
    print(f"  Saved {len(doc_texts)} source documents → data/documents/")

    return records


def _load_financebench_github():
    """
    Fallback: loads FinanceBench directly from the GitHub repo CSV.
    Used if HuggingFace is unavailable.
    """
    url = (
        "https://raw.githubusercontent.com/patronus-ai/financebench"
        "/main/data/financebench_open_source.jsonl"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    records = []
    for line in response.text.strip().split("\n"):
        if line:
            records.append(json.loads(line))

    # Wrap in a simple iterable with consistent keys
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2. FinQA
# ─────────────────────────────────────────────────────────────────────────────

def load_finqa():
    """
    Loads FinQA from HuggingFace (ibm-research/finqa).

    Each entry contains:
        - question: financial question requiring numerical reasoning
        - answer: ground truth answer (often a number or formula)
        - context: pre and post text surrounding the relevant table/passage
        - table: structured table data as list of rows
        - gold_inds: indices of gold evidence in the context

    Returns:
        list[dict]: cleaned Q&A records for all splits
    """
    print("\n── Loading FinQA ─────────────────────────────────────")

    dataset = load_dataset("ibm-research/finqa")
    print(f"  Splits available: {list(dataset.keys())}")

    all_records = []

    for split in ["train", "validation", "test"]:
        if split not in dataset:
            continue

        split_data = dataset[split]
        print(f"  Processing {split}: {len(split_data)} records")

        for row in tqdm(split_data, desc=f"  {split}"):
            # Combine pre_text and post_text into a single context string
            pre_text = " ".join(row.get("pre_text", []))
            post_text = " ".join(row.get("post_text", []))
            table_ori = row.get("table_ori", [])

            # Convert table to readable string
            table_str = _table_to_string(table_ori)

            full_context = f"{pre_text}\n\nTABLE:\n{table_str}\n\n{post_text}".strip()

            record = {
                "id": row.get("id", f"fq_{len(all_records)}"),
                "source": "finqa",
                "split": split,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "program": row.get("program", ""),   # arithmetic program for the answer
                "context": full_context,
                "pre_text": pre_text,
                "post_text": post_text,
                "table": table_ori,
                "gold_inds": row.get("gold_inds", {}),
            }
            all_records.append(record)

            # Save document context for retrieval corpus
            doc_id = row.get("id", f"finqa_{len(all_records)}")
            doc_path = f"data/documents/finqa_{doc_id}.txt"
            with open(doc_path, "w") as f:
                f.write(full_context)

    # Save all Q&A pairs
    out_path = "data/finqa_qa.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"  Saved {len(all_records)} Q&A pairs → {out_path}")

    return all_records


def _table_to_string(table_ori):
    """
    Converts a FinQA table (list of lists) into a readable string.

    Example:
        [["Year", "Revenue"], ["2022", "$89.5B"]]
        → "Year | Revenue\n2022 | $89.5B"
    """
    if not table_ori:
        return ""
    rows = []
    for row in table_ori:
        rows.append(" | ".join(str(cell) for cell in row))
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset Statistics & Inspection
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_stats(financebench_records, finqa_records):
    """
    Prints a summary of both datasets so the team can understand
    what they're working with before building the pipeline.
    """
    print("\n── Dataset Statistics ────────────────────────────────")

    # FinanceBench stats
    fb_df = pd.DataFrame(financebench_records)
    print(f"\n  FinanceBench:")
    print(f"    Total Q&A pairs  : {len(fb_df)}")
    if "doc_name" in fb_df.columns:
        print(f"    Unique documents : {fb_df['doc_name'].nunique()}")
    if "question_type" in fb_df.columns:
        print(f"    Question types   :\n{fb_df['question_type'].value_counts().to_string()}")

    # FinQA stats
    fq_df = pd.DataFrame(finqa_records)
    print(f"\n  FinQA:")
    print(f"    Total Q&A pairs  : {len(fq_df)}")
    if "split" in fq_df.columns:
        print(f"    By split         :\n{fq_df['split'].value_counts().to_string()}")

    # Show a sample from each
    print("\n── Sample Records ────────────────────────────────────")

    if financebench_records:
        sample_fb = financebench_records[0]
        print(f"\n  FinanceBench sample:")
        print(f"    Q: {sample_fb['question']}")
        print(f"    A: {sample_fb['answer']}")
        print(f"    Doc: {sample_fb['doc_name']}")

    if finqa_records:
        sample_fq = finqa_records[0]
        print(f"\n  FinQA sample:")
        print(f"    Q: {sample_fq['question']}")
        print(f"    A: {sample_fq['answer']}")
        print(f"    Context (first 200 chars): {sample_fq['context'][:200]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Finance — Data Loader")
    print("=" * 60)

    # Load both datasets
    financebench_records = load_financebench()
    finqa_records = load_finqa()

    # Print stats for team inspection
    print_dataset_stats(financebench_records, finqa_records)

    print("\n✓ Data loading complete.")
    print("  Next step: run chunker.py to segment documents into passages.")
