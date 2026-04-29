"""
finrag.pipeline.load_datasets
------------------------------
Loads FinanceBench, FinQA, and TAT-QA from HuggingFace (or GitHub fallback)
and saves cleaned outputs under data/ for downstream use.

Outputs
-------
data/financebench_qa.json    Q&A pairs with evidence passages
data/finqa_qa.json           Q&A pairs with table + text context
data/tatqa_qa.json           Q&A pairs with table + paragraph context
data/documents/              Raw document texts for the retrieval corpus

Run standalone:
    python -m finrag.pipeline.load_datasets [--datasets financebench finqa tatqa]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests
from tqdm import tqdm

from finrag.config import DATA_DIR, ensure_data_dirs

DOCUMENTS_DIR = DATA_DIR / "documents"
TATQA_DEV_URL = (
    "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# FinanceBench
# ─────────────────────────────────────────────────────────────────────────────

def load_financebench(limit: int | None = None) -> list[dict]:
    """
    Load FinanceBench from HuggingFace (PatronusAI/financebench).

    Each record contains:
        id, source, question, answer, evidence_passages, doc_name, question_type
    """
    from datasets import load_dataset

    print("\n── Loading FinanceBench ──────────────────────────────")
    try:
        dataset = load_dataset("PatronusAI/financebench", split="train")
        print(f"  {len(dataset)} records from HuggingFace")
    except Exception as exc:
        print(f"  HuggingFace failed ({exc}), trying GitHub fallback…")
        dataset = _financebench_github_fallback()

    if limit:
        dataset = list(dataset)[:limit]

    records: list[dict] = []
    doc_texts: dict[str, list[str]] = {}

    for row in tqdm(dataset, desc="  FinanceBench"):
        passages = []
        for ev in row.get("evidence", []) or []:
            text = ev.get("evidence_text", "")
            if text:
                passages.append({"text": text, "page": ev.get("page_number")})
                doc_texts.setdefault(row.get("doc_name", "unknown"), []).append(text)

        records.append({
            "id": row.get("financebench_id", f"fb_{len(records)}"),
            "source": "financebench",
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "evidence_passages": passages,
            "doc_name": row.get("doc_name", "unknown"),
            "question_type": row.get("question_type", "unknown"),
        })

    _save_json(records, DATA_DIR / "financebench_qa.json")
    _save_documents(doc_texts)
    print(f"  Saved {len(records)} Q&A pairs and {len(doc_texts)} documents")
    return records


def _financebench_github_fallback() -> list[dict]:
    url = (
        "https://raw.githubusercontent.com/patronus-ai/financebench"
        "/main/data/financebench_open_source.jsonl"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.strip().splitlines() if line]


# ─────────────────────────────────────────────────────────────────────────────
# FinQA
# ─────────────────────────────────────────────────────────────────────────────

def load_finqa(splits: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """
    Load FinQA from GitHub (czyssrs/FinQA).

    The ibm-research/finqa HuggingFace dataset uses a legacy loading script
    that newer versions of the datasets library no longer support, so we
    download the JSON files directly from the source GitHub repo instead.

    Each record contains:
        id, source, split, question, answer, program, context, table
    """
    print("\n── Loading FinQA ─────────────────────────────────────")

    FINQA_URLS = {
        "train":      "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
        "validation": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
        "test":       "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
    }
    target_splits = splits or list(FINQA_URLS.keys())

    all_records: list[dict] = []
    for split in target_splits:
        print(f"  Downloading {split} from GitHub…")
        resp = requests.get(FINQA_URLS[split], timeout=60)
        resp.raise_for_status()
        rows = resp.json()

        if limit:
            rows = rows[:limit]
        print(f"  {split}: {len(rows)} records")

        for row in tqdm(rows, desc=f"  {split}"):
            qa = row.get("qa", {})
            pre = " ".join(row.get("pre_text", []))
            post = " ".join(row.get("post_text", []))
            table_str = _table_to_string(row.get("table_ori") or row.get("table", []))
            context = f"{pre}\n\nTABLE:\n{table_str}\n\n{post}".strip()

            rec = {
                "id": row.get("id", f"fq_{len(all_records)}"),
                "source": "finqa",
                "split": split,
                "question": qa.get("question", row.get("question", "")),
                "answer": str(qa.get("exe_ans", qa.get("answer", row.get("answer", "")))),
                "program": qa.get("program", row.get("program", "")),
                "context": context,
                "pre_text": pre,
                "post_text": post,
                "table": row.get("table_ori") or row.get("table", []),
                "gold_inds": qa.get("gold_inds", row.get("gold_inds", {})),
            }
            all_records.append(rec)

            safe_id = re.sub(r"[^\w\-]", "_", rec["id"])
            doc_path = DOCUMENTS_DIR / f"finqa_{safe_id}.txt"
            doc_path.write_text(context, encoding="utf-8")

    _save_json(all_records, DATA_DIR / "finqa_qa.json")
    print(f"  Saved {len(all_records)} FinQA Q&A pairs")
    return all_records


# ─────────────────────────────────────────────────────────────────────────────
# TAT-QA
# ─────────────────────────────────────────────────────────────────────────────

def load_tatqa(split: str = "dev", limit: int | None = None) -> list[dict]:
    """
    Download TAT-QA from GitHub and flatten into Q&A records.

    Each record contains:
        id, source, question, answer, answer_type, derivation, gold_evidence
    """
    print(f"\n── Loading TAT-QA ({split}) ───────────────────────────")
    resp = requests.get(
        f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_{split}.json",
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    records: list[dict] = []
    for example in data:
        paragraphs = example.get("paragraphs", [])
        para_text = "\n".join(p.get("text", "") for p in paragraphs)
        table_text = _table_to_string(example.get("table", {}).get("table", []))
        context = f"Table:\n{table_text}\n\nParagraphs:\n{para_text}".strip()

        for q in example.get("questions", []):
            ans = q.get("answer", "")
            if isinstance(ans, list):
                ans = "; ".join(str(a) for a in ans)
            records.append({
                "id": q.get("uid", f"tatqa_{len(records)}"),
                "source": "tatqa",
                "split": split,
                "question": q.get("question", ""),
                "answer": str(ans),
                "answer_type": q.get("answer_type", ""),
                "derivation": q.get("derivation", ""),
                "scale": q.get("scale", ""),
                "gold_evidence": context,
                "source_uid": example.get("table", {}).get("uid", ""),
            })
            if limit and len(records) >= limit:
                break
        if limit and len(records) >= limit:
            break

    _save_json(records, DATA_DIR / "tatqa_qa.json")
    print(f"  Saved {len(records)} TAT-QA Q&A pairs")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _table_to_string(table: list) -> str:
    if not table:
        return ""
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def _save_json(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_documents(doc_texts: dict[str, list[str]]) -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for doc_name, passages in doc_texts.items():
        (DOCUMENTS_DIR / f"{doc_name}.txt").write_text(
            "\n\n".join(passages), encoding="utf-8"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and save financial benchmark datasets.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["financebench", "finqa", "tatqa"],
        choices=["financebench", "finqa", "tatqa"],
    )
    p.add_argument("--limit", type=int, default=None, help="Cap records per dataset (for quick testing).")
    p.add_argument("--tatqa-split", default="dev", choices=["train", "dev", "test"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if "financebench" in args.datasets:
        load_financebench(limit=args.limit)
    if "finqa" in args.datasets:
        load_finqa(limit=args.limit)
    if "tatqa" in args.datasets:
        load_tatqa(split=args.tatqa_split, limit=args.limit)

    print("\n✓ Dataset loading complete.")


if __name__ == "__main__":
    main()
