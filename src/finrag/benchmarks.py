from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from datasets import load_dataset

from finrag.config import BENCHMARKS_DIR, EVALUATION_DIR, ensure_data_dirs


TATQA_BASE_URL = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw"


def evidence_to_text(evidence: Any) -> str:
    if not evidence:
        return ""
    texts: list[str] = []
    for item in evidence:
        if isinstance(item, dict):
            text = item.get("evidence_text") or item.get("evidence_text_full_page") or ""
            if text:
                texts.append(str(text))
        elif item:
            texts.append(str(item))
    return "\n\n".join(texts)


def prepare_financebench(limit: int | None = None) -> Path:
    ensure_data_dirs()
    dataset = load_dataset("PatronusAI/financebench", split="train")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    rows = []
    raw_path = BENCHMARKS_DIR / "financebench_raw.jsonl"
    eval_path = EVALUATION_DIR / "financebench_eval.csv"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(
                {
                    "benchmark": "financebench",
                    "benchmark_id": row.get("financebench_id", ""),
                    "company": row.get("company", ""),
                    "doc_name": row.get("doc_name", ""),
                    "doc_type": row.get("doc_type", ""),
                    "doc_period": row.get("doc_period", ""),
                    "question_type": row.get("question_type", ""),
                    "question_reasoning": row.get("question_reasoning", ""),
                    "question": row.get("question", ""),
                    "gold_answer": row.get("answer", ""),
                    "justification": row.get("justification", ""),
                    "gold_evidence": evidence_to_text(row.get("evidence")),
                    "doc_link": row.get("doc_link", ""),
                }
            )

    pd.DataFrame(rows).to_csv(eval_path, index=False)
    return eval_path


def table_to_text(table: dict[str, Any]) -> str:
    rows = table.get("table", [])
    return "\n".join(" | ".join(str(cell) for cell in row) for row in rows)


def prepare_tatqa(split: str = "dev", limit: int | None = None) -> Path:
    ensure_data_dirs()
    url = f"{TATQA_BASE_URL}/tatqa_dataset_{split}.json"
    raw_path = BENCHMARKS_DIR / f"tatqa_{split}.json"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    raw_path.write_text(response.text, encoding="utf-8")

    data = response.json()
    rows = []
    count = 0
    for example in data:
        paragraphs = example.get("paragraphs", [])
        paragraph_text = "\n".join(str(paragraph.get("text", "")) for paragraph in paragraphs)
        table_text = table_to_text(example.get("table", {}))
        context = f"Table:\n{table_text}\n\nParagraphs:\n{paragraph_text}".strip()
        for question in example.get("questions", []):
            answer = question.get("answer", "")
            if isinstance(answer, list):
                answer = "; ".join(str(item) for item in answer)
            rows.append(
                {
                    "benchmark": "tatqa",
                    "benchmark_id": question.get("uid", ""),
                    "source_uid": example.get("table", {}).get("uid", ""),
                    "question": question.get("question", ""),
                    "gold_answer": answer,
                    "derivation": question.get("derivation", ""),
                    "answer_type": question.get("answer_type", ""),
                    "answer_from": question.get("answer_from", ""),
                    "scale": question.get("scale", ""),
                    "gold_evidence": context,
                }
            )
            count += 1
            if limit and count >= limit:
                break
        if limit and count >= limit:
            break

    eval_path = EVALUATION_DIR / f"tatqa_{split}_eval.csv"
    pd.DataFrame(rows).to_csv(eval_path, index=False)
    return eval_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark files for FinRAG evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    financebench = subparsers.add_parser("financebench", help="Prepare FinanceBench evaluation CSV.")
    financebench.add_argument("--limit", type=int, default=None)

    tatqa = subparsers.add_parser("tatqa", help="Download and flatten TAT-QA.")
    tatqa.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    tatqa.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "financebench":
        output = prepare_financebench(limit=args.limit)
    elif args.command == "tatqa":
        output = prepare_tatqa(split=args.split, limit=args.limit)
    else:
        raise ValueError(f"Unknown benchmark command: {args.command}")
    print(f"Wrote benchmark evaluation file to {output}")


if __name__ == "__main__":
    main()
