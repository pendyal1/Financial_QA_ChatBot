from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import requests
from datasets import load_dataset

from finrag.config import DATA_DIR


DEFAULT_OUTPUT_PATH = DATA_DIR / "fine_tuning" / "financial_qa_mix_train.jsonl"
DEFAULT_MANIFEST_PATH = DATA_DIR / "fine_tuning" / "financial_qa_mix_manifest.json"
CONVFINQA_TRAIN_URL = "https://huggingface.co/datasets/AdaptLLM/ConvFinQA/resolve/main/train_turn.json"
TATQA_TRAIN_URL = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_train.json"

SYSTEM_MESSAGE = (
    "You are a financial analyst assistant. Answer only from the supplied evidence. "
    "If the evidence does not support a claim, say the available evidence is insufficient."
)


def compact_text(parts: Iterable[Any], max_chars: int = 5000) -> str:
    text = " ".join(str(part).strip() for part in parts if str(part).strip())
    text = " ".join(text.split())
    return text[:max_chars].strip()


def render_table(table: Any, max_rows: int = 12, max_chars: int = 2500) -> str:
    if not table:
        return ""
    rows: list[str] = []
    rows_iterable = table.get("table", []) if isinstance(table, dict) else table
    for raw_row in list(rows_iterable)[:max_rows]:
        if isinstance(raw_row, list):
            rows.append(" | ".join(str(cell).strip() for cell in raw_row if str(cell).strip()))
        elif isinstance(raw_row, dict):
            rows.append(" | ".join(f"{key}: {value}" for key, value in raw_row.items()))
        else:
            rows.append(str(raw_row).strip())
    return compact_text(rows, max_chars=max_chars)


def make_record(
    *,
    dataset_name: str,
    question: str,
    answer: Any,
    context_parts: Iterable[Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    clean_question = str(question).strip()
    clean_answer = str(answer).strip()
    context = compact_text(context_parts)
    if not clean_question or not clean_answer or not context:
        return None
    return {
        "dataset": dataset_name,
        "metadata": metadata or {},
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": f"Evidence:\n{context}\n\nQuestion: {clean_question}",
            },
            {"role": "assistant", "content": clean_answer},
        ],
    }


def fetch_json(url: str) -> Any:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def prepare_finqa(limit: int | None, trust_remote_code: bool) -> list[dict[str, Any]]:
    dataset = load_dataset(
        "ibm-research/finqa",
        split="train",
        trust_remote_code=trust_remote_code,
    )
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records: list[dict[str, Any]] = []
    for row in dataset:
        context_parts = [
            " ".join(row.get("pre_text", []) or []),
            render_table(row.get("table")),
            " ".join(row.get("post_text", []) or []),
        ]
        record = make_record(
            dataset_name="FinQA",
            question=row.get("question", ""),
            answer=row.get("answer", "") or row.get("final_result", ""),
            context_parts=context_parts,
            metadata={"id": row.get("id", ""), "gold_inds": row.get("gold_inds", "")},
        )
        if record:
            records.append(record)
    return records


def prepare_convfinqa(limit: int | None) -> list[dict[str, Any]]:
    payload = fetch_json(CONVFINQA_TRAIN_URL)
    rows = payload[:limit] if limit else payload

    records: list[dict[str, Any]] = []
    for row in rows:
        qa = row.get("qa", {}) or {}
        context_parts = [
            " ".join(row.get("pre_text", []) or []),
            render_table(row.get("table_ori") or row.get("table")),
            " ".join(row.get("post_text", []) or []),
        ]
        record = make_record(
            dataset_name="ConvFinQA",
            question=qa.get("question", ""),
            answer=qa.get("answer", "") or qa.get("exe_ans", ""),
            context_parts=context_parts,
            metadata={"id": row.get("id", ""), "filename": row.get("filename", "")},
        )
        if record:
            records.append(record)
    return records


def prepare_tatqa(limit: int | None) -> list[dict[str, Any]]:
    payload = fetch_json(TATQA_TRAIN_URL)
    records: list[dict[str, Any]] = []
    question_count = 0
    for row in payload:
        paragraphs = row.get("paragraphs", []) or []
        paragraph_text = compact_text(
            [paragraph.get("text", "") for paragraph in paragraphs if isinstance(paragraph, dict)],
            max_chars=2500,
        )
        table_text = render_table(row.get("table"))
        for question_row in row.get("questions", []) or []:
            answer = question_row.get("answer", "")
            if isinstance(answer, list):
                answer = ", ".join(str(item) for item in answer)
            record = make_record(
                dataset_name="TAT-QA",
                question=question_row.get("question", ""),
                answer=answer,
                context_parts=[table_text, paragraph_text],
                metadata={
                    "uid": question_row.get("uid", ""),
                    "answer_type": question_row.get("answer_type", ""),
                    "scale": question_row.get("scale", ""),
                },
            )
            if record:
                records.append(record)
                question_count += 1
                if limit and question_count >= limit:
                    return records
    return records


def write_records(records: list[dict[str, Any]], output_path: Path, manifest_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            dataset_counts[record["dataset"]] = dataset_counts.get(record["dataset"], 0) + 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {"total_examples": len(records), "datasets": dataset_counts}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a curated financial QA fine-tuning mixture for QLoRA."
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--finqa-limit", type=int, default=None)
    parser.add_argument("--convfinqa-limit", type=int, default=None)
    parser.add_argument("--tatqa-limit", type=int, default=None)
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable execution of the Hugging Face FinQA dataset loader script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    records.extend(prepare_finqa(args.finqa_limit, trust_remote_code=not args.no_trust_remote_code))
    records.extend(prepare_convfinqa(args.convfinqa_limit))
    records.extend(prepare_tatqa(args.tatqa_limit))
    write_records(records, args.output_path, args.manifest_path)
    print(f"Wrote {len(records)} fine-tuning examples to {args.output_path}")
    print(f"Wrote dataset counts to {args.manifest_path}")


if __name__ == "__main__":
    main()
