from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from finrag.config import DATA_DIR


def prepare_finqa(
    output_path: Path,
    split: str,
    limit: int | None,
    trust_remote_code: bool,
) -> None:
    dataset = load_dataset(
        "ibm-research/finqa",
        split=split,
        trust_remote_code=trust_remote_code,
    )
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            question = row.get("question", "")
            answer = row.get("answer", "")
            context = " ".join(str(row.get(key, "")) for key in ["pre_text", "post_text"])
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer financial questions accurately using the provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {question}",
                    },
                    {"role": "assistant", "content": str(answer)},
                ]
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small FinQA JSONL file for later fine-tuning experiments."
    )
    parser.add_argument("--output-path", type=Path, default=DATA_DIR / "fine_tuning" / "finqa_train.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable execution of the Hugging Face FinQA dataset loader script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_finqa(
        args.output_path,
        args.split,
        args.limit,
        trust_remote_code=not args.no_trust_remote_code,
    )
    print(f"Wrote fine-tuning preparation data to {args.output_path}")


if __name__ == "__main__":
    main()
