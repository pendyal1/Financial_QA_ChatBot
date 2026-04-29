from __future__ import annotations

import argparse
import json
from pathlib import Path

from finrag.config import DATA_DIR

# Local cache written by finrag.pipeline.load_datasets
_FINQA_CACHE = DATA_DIR / "finqa_qa.json"


def prepare_finqa(
    output_path: Path,
    split: str,
    limit: int | None,
) -> None:
    """
    Convert the locally-cached FinQA Q&A pairs into the chat-message JSONL
    format expected by train_qlora.py.

    Reads from data/finqa_qa.json (written by finrag.pipeline.load_datasets).
    That file already contains train / validation / test records with the
    'context' field pre-built, so we don't need to re-download anything.
    """
    if not _FINQA_CACHE.exists():
        raise FileNotFoundError(
            f"FinQA cache not found at {_FINQA_CACHE}. "
            "Run `python -m finrag.pipeline.load_datasets --datasets finqa` first."
        )

    with _FINQA_CACHE.open(encoding="utf-8") as fh:
        all_records = json.load(fh)

    # Filter to the requested split
    records = [r for r in all_records if r.get("split", "train") == split]
    if not records:
        raise ValueError(
            f"No records found for split='{split}' in {_FINQA_CACHE}. "
            f"Available splits: {sorted({r.get('split') for r in all_records})}"
        )

    if limit:
        records = records[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in records:
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer financial questions accurately using the provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"Context: {row.get('context', '')}\n\nQuestion: {row.get('question', '')}",
                    },
                    {"role": "assistant", "content": str(row.get("answer", ""))},
                ]
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert cached FinQA records into chat-message JSONL for QLoRA training."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "fine_tuning" / "finqa_train.jsonl",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of training examples (default: use all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_finqa(args.output_path, args.split, args.limit)
    print(f"Wrote fine-tuning data to {args.output_path}")


if __name__ == "__main__":
    main()
