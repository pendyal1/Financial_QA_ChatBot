from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from finrag.answer import answer_question
from finrag.config import EVALUATION_DIR


def evaluate(input_csv: Path, output_csv: Path, top_k: int) -> pd.DataFrame:
    questions = pd.read_csv(input_csv)
    rows = []
    for _, row in questions.iterrows():
        response = answer_question(row["question"], top_k=top_k)
        retrieved_tickers = {result.ticker for result in response.retrieved}
        retrieved_text = " ".join(result.text.lower() for result in response.retrieved)
        expected_topic = str(row.get("expected_topic", "")).lower()
        topic_terms = [term for term in expected_topic.split() if term]

        rows.append(
            {
                "question": row["question"],
                "expected_ticker": row.get("expected_ticker", ""),
                "expected_topic": row.get("expected_topic", ""),
                "answer": response.answer,
                "citations": ", ".join(response.citations),
                "top_retrieved": response.retrieved[0].chunk_id if response.retrieved else "",
                "top_retrieved_expected_ticker": (
                    bool(response.retrieved)
                    and response.retrieved[0].ticker == row.get("expected_ticker", "")
                ),
                "retrieved_expected_ticker": row.get("expected_ticker", "") in retrieved_tickers,
                "retrieved_expected_topic": all(term in retrieved_text for term in topic_terms),
                "confidence_score": response.verification.confidence_score,
                "hallucination_risk": response.verification.hallucination_risk,
            }
        )

    results = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the FinRAG baseline.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=EVALUATION_DIR / "sample_questions.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=EVALUATION_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args.input_csv, args.output_csv, args.top_k)
    print(
        results[
            [
                "question",
                "top_retrieved_expected_ticker",
                "retrieved_expected_ticker",
                "confidence_score",
                "hallucination_risk",
            ]
        ]
    )
    print(f"Wrote evaluation results to {args.output_csv}")


if __name__ == "__main__":
    main()
