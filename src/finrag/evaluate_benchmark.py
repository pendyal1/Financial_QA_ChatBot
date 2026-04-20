from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from finrag.answer import extractive_answer, is_low_content_answer
from finrag.config import EVALUATION_DIR
from finrag.hallucination_detection import extract_citations, verify_answer
from finrag.models import RetrievalResult
from finrag.remote_qwen import DEFAULT_QWEN_ENDPOINT, endpoint_generate


def normalize(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", str(text).lower())
    text = text.replace("$", "")
    tokens = []
    for token in re.findall(r"[a-z0-9.%-]+", text):
        if re.fullmatch(r"\d+\.0+", token):
            token = token.split(".", 1)[0]
        tokens.append(token)
    return " ".join(tokens)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if gold_counts.get(token, 0) > 0:
            overlap += 1
            gold_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return round((2 * precision * recall) / (precision + recall), 3)


def make_gold_retrieval(row: pd.Series, index: int) -> RetrievalResult:
    benchmark = str(row.get("benchmark", "BENCH")).upper()
    citation_prefix = re.sub(r"[^A-Z0-9_.-]", "", benchmark)[:8] or "BENCH"
    chunk_id = f"{citation_prefix}-2024-01-01-{index + 1:04d}"
    source = str(row.get("doc_name") or row.get("source_uid") or row.get("benchmark_id") or benchmark)
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        ticker=citation_prefix,
        company=str(row.get("company") or benchmark),
        source=source,
        source_url=str(row.get("doc_link") or ""),
        text=str(row.get("gold_evidence") or row.get("justification") or ""),
    )


def evaluate_benchmark(
    input_csv: Path,
    output_csv: Path,
    backend: str,
    endpoint: str,
    limit: int | None,
) -> pd.DataFrame:
    dataset = pd.read_csv(input_csv)
    if limit:
        dataset = dataset.head(limit)

    rows = []
    for idx, row in dataset.iterrows():
        question = str(row["question"])
        gold_answer = str(row.get("gold_answer", ""))
        retrieved = [make_gold_retrieval(row, idx)]
        if backend == "remote-qwen":
            answer = endpoint_generate(
                endpoint=endpoint,
                question=question,
                retrieved=retrieved,
                max_new_tokens=350,
            )
            if is_low_content_answer(answer):
                answer = extractive_answer(question, retrieved)
        else:
            answer = extractive_answer(question, retrieved)

        citations = extract_citations(answer)
        verification = verify_answer(answer, retrieved)
        normalized_answer = normalize(answer)
        normalized_gold = normalize(gold_answer)
        rows.append(
            {
                "benchmark": row.get("benchmark", ""),
                "benchmark_id": row.get("benchmark_id", ""),
                "question": question,
                "gold_answer": gold_answer,
                "answer": answer,
                "gold_answer_contained": bool(normalized_gold and normalized_gold in normalized_answer),
                "answer_token_f1": token_f1(answer, gold_answer),
                "citations": ", ".join(citations),
                "confidence_score": verification.confidence_score,
                "hallucination_risk": verification.hallucination_risk,
                "notes": " ".join(verification.notes),
            }
        )

    results = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FinRAG on prepared benchmark CSV files.")
    parser.add_argument("--input-csv", type=Path, default=EVALUATION_DIR / "financebench_eval.csv")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=EVALUATION_DIR / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    parser.add_argument("--backend", choices=["extractive", "remote-qwen"], default="extractive")
    parser.add_argument("--endpoint", default=DEFAULT_QWEN_ENDPOINT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "remote-qwen" and not args.endpoint:
        raise ValueError("Set --endpoint or COLAB_QWEN_ENDPOINT when using --backend remote-qwen.")
    results = evaluate_benchmark(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        backend=args.backend,
        endpoint=args.endpoint,
        limit=args.limit,
    )
    print(
        results[
            [
                "benchmark",
                "benchmark_id",
                "gold_answer_contained",
                "answer_token_f1",
                "confidence_score",
                "hallucination_risk",
            ]
        ].head(20)
    )
    print(f"Wrote benchmark results to {args.output_csv}")


if __name__ == "__main__":
    main()
