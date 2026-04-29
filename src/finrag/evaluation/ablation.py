"""
finrag.evaluation.ablation
---------------------------
Runs the four-configuration ablation study described in the project spec.

Configurations
--------------
  BASE_ONLY          — LLM with no retrieval (measures domain fluency)
  BASE_RAG           — Base LLM + FAISS retrieval (no fine-tuning)
  FINETUNED_RAG      — Fine-tuned LLM (remote Qwen) + FAISS retrieval
  FINETUNED_RAG_HD   — Fine-tuned + RAG + NLI hallucination detection

Usage
-----
    # Run from the command line (reads data/evaluation/financebench_eval.csv):
    python -m finrag.evaluation.ablation --configs base_rag finetuned_rag_hd

    # Or import and call programmatically:
    from finrag.evaluation.ablation import run_ablation, AblationConfig
    results_df = run_ablation(questions, [AblationConfig.BASE_RAG,
                                          AblationConfig.FINETUNED_RAG_HD])
"""
from __future__ import annotations

import argparse
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from finrag.config import DEFAULT_OPENAI_MODEL, EVALUATION_DIR


class AblationConfig(str, Enum):
    BASE_ONLY = "base_only"
    BASE_RAG = "base_rag"
    FINETUNED_RAG = "finetuned_rag"
    FINETUNED_RAG_HD = "finetuned_rag_hd"


def run_ablation(
    questions: list[dict],
    configs: list[AblationConfig] | None = None,
    qwen_endpoint: str = "",
    base_endpoint: str = "",
    top_k: int = 5,
    openai_model: str = DEFAULT_OPENAI_MODEL,
) -> pd.DataFrame:
    """
    Run all specified configurations on every question and return a
    comparison DataFrame.

    Parameters
    ----------
    questions : list[dict]
        Each dict must have at least "question" and "gold_answer" keys.
        Optional: "benchmark_id", "company", "gold_evidence".
    configs : list[AblationConfig]
        Which configurations to run. Defaults to all four.
    qwen_endpoint : str
        URL of the fine-tuned Qwen server (FINETUNED_RAG / FINETUNED_RAG_HD).
    base_endpoint : str
        URL of the base Qwen server (no LoRA adapter) for BASE_ONLY / BASE_RAG.
        Falls back to qwen_endpoint if not set, then falls back to OpenAI.
    top_k : int
        Number of passages to retrieve per question.
    openai_model : str
        OpenAI model fallback when no Qwen endpoint is available.

    Returns
    -------
    pd.DataFrame
        One row per (question × config), with metrics columns.
    """
    from finrag.evaluation.metrics import exact_match, numerical_accuracy, token_f1

    if configs is None:
        configs = list(AblationConfig)

    # BASE_ONLY and BASE_RAG use base_endpoint if provided, else qwen_endpoint, else OpenAI
    resolved_base = base_endpoint or qwen_endpoint

    rows: list[dict] = []

    for q in questions:
        question = q["question"]
        gold = str(q.get("gold_answer", ""))

        for config in configs:
            answer, report = _run_config(
                config=config,
                question=question,
                top_k=top_k,
                qwen_endpoint=qwen_endpoint,
                base_endpoint=resolved_base,
                openai_model=openai_model,
            )

            row: dict = {
                "config": config.value,
                "benchmark_id": q.get("benchmark_id", ""),
                "question": question,
                "gold_answer": gold,
                "answer": answer,
                "exact_match": exact_match(answer, gold),
                "token_f1": token_f1(answer, gold),
                "numerical_accuracy": numerical_accuracy(answer, gold),
            }

            if report is not None:
                row.update({
                    "confidence_score": report.confidence_score,
                    "overall_risk": report.overall_risk,
                    "grounded_claims": report.grounded_count,
                    "partial_claims": report.partial_count,
                    "unsupported_claims": report.unsupported_count,
                })
            else:
                # Configs that don't run HD still get the fast lexical check
                from finrag.answer import answer_question as _aq

                row.update({
                    "confidence_score": None,
                    "overall_risk": None,
                    "grounded_claims": None,
                    "partial_claims": None,
                    "unsupported_claims": None,
                })

            rows.append(row)

    return pd.DataFrame(rows)


def _run_config(
    config: AblationConfig,
    question: str,
    top_k: int,
    qwen_endpoint: str,
    base_endpoint: str,
    openai_model: str,
) -> tuple[str, object | None]:
    """
    Run a single config on a single question.

    Returns (answer_text, HallucinationReport_or_None).
    """
    if config == AblationConfig.BASE_ONLY:
        if base_endpoint:
            return _base_only_qwen(question, base_endpoint), None
        return _base_only_openai(question, openai_model), None

    if config == AblationConfig.BASE_RAG:
        if base_endpoint:
            from finrag.remote_qwen import answer_with_remote_qwen
            response = answer_with_remote_qwen(question, endpoint=base_endpoint, top_k=top_k)
            return response.answer, None
        from finrag.answer import answer_question
        response = answer_question(question, top_k=top_k, model=openai_model)
        return response.answer, None

    if config == AblationConfig.FINETUNED_RAG:
        if not qwen_endpoint:
            return "[FINETUNED_RAG requires --qwen-endpoint]", None
        from finrag.remote_qwen import answer_with_remote_qwen
        response = answer_with_remote_qwen(question, endpoint=qwen_endpoint, top_k=top_k)
        return response.answer, None

    if config == AblationConfig.FINETUNED_RAG_HD:
        from finrag.hallucination import detect_hallucinations
        endpoint = qwen_endpoint or base_endpoint
        if endpoint:
            from finrag.remote_qwen import answer_with_remote_qwen
            response = answer_with_remote_qwen(question, endpoint=endpoint, top_k=top_k)
        else:
            from finrag.answer import answer_question
            response = answer_question(question, top_k=top_k, model=openai_model)
        report = detect_hallucinations(response.answer, response.retrieved)
        return response.answer, report

    raise ValueError(f"Unknown config: {config}")


def _base_only_qwen(question: str, endpoint: str) -> str:
    """Call the Qwen server with no retrieval context."""
    import requests
    payload = {
        "question": question,
        "context": "No external context provided. Answer from your training knowledge.",
        "allowed_citations": [],
        "max_new_tokens": 350,
    }
    response = requests.post(f"{endpoint.rstrip('/')}/generate", json=payload, timeout=180)
    response.raise_for_status()
    return str(response.json().get("answer", "")).strip()


def _base_only_openai(question: str, model: str) -> str:
    """Call OpenAI with no retrieval context (fallback when no Qwen endpoint)."""
    import os
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        return "[BASE_ONLY requires --base-endpoint or OPENAI_API_KEY]"

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analyst. Answer the question concisely "
                    "using only your knowledge. If unsure, say so."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def print_summary(df: pd.DataFrame) -> None:
    """Print a per-config summary table to stdout."""
    numeric_cols = [c for c in ["exact_match", "token_f1", "numerical_accuracy",
                                "confidence_score"] if c in df.columns]
    summary = df.groupby("config")[numeric_cols].mean().round(4)
    print("\n── Ablation Study Summary ─────────────────────────────")
    print(summary.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FinRAG ablation study.")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=EVALUATION_DIR / "financebench_eval.csv",
        help="Benchmark CSV with question/gold_answer columns.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=EVALUATION_DIR / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=["base_rag", "finetuned_rag_hd"],
        choices=[c.value for c in AblationConfig],
    )
    p.add_argument("--qwen-endpoint", default="", help="Fine-tuned Qwen server URL (FINETUNED_RAG configs).")
    p.add_argument("--base-endpoint", default="", help="Base Qwen server URL (BASE_ONLY / BASE_RAG). Falls back to --qwen-endpoint.")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data = pd.read_csv(args.input_csv)
    if args.limit:
        data = data.head(args.limit)

    questions = data.to_dict("records")
    configs = [AblationConfig(c) for c in args.configs]

    results = run_ablation(
        questions=questions,
        configs=configs,
        qwen_endpoint=args.qwen_endpoint,
        base_endpoint=args.base_endpoint,
        top_k=args.top_k,
    )

    results.to_csv(args.output_csv, index=False)
    print_summary(results)
    print(f"\nFull results → {args.output_csv}")


if __name__ == "__main__":
    main()
