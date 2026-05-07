#!/usr/bin/env python3
"""Run one system of the FinRAG ablation study on FinanceBench.

Uses gold evidence passages (not live retrieval) to isolate generation quality.
Run once per system, then use compare_results.py to build the comparison table.

Usage
-----
# System A — extractive fallback (no Colab needed, ~5 min)
python scripts/run_ablation.py --system extractive

# System B — Qwen base, NO adapter (start Colab without --adapter-path first)
python scripts/run_ablation.py --system qwen_base --endpoint https://xxx.trycloudflare.com

# System C — Qwen + LoRA (restart Colab WITH --adapter-path)
python scripts/run_ablation.py --system qwen_lora --endpoint https://xxx.trycloudflare.com

# Retrieval ablation (no Colab, ~20 min) — reranker on vs off on live retrieval
python scripts/run_ablation.py --system retrieval_reranker
python scripts/run_ablation.py --system retrieval_no_reranker
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from finrag.answer import extractive_answer, is_low_content_answer
from finrag.hallucination_detection import extract_citations, verify_answer
from finrag.models import RetrievalResult
from finrag.remote_qwen import endpoint_generate

OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"
FINANCEBENCH_CSV = OUTPUT_DIR / "financebench_eval.csv"
RETRIEVAL_SAMPLE_SIZE = 50


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", str(text).lower())
    text = text.replace("$", "").replace("%", "")
    tokens = []
    for token in re.findall(r"[a-z0-9.]+", text):
        if re.fullmatch(r"\d+\.0+", token):
            token = token.split(".", 1)[0]
        tokens.append(token)
    return " ".join(tokens)


def token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    counts: dict[str, int] = {}
    for t in gold_toks:
        counts[t] = counts.get(t, 0) + 1
    overlap = sum(min(pred_toks.count(t), counts.get(t, 0)) for t in set(pred_toks))
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_toks)
    r = overlap / len(gold_toks)
    return round(2 * p * r / (p + r), 4)


def rouge_l(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    m, n = len(g_toks), len(p_toks)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if g_toks[i - 1] == p_toks[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    p = lcs / n
    r = lcs / m
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def numerical_em(pred: str, gold: str) -> bool:
    """True if first numbers in pred and gold match within 1%."""
    def extract(text: str) -> float | None:
        text = text.replace(",", "").replace("$", "").replace("%", "")
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(m.group()) if m else None
    p, g = extract(pred), extract(gold)
    if p is None or g is None:
        return False
    return abs(p - g) / (abs(g) + 1e-9) <= 0.01


def gold_retrieval_result(row: pd.Series, idx: int) -> RetrievalResult:
    benchmark = str(row.get("benchmark", "BENCH")).upper()
    cid_prefix = re.sub(r"[^A-Z0-9]", "", benchmark)[:8] or "BENCH"
    chunk_id = f"{cid_prefix}-2024-01-01-{idx + 1:04d}"
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        ticker=str(row.get("company", benchmark))[:6].upper(),
        company=str(row.get("company", benchmark)),
        source=str(row.get("doc_name", benchmark)),
        source_url=str(row.get("doc_link", "")),
        text=str(row.get("gold_evidence", "")),
    )


# ---------------------------------------------------------------------------
# Generation ablation (A / B / C)
# ---------------------------------------------------------------------------

def run_generation_ablation(system: str, endpoint: str, limit: int | None) -> list[dict]:
    dataset = pd.read_csv(FINANCEBENCH_CSV)
    if limit:
        dataset = dataset.head(limit)

    results = []
    for idx, row in dataset.iterrows():
        question = str(row["question"])
        gold = str(row.get("gold_answer", ""))
        qtype = str(row.get("question_type", "unknown"))
        retrieved = [gold_retrieval_result(row, idx)]

        print(f"[{idx + 1:03d}/{len(dataset)}] ({qtype}) {question[:65]}...")

        try:
            if system == "extractive":
                answer = extractive_answer(question, retrieved)
                if is_low_content_answer(answer):
                    answer = extractive_answer(question, retrieved)
            else:
                answer = endpoint_generate(endpoint=endpoint, question=question, retrieved=retrieved)
                if is_low_content_answer(answer):
                    answer = extractive_answer(question, retrieved)

            citations = extract_citations(answer)
            verification = verify_answer(answer, retrieved)
            is_numerical = qtype == "metrics-generated"

            record = {
                "system": system,
                "benchmark_id": str(row.get("benchmark_id", "")),
                "question_type": qtype,
                "question": question,
                "gold_answer": gold,
                "answer": answer,
                "token_f1": token_f1(answer, gold),
                "rouge_l": rouge_l(answer, gold),
                "numerical_em": numerical_em(answer, gold) if is_numerical else None,
                "gold_contained": bool(gold and _normalize(gold) in _normalize(answer)),
                "num_citations": len(citations),
                "confidence_score": verification.confidence_score,
                "hallucination_risk": verification.hallucination_risk,
            }
            results.append(record)
            print(f"         F1={record['token_f1']:.2f}  ROUGE-L={record['rouge_l']:.2f}  risk={record['hallucination_risk']}")

        except Exception as exc:
            print(f"         ERROR: {exc}")
            results.append({
                "system": system,
                "benchmark_id": str(row.get("benchmark_id", "")),
                "question_type": qtype,
                "question": question,
                "gold_answer": gold,
                "error": str(exc),
            })

        time.sleep(0.2)

    return results


# ---------------------------------------------------------------------------
# Retrieval ablation (D / E)
# ---------------------------------------------------------------------------

def run_retrieval_ablation(use_reranker: bool) -> list[dict]:
    """Live retrieval on first RETRIEVAL_SAMPLE_SIZE FinanceBench questions.
    Measures whether top-5 chunks contain the gold evidence passage."""
    import random
    from finrag.sec_live import get_live_retriever

    if not use_reranker:
        import finrag.rerank as rerank_mod
        import finrag.sec_live as sec_live_mod
        def _noop(question, results):  # noqa: ARG001
            return results
        rerank_mod.rerank_results = _noop
        sec_live_mod.rerank_results = _noop
        print("Reranker disabled.")

    dataset = pd.read_csv(FINANCEBENCH_CSV)
    sample = dataset.sample(n=min(RETRIEVAL_SAMPLE_SIZE, len(dataset)), random_state=42)

    retriever = get_live_retriever()
    results = []

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        question = str(row["question"])
        gold_evidence = _normalize(str(row.get("gold_evidence", "")))
        print(f"[{i:02d}/{len(sample)}] {question[:65]}...")

        try:
            _company, retrieved = retriever.retrieve(question, top_k=5)
            retrieved_text = " ".join(_normalize(r.text) for r in retrieved)
            # Check if any retrieved chunk contains key phrases from the gold evidence
            gold_tokens = set(gold_evidence.split())
            retrieved_tokens = set(retrieved_text.split())
            overlap_ratio = len(gold_tokens & retrieved_tokens) / max(len(gold_tokens), 1)

            results.append({
                "system": "retrieval_reranker" if use_reranker else "retrieval_no_reranker",
                "benchmark_id": str(row.get("benchmark_id", "")),
                "question_type": str(row.get("question_type", "")),
                "question": question,
                "top_chunk_id": retrieved[0].chunk_id if retrieved else "",
                "gold_in_top1": gold_evidence[:100] in _normalize(retrieved[0].text) if retrieved else False,
                "gold_token_overlap": round(overlap_ratio, 4),
            })
            print(f"         overlap={overlap_ratio:.2f}  top={retrieved[0].chunk_id if retrieved else 'none'}")

        except Exception as exc:
            print(f"         ERROR: {exc}")
            results.append({
                "system": "retrieval_reranker" if use_reranker else "retrieval_no_reranker",
                "benchmark_id": str(row.get("benchmark_id", "")),
                "question_type": str(row.get("question_type", "")),
                "question": question,
                "error": str(exc),
            })

        time.sleep(0.5)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        choices=["extractive", "qwen_base", "qwen_lora", "retrieval_reranker", "retrieval_no_reranker"],
        required=True,
    )
    parser.add_argument("--endpoint", default="", help="Colab endpoint URL (required for qwen_* systems).")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of questions (useful for testing).")
    args = parser.parse_args()

    if args.system.startswith("qwen") and not args.endpoint.strip():
        sys.exit("ERROR: --endpoint required for Qwen systems.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"ablation_{args.system}.json"

    print(f"\n{'='*60}")
    print(f"Running system: {args.system}")
    print(f"Output:         {out_path}")
    print(f"{'='*60}\n")

    if args.system in ("retrieval_reranker", "retrieval_no_reranker"):
        results = run_retrieval_ablation(use_reranker=(args.system == "retrieval_reranker"))
    else:
        results = run_generation_ablation(args.system, args.endpoint.strip(), args.limit)

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Quick summary
    valid = [r for r in results if "token_f1" in r]
    if valid:
        avg_f1 = sum(r["token_f1"] for r in valid) / len(valid)
        avg_rl = sum(r["rouge_l"] for r in valid) / len(valid)
        high_pct = sum(1 for r in valid if r.get("hallucination_risk") == "High") / len(valid) * 100
        num_rows = [r for r in valid if r.get("numerical_em") is not None]
        num_em_pct = sum(1 for r in num_rows if r["numerical_em"]) / len(num_rows) * 100 if num_rows else 0
        print(f"\nSummary ({len(valid)}/{len(results)} successful):")
        print(f"  Token F1:      {avg_f1:.3f}")
        print(f"  ROUGE-L:       {avg_rl:.3f}")
        print(f"  Numerical EM:  {num_em_pct:.1f}%  (metrics-generated questions)")
        print(f"  High halluc:   {high_pct:.1f}%")

    valid_ret = [r for r in results if "gold_token_overlap" in r]
    if valid_ret:
        avg_overlap = sum(r["gold_token_overlap"] for r in valid_ret) / len(valid_ret)
        top1_pct = sum(1 for r in valid_ret if r.get("gold_in_top1")) / len(valid_ret) * 100
        print(f"\nRetrieval summary ({len(valid_ret)}/{len(results)} successful):")
        print(f"  Gold token overlap (avg): {avg_overlap:.3f}")
        print(f"  Gold in top-1 chunk:      {top1_pct:.1f}%")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
