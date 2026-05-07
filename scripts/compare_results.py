#!/usr/bin/env python3
"""Print the ablation comparison table from saved result files.

Run after all systems in run_ablation.py have completed.

Usage
-----
python scripts/compare_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"

GENERATION_SYSTEMS = {
    "extractive":           "A: Extractive (baseline)",
    "qwen_base":            "B: Qwen 7B base (no LoRA)",
    "qwen_lora":            "C: Qwen 7B + LoRA",
}

RETRIEVAL_SYSTEMS = {
    "retrieval_no_reranker": "D: Dense only (no reranker)",
    "retrieval_reranker":    "E: Dense + bge-reranker",
}

QTYPES = ["metrics-generated", "domain-relevant", "novel-generated"]


def load(system: str) -> list[dict]:
    path = OUTPUT_DIR / f"ablation_{system}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def gen_stats(rows: list[dict], qtype: str | None = None) -> dict:
    rows = [r for r in rows if "token_f1" in r]
    if qtype:
        rows = [r for r in rows if r.get("question_type") == qtype]
    if not rows:
        return {}
    n = len(rows)
    num_rows = [r for r in rows if r.get("numerical_em") is not None]
    return {
        "n":        n,
        "f1":       sum(r["token_f1"] for r in rows) / n,
        "rouge_l":  sum(r["rouge_l"] for r in rows) / n,
        "num_em":   sum(1 for r in num_rows if r["numerical_em"]) / len(num_rows) if num_rows else None,
        "high_pct": sum(1 for r in rows if r.get("hallucination_risk") == "High") / n * 100,
        "conf":     sum(r.get("confidence_score", 0) for r in rows) / n,
    }


def ret_stats(rows: list[dict]) -> dict:
    rows = [r for r in rows if "gold_token_overlap" in r]
    if not rows:
        return {}
    n = len(rows)
    return {
        "n":        n,
        "overlap":  sum(r["gold_token_overlap"] for r in rows) / n,
        "top1_pct": sum(1 for r in rows if r.get("gold_in_top1")) / n * 100,
    }


def hdr(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def fmt(val, fmt_str: str, missing: str = "  n/a") -> str:
    return format(val, fmt_str) if val is not None else missing


def print_generation_table(label: str, data: dict[str, dict]) -> None:
    hdr(label)
    print(f"{'System':<30} {'N':>4}  {'F1':>6}  {'ROUGE-L':>7}  {'Num-EM':>7}  {'High%':>6}  {'Conf':>5}")
    print("-" * 72)
    for key, name in GENERATION_SYSTEMS.items():
        s = data.get(key, {})
        if not s:
            print(f"{name:<30}  {'—':>4}  {'—':>6}  {'—':>7}  {'—':>7}  {'—':>6}  {'—':>5}")
            continue
        num_em_str = fmt(s.get("num_em"), ".1%") if s.get("num_em") is not None else "  n/a"
        print(
            f"{name:<30} {s['n']:>4}  {s['f1']:>6.3f}  {s['rouge_l']:>7.3f}  "
            f"{num_em_str:>7}  {s['high_pct']:>5.1f}%  {s['conf']:>5.2f}"
        )


def main() -> None:
    # ---- Generation ablation: overall ----
    gen_data_overall: dict[str, dict] = {}
    for key in GENERATION_SYSTEMS:
        rows = load(key)
        s = gen_stats(rows)
        if s:
            gen_data_overall[key] = s

    print_generation_table("GENERATION ABLATION — Overall (FinanceBench gold evidence)", gen_data_overall)

    # ---- Generation ablation: per question type ----
    for qtype in QTYPES:
        type_data: dict[str, dict] = {}
        for key in GENERATION_SYSTEMS:
            rows = load(key)
            s = gen_stats(rows, qtype)
            if s:
                type_data[key] = s
        print_generation_table(f"GENERATION ABLATION — {qtype}", type_data)

    # ---- Retrieval ablation ----
    hdr("RETRIEVAL ABLATION — Live retrieval with/without bge-reranker")
    print(f"{'System':<35} {'N':>4}  {'Gold overlap':>13}  {'Gold in top-1':>14}")
    print("-" * 72)
    for key, name in RETRIEVAL_SYSTEMS.items():
        rows = load(key)
        s = ret_stats(rows)
        if s:
            print(f"{name:<35} {s['n']:>4}  {s['overlap']:>13.3f}  {s['top1_pct']:>13.1f}%")
        else:
            print(f"{name:<35}  {'not run yet':>30}")

    # ---- Key findings ----
    hdr("KEY FINDINGS")
    a = gen_data_overall.get("extractive", {})
    b = gen_data_overall.get("qwen_base", {})
    c = gen_data_overall.get("qwen_lora", {})
    if a and b:
        delta_f1 = b.get("f1", 0) - a.get("f1", 0)
        print(f"  LLM generation vs extractive:   ΔF1 = {delta_f1:+.3f}")
    if b and c:
        delta_f1 = c.get("f1", 0) - b.get("f1", 0)
        print(f"  LoRA fine-tuning vs base model: ΔF1 = {delta_f1:+.3f}")
    d = ret_stats(load("retrieval_no_reranker"))
    e = ret_stats(load("retrieval_reranker"))
    if d and e:
        delta_overlap = e.get("overlap", 0) - d.get("overlap", 0)
        print(f"  Reranker vs dense-only:         Δoverlap = {delta_overlap:+.3f}")

    print()


if __name__ == "__main__":
    main()
