"""
finrag.evaluation — evaluation metrics and ablation study runner.

Metrics:
    from finrag.evaluation.metrics import token_f1, numerical_accuracy, hallucination_metrics

Ablation study:
    from finrag.evaluation.ablation import run_ablation, AblationConfig
    results = run_ablation(questions, configs=[AblationConfig.BASE_RAG,
                                               AblationConfig.FINETUNED_RAG_HD])
"""
from finrag.evaluation.metrics import (
    exact_match,
    hallucination_metrics,
    numerical_accuracy,
    token_f1,
)

__all__ = [
    "exact_match",
    "token_f1",
    "numerical_accuracy",
    "hallucination_metrics",
]
