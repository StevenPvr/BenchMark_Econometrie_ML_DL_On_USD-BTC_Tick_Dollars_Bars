"""Evaluation package for primary labeling models."""

from src.labelling.label_primaire.eval.logic import (
    ClassificationMetrics,
    EvaluationResult,
    compute_metrics,
    evaluate_model,
    get_features_path,
    get_labeled_features_path,
    get_trained_models,
    load_data,
    load_model,
    print_comparison,
    print_confusion_matrix,
    print_metrics,
    print_results,
)

__all__ = [
    "ClassificationMetrics",
    "EvaluationResult",
    "compute_metrics",
    "evaluate_model",
    "get_features_path",
    "get_labeled_features_path",
    "get_trained_models",
    "load_data",
    "load_model",
    "print_comparison",
    "print_confusion_matrix",
    "print_metrics",
    "print_results",
]
