"""Evaluation module for classification models.

This module provides a unified evaluation interface compatible with:
- Econometric classifiers (RidgeClassifier, LassoClassifier, Logistic)
- Machine Learning classifiers (XGBoost, LightGBM, CatBoost, RandomForest)
- Deep Learning classifiers (LSTM)

Key Features:
- Multi-class classification metrics (De Prado triple-barrier: -1, 0, 1)
- Accuracy, F1, Precision, Recall (macro and weighted)
- Confusion matrix analysis
- Model comparison with statistical significance tests

Example Usage:
    >>> from src.model.machine_learning.xgboost_model import XGBoostModel
    >>> from src.evaluation import evaluate_model
    >>>
    >>> # Evaluate a single model
    >>> model = XGBoostModel(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> result = evaluate_model(model, X_test, y_test)
    >>> print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    >>> print(f"F1 (macro): {result.metrics['f1_macro']:.4f}")

Computing specific metrics:
    >>> from src.evaluation import accuracy, f1_macro, balanced_accuracy
    >>> acc = accuracy(y_true, y_pred)
    >>> f1 = f1_macro(y_true, y_pred)
"""

from __future__ import annotations

# Metrics
from src.evaluation.metrics import (
    ALL_METRICS,
    CLASSIFICATION_METRICS,
    DEFAULT_CLASSIFICATION_METRICS,
    accuracy,
    balanced_accuracy,
    compute_classification_report,
    compute_metrics,
    confusion_matrix,
    f1_macro,
    f1_weighted,
    get_metric,
    list_available_metrics,
    log_loss_multiclass,
    per_class_accuracy,
    precision_macro,
    recall_macro,
)

# Evaluator
from src.evaluation.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    PredictableModel,
    evaluate_model,
    quick_evaluate,
    save_evaluation_results,
)

# Comparison
from src.evaluation.comparison import (
    ModelComparisonResult,
    ModelComparator,
    PairwiseTestResult,
    compare_models,
    diebold_mariano_test,
    pairwise_dm_tests,
    rank_models,
    save_comparison_results,
)

__all__ = [
    # Classification metric functions
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "log_loss_multiclass",
    "confusion_matrix",
    "per_class_accuracy",
    "compute_classification_report",
    # Metric utilities
    "get_metric",
    "compute_metrics",
    "list_available_metrics",
    # Metric registries
    "ALL_METRICS",
    "CLASSIFICATION_METRICS",
    "DEFAULT_CLASSIFICATION_METRICS",
    # Data classes
    "EvaluationResult",
    "EvaluationConfig",
    "ModelComparisonResult",
    "PairwiseTestResult",
    # Evaluator
    "Evaluator",
    "PredictableModel",
    "evaluate_model",
    "quick_evaluate",
    "save_evaluation_results",
    # Comparison
    "ModelComparator",
    "compare_models",
    "rank_models",
    "diebold_mariano_test",
    "pairwise_dm_tests",
    "save_comparison_results",
]
