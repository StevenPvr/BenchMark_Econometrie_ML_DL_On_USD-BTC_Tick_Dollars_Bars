"""Evaluation metrics for classification models.

This module provides comprehensive metrics for:
- Multi-class classification (De Prado triple-barrier labeling: -1, 0, 1)
- Accuracy, F1, Precision, Recall (macro and weighted)
- Confusion matrix analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# MULTI-CLASS CLASSIFICATION METRICS (De Prado Triple-Barrier: -1, 0, 1)
# ============================================================================


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy.

    Returns:
        Fraction of correct predictions (0-1).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy (average recall per class).

    Useful for imbalanced multi-class classification.
    Returns average of per-class recall.

    Returns:
        Balanced accuracy (0-1).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else 0.0


def precision_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged precision for multi-class classification.

    Returns:
        Macro precision (0-1).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)

    return float(np.mean(precisions)) if precisions else 0.0


def recall_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged recall for multi-class classification.

    Returns:
        Macro recall (0-1).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(y_true)
    recalls = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0.0)

    return float(np.mean(recalls)) if recalls else 0.0


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 score for multi-class classification.

    Calculates F1 for each class and averages them.

    Returns:
        Macro F1 score (0-1).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted F1 score for multi-class classification.

    Weights F1 for each class by its support (number of samples).

    Returns:
        Weighted F1 score (0-1).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = []
    weights = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        support = np.sum(y_true == cls)
        f1_scores.append(f1)
        weights.append(support)

    total_weight = sum(weights)
    if total_weight > 0:
        return float(sum(f * w for f, w in zip(f1_scores, weights)) / total_weight)
    return 0.0


def log_loss_multiclass(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    """Multi-class logarithmic loss (cross-entropy).

    Args:
        y_true: True class labels.
        y_pred_proba: Predicted probabilities of shape (n_samples, n_classes).
        epsilon: Small value for numerical stability.

    Returns:
        Log loss value (lower is better).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba)

    # Clip probabilities to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    # Get unique classes
    classes = np.unique(y_true)
    n_samples = len(y_true)

    # One-hot encode y_true
    y_true_onehot = np.zeros_like(y_pred_proba)
    for i, cls in enumerate(classes):
        y_true_onehot[y_true == cls, i] = 1

    # Calculate log loss
    return float(-np.sum(y_true_onehot * np.log(y_pred_proba)) / n_samples)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix for multi-class classification.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.

    Returns:
        Confusion matrix of shape (n_classes, n_classes).
        Entry [i, j] is count of samples with true class i and predicted class j.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_val, pred_val in zip(y_true, y_pred):
        cm[class_to_idx[true_val], class_to_idx[pred_val]] += 1

    return cm


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    """Compute per-class accuracy for multi-class classification.

    Returns:
        Dict mapping class label to accuracy for that class.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(y_true)
    result = {}

    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            result[int(cls)] = float(np.mean(y_pred[mask] == cls))
        else:
            result[int(cls)] = 0.0

    return result


# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================


@dataclass
class ClassificationReport:
    """Classification report for multi-class problems.

    Attributes:
        accuracy: Overall accuracy.
        balanced_accuracy: Balanced accuracy (macro recall).
        f1_macro: Macro-averaged F1 score.
        f1_weighted: Weighted F1 score.
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        per_class_accuracy: Per-class accuracy dict.
        confusion_matrix: Confusion matrix.
        class_labels: List of class labels.
    """

    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    per_class_accuracy: dict[int, float]
    confusion_matrix: np.ndarray
    class_labels: list[int]


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ClassificationReport:
    """Compute comprehensive classification report.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.

    Returns:
        ClassificationReport dataclass.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(np.concatenate([y_true, y_pred]))

    return ClassificationReport(
        accuracy=accuracy(y_true, y_pred),
        balanced_accuracy=balanced_accuracy(y_true, y_pred),
        f1_macro=f1_macro(y_true, y_pred),
        f1_weighted=f1_weighted(y_true, y_pred),
        precision_macro=precision_macro(y_true, y_pred),
        recall_macro=recall_macro(y_true, y_pred),
        per_class_accuracy=per_class_accuracy(y_true, y_pred),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        class_labels=[int(c) for c in classes],
    )


# ============================================================================
# METRIC REGISTRY
# ============================================================================


CLASSIFICATION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "f1_weighted": f1_weighted,
}

# All metrics available (classification only)
ALL_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    **CLASSIFICATION_METRICS,
}

# Default metrics for classification
DEFAULT_CLASSIFICATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
]


def get_metric(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get a metric function by name.

    Args:
        name: Metric name (case-insensitive).

    Returns:
        Metric function.

    Raises:
        ValueError: If metric not found.
    """
    name_lower = name.lower()
    if name_lower not in ALL_METRICS:
        raise ValueError(f"Unknown metric '{name}'. Available: {list(ALL_METRICS.keys())}")
    return ALL_METRICS[name_lower]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute multiple metrics at once.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        metrics: List of metric names. If None, computes all classification metrics.

    Returns:
        Dict mapping metric names to values.
    """
    if metrics is None:
        metrics = DEFAULT_CLASSIFICATION_METRICS

    results = {}
    for metric_name in metrics:
        try:
            metric_fn = get_metric(metric_name)
            results[metric_name] = metric_fn(y_true, y_pred)
        except Exception as e:
            logger.warning("Failed to compute %s: %s", metric_name, e)
            results[metric_name] = float("nan")

    return results


def list_available_metrics() -> list[str]:
    """List all available metric names."""
    return list(ALL_METRICS.keys())
