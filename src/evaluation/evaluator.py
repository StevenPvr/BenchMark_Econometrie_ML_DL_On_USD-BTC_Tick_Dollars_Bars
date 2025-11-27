"""Generic model evaluator for classification models.

This module provides a unified evaluation interface compatible with:
- Econometric classifiers (Ridge, Lasso, Logistic)
- Machine Learning classifiers (XGBoost, LightGBM, CatBoost, RandomForest)
- Deep Learning classifiers (LSTM)
- Baseline classifiers (Random, Persistence, AR1)

Supports De Prado's triple-barrier labeling (-1, 0, 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.evaluation.metrics import (
    ALL_METRICS,
    CLASSIFICATION_METRICS,
    DEFAULT_CLASSIFICATION_METRICS,
    ClassificationReport,
    compute_classification_report,
    compute_metrics,
    confusion_matrix,
    per_class_accuracy,
)
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


# ============================================================================
# PROTOCOLS
# ============================================================================


class PredictableModel(Protocol):
    """Protocol for models that can make predictions."""

    name: str
    is_fitted: bool

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Get model hyperparameters."""
        ...


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.

    Attributes:
        metrics: List of metrics to compute (None = all classification metrics).
        compute_confusion_matrix: Whether to compute confusion matrix.
        compute_per_class: Whether to compute per-class metrics.
        verbose: Verbosity level (0=silent, 1=summary, 2=detailed).
        output_dir: Directory for saving outputs.
    """

    metrics: list[str] | None = None
    compute_confusion_matrix: bool = True
    compute_per_class: bool = True
    verbose: int = 1
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.metrics is not None:
            for metric in self.metrics:
                if metric.lower() not in ALL_METRICS:
                    raise ValueError(
                        f"Unknown metric '{metric}'. Available: {list(ALL_METRICS.keys())}"
                    )


# ============================================================================
# EVALUATION RESULTS
# ============================================================================


@dataclass
class EvaluationResult:
    """Result from model evaluation.

    Attributes:
        model_name: Name of the evaluated model.
        metrics: Dict of metric name to value.
        predictions: Model predictions (class labels).
        actuals: Actual class labels.
        confusion_matrix: Confusion matrix (n_classes x n_classes).
        per_class_accuracy: Per-class accuracy dict.
        class_labels: List of class labels.
        evaluation_time: Evaluation duration in seconds.
        n_samples: Number of samples evaluated.
        model_params: Model hyperparameters.
    """

    model_name: str
    metrics: dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    confusion_matrix: np.ndarray | None
    per_class_accuracy: dict[int, float] | None
    class_labels: list[int] | None
    evaluation_time: float
    n_samples: int
    model_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result = {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "n_samples": self.n_samples,
            "evaluation_time": self.evaluation_time,
            "model_params": self.model_params,
        }

        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix.tolist()
            result["class_labels"] = self.class_labels

        if self.per_class_accuracy is not None:
            result["per_class_accuracy"] = self.per_class_accuracy

        return result

    def summary(self) -> str:
        """Generate a text summary of the evaluation."""
        lines = [
            f"=== Classification Results: {self.model_name} ===",
            f"Samples: {self.n_samples}",
            f"Evaluation time: {self.evaluation_time:.2f}s",
            "",
            "Metrics:",
        ]

        for metric, value in self.metrics.items():
            lines.append(f"  {metric}: {value:.4f}")

        if self.per_class_accuracy is not None:
            lines.extend(["", "Per-class Accuracy:"])
            for cls, acc in sorted(self.per_class_accuracy.items()):
                lines.append(f"  Class {cls}: {acc:.4f}")

        if self.confusion_matrix is not None and self.class_labels is not None:
            lines.extend(["", "Confusion Matrix:"])
            # Header
            header = "      " + "  ".join(f"{c:>5}" for c in self.class_labels)
            lines.append(header)
            # Rows
            for i, cls in enumerate(self.class_labels):
                row = f"{cls:>5} " + "  ".join(f"{v:>5}" for v in self.confusion_matrix[i])
                lines.append(row)

        return "\n".join(lines)


# ============================================================================
# EVALUATOR CLASS
# ============================================================================


class Evaluator:
    """Generic evaluator for classification models.

    This evaluator provides a unified interface for evaluating any model
    that implements the PredictableModel protocol.

    Example:
        >>> from src.model.machine_learning.lightgbm.lightgbm_model import LightGBMModel
        >>> from src.evaluation import Evaluator, EvaluationConfig
        >>>
        >>> model = LightGBMModel(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>>
        >>> evaluator = Evaluator(config=EvaluationConfig(
        ...     metrics=["accuracy", "f1_macro", "balanced_accuracy"],
        ... ))
        >>> result = evaluator.evaluate(model, X_test, y_test)
        >>> print(result.summary())
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration.
        """
        self.config = config or EvaluationConfig()

    def evaluate(
        self,
        model: PredictableModel,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> EvaluationResult:
        """Evaluate a model on test data.

        Args:
            model: Trained model to evaluate.
            X: Test features.
            y: Test target (class labels).

        Returns:
            EvaluationResult with all computed metrics.
        """
        start_time = datetime.now()

        # Convert to numpy
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        if self.config.verbose >= 1:
            logger.info("Evaluating %s on %d samples", model.name, len(y_arr))

        # Get predictions
        predictions = model.predict(X_arr)

        # Compute metrics
        metrics_to_compute = self.config.metrics or DEFAULT_CLASSIFICATION_METRICS
        metrics = compute_metrics(y_arr, predictions, metrics_to_compute)

        if self.config.verbose >= 2:
            for metric, value in metrics.items():
                logger.info("  %s: %.4f", metric, value)

        # Confusion matrix
        cm = None
        class_labels = None
        if self.config.compute_confusion_matrix:
            cm = confusion_matrix(y_arr, predictions)
            class_labels = sorted([int(c) for c in np.unique(np.concatenate([y_arr, predictions]))])

        # Per-class accuracy
        per_class_acc = None
        if self.config.compute_per_class:
            per_class_acc = per_class_accuracy(y_arr, predictions)

        evaluation_time = (datetime.now() - start_time).total_seconds()

        if self.config.verbose >= 1:
            logger.info(
                "Evaluation complete: accuracy=%.4f, f1_macro=%.4f (%d samples, %.2fs)",
                metrics.get("accuracy", float("nan")),
                metrics.get("f1_macro", float("nan")),
                len(y_arr),
                evaluation_time,
            )

        return EvaluationResult(
            model_name=model.name,
            metrics=metrics,
            predictions=predictions,
            actuals=y_arr,
            confusion_matrix=cm,
            per_class_accuracy=per_class_acc,
            class_labels=class_labels,
            evaluation_time=evaluation_time,
            n_samples=len(y_arr),
            model_params=model.get_params(),
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def evaluate_model(
    model: PredictableModel,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    metrics: list[str] | None = None,
    verbose: bool = True,
) -> EvaluationResult:
    """Convenience function to evaluate a model.

    Args:
        model: Trained model to evaluate.
        X: Test features.
        y: Test target (class labels).
        metrics: List of metrics to compute.
        verbose: Show progress.

    Returns:
        EvaluationResult with computed metrics.

    Example:
        >>> result = evaluate_model(model, X_test, y_test)
        >>> print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    """
    config = EvaluationConfig(
        metrics=metrics,
        verbose=1 if verbose else 0,
    )
    evaluator = Evaluator(config)
    return evaluator.evaluate(model, X, y)


def quick_evaluate(
    model: PredictableModel,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Quick evaluation returning only metrics dict.

    Args:
        model: Trained model.
        X: Test features.
        y: Test target (class labels).

    Returns:
        Dict of metric names to values.
    """
    config = EvaluationConfig(
        verbose=0,
        compute_confusion_matrix=False,
        compute_per_class=False,
    )
    evaluator = Evaluator(config)
    result = evaluator.evaluate(model, X, y)
    return result.metrics


def save_evaluation_results(
    result: EvaluationResult,
    output_file: str | Path,
    include_predictions: bool = False,
) -> None:
    """Save evaluation results to JSON file.

    Args:
        result: Evaluation result to save.
        output_file: Output file path.
        include_predictions: If True, include predictions array.
    """
    output_path = Path(output_file)

    data = result.to_dict()
    if include_predictions:
        data["predictions"] = result.predictions.tolist()
        data["actuals"] = result.actuals.tolist()

    save_json_pretty(data, output_path)
    logger.info("Saved evaluation results to %s", output_path)
