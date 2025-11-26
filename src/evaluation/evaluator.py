"""Generic model evaluator for all model types.

This module provides a unified evaluation interface compatible with:
- Econometric models (Ridge, Lasso)
- Machine Learning models (XGBoost, LightGBM, CatBoost, RandomForest)
- Deep Learning models (LSTM)
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
    REGRESSION_METRICS,
    ResidualDiagnostics,
    compute_metrics,
    compute_residual_diagnostics,
    mincer_zarnowitz_r2,
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
        metrics: List of metrics to compute (None = all regression metrics).
        compute_residual_diagnostics: Whether to compute residual diagnostics.
        compute_mz_regression: Whether to compute Mincer-Zarnowitz regression.
        verbose: Verbosity level (0=silent, 1=summary, 2=detailed).
        output_dir: Directory for saving outputs.
    """

    metrics: list[str] | None = None
    compute_residual_diagnostics: bool = True
    compute_mz_regression: bool = False
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
        predictions: Model predictions.
        actuals: Actual values.
        residuals: Prediction residuals (actuals - predictions).
        residual_diagnostics: Residual analysis results.
        mz_r2: Mincer-Zarnowitz R².
        mz_alpha: Mincer-Zarnowitz intercept.
        mz_beta: Mincer-Zarnowitz slope.
        evaluation_time: Evaluation duration in seconds.
        n_samples: Number of samples evaluated.
        model_params: Model hyperparameters.
    """

    model_name: str
    metrics: dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    residuals: np.ndarray
    residual_diagnostics: ResidualDiagnostics | None
    mz_r2: float | None
    mz_alpha: float | None
    mz_beta: float | None
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

        if self.mz_r2 is not None:
            result["mincer_zarnowitz"] = {
                "r2": self.mz_r2,
                "alpha": self.mz_alpha,
                "beta": self.mz_beta,
            }

        if self.residual_diagnostics is not None:
            result["residual_diagnostics"] = {
                "mean": self.residual_diagnostics.mean,
                "std": self.residual_diagnostics.std,
                "skewness": self.residual_diagnostics.skewness,
                "kurtosis": self.residual_diagnostics.kurtosis,
                "jarque_bera_stat": self.residual_diagnostics.jarque_bera_stat,
                "autocorr_lag1": self.residual_diagnostics.autocorr_lag1,
            }

        return result

    def summary(self) -> str:
        """Generate a text summary of the evaluation."""
        lines = [
            f"=== Evaluation Results: {self.model_name} ===",
            f"Samples: {self.n_samples}",
            f"Evaluation time: {self.evaluation_time:.2f}s",
            "",
            "Metrics:",
        ]

        for metric, value in self.metrics.items():
            lines.append(f"  {metric}: {value:.6f}")

        if self.mz_r2 is not None:
            lines.extend([
                "",
                "Mincer-Zarnowitz Regression:",
                f"  R²: {self.mz_r2:.4f}",
                f"  alpha: {self.mz_alpha:.4f}",
                f"  beta: {self.mz_beta:.4f}",
            ])

        if self.residual_diagnostics is not None:
            diag = self.residual_diagnostics
            lines.extend([
                "",
                "Residual Diagnostics:",
                f"  Mean: {diag.mean:.6f}",
                f"  Std: {diag.std:.6f}",
                f"  Skewness: {diag.skewness:.4f}",
                f"  Kurtosis: {diag.kurtosis:.4f}",
                f"  Autocorr(1): {diag.autocorr_lag1:.4f}",
            ])

        return "\n".join(lines)


# ============================================================================
# EVALUATOR CLASS
# ============================================================================


class Evaluator:
    """Generic evaluator for all model types.

    This evaluator provides a unified interface for evaluating any model
    that implements the PredictableModel protocol.

    Example:
        >>> from src.model.machine_learning.xgboost_model import XGBoostModel
        >>> from src.evaluation import Evaluator, EvaluationConfig
        >>>
        >>> model = XGBoostModel(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>>
        >>> evaluator = Evaluator(config=EvaluationConfig(
        ...     metrics=["mse", "rmse", "mae", "r2"],
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
            y: Test target.

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
        residuals = y_arr - predictions

        # Compute metrics
        metrics_to_compute = self.config.metrics or list(REGRESSION_METRICS.keys())
        metrics = compute_metrics(y_arr, predictions, metrics_to_compute)

        if self.config.verbose >= 2:
            for metric, value in metrics.items():
                logger.info("  %s: %.6f", metric, value)

        # Residual diagnostics
        residual_diag = None
        if self.config.compute_residual_diagnostics:
            residual_diag = compute_residual_diagnostics(y_arr, predictions)

        # Mincer-Zarnowitz regression
        mz_r2, mz_alpha, mz_beta = None, None, None
        if self.config.compute_mz_regression:
            mz_r2, mz_alpha, mz_beta = mincer_zarnowitz_r2(y_arr, predictions)

        evaluation_time = (datetime.now() - start_time).total_seconds()

        if self.config.verbose >= 1:
            primary_metric = metrics_to_compute[0]
            logger.info(
                "Evaluation complete: %s=%.6f (%d samples, %.2fs)",
                primary_metric,
                metrics.get(primary_metric, float("nan")),
                len(y_arr),
                evaluation_time,
            )

        return EvaluationResult(
            model_name=model.name,
            metrics=metrics,
            predictions=predictions,
            actuals=y_arr,
            residuals=residuals,
            residual_diagnostics=residual_diag,
            mz_r2=mz_r2,
            mz_alpha=mz_alpha,
            mz_beta=mz_beta,
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
        y: Test target.
        metrics: List of metrics to compute.
        verbose: Show progress.

    Returns:
        EvaluationResult with computed metrics.

    Example:
        >>> result = evaluate_model(model, X_test, y_test)
        >>> print(f"MSE: {result.metrics['mse']:.4f}")
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
        y: Test target.

    Returns:
        Dict of metric names to values.
    """
    config = EvaluationConfig(verbose=0)
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
