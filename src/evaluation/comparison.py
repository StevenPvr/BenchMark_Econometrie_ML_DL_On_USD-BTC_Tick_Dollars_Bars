"""Model comparison utilities.

This module provides tools for comparing multiple models:
- Side-by-side metric comparison
- Statistical significance tests (Diebold-Mariano)
- Ranking and selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.evaluation.evaluator import EvaluationResult, Evaluator, EvaluationConfig, PredictableModel
from src.evaluation.metrics import REGRESSION_METRICS
from src.utils import get_logger, save_json_pretty
from pathlib import Path

logger = get_logger(__name__)


# ============================================================================
# COMPARISON RESULTS
# ============================================================================


@dataclass
class ModelComparisonResult:
    """Result from comparing multiple models.

    Attributes:
        model_names: List of model names compared.
        metrics_df: DataFrame with metrics for each model.
        rankings: Dict mapping metric to ranking of models.
        best_model: Name of the best model (by primary metric).
        primary_metric: Metric used for determining best model.
        evaluation_results: Individual evaluation results.
    """

    model_names: list[str]
    metrics_df: pd.DataFrame
    rankings: dict[str, list[str]]
    best_model: str
    primary_metric: str
    evaluation_results: dict[str, EvaluationResult]

    def summary(self) -> str:
        """Generate a text summary of the comparison."""
        lines = [
            "=== Model Comparison Summary ===",
            f"Models compared: {len(self.model_names)}",
            f"Primary metric: {self.primary_metric}",
            f"Best model: {self.best_model}",
            "",
            "Metrics by model:",
            self.metrics_df.to_string(),
            "",
            f"Rankings (by {self.primary_metric}):",
        ]

        for i, model in enumerate(self.rankings[self.primary_metric], 1):
            metric_value = self.metrics_df.loc[model, self.primary_metric]
            lines.append(f"  {i}. {model}: {metric_value:.6f}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_names": self.model_names,
            "metrics": self.metrics_df.to_dict(),
            "rankings": self.rankings,
            "best_model": self.best_model,
            "primary_metric": self.primary_metric,
        }


# ============================================================================
# STATISTICAL TESTS
# ============================================================================


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> tuple[float, float]:
    """Diebold-Mariano test for comparing forecast accuracy.

    Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)
    and L is the loss function (default: squared error).

    Args:
        errors1: Forecast errors from model 1.
        errors2: Forecast errors from model 2.
        h: Forecast horizon (for HAC standard errors).
        power: Power of the loss function (2 = MSE, 1 = MAE).

    Returns:
        Tuple of (DM statistic, p-value).
        Negative DM -> model 1 is better.
        Positive DM -> model 2 is better.
    """
    errors1 = np.asarray(errors1).ravel()
    errors2 = np.asarray(errors2).ravel()

    # Loss differential
    d = np.abs(errors1) ** power - np.abs(errors2) ** power
    n = len(d)

    if n < 2:
        return 0.0, 1.0

    # Mean loss differential
    d_bar = np.mean(d)

    # HAC variance estimate (Newey-West)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0

    for k in range(1, h):
        gamma_k = np.cov(d[:-k], d[k:])[0, 1] if len(d) > k else 0.0
        weight = 1 - k / h  # Bartlett kernel
        gamma_sum += 2 * weight * gamma_k

    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return 0.0, 1.0

    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d)

    # P-value (two-sided, using normal approximation)
    try:
        from scipy.stats import norm  # type: ignore
        p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    except ImportError:
        # Approximate using standard normal
        p_value = 2 * (1 - 0.5 * (1 + np.tanh(abs(dm_stat) / np.sqrt(2))))

    return float(dm_stat), float(p_value)


@dataclass
class PairwiseTestResult:
    """Result from pairwise statistical test.

    Attributes:
        model1: First model name.
        model2: Second model name.
        test_statistic: Test statistic value.
        p_value: P-value.
        significant: Whether difference is significant at alpha level.
        better_model: Name of significantly better model (or None).
    """

    model1: str
    model2: str
    test_statistic: float
    p_value: float
    significant: bool
    better_model: str | None


def pairwise_dm_tests(
    evaluation_results: dict[str, EvaluationResult],
    alpha: float = 0.05,
) -> list[PairwiseTestResult]:
    """Perform pairwise Diebold-Mariano tests between all models.

    Args:
        evaluation_results: Dict mapping model names to evaluation results.
        alpha: Significance level.

    Returns:
        List of PairwiseTestResult for each pair of models.
    """
    model_names = list(evaluation_results.keys())
    results = []

    for i, model1 in enumerate(model_names):
        for model2 in model_names[i + 1:]:
            errors1 = evaluation_results[model1].residuals
            errors2 = evaluation_results[model2].residuals

            dm_stat, p_value = diebold_mariano_test(errors1, errors2)
            significant = p_value < alpha

            better_model = None
            if significant:
                # Negative DM -> model 1 better, Positive DM -> model 2 better
                better_model = model1 if dm_stat < 0 else model2

            results.append(PairwiseTestResult(
                model1=model1,
                model2=model2,
                test_statistic=dm_stat,
                p_value=p_value,
                significant=significant,
                better_model=better_model,
            ))

    return results


# ============================================================================
# MODEL COMPARISON
# ============================================================================


class ModelComparator:
    """Compare multiple models on the same test set.

    Example:
        >>> comparator = ModelComparator(metrics=["mse", "rmse", "mae"])
        >>> result = comparator.compare(
        ...     models={"xgb": xgb_model, "lgb": lgb_model, "rf": rf_model},
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        primary_metric: str = "mse",
        lower_is_better: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialize the comparator.

        Args:
            metrics: List of metrics to compute.
            primary_metric: Metric for ranking/best model selection.
            lower_is_better: If True, lower metric values are better.
            verbose: Show progress.
        """
        self.metrics = metrics or list(REGRESSION_METRICS.keys())
        self.primary_metric = primary_metric
        self.lower_is_better = lower_is_better
        self.verbose = verbose

        if primary_metric not in self.metrics:
            self.metrics.append(primary_metric)

    def compare(
        self,
        models: dict[str, PredictableModel],
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
    ) -> ModelComparisonResult:
        """Compare multiple models on test data.

        Args:
            models: Dict mapping model names to trained models.
            X_test: Test features.
            y_test: Test target.

        Returns:
            ModelComparisonResult with comparison results.
        """
        if self.verbose:
            logger.info("Comparing %d models on %d samples", len(models), len(y_test))

        # Evaluate each model
        config = EvaluationConfig(
            metrics=self.metrics,
            verbose=0,
        )
        evaluator = Evaluator(config)

        evaluation_results: dict[str, EvaluationResult] = {}
        metrics_data: dict[str, dict[str, float]] = {}

        for name, model in models.items():
            if self.verbose:
                logger.info("Evaluating %s...", name)

            result = evaluator.evaluate(model, X_test, y_test)
            evaluation_results[name] = result
            metrics_data[name] = result.metrics

        # Create metrics DataFrame
        metrics_df = pd.DataFrame(metrics_data).T
        metrics_df.index.name = "model"

        # Compute rankings for each metric
        rankings: dict[str, list[str]] = {}
        for metric in self.metrics:
            if metric in metrics_df.columns:
                # metrics_df[metric] returns a Series when metric is a column name
                metric_series = cast(pd.Series, metrics_df[metric])
                sorted_series = metric_series.sort_values(ascending=self.lower_is_better)
                sorted_models = [str(m) for m in sorted_series.index.tolist()]
                rankings[metric] = sorted_models

        # Determine best model
        best_model = rankings[self.primary_metric][0]

        if self.verbose:
            logger.info(
                "Comparison complete. Best model: %s (%s=%.6f)",
                best_model,
                self.primary_metric,
                metrics_df.loc[best_model, self.primary_metric],
            )

        return ModelComparisonResult(
            model_names=list(models.keys()),
            metrics_df=metrics_df,
            rankings=rankings,
            best_model=best_model,
            primary_metric=self.primary_metric,
            evaluation_results=evaluation_results,
        )

    def compare_with_significance(
        self,
        models: dict[str, PredictableModel],
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        alpha: float = 0.05,
    ) -> tuple[ModelComparisonResult, list[PairwiseTestResult]]:
        """Compare models with statistical significance tests.

        Args:
            models: Dict mapping model names to trained models.
            X_test: Test features.
            y_test: Test target.
            alpha: Significance level for DM tests.

        Returns:
            Tuple of (comparison result, pairwise test results).
        """
        comparison = self.compare(models, X_test, y_test)
        dm_tests = pairwise_dm_tests(comparison.evaluation_results, alpha)

        if self.verbose:
            significant_diffs = [t for t in dm_tests if t.significant]
            logger.info(
                "Statistical tests: %d/%d pairs show significant difference (alpha=%.2f)",
                len(significant_diffs),
                len(dm_tests),
                alpha,
            )

        return comparison, dm_tests


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def compare_models(
    models: dict[str, PredictableModel],
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    metrics: list[str] | None = None,
    primary_metric: str = "mse",
    verbose: bool = True,
) -> ModelComparisonResult:
    """Convenience function to compare models.

    Args:
        models: Dict mapping model names to trained models.
        X_test: Test features.
        y_test: Test target.
        metrics: Metrics to compute.
        primary_metric: Metric for ranking.
        verbose: Show progress.

    Returns:
        ModelComparisonResult.

    Example:
        >>> result = compare_models(
        ...     {"xgb": xgb_model, "lgb": lgb_model},
        ...     X_test, y_test,
        ...     metrics=["mse", "rmse", "r2"],
        ... )
        >>> print(f"Best model: {result.best_model}")
    """
    comparator = ModelComparator(
        metrics=metrics,
        primary_metric=primary_metric,
        verbose=verbose,
    )
    return comparator.compare(models, X_test, y_test)


def rank_models(
    models: dict[str, PredictableModel],
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    metric: str = "mse",
) -> list[tuple[str, float]]:
    """Rank models by a single metric.

    Args:
        models: Dict mapping model names to trained models.
        X_test: Test features.
        y_test: Test target.
        metric: Metric for ranking.

    Returns:
        List of (model_name, metric_value) tuples, sorted best to worst.
    """
    comparator = ModelComparator(
        metrics=[metric],
        primary_metric=metric,
        verbose=False,
    )
    result = comparator.compare(models, X_test, y_test)

    return [
        (name, result.metrics_df.loc[name, metric])
        for name in result.rankings[metric]
    ]


def save_comparison_results(
    result: ModelComparisonResult,
    output_file: str | Path,
) -> None:
    """Save comparison results to JSON file.

    Args:
        result: Comparison result to save.
        output_file: Output file path.
    """
    output_path = Path(output_file)
    save_json_pretty(result.to_dict(), output_path)
    logger.info("Saved comparison results to %s", output_path)
