"""Evaluation metrics for all model types.

This module provides comprehensive metrics for:
- Regression tasks (MSE, RMSE, MAE, R², MAPE)
- Volatility forecasting (QLIKE, MZ regression)
- Classification tasks (if needed)
- Financial metrics (Sharpe-like ratios)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# REGRESSION METRICS
# ============================================================================


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error.

    MSE = mean((y_true - y_pred)^2)

    Lower is better. Range: [0, inf).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    RMSE = sqrt(MSE)

    Lower is better. Same units as target. Range: [0, inf).
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    MAE = mean(|y_true - y_pred|)

    Lower is better. Same units as target. Range: [0, inf).
    More robust to outliers than MSE.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Mean Absolute Percentage Error.

    MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    Lower is better. Range: [0, inf). In percentage.
    Warning: undefined when y_true contains zeros.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Symmetric Mean Absolute Percentage Error.

    SMAPE = mean(|y_true - y_pred| / (|y_true| + |y_pred|)) * 200

    Lower is better. Range: [0, 200]. More symmetric than MAPE.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return float(np.mean(numerator / denominator) * 200)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²).

    R² = 1 - SS_res / SS_tot

    Range: (-inf, 1]. Higher is better. 1 = perfect prediction.
    Can be negative if model is worse than predicting the mean.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Adjusted R² (penalizes for number of features).

    Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)

    where n = number of samples, p = number of features.
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n <= n_features + 1:
        return r2
    return float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum absolute error.

    Useful to identify worst-case prediction errors.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.max(np.abs(y_true - y_pred)))


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Error.

    More robust to outliers than MAE.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.median(np.abs(y_true - y_pred)))


# ============================================================================
# VOLATILITY/VARIANCE FORECASTING METRICS
# ============================================================================


def qlike(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """QLIKE loss for variance forecasting.

    QLIKE = mean(y_true / y_pred + log(y_pred))

    This is the quasi-likelihood loss, optimal for variance forecasting
    under Gaussian assumptions.

    Args:
        y_true: Actual variance/squared returns (must be positive).
        y_pred: Predicted variance (must be positive).
        epsilon: Small value to avoid division by zero.

    Returns:
        QLIKE loss value. Lower is better.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_pred_safe = np.maximum(y_pred, epsilon)
    return float(np.mean(y_true / y_pred_safe + np.log(y_pred_safe)))


def mse_log(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """MSE on log-transformed values.

    Useful for variance forecasting where we predict log-variance.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    log_true = np.log(np.maximum(y_true, epsilon))
    log_pred = np.log(np.maximum(y_pred, epsilon))
    return float(np.mean((log_true - log_pred) ** 2))


def mincer_zarnowitz_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """Mincer-Zarnowitz regression for forecast evaluation.

    Regresses: y_true = alpha + beta * y_pred + epsilon

    Returns:
        Tuple of (R², alpha, beta).
        - R² near 1 indicates good forecasts
        - alpha near 0 and beta near 1 indicate unbiased forecasts
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Add constant for regression
    X = np.column_stack([np.ones(len(y_pred)), y_pred])

    # OLS: (X'X)^-1 X'y
    try:
        coeffs = np.linalg.lstsq(X, y_true, rcond=None)[0]
        alpha, beta = coeffs[0], coeffs[1]

        # Compute R²
        y_fitted = alpha + beta * y_pred
        r2 = r2_score(y_true, y_fitted)

        return float(r2), float(alpha), float(beta)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 1.0


# ============================================================================
# DIRECTIONAL/SIGN METRICS
# ============================================================================


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Direction accuracy (percentage of correct sign predictions).

    Useful for financial returns where direction matters more than magnitude.

    Returns:
        Percentage of correct direction predictions (0-100).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Compare signs
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(correct) * 100)


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
    """Hit rate for binary predictions above/below threshold.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        threshold: Classification threshold.

    Returns:
        Hit rate percentage (0-100).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    true_class = y_true > threshold
    pred_class = y_pred > threshold
    return float(np.mean(true_class == pred_class) * 100)


# ============================================================================
# INFORMATION CRITERIA
# ============================================================================


def aic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_params: int,
) -> float:
    """Akaike Information Criterion.

    AIC = 2k + n * log(RSS/n)

    where k = number of parameters, n = number of samples.
    Lower is better.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)

    if rss <= 0 or n <= 0:
        return float("inf")

    return float(2 * n_params + n * np.log(rss / n))


def bic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_params: int,
) -> float:
    """Bayesian Information Criterion.

    BIC = k * log(n) + n * log(RSS/n)

    Penalizes model complexity more than AIC.
    Lower is better.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)

    if rss <= 0 or n <= 0:
        return float("inf")

    return float(n_params * np.log(n) + n * np.log(rss / n))


# ============================================================================
# RESIDUAL DIAGNOSTICS
# ============================================================================


@dataclass
class ResidualDiagnostics:
    """Diagnostics computed from residuals.

    Attributes:
        mean: Mean of residuals (should be ~0 for unbiased).
        std: Standard deviation of residuals.
        skewness: Skewness (0 for symmetric).
        kurtosis: Excess kurtosis (0 for normal).
        jarque_bera_stat: Jarque-Bera test statistic.
        ljung_box_stat: Ljung-Box test statistic (first lag).
        autocorr_lag1: First-order autocorrelation.
    """

    mean: float
    std: float
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    ljung_box_stat: float | None
    autocorr_lag1: float


def compute_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lag: int = 10,
) -> ResidualDiagnostics:
    """Compute comprehensive residual diagnostics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        max_lag: Maximum lag for autocorrelation tests.

    Returns:
        ResidualDiagnostics dataclass.
    """
    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    n = len(residuals)

    # Basic statistics
    mean = float(np.mean(residuals))
    std = float(np.std(residuals, ddof=1))

    # Standardized residuals for higher moments
    if std > 0:
        z = (residuals - mean) / std
    else:
        z = residuals - mean

    # Skewness and kurtosis
    skewness = float(np.mean(z**3))
    kurtosis = float(np.mean(z**4) - 3)  # Excess kurtosis

    # Jarque-Bera test statistic
    jb_stat = float((n / 6) * (skewness**2 + (kurtosis**2) / 4))

    # Autocorrelation at lag 1
    if n > 1:
        autocorr_lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    else:
        autocorr_lag1 = 0.0

    # Ljung-Box statistic (simplified, first lag only)
    lb_stat = None
    if n > max_lag:
        acf_values = []
        for lag in range(1, max_lag + 1):
            if n > lag:
                acf = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                acf_values.append(acf**2 / (n - lag))
        if acf_values:
            lb_stat = float(n * (n + 2) * sum(acf_values))

    return ResidualDiagnostics(
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        jarque_bera_stat=jb_stat,
        ljung_box_stat=lb_stat,
        autocorr_lag1=autocorr_lag1,
    )


# ============================================================================
# METRIC REGISTRY
# ============================================================================


REGRESSION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
    "smape": smape,
    "r2": r2_score,
    "max_error": max_error,
    "median_ae": median_absolute_error,
}

VOLATILITY_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "qlike": qlike,
    "mse_log": mse_log,
}

DIRECTION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "direction_accuracy": direction_accuracy,
}

ALL_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    **REGRESSION_METRICS,
    **VOLATILITY_METRICS,
    **DIRECTION_METRICS,
}


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
        y_true: Actual values.
        y_pred: Predicted values.
        metrics: List of metric names. If None, computes all regression metrics.

    Returns:
        Dict mapping metric names to values.
    """
    if metrics is None:
        metrics = list(REGRESSION_METRICS.keys())

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
