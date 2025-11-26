"""Fractionally Differentiated Features (De Prado).

This module implements fractional differentiation to create stationary
features while preserving long memory, as described in:

    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 5: Fractionally Differentiated Features.

Standard differentiation (d=1) makes series stationary but loses memory.
Fractional differentiation (0 < d < 1) balances stationarity and memory:

    (1 - B)^d * X_t = Σ_{k=0}^{∞} w_k * X_{t-k}

Where:
    - B is the backshift operator
    - d is the fractional order (typically 0.3-0.7)
    - w_k are the binomial weights

The weights decay as:
    w_k = -w_{k-1} * (d - k + 1) / k

Key insight:
    - d ≈ 0: Original series (non-stationary, full memory)
    - d ≈ 1: First difference (stationary, no memory)
    - d ≈ 0.4-0.6: Sweet spot (stationary enough, retains memory)

Two methods implemented:
    1. Fixed-window (FFD): Truncates weights at threshold
    2. Expanding window: Uses all available history

Reference:
    Hosking, J. R. M. (1981). Fractional Differencing. Biometrika.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.config_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_frac_diff_weights",
    "compute_frac_diff",
    "compute_frac_diff_ffd",
    "find_min_frac_diff_order",
    "compute_frac_diff_features",
]


# =============================================================================
# WEIGHT COMPUTATION
# =============================================================================


@njit(cache=True)
def _compute_weights(d: float, size: int) -> NDArray[np.float64]:
    """Compute fractional differentiation weights (numba optimized).

    w_k = -w_{k-1} * (d - k + 1) / k

    Args:
        d: Fractional differentiation order.
        size: Number of weights to compute.

    Returns:
        Array of weights.
    """
    weights = np.zeros(size, dtype=np.float64)
    weights[0] = 1.0

    for k in range(1, size):
        weights[k] = -weights[k - 1] * (d - k + 1) / k

    return weights


@njit(cache=True)
def _compute_weights_ffd(
    d: float,
    threshold: float,
    max_size: int,
) -> NDArray[np.float64]:
    """Compute FFD weights with threshold cutoff (numba optimized).

    Stops when |w_k| < threshold to create fixed-width window.

    Args:
        d: Fractional differentiation order.
        threshold: Weight cutoff threshold.
        max_size: Maximum number of weights.

    Returns:
        Array of weights (truncated at threshold).
    """
    weights = np.zeros(max_size, dtype=np.float64)
    weights[0] = 1.0

    k = 1
    while k < max_size:
        w = -weights[k - 1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights[k] = w
        k += 1

    return weights[:k]


def compute_frac_diff_weights(
    d: float,
    threshold: float = 1e-5,
    max_lags: int = 1000,
) -> np.ndarray:
    """Compute fractional differentiation weights.

    Args:
        d: Fractional differentiation order (0 < d < 1).
        threshold: Weight cutoff threshold for FFD.
        max_lags: Maximum number of lags.

    Returns:
        Array of weights.

    Example:
        >>> weights = compute_frac_diff_weights(0.5)
        >>> print(f"Number of weights: {len(weights)}")
    """
    weights = _compute_weights_ffd(d, threshold, max_lags)

    logger.debug(
        "Frac diff weights (d=%.2f): %d lags, sum=%.4f",
        d,
        len(weights),
        np.sum(weights),
    )

    return weights


# =============================================================================
# FRACTIONAL DIFFERENTIATION
# =============================================================================


@njit(cache=True)
def _apply_frac_diff(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply fractional differentiation (numba optimized).

    Args:
        values: Input series.
        weights: Fractional diff weights.

    Returns:
        Fractionally differentiated series.
    """
    n = len(values)
    n_weights = len(weights)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(n_weights - 1, n):
        if np.isnan(values[i]):
            continue

        # Apply convolution with weights
        val = 0.0
        valid = True
        for j in range(n_weights):
            if np.isnan(values[i - j]):
                valid = False
                break
            val += weights[j] * values[i - j]

        if valid:
            result[i] = val

    return result


@njit(cache=True)
def _apply_frac_diff_expanding(
    values: NDArray[np.float64],
    d: float,
    min_periods: int,
) -> NDArray[np.float64]:
    """Apply expanding fractional differentiation (numba optimized).

    Uses all available history at each point.

    Args:
        values: Input series.
        d: Fractional differentiation order.
        min_periods: Minimum observations required.

    Returns:
        Fractionally differentiated series.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(min_periods - 1, n):
        # Compute weights up to current position
        n_weights = i + 1
        weights = np.zeros(n_weights, dtype=np.float64)
        weights[0] = 1.0

        for k in range(1, n_weights):
            weights[k] = -weights[k - 1] * (d - k + 1) / k

        # Apply convolution
        val = 0.0
        valid = True
        for j in range(n_weights):
            if np.isnan(values[i - j]):
                valid = False
                break
            val += weights[j] * values[i - j]

        if valid:
            result[i] = val

    return result


def compute_frac_diff(
    series: pd.Series | np.ndarray,
    d: float,
    threshold: float = 1e-5,
    max_lags: int = 1000,
) -> np.ndarray:
    """Compute fractional differentiation of a series.

    Standard fractional differentiation with fixed weight window.

    Args:
        series: Input series (e.g., log prices).
        d: Fractional differentiation order (0 < d < 1).
        threshold: Weight cutoff threshold.
        max_lags: Maximum number of lags.

    Returns:
        Fractionally differentiated series.

    Example:
        >>> log_prices = np.log(df_bars["close"])
        >>> frac_diff = compute_frac_diff(log_prices, d=0.5)
    """
    if isinstance(series, pd.Series):
        values = series.values.astype(np.float64)
    else:
        values = series.astype(np.float64)

    weights = _compute_weights_ffd(d, threshold, max_lags)
    result = _apply_frac_diff(values, weights)

    return result


def compute_frac_diff_ffd(
    series: pd.Series | np.ndarray,
    d: float,
    threshold: float = 1e-5,
) -> np.ndarray:
    """Compute Fixed-Width Window Fractional Differentiation (FFD).

    FFD is preferred over standard frac diff because:
    - Fixed window width = constant memory usage
    - No look-ahead bias
    - Suitable for real-time applications

    Args:
        series: Input series (e.g., log prices).
        d: Fractional differentiation order (0 < d < 1).
        threshold: Weight cutoff threshold.

    Returns:
        FFD series.

    Example:
        >>> log_prices = np.log(df_bars["close"])
        >>> ffd = compute_frac_diff_ffd(log_prices, d=0.5)
    """
    return compute_frac_diff(series, d, threshold)


# =============================================================================
# MINIMUM D ESTIMATION
# =============================================================================


def _adf_test(series: np.ndarray, max_lags: int = 10) -> float:
    """Simple ADF test statistic computation.

    Returns the t-statistic for the unit root test.
    More negative = more stationary.
    """
    # Remove NaN
    valid = series[~np.isnan(series)]
    if len(valid) < max_lags + 10:
        return 0.0

    # Compute first difference
    diff = np.diff(valid)

    # Lag the level
    y_lag = valid[:-1]

    # Simple OLS: diff = alpha + beta * y_lag + error
    # We want to test if beta < 0 (stationary)
    n = len(diff)

    # Include lagged differences for augmented test
    X = np.column_stack([
        np.ones(n - max_lags),
        y_lag[max_lags:],
    ])

    # Add lagged differences
    for lag in range(1, max_lags + 1):
        X = np.column_stack([X, diff[max_lags - lag:-lag if lag > 0 else None]])

    y = diff[max_lags:]

    # OLS estimation
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        sigma2 = np.sum(residuals ** 2) / (len(y) - X.shape[1])

        # Standard error of beta[1] (coefficient on y_lag)
        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])

        t_stat = beta[1] / se_beta
        return t_stat
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def find_min_frac_diff_order(
    series: pd.Series | np.ndarray,
    d_range: tuple[float, float] = (0.0, 1.0),
    threshold: float = 1e-5,
    adf_critical: float = -2.86,  # 5% critical value
    n_steps: int = 20,
) -> float:
    """Find minimum d that makes series stationary.

    Searches for the smallest d such that the ADF test rejects
    the unit root hypothesis.

    Args:
        series: Input series (e.g., log prices).
        d_range: Range of d values to search.
        threshold: Weight cutoff threshold.
        adf_critical: ADF critical value (default: -2.86 for 5%).
        n_steps: Number of d values to test.

    Returns:
        Minimum d for stationarity.

    Example:
        >>> log_prices = np.log(df_bars["close"])
        >>> min_d = find_min_frac_diff_order(log_prices)
        >>> print(f"Minimum d for stationarity: {min_d:.2f}")
    """
    if isinstance(series, pd.Series):
        values = series.values.astype(np.float64)
    else:
        values = series.astype(np.float64)

    d_values = np.linspace(d_range[0], d_range[1], n_steps)

    for d in d_values:
        if d == 0:
            frac_diff = values
        else:
            frac_diff = compute_frac_diff(values, d, threshold)

        adf_stat = _adf_test(frac_diff)

        if adf_stat < adf_critical:
            logger.info(
                "Minimum d for stationarity: %.3f (ADF=%.2f < %.2f)",
                d,
                adf_stat,
                adf_critical,
            )
            return d

    logger.warning(
        "Could not find d for stationarity in range [%.2f, %.2f]",
        d_range[0],
        d_range[1],
    )
    return d_range[1]


# =============================================================================
# FEATURE GENERATION
# =============================================================================


def compute_frac_diff_features(
    df_bars: pd.DataFrame,
    price_col: str = "close",
    d_values: list[float] | None = None,
    threshold: float = 1e-5,
    auto_find_d: bool = False,
) -> pd.DataFrame:
    """Compute fractionally differentiated features.

    Creates 1-2 fractionally differentiated series of log-close
    to inject long memory for ML models.

    Args:
        df_bars: DataFrame with price data.
        price_col: Name of price column.
        d_values: List of d values (default: [0.3, 0.5]).
        threshold: Weight cutoff threshold for FFD.
        auto_find_d: If True, automatically find minimum d.

    Returns:
        DataFrame with fractionally differentiated features.

    Example:
        >>> df_frac = compute_frac_diff_features(df_bars)
        >>> df_bars = pd.concat([df_bars, df_frac], axis=1)
    """
    result = pd.DataFrame(index=df_bars.index)

    # Compute log prices
    prices = df_bars[price_col].values.astype(np.float64)
    log_prices = np.log(prices)

    # Default d values
    if d_values is None:
        if auto_find_d:
            # Find minimum d for stationarity
            min_d = find_min_frac_diff_order(log_prices, threshold=threshold)
            # Use min_d and a slightly higher value
            d_values = [min_d, min(min_d + 0.2, 0.9)]
            logger.info("Auto-selected d values: %s", d_values)
        else:
            # Default values that typically work well
            d_values = [0.3, 0.5]

    # Compute FFD for each d value
    for d in d_values:
        frac_diff = compute_frac_diff_ffd(log_prices, d, threshold)

        # Column name
        d_str = f"{d:.1f}".replace(".", "")
        col_name = f"frac_diff_d{d_str}"
        result[col_name] = frac_diff

        # Log statistics
        valid = frac_diff[~np.isnan(frac_diff)]
        if len(valid) > 0:
            # Compute ADF statistic
            adf_stat = _adf_test(frac_diff)

            logger.info(
                "Frac diff (d=%.2f): mean=%.6f, std=%.6f, ADF=%.2f, "
                "NaN=%d/%d",
                d,
                np.mean(valid),
                np.std(valid),
                adf_stat,
                len(frac_diff) - len(valid),
                len(frac_diff),
            )

    # Also add log price for reference (d=0)
    result["log_price"] = log_prices

    return result
