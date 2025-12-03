"""Realized volatility and return-volatility ratio features.

This module computes volatility-based features from past returns:

1. Realized volatility over rolling window:
   σ_t^{(k)} = √(Σ_{j=1}^{k} r_{t-j}²)

2. Return-volatility ratio (NOT a true Sharpe ratio - no risk-free rate or annualization):
   RVR_t^{(k)} = R_t^{(k)} / σ_t^{(k)}

3. Realized skewness (third moment):
   Skew_t^{(k)} = E[(r - μ)³] / σ³

4. Realized kurtosis (fourth moment, excess):
   Kurt_t^{(k)} = E[(r - μ)⁴] / σ⁴ - 3

Interpretation:
    - High realized vol: Turbulent market regime
    - Low realized vol: Calm market regime
    - High RVR: Strong momentum with low risk
    - Low RVR: Weak momentum or high risk
    - High skewness: Asymmetric returns (more positive or negative outliers)
    - High kurtosis: Fat tails, more extreme events

Reference:
    Andersen, T. G., & Bollerslev, T. (1998). Answering the Skeptics:
    Yes, Standard Volatility Models do Provide Accurate Forecasts.
    International Economic Review, 39(4), 885-905.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.constants import EPS
from src.utils import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_realized_volatility",
    "compute_return_volatility_ratio",
    "compute_realized_skewness",
    "compute_realized_kurtosis",
    # Deprecated alias for backward compatibility
    "compute_local_sharpe",
]

# NOTE: Numba functions use literal values equivalent to EPS (1e-10)
# because Numba cannot import Python constants at compile time.


def _rolling_sum_squares(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling sum of squared returns using pandas (numerically stable).

    Uses pandas rolling which is implemented in C with proper numerical stability.

    Args:
        returns: Array of returns.
        window: Rolling window size.

    Returns:
        Array of rolling sum of squares.
    """
    # Use pandas rolling sum - numerically stable implementation
    series = pd.Series(returns ** 2)
    result = series.rolling(window=window, min_periods=window).sum()
    return np.asarray(result, dtype=np.float64)


def _rolling_sum(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling sum using pandas (numerically stable).

    Args:
        returns: Array of returns.
        window: Rolling window size.

    Returns:
        Array of rolling sums.
    """
    series = pd.Series(returns)
    result = series.rolling(window=window, min_periods=window).sum()
    return np.asarray(result, dtype=np.float64)


def compute_realized_volatility(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute realized volatility over multiple horizons.

    σ_t^{(k)} = √(Σ_{j=1}^{k} r_{t-j}²)

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [5, 10, 20, 50]).

    Returns:
        DataFrame with realized volatility columns for each horizon.

    Example:
        >>> df_vol = compute_realized_volatility(df_bars)
        >>> df_bars = pd.concat([df_bars, df_vol], axis=1)
    """
    if horizons is None:
        horizons = [5, 10, 20, 50]

    returns = df_bars[return_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in horizons:
        sum_sq = _rolling_sum_squares(returns, k)
        realized_vol = np.sqrt(sum_sq)

        col_name = f"realized_vol_{k}"
        result[col_name] = realized_vol

        # Log statistics
        valid = realized_vol[~np.isnan(realized_vol)]
        if len(valid) > 0:
            logger.info(
                "Realized volatility (k=%d) stats: mean=%.4f, std=%.4f, "
                "min=%.4f, max=%.4f",
                k,
                np.mean(valid),
                np.std(valid),
                np.min(valid),
                np.max(valid),
            )

    return result


def compute_return_volatility_ratio(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute return-volatility ratio (return / volatility).

    NOTE: This is NOT a true Sharpe ratio as it lacks:
    - Risk-free rate subtraction
    - Annualization factor

    RVR_t^{(k)} = R_t^{(k)} / σ_t^{(k)}

    Where:
        R_t^{(k)} = Σ_{j=0}^{k-1} r_{t-j} (cumulative return)
        σ_t^{(k)} = √(Σ_{j=1}^{k} r_{t-j}²) (realized volatility)

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [5, 10, 20]).

    Returns:
        DataFrame with return-volatility ratio columns for each horizon.

    Example:
        >>> df_rvr = compute_return_volatility_ratio(df_bars)
        >>> df_bars = pd.concat([df_bars, df_rvr], axis=1)
    """
    if horizons is None:
        horizons = [5, 10, 20]

    returns = df_bars[return_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in horizons:
        # Cumulative return
        cum_ret = _rolling_sum(returns, k)

        # Realized volatility
        sum_sq = _rolling_sum_squares(returns, k)
        realized_vol = np.sqrt(sum_sq)

        # Return-volatility ratio (0 when vol is ~0)
        rvr = np.where(
            realized_vol > 1e-10,
            cum_ret / realized_vol,
            0.0,  # No vol = no risk-adjusted return
        )

        col_name = f"return_vol_ratio_{k}"
        result[col_name] = rvr

        # Log statistics
        valid = rvr[~np.isnan(rvr)]
        if len(valid) > 0:
            logger.info(
                "Return-volatility ratio (k=%d) stats: mean=%.4f, std=%.4f, "
                "min=%.4f, max=%.4f",
                k,
                np.mean(valid),
                np.std(valid),
                np.min(valid),
                np.max(valid),
            )

    return result


# Deprecated alias for backward compatibility
def compute_local_sharpe(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Deprecated: Use compute_return_volatility_ratio instead.

    This function is kept for backward compatibility but will be removed
    in a future version. Note that this is NOT a true Sharpe ratio.
    """
    import warnings
    warnings.warn(
        "compute_local_sharpe is deprecated. Use compute_return_volatility_ratio instead. "
        "Note: This metric is NOT a true Sharpe ratio (missing rf and annualization).",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_return_volatility_ratio(df_bars, return_col, horizons)


def compute_realized_skewness(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute realized skewness (third standardized moment) over rolling windows.

    Skew_t^{(k)} = E[(r - μ)³] / σ³

    Where:
        μ = rolling mean of returns
        σ = rolling standard deviation

    Interpretation:
        - Positive skew: More positive outliers (right tail heavier)
        - Negative skew: More negative outliers (left tail heavier)
        - Near zero: Symmetric distribution

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [10, 20, 50]).

    Returns:
        DataFrame with realized skewness columns for each horizon.

    Example:
        >>> df_skew = compute_realized_skewness(df_bars)
        >>> df_bars = pd.concat([df_bars, df_skew], axis=1)
    """
    if horizons is None:
        horizons = [10, 20, 50]

    returns = df_bars[return_col]
    result = pd.DataFrame(index=df_bars.index)

    for k in horizons:
        # Rolling skewness using pandas (handles edge cases properly)
        skew = returns.rolling(window=k, min_periods=k).skew()

        col_name = f"realized_skew_{k}"
        result[col_name] = skew

        # Log statistics
        valid = skew.dropna()
        if len(valid) > 0:
            logger.info(
                "Realized skewness (k=%d) stats: mean=%.4f, std=%.4f, "
                "min=%.4f, max=%.4f",
                k,
                valid.mean(),
                valid.std(),
                valid.min(),
                valid.max(),
            )

    return result


def compute_realized_kurtosis(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute realized excess kurtosis (fourth standardized moment - 3) over rolling windows.

    Kurt_t^{(k)} = E[(r - μ)⁴] / σ⁴ - 3

    Where:
        μ = rolling mean of returns
        σ = rolling standard deviation

    Note: Returns EXCESS kurtosis (normal distribution = 0, not 3).

    Interpretation:
        - Positive kurtosis: Fat tails, more extreme events (leptokurtic)
        - Negative kurtosis: Thin tails, fewer extreme events (platykurtic)
        - Near zero: Normal-like tails (mesokurtic)

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [10, 20, 50]).

    Returns:
        DataFrame with realized kurtosis columns for each horizon.

    Example:
        >>> df_kurt = compute_realized_kurtosis(df_bars)
        >>> df_bars = pd.concat([df_bars, df_kurt], axis=1)
    """
    if horizons is None:
        horizons = [10, 20, 50]

    returns = df_bars[return_col]
    result = pd.DataFrame(index=df_bars.index)

    for k in horizons:
        # Rolling kurtosis using pandas (returns excess kurtosis by default)
        kurt = returns.rolling(window=k, min_periods=k).kurt()

        col_name = f"realized_kurt_{k}"
        result[col_name] = kurt

        # Log statistics
        valid = kurt.dropna()
        if len(valid) > 0:
            logger.info(
                "Realized kurtosis (k=%d) stats: mean=%.4f, std=%.4f, "
                "min=%.4f, max=%.4f",
                k,
                valid.mean(),
                valid.std(),
                valid.min(),
                valid.max(),
            )

    return result
