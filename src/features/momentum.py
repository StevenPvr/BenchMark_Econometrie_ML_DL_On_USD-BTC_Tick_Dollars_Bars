"""Multi-horizon returns and momentum features.

This module computes momentum-based features from past returns:

1. Cumulative returns over multiple horizons:
   R_t^{(k)} = Σ_{j=0}^{k-1} r_{t-j}
   for k ∈ {1, 3, 5, 10, 20} bars.

2. Recent extreme returns (shocks):
   max_{j≤k} r_{t-j} and min_{j≤k} r_{t-j}

Interpretation:
    - Positive cumulative returns: Upward momentum
    - Negative cumulative returns: Downward momentum
    - Large max/min: Recent shock events

Reference:
    De Bondt, W. F., & Thaler, R. (1985). Does the Stock Market Overreact?
    Journal of Finance, 40(3), 793-805.
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
    "compute_cumulative_returns",
    "compute_recent_extremes",
]


@njit(cache=True)
def _rolling_sum(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling sum (numba optimized).

    Args:
        returns: Array of returns.
        window: Rolling window size.

    Returns:
        Array of rolling sums.
    """
    n = len(returns)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    # First valid window
    window_sum = 0.0
    valid_count = 0
    for j in range(window):
        if not np.isnan(returns[j]):
            window_sum += returns[j]
            valid_count += 1

    if valid_count >= window // 2:
        result[window - 1] = window_sum

    # Rolling forward
    for i in range(window, n):
        old_val = returns[i - window]
        new_val = returns[i]

        if not np.isnan(old_val):
            window_sum -= old_val
        if not np.isnan(new_val):
            window_sum += new_val

        result[i] = window_sum

    return result


@njit(cache=True)
def _rolling_max(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling maximum (numba optimized).

    Args:
        returns: Array of returns.
        window: Rolling window size.

    Returns:
        Array of rolling maxima.
    """
    n = len(returns)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        max_val = -np.inf
        for j in range(window):
            val = returns[i - j]
            if not np.isnan(val) and val > max_val:
                max_val = val
        if max_val > -np.inf:
            result[i] = max_val

    return result


@njit(cache=True)
def _rolling_min(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling minimum (numba optimized).

    Args:
        returns: Array of returns.
        window: Rolling window size.

    Returns:
        Array of rolling minima.
    """
    n = len(returns)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        min_val = np.inf
        for j in range(window):
            val = returns[i - j]
            if not np.isnan(val) and val < min_val:
                min_val = val
        if min_val < np.inf:
            result[i] = min_val

    return result


def compute_cumulative_returns(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute cumulative returns over multiple horizons.

    R_t^{(k)} = Σ_{j=0}^{k-1} r_{t-j}

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [1, 3, 5, 10, 20]).

    Returns:
        DataFrame with cumulative return columns for each horizon.

    Example:
        >>> df_momentum = compute_cumulative_returns(df_bars)
        >>> df_bars = pd.concat([df_bars, df_momentum], axis=1)
    """
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]

    returns = df_bars[return_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in horizons:
        cum_ret = _rolling_sum(returns, k)
        col_name = f"cum_return_{k}"
        result[col_name] = cum_ret

        # Log statistics
        valid = cum_ret[~np.isnan(cum_ret)]
        if len(valid) > 0:
            logger.info(
                "Cumulative return (k=%d) stats: mean=%.4f, std=%.4f, "
                "min=%.4f, max=%.4f",
                k,
                np.mean(valid),
                np.std(valid),
                np.min(valid),
                np.max(valid),
            )

    return result


def compute_recent_extremes(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute recent maximum and minimum returns (shock detection).

    max_{j≤k} r_{t-j} and min_{j≤k} r_{t-j}

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [5, 10, 20]).

    Returns:
        DataFrame with max/min return columns for each horizon.

    Example:
        >>> df_extremes = compute_recent_extremes(df_bars)
        >>> df_bars = pd.concat([df_bars, df_extremes], axis=1)
    """
    if horizons is None:
        horizons = [5, 10, 20]

    returns = df_bars[return_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in horizons:
        # Maximum return
        max_ret = _rolling_max(returns, k)
        result[f"max_return_{k}"] = max_ret

        # Minimum return
        min_ret = _rolling_min(returns, k)
        result[f"min_return_{k}"] = min_ret

        # Log statistics
        valid_max = max_ret[~np.isnan(max_ret)]
        valid_min = min_ret[~np.isnan(min_ret)]
        if len(valid_max) > 0:
            logger.info(
                "Recent extremes (k=%d): max_mean=%.4f, min_mean=%.4f",
                k,
                np.mean(valid_max),
                np.mean(valid_min),
            )

    return result
