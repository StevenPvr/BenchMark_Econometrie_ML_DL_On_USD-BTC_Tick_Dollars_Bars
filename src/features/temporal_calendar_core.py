"""Numba-optimized core functions for temporal and calendar features.

This module contains the low-level Numba functions for computing
temporal features. These are used by temporal_calendar.py.

NOTE: Numba functions use literal values equivalent to EPS (1e-10)
because Numba cannot import Python constants at compile time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "_time_since_shock",
    "_time_since_positive_shock",
    "_time_since_negative_shock",
    "_expanding_mean",
    "_expanding_std",
    "_compute_drawdown",
    "_compute_drawup",
]


# =============================================================================
# TIME SINCE SHOCK (Numba optimized)
# =============================================================================


@njit(cache=True)
def _time_since_shock(
    returns: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    """Compute bars since last shock (numba optimized).

    Args:
        returns: Array of returns.
        threshold: Shock threshold (absolute return).

    Returns:
        Array of bars since last shock.
    """
    n = len(returns)
    result = np.zeros(n, dtype=np.float64)

    bars_since = 0.0

    for i in range(n):
        if np.isnan(returns[i]):
            result[i] = bars_since
            bars_since += 1.0
        elif np.abs(returns[i]) > threshold:
            result[i] = 0.0
            bars_since = 1.0
        else:
            result[i] = bars_since
            bars_since += 1.0

    return result


@njit(cache=True)
def _time_since_positive_shock(
    returns: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    """Compute bars since last positive shock (numba optimized).

    Args:
        returns: Array of returns.
        threshold: Shock threshold.

    Returns:
        Array of bars since last positive shock.
    """
    n = len(returns)
    result = np.zeros(n, dtype=np.float64)

    bars_since = 0.0

    for i in range(n):
        if np.isnan(returns[i]):
            result[i] = bars_since
            bars_since += 1.0
        elif returns[i] > threshold:
            result[i] = 0.0
            bars_since = 1.0
        else:
            result[i] = bars_since
            bars_since += 1.0

    return result


@njit(cache=True)
def _time_since_negative_shock(
    returns: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    """Compute bars since last negative shock (numba optimized).

    Args:
        returns: Array of returns.
        threshold: Shock threshold (positive value, will check < -threshold).

    Returns:
        Array of bars since last negative shock.
    """
    n = len(returns)
    result = np.zeros(n, dtype=np.float64)

    bars_since = 0.0

    for i in range(n):
        if np.isnan(returns[i]):
            result[i] = bars_since
            bars_since += 1.0
        elif returns[i] < -threshold:
            result[i] = 0.0
            bars_since = 1.0
        else:
            result[i] = bars_since
            bars_since += 1.0

    return result


# =============================================================================
# EXPANDING STATISTICS (Numba optimized)
# =============================================================================


@njit(cache=True)
def _expanding_mean(
    values: NDArray[np.float64],
    min_periods: int,
) -> NDArray[np.float64]:
    """Compute expanding mean (numba optimized)."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    running_sum = 0.0
    count = 0

    for i in range(n):
        if not np.isnan(values[i]):
            running_sum += values[i]
            count += 1

        if count >= min_periods:
            result[i] = running_sum / count

    return result


@njit(cache=True)
def _expanding_std(
    values: NDArray[np.float64],
    min_periods: int,
) -> NDArray[np.float64]:
    """Compute expanding standard deviation using Welford's algorithm (O(n)).

    Uses Welford's online algorithm for numerically stable single-pass
    computation of variance/standard deviation.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    count = 0
    mean = 0.0
    m2 = 0.0  # Sum of squared differences from mean

    for i in range(n):
        if np.isnan(values[i]):
            if count >= min_periods:
                result[i] = np.sqrt(m2 / (count - 1)) if count > 1 else 0.0
            continue

        count += 1
        delta = values[i] - mean
        mean += delta / count
        delta2 = values[i] - mean
        m2 += delta * delta2

        if count >= min_periods and count > 1:
            result[i] = np.sqrt(m2 / (count - 1))

    return result


# =============================================================================
# DRAWDOWN / DRAWUP (Numba optimized)
# =============================================================================


@njit(cache=True)
def _compute_drawdown(
    prices: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute drawdown from rolling maximum (numba optimized).

    Args:
        prices: Array of prices.

    Returns:
        Tuple of (drawdown, rolling_max, bars_since_high).
    """
    n = len(prices)
    drawdown = np.full(n, np.nan, dtype=np.float64)
    rolling_max = np.full(n, np.nan, dtype=np.float64)
    bars_since_high = np.zeros(n, dtype=np.float64)

    if n == 0:
        return drawdown, rolling_max, bars_since_high

    current_max = prices[0]
    bars_since = 0.0

    for i in range(n):
        if np.isnan(prices[i]):
            drawdown[i] = np.nan
            rolling_max[i] = current_max
            bars_since_high[i] = bars_since
            bars_since += 1.0
            continue

        if prices[i] > current_max:
            current_max = prices[i]
            bars_since = 0.0
        else:
            bars_since += 1.0

        rolling_max[i] = current_max
        bars_since_high[i] = bars_since

        if current_max > 0:
            drawdown[i] = (prices[i] - current_max) / current_max

    return drawdown, rolling_max, bars_since_high


@njit(cache=True)
def _compute_drawup(
    prices: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute drawup from rolling minimum (numba optimized).

    Args:
        prices: Array of prices.

    Returns:
        Tuple of (drawup, rolling_min, bars_since_low).
    """
    n = len(prices)
    drawup = np.full(n, np.nan, dtype=np.float64)
    rolling_min = np.full(n, np.nan, dtype=np.float64)
    bars_since_low = np.zeros(n, dtype=np.float64)

    if n == 0:
        return drawup, rolling_min, bars_since_low

    current_min = prices[0]
    bars_since = 0.0

    for i in range(n):
        if np.isnan(prices[i]):
            drawup[i] = np.nan
            rolling_min[i] = current_min
            bars_since_low[i] = bars_since
            bars_since += 1.0
            continue

        if prices[i] < current_min:
            current_min = prices[i]
            bars_since = 0.0
        else:
            bars_since += 1.0

        rolling_min[i] = current_min
        bars_since_low[i] = bars_since

        if current_min > 0:
            drawup[i] = (prices[i] - current_min) / current_min

    return drawup, rolling_min, bars_since_low
