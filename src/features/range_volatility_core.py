"""Numba-optimized core functions for range-based volatility estimators.

This module contains the low-level Numba functions for computing
range-based volatility estimators. These are used by range_volatility.py.

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
    "_parkinson_single",
    "_rolling_parkinson",
    "_garman_klass_single",
    "_rolling_garman_klass",
    "_rogers_satchell_single",
    "_rolling_rogers_satchell",
    "_rolling_yang_zhang",
    "_compute_range_ratios",
]


# =============================================================================
# PARKINSON VOLATILITY
# =============================================================================


@njit(cache=True)
def _parkinson_single(high: float, low: float) -> float:
    """Compute single-bar Parkinson variance.

    σ²_Park = (1 / 4·ln(2)) · (ln(H/L))²

    Args:
        high: High price.
        low: Low price.

    Returns:
        Parkinson variance estimate.
    """
    if np.isnan(high) or np.isnan(low) or low <= 0 or high <= 0:
        return np.nan

    log_hl = np.log(high / low)
    return (log_hl ** 2) / (4.0 * np.log(2.0))


@njit(cache=True)
def _rolling_parkinson(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling Parkinson volatility (numba optimized).

    Args:
        high: Array of high prices.
        low: Array of low prices.
        window: Rolling window size.

    Returns:
        Array of Parkinson volatility (standard deviation).
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        var_sum = 0.0
        count = 0

        for j in range(window):
            idx = i - j
            var = _parkinson_single(high[idx], low[idx])
            if not np.isnan(var):
                var_sum += var
                count += 1

        if count > 0:
            result[i] = np.sqrt(var_sum / count)

    return result


# =============================================================================
# GARMAN-KLASS VOLATILITY
# =============================================================================


@njit(cache=True)
def _garman_klass_single(
    high: float,
    low: float,
    open_: float,
    close: float,
) -> float:
    """Compute single-bar Garman-Klass variance.

    σ²_GK = 0.5·(ln(H/L))² - (2·ln(2) - 1)·(ln(C/O))²

    Args:
        high: High price.
        low: Low price.
        open_: Open price.
        close: Close price.

    Returns:
        Garman-Klass variance estimate.
    """
    if (np.isnan(high) or np.isnan(low) or np.isnan(open_) or np.isnan(close)
            or low <= 0 or high <= 0 or open_ <= 0 or close <= 0):
        return np.nan

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    return 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)


@njit(cache=True)
def _rolling_garman_klass(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    open_: NDArray[np.float64],
    close: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling Garman-Klass volatility (numba optimized).

    Args:
        high: Array of high prices.
        low: Array of low prices.
        open_: Array of open prices.
        close: Array of close prices.
        window: Rolling window size.

    Returns:
        Array of Garman-Klass volatility.
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        var_sum = 0.0
        count = 0

        for j in range(window):
            idx = i - j
            var = _garman_klass_single(high[idx], low[idx], open_[idx], close[idx])
            if not np.isnan(var):
                var_sum += var
                count += 1

        if count > 0:
            avg_var = var_sum / count
            if avg_var > 0:
                result[i] = np.sqrt(avg_var)
            else:
                result[i] = 0.0

    return result


# =============================================================================
# ROGERS-SATCHELL VOLATILITY
# =============================================================================


@njit(cache=True)
def _rogers_satchell_single(
    high: float,
    low: float,
    open_: float,
    close: float,
) -> float:
    """Compute single-bar Rogers-Satchell variance.

    σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)

    Args:
        high: High price.
        low: Low price.
        open_: Open price.
        close: Close price.

    Returns:
        Rogers-Satchell variance estimate.
    """
    if (np.isnan(high) or np.isnan(low) or np.isnan(open_) or np.isnan(close)
            or low <= 0 or high <= 0 or open_ <= 0 or close <= 0):
        return np.nan

    log_hc = np.log(high / close)
    log_ho = np.log(high / open_)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_)

    return log_hc * log_ho + log_lc * log_lo


@njit(cache=True)
def _rolling_rogers_satchell(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    open_: NDArray[np.float64],
    close: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling Rogers-Satchell volatility (numba optimized).

    Args:
        high: Array of high prices.
        low: Array of low prices.
        open_: Array of open prices.
        close: Array of close prices.
        window: Rolling window size.

    Returns:
        Array of Rogers-Satchell volatility.
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        var_sum = 0.0
        count = 0

        for j in range(window):
            idx = i - j
            var = _rogers_satchell_single(high[idx], low[idx], open_[idx], close[idx])
            if not np.isnan(var):
                var_sum += var
                count += 1

        if count > 0:
            avg_var = var_sum / count
            if avg_var > 0:
                result[i] = np.sqrt(avg_var)
            else:
                result[i] = 0.0

    return result


# =============================================================================
# YANG-ZHANG VOLATILITY
# =============================================================================


@njit(cache=True)
def _rolling_yang_zhang(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    open_: NDArray[np.float64],
    close: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling Yang-Zhang volatility (numba optimized).

    σ²_YZ = σ²_overnight + k·σ²_open + (1-k)·σ²_RS

    Where:
        σ²_overnight = Var(ln(O_t / C_{t-1}))
        σ²_open = Var(ln(C_t / O_t))
        k = 0.34 / (1.34 + (n+1)/(n-1))

    Args:
        high: Array of high prices.
        low: Array of low prices.
        open_: Array of open prices.
        close: Array of close prices.
        window: Rolling window size.

    Returns:
        Array of Yang-Zhang volatility.
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window + 1:
        return result

    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))

    for i in range(window, n):
        overnight_returns = np.zeros(window, dtype=np.float64)
        open_close_returns = np.zeros(window, dtype=np.float64)
        rs_vars = np.zeros(window, dtype=np.float64)
        count = 0

        for j in range(window):
            idx = i - j
            prev_close = close[idx - 1]

            if (np.isnan(high[idx]) or np.isnan(low[idx]) or
                    np.isnan(open_[idx]) or np.isnan(close[idx]) or
                    np.isnan(prev_close) or prev_close <= 0 or
                    open_[idx] <= 0 or close[idx] <= 0 or
                    high[idx] <= 0 or low[idx] <= 0):
                continue

            overnight_returns[count] = np.log(open_[idx] / prev_close)
            open_close_returns[count] = np.log(close[idx] / open_[idx])
            rs_vars[count] = _rogers_satchell_single(
                high[idx], low[idx], open_[idx], close[idx]
            )
            count += 1

        if count < 2:
            continue

        overnight = overnight_returns[:count]
        open_close = open_close_returns[:count]
        rs = rs_vars[:count]

        overnight_mean = 0.0
        for val in overnight:
            overnight_mean += val
        overnight_mean /= count

        overnight_var = 0.0
        for val in overnight:
            overnight_var += (val - overnight_mean) ** 2
        overnight_var /= (count - 1)

        open_close_mean = 0.0
        for val in open_close:
            open_close_mean += val
        open_close_mean /= count

        open_close_var = 0.0
        for val in open_close:
            open_close_var += (val - open_close_mean) ** 2
        open_close_var /= (count - 1)

        rs_var = 0.0
        rs_count = 0
        for val in rs:
            if not np.isnan(val):
                rs_var += val
                rs_count += 1
        if rs_count > 0:
            rs_var /= rs_count

        yz_var = overnight_var + k * open_close_var + (1.0 - k) * rs_var

        if yz_var > 0:
            result[i] = np.sqrt(yz_var)
        else:
            result[i] = 0.0

    return result


# =============================================================================
# RANGE RATIOS
# =============================================================================


@njit(cache=True)
def _compute_range_ratios(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    open_: NDArray[np.float64],
    close: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute range-based ratios (numba optimized).

    Args:
        high: Array of high prices.
        low: Array of low prices.
        open_: Array of open prices.
        close: Array of close prices.

    Returns:
        Tuple of (range_ratio, body_ratio).
    """
    n = len(high)
    range_ratio = np.full(n, np.nan, dtype=np.float64)
    body_ratio = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        h, l, o, c = high[i], low[i], open_[i], close[i]

        if np.isnan(h) or np.isnan(l) or np.isnan(o) or np.isnan(c):
            continue

        if c <= 0:
            continue

        hl_range = h - l

        range_ratio[i] = hl_range / c

        if hl_range > 1e-10:
            body_ratio[i] = abs(c - o) / hl_range
        else:
            body_ratio[i] = 0.0

    return range_ratio, body_ratio
