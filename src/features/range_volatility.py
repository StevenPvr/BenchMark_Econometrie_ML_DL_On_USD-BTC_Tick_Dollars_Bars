"""Range-based volatility estimators.

This module computes volatility estimators using intrabar price information
(OHLC data), which are more efficient than close-to-close estimators.

1. Parkinson (1980):
   σ²_Park = (1 / 4·ln(2)) · (ln(H/L))²
   - Uses high-low range only
   - 5x more efficient than close-to-close

2. Garman-Klass (1980):
   σ²_GK = 0.5·(ln(H/L))² - (2·ln(2) - 1)·(ln(C/O))²
   - Uses OHLC data
   - More efficient than Parkinson

3. Rogers-Satchell (1991):
   σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)
   - Handles drift (trending markets)
   - Unbiased for non-zero mean returns

4. Yang-Zhang (2000):
   σ²_YZ = σ²_overnight + k·σ²_open + (1-k)·σ²_RS
   - Combines overnight, open-to-close, and RS
   - Handles both drift and opening jumps

5. Simple ratios:
   - Range ratio: (H - L) / C
   - Body ratio: |C - O| / (H - L)

Reference:
    Parkinson, M. (1980). The Extreme Value Method for Estimating
    the Variance of the Rate of Return. Journal of Business.

    Yang, D., & Zhang, Q. (2000). Drift Independent Volatility Estimation
    Based on High, Low, Open, and Close Prices.
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
    "compute_parkinson_volatility",
    "compute_garman_klass_volatility",
    "compute_rogers_satchell_volatility",
    "compute_yang_zhang_volatility",
    "compute_range_ratios",
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
            # Average variance, then sqrt for volatility
            result[i] = np.sqrt(var_sum / count)

    return result


def compute_parkinson_volatility(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Parkinson volatility estimator.

    σ²_Park = (1 / 4·ln(2)) · (ln(H/L))²

    The Parkinson estimator uses the high-low range and is approximately
    5x more efficient than the close-to-close estimator.

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        windows: List of window sizes (default: [5, 10, 20]).

    Returns:
        DataFrame with Parkinson volatility columns.

    Example:
        >>> df_vol = compute_parkinson_volatility(df_bars)
        >>> df_bars = pd.concat([df_bars, df_vol], axis=1)
    """
    if windows is None:
        windows = [5, 10, 20]

    high = df_bars[high_col].values.astype(np.float64)
    low = df_bars[low_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in windows:
        vol = _rolling_parkinson(high, low, k)
        col_name = f"parkinson_vol_{k}"
        result[col_name] = vol

        valid = vol[~np.isnan(vol)]
        if len(valid) > 0:
            logger.info(
                "Parkinson volatility (k=%d) stats: mean=%.6f, std=%.6f",
                k,
                np.mean(valid),
                np.std(valid),
            )

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
            # Handle potential negative variance (can happen with GK)
            if avg_var > 0:
                result[i] = np.sqrt(avg_var)

    return result


def compute_garman_klass_volatility(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Garman-Klass volatility estimator.

    σ²_GK = 0.5·(ln(H/L))² - (2·ln(2) - 1)·(ln(C/O))²

    More efficient than Parkinson as it uses all OHLC information.

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        open_col: Name of open price column.
        close_col: Name of close price column.
        windows: List of window sizes (default: [5, 10, 20]).

    Returns:
        DataFrame with Garman-Klass volatility columns.

    Example:
        >>> df_vol = compute_garman_klass_volatility(df_bars)
        >>> df_bars = pd.concat([df_bars, df_vol], axis=1)
    """
    if windows is None:
        windows = [5, 10, 20]

    high = df_bars[high_col].values.astype(np.float64)
    low = df_bars[low_col].values.astype(np.float64)
    open_ = df_bars[open_col].values.astype(np.float64)
    close = df_bars[close_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in windows:
        vol = _rolling_garman_klass(high, low, open_, close, k)
        col_name = f"garman_klass_vol_{k}"
        result[col_name] = vol

        valid = vol[~np.isnan(vol)]
        if len(valid) > 0:
            logger.info(
                "Garman-Klass volatility (k=%d) stats: mean=%.6f, std=%.6f",
                k,
                np.mean(valid),
                np.std(valid),
            )

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

    return result


def compute_rogers_satchell_volatility(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Rogers-Satchell volatility estimator.

    σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)

    This estimator is unbiased for non-zero mean returns (handles drift).

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        open_col: Name of open price column.
        close_col: Name of close price column.
        windows: List of window sizes (default: [5, 10, 20]).

    Returns:
        DataFrame with Rogers-Satchell volatility columns.

    Example:
        >>> df_vol = compute_rogers_satchell_volatility(df_bars)
        >>> df_bars = pd.concat([df_bars, df_vol], axis=1)
    """
    if windows is None:
        windows = [5, 10, 20]

    high = df_bars[high_col].values.astype(np.float64)
    low = df_bars[low_col].values.astype(np.float64)
    open_ = df_bars[open_col].values.astype(np.float64)
    close = df_bars[close_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in windows:
        vol = _rolling_rogers_satchell(high, low, open_, close, k)
        col_name = f"rogers_satchell_vol_{k}"
        result[col_name] = vol

        valid = vol[~np.isnan(vol)]
        if len(valid) > 0:
            logger.info(
                "Rogers-Satchell volatility (k=%d) stats: mean=%.6f, std=%.6f",
                k,
                np.mean(valid),
                np.std(valid),
            )

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

    # k coefficient
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))

    for i in range(window, n):
        # Collect window data (need previous close for overnight)
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

            # Overnight return: ln(O_t / C_{t-1})
            overnight_returns[count] = np.log(open_[idx] / prev_close)

            # Open-to-close return: ln(C_t / O_t)
            open_close_returns[count] = np.log(close[idx] / open_[idx])

            # Rogers-Satchell variance
            rs_vars[count] = _rogers_satchell_single(
                high[idx], low[idx], open_[idx], close[idx]
            )
            count += 1

        if count < 2:
            continue

        # Compute variances
        overnight = overnight_returns[:count]
        open_close = open_close_returns[:count]
        rs = rs_vars[:count]

        # Overnight variance
        overnight_mean = 0.0
        for val in overnight:
            overnight_mean += val
        overnight_mean /= count

        overnight_var = 0.0
        for val in overnight:
            overnight_var += (val - overnight_mean) ** 2
        overnight_var /= (count - 1)

        # Open-close variance
        open_close_mean = 0.0
        for val in open_close:
            open_close_mean += val
        open_close_mean /= count

        open_close_var = 0.0
        for val in open_close:
            open_close_var += (val - open_close_mean) ** 2
        open_close_var /= (count - 1)

        # Average RS variance
        rs_var = 0.0
        rs_count = 0
        for val in rs:
            if not np.isnan(val):
                rs_var += val
                rs_count += 1
        if rs_count > 0:
            rs_var /= rs_count

        # Yang-Zhang variance
        yz_var = overnight_var + k * open_close_var + (1.0 - k) * rs_var

        if yz_var > 0:
            result[i] = np.sqrt(yz_var)

    return result


def compute_yang_zhang_volatility(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Yang-Zhang volatility estimator.

    σ²_YZ = σ²_overnight + k·σ²_open + (1-k)·σ²_RS

    This is the most efficient estimator that handles both drift and
    opening price jumps. Requires previous close price.

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        open_col: Name of open price column.
        close_col: Name of close price column.
        windows: List of window sizes (default: [5, 10, 20]).

    Returns:
        DataFrame with Yang-Zhang volatility columns.

    Example:
        >>> df_vol = compute_yang_zhang_volatility(df_bars)
        >>> df_bars = pd.concat([df_bars, df_vol], axis=1)
    """
    if windows is None:
        windows = [5, 10, 20]

    high = df_bars[high_col].values.astype(np.float64)
    low = df_bars[low_col].values.astype(np.float64)
    open_ = df_bars[open_col].values.astype(np.float64)
    close = df_bars[close_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in windows:
        vol = _rolling_yang_zhang(high, low, open_, close, k)
        col_name = f"yang_zhang_vol_{k}"
        result[col_name] = vol

        valid = vol[~np.isnan(vol)]
        if len(valid) > 0:
            logger.info(
                "Yang-Zhang volatility (k=%d) stats: mean=%.6f, std=%.6f",
                k,
                np.mean(valid),
                np.std(valid),
            )

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

        # Range ratio: (H - L) / C
        range_ratio[i] = hl_range / c

        # Body ratio: |C - O| / (H - L)
        if hl_range > 1e-10:
            body_ratio[i] = abs(c - o) / hl_range

    return range_ratio, body_ratio


def compute_range_ratios(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute simple range-based ratios.

    1. Range ratio: (H - L) / C
       - Normalized range (relative to price level)
       - Higher values indicate higher intrabar volatility

    2. Body ratio: |C - O| / (H - L)
       - Proportion of range captured by open-close move
       - High value: Directional bar (trending)
       - Low value: Indecisive bar (doji-like)

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        open_col: Name of open price column.
        close_col: Name of close price column.

    Returns:
        DataFrame with range_ratio and body_ratio columns.

    Example:
        >>> df_ratios = compute_range_ratios(df_bars)
        >>> df_bars = pd.concat([df_bars, df_ratios], axis=1)
    """
    high = df_bars[high_col].values.astype(np.float64)
    low = df_bars[low_col].values.astype(np.float64)
    open_ = df_bars[open_col].values.astype(np.float64)
    close = df_bars[close_col].values.astype(np.float64)

    range_ratio, body_ratio = _compute_range_ratios(high, low, open_, close)

    result = pd.DataFrame(index=df_bars.index)
    result["range_ratio"] = range_ratio
    result["body_ratio"] = body_ratio

    # Log statistics
    valid_range = range_ratio[~np.isnan(range_ratio)]
    valid_body = body_ratio[~np.isnan(body_ratio)]

    if len(valid_range) > 0:
        logger.info(
            "Range ratio stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
            np.mean(valid_range),
            np.std(valid_range),
            np.min(valid_range),
            np.max(valid_range),
        )

    if len(valid_body) > 0:
        logger.info(
            "Body ratio stats: mean=%.4f, std=%.4f (1=directional, 0=doji)",
            np.mean(valid_body),
            np.std(valid_body),
        )

    return result
