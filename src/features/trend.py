"""Trend and mean reversion features.

This module computes trend-based and mean reversion features:

1. Moving Averages (MA):
   MA_t^{(k)} = (1/k) * Σ_{j=1}^{k} P_{t-j}

2. Price z-score (deviation from MA):
   z_t^{price,(k)} = (P_t - MA_t^{(k)}) / std(P_{t-k:t})

3. Cross MA (trend regime):
   sign(MA^{(k1)}_t - MA^{(k2)}_t)
   e.g., 20 vs 100 bars

4. Return streak:
   Number of consecutive bars with same return sign

Interpretation:
    - z > 0: Price above MA (bullish)
    - z < 0: Price below MA (bearish)
    - Large |z|: Potential mean reversion opportunity
    - Cross MA > 0: Short-term trend above long-term (bullish)
    - Long streak: Strong momentum, potential exhaustion

Reference:
    Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple Technical
    Trading Rules and the Stochastic Properties of Stock Returns.
    Journal of Finance, 47(5), 1731-1764.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]  # Still used by _compute_streak

from src.config_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_moving_averages",
    "compute_price_zscore",
    "compute_cross_ma",
    "compute_return_streak",
]


# =============================================================================
# MOVING AVERAGES
# =============================================================================


def _rolling_mean(
    values: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling mean using pandas (numerically stable).

    Args:
        values: Input array.
        window: Rolling window size.

    Returns:
        Array of rolling means.
    """
    series = pd.Series(values)
    result = np.asarray(series.rolling(window=window, min_periods=window).mean(), dtype=np.float64)
    return result


def _rolling_std(
    values: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling standard deviation using pandas (numerically stable).

    Args:
        values: Input array.
        window: Rolling window size.

    Returns:
        Array of rolling standard deviations.
    """
    series = pd.Series(values)
    result = np.asarray(series.rolling(window=window, min_periods=window).std(), dtype=np.float64)
    return result


def compute_moving_averages(
    df_bars: pd.DataFrame,
    price_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute simple moving averages over multiple windows.

    MA_t^{(k)} = (1/k) * Σ_{j=1}^{k} P_{t-j}

    Args:
        df_bars: DataFrame with price data.
        price_col: Name of price column.
        windows: List of window sizes (default: [5, 10, 20, 50, 100]).

    Returns:
        DataFrame with MA columns for each window.

    Example:
        >>> df_ma = compute_moving_averages(df_bars)
        >>> df_bars = pd.concat([df_bars, df_ma], axis=1)
    """
    if windows is None:
        windows = [5, 10, 20, 50, 100]

    prices = df_bars[price_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in windows:
        ma = _rolling_mean(prices, k)
        col_name = f"ma_{k}"
        result[col_name] = ma

        # Log statistics
        valid = ma[~np.isnan(ma)]
        if len(valid) > 0:
            logger.info(
                "MA (k=%d) stats: mean=%.2f, std=%.2f",
                k,
                np.mean(valid),
                np.std(valid),
            )

    return result


# =============================================================================
# PRICE Z-SCORE
# =============================================================================


def _compute_zscore(
    prices: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute price z-score relative to MA using pandas (numerically stable).

    z_t = (P_t - MA_t) / std(P_{t-k:t})

    Args:
        prices: Array of prices.
        window: Rolling window size.

    Returns:
        Array of z-scores.
    """
    series = pd.Series(prices)
    rolling = series.rolling(window=window, min_periods=window)

    mean = rolling.mean()
    std = rolling.std()

    # Z-score: 0 when std is ~0 (price constant = no deviation)
    zscore = np.where(
        std > 1e-10,
        (series - mean) / std,
        0.0,  # No variance = price at mean = z-score of 0
    )

    return zscore


def compute_price_zscore(
    df_bars: pd.DataFrame,
    price_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute price z-score (deviation from rolling mean).

    z_t^{price,(k)} = (P_t - MA_t^{(k)}) / std(P_{t-k:t})

    Args:
        df_bars: DataFrame with price data.
        price_col: Name of price column.
        windows: List of window sizes (default: [10, 20, 50]).

    Returns:
        DataFrame with z-score columns for each window.

    Example:
        >>> df_zscore = compute_price_zscore(df_bars)
        >>> df_bars = pd.concat([df_bars, df_zscore], axis=1)
    """
    if windows is None:
        windows = [10, 20, 50]

    prices = df_bars[price_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in windows:
        zscore = _compute_zscore(prices, k)
        col_name = f"price_zscore_{k}"
        result[col_name] = zscore

        # Log statistics
        valid = zscore[~np.isnan(zscore)]
        if len(valid) > 0:
            logger.info(
                "Price z-score (k=%d) stats: mean=%.4f, std=%.4f, "
                "min=%.4f, max=%.4f",
                k,
                np.mean(valid),
                np.std(valid),
                np.min(valid),
                np.max(valid),
            )

    return result


# =============================================================================
# CROSS MA (TREND REGIME)
# =============================================================================


def compute_cross_ma(
    df_bars: pd.DataFrame,
    price_col: str = "close",
    pairs: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Compute cross moving average indicators (trend regime).

    CrossMA = sign(MA^{(k1)} - MA^{(k2)})

    Where k1 < k2 (short-term vs long-term).

    Args:
        df_bars: DataFrame with price data.
        price_col: Name of price column.
        pairs: List of (short, long) window pairs (default: [(5,20), (10,50), (20,100)]).

    Returns:
        DataFrame with cross MA columns (values: -1, 0, +1).

    Example:
        >>> df_cross = compute_cross_ma(df_bars)
        >>> df_bars = pd.concat([df_bars, df_cross], axis=1)
    """
    if pairs is None:
        pairs = [(5, 20), (10, 50), (20, 100)]

    prices = df_bars[price_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k_short, k_long in pairs:
        ma_short = _rolling_mean(prices, k_short)
        ma_long = _rolling_mean(prices, k_long)

        # Sign of difference
        diff = ma_short - ma_long
        cross_ma = np.sign(diff)

        col_name = f"cross_ma_{k_short}_{k_long}"
        result[col_name] = cross_ma

        # Log statistics
        valid = cross_ma[~np.isnan(cross_ma)]
        if len(valid) > 0:
            pct_bullish = 100 * np.mean(valid > 0)
            pct_bearish = 100 * np.mean(valid < 0)
            logger.info(
                "Cross MA (%d/%d): bullish=%.1f%%, bearish=%.1f%%",
                k_short,
                k_long,
                pct_bullish,
                pct_bearish,
            )

    return result


# =============================================================================
# RETURN STREAK
# =============================================================================


@njit(cache=True)
def _compute_streak(
    returns: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute return streak (consecutive same-sign returns) (numba optimized).

    Positive streak: consecutive positive returns
    Negative streak: consecutive negative returns

    Args:
        returns: Array of returns.

    Returns:
        Array of streak values (positive for up streak, negative for down streak).
    """
    n = len(returns)
    streak = np.zeros(n, dtype=np.float64)

    if n == 0:
        return streak

    # Initialize
    if np.isnan(returns[0]):
        streak[0] = 0.0
    elif returns[0] > 0:
        streak[0] = 1.0
    elif returns[0] < 0:
        streak[0] = -1.0
    else:
        streak[0] = 0.0

    for i in range(1, n):
        if np.isnan(returns[i]):
            streak[i] = streak[i - 1]
            continue

        prev_streak = streak[i - 1]

        if returns[i] > 0:
            # Positive return
            if prev_streak > 0:
                streak[i] = prev_streak + 1.0
            else:
                streak[i] = 1.0
        elif returns[i] < 0:
            # Negative return
            if prev_streak < 0:
                streak[i] = prev_streak - 1.0
            else:
                streak[i] = -1.0
        else:
            # Zero return - reset streak
            streak[i] = 0.0

    return streak


def compute_return_streak(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
) -> pd.Series:
    """Compute return streak (consecutive same-sign returns).

    Streak measures the number of consecutive bars with positive or
    negative returns:
        - Positive values: consecutive up bars (e.g., +3 = 3 up bars)
        - Negative values: consecutive down bars (e.g., -4 = 4 down bars)

    Interpretation:
        - Long positive streak: Strong upward momentum
        - Long negative streak: Strong downward momentum
        - Streak reversal: Potential trend change

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.

    Returns:
        Series with streak values.

    Example:
        >>> df_bars["streak"] = compute_return_streak(df_bars)
    """
    returns = df_bars[return_col].values.astype(np.float64)
    streak = _compute_streak(returns)

    # Log statistics
    valid = streak[streak != 0]
    if len(valid) > 0:
        max_up = np.max(streak)
        max_down = np.min(streak)
        avg_abs = np.mean(np.abs(valid))
        logger.info(
            "Return streak stats: max_up=%d, max_down=%d, avg_abs=%.2f",
            int(max_up),
            int(max_down),
            avg_abs,
        )

    return pd.Series(streak, index=df_bars.index, name="return_streak")
