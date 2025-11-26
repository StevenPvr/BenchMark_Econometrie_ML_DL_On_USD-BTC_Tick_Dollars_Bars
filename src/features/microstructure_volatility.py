"""Microstructure volatility features from tick data within bars.

This module computes volatility measures using tick-level information
within each bar, providing finer granularity than OHLC-based estimators.

1. Intrabar tick return variance:
   σ²_intrabar = Var(r_tick) within each bar
   Where r_tick = ln(P_tick / P_tick-1)

2. Intrabar tick range:
   Range_tick = (max(P_tick) - min(P_tick)) / close
   More granular than OHLC high-low

3. Tick count volatility proxy:
   Higher tick count often correlates with higher volatility

4. Realized variance from ticks:
   RV = Σ r²_tick (sum of squared tick returns)

Interpretation:
    - High intrabar variance: Volatile price discovery within bar
    - High tick range vs OHLC range: Hidden volatility not captured by OHLC
    - High tick count: Active trading, potentially higher information flow

Reference:
    Ait-Sahalia, Y., & Jacod, J. (2014). High-Frequency Financial Econometrics.
    Princeton University Press.
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
    "compute_intrabar_volatility",
    "compute_microstructure_features",
]


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================


@njit(cache=True)
def _compute_tick_stats(
    prices: NDArray[np.float64],
    start_idx: int,
    end_idx: int,
) -> tuple[float, float, float, float, int]:
    """Compute tick statistics for a single bar (numba optimized).

    Args:
        prices: Array of all tick prices.
        start_idx: Start index for this bar.
        end_idx: End index for this bar (exclusive).

    Returns:
        Tuple of (variance, range_ratio, realized_var, mean_return, tick_count).
    """
    n_ticks = end_idx - start_idx

    if n_ticks < 2:
        return np.nan, np.nan, np.nan, np.nan, n_ticks

    # Compute tick returns
    returns = np.zeros(n_ticks - 1, dtype=np.float64)
    valid_returns = 0

    for i in range(n_ticks - 1):
        p0 = prices[start_idx + i]
        p1 = prices[start_idx + i + 1]

        if p0 > 0 and p1 > 0 and not np.isnan(p0) and not np.isnan(p1):
            returns[valid_returns] = np.log(p1 / p0)
            valid_returns += 1

    if valid_returns < 2:
        return np.nan, np.nan, np.nan, np.nan, n_ticks

    # Compute mean
    mean_ret = 0.0
    for i in range(valid_returns):
        mean_ret += returns[i]
    mean_ret /= valid_returns

    # Compute variance
    var_sum = 0.0
    for i in range(valid_returns):
        var_sum += (returns[i] - mean_ret) ** 2
    variance = var_sum / (valid_returns - 1)

    # Compute realized variance (sum of squared returns)
    realized_var = 0.0
    for i in range(valid_returns):
        realized_var += returns[i] ** 2

    # Compute tick range
    min_price = prices[start_idx]
    max_price = prices[start_idx]
    close_price = prices[end_idx - 1]

    for i in range(start_idx, end_idx):
        p = prices[i]
        if not np.isnan(p):
            if p < min_price:
                min_price = p
            if p > max_price:
                max_price = p

    if close_price > 0:
        range_ratio = (max_price - min_price) / close_price
    else:
        range_ratio = np.nan

    return variance, range_ratio, realized_var, mean_ret, n_ticks


@njit(cache=True)
def _aggregate_tick_stats_by_bar(
    tick_prices: NDArray[np.float64],
    tick_timestamps: NDArray[np.int64],
    bar_open_ts: NDArray[np.int64],
    bar_close_ts: NDArray[np.int64],
    n_bars: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Aggregate tick statistics by bar (numba optimized).

    Args:
        tick_prices: Array of tick prices.
        tick_timestamps: Array of tick timestamps (int64).
        bar_open_ts: Array of bar open timestamps.
        bar_close_ts: Array of bar close timestamps.
        n_bars: Number of bars.

    Returns:
        Tuple of arrays: (variance, range_ratio, realized_var, mean_return, tick_count).
    """
    intrabar_var = np.full(n_bars, np.nan, dtype=np.float64)
    tick_range = np.full(n_bars, np.nan, dtype=np.float64)
    realized_var = np.full(n_bars, np.nan, dtype=np.float64)
    mean_return = np.full(n_bars, np.nan, dtype=np.float64)
    tick_count = np.zeros(n_bars, dtype=np.int64)

    n_ticks = len(tick_prices)
    tick_idx = 0

    for bar_id in range(n_bars):
        bar_start = bar_open_ts[bar_id]
        bar_end = bar_close_ts[bar_id]

        # Find ticks within this bar
        start_idx = -1
        end_idx = -1

        # Advance to first tick in bar
        while tick_idx < n_ticks and tick_timestamps[tick_idx] < bar_start:
            tick_idx += 1

        if tick_idx >= n_ticks:
            break

        # Mark start
        if tick_timestamps[tick_idx] <= bar_end:
            start_idx = tick_idx

        # Find end of bar
        temp_idx = tick_idx
        while temp_idx < n_ticks and tick_timestamps[temp_idx] <= bar_end:
            temp_idx += 1
        end_idx = temp_idx

        if start_idx >= 0 and end_idx > start_idx:
            var, rng, rv, mean_ret, count = _compute_tick_stats(
                tick_prices, start_idx, end_idx
            )
            intrabar_var[bar_id] = var
            tick_range[bar_id] = rng
            realized_var[bar_id] = rv
            mean_return[bar_id] = mean_ret
            tick_count[bar_id] = count

    return intrabar_var, tick_range, realized_var, mean_return, tick_count


# =============================================================================
# PUBLIC API
# =============================================================================


def compute_intrabar_volatility(
    df_ticks: pd.DataFrame,
    df_bars: pd.DataFrame,
    price_col: str = "price",
    timestamp_col: str = "timestamp",
    bar_timestamp_open: str = "timestamp_open",
    bar_timestamp_close: str = "timestamp_close",
) -> pd.DataFrame:
    """Compute intrabar volatility measures from tick data.

    This function computes volatility statistics using tick-level data
    within each bar:

    1. intrabar_variance: Variance of tick log-returns within bar
    2. tick_range: (max - min) / close of tick prices (more granular than OHLC)
    3. realized_variance: Sum of squared tick returns (RV)
    4. tick_mean_return: Mean tick return within bar
    5. tick_count: Number of ticks in bar

    Args:
        df_ticks: DataFrame with tick-level data.
        df_bars: DataFrame with dollar bars.
        price_col: Name of price column in ticks.
        timestamp_col: Name of timestamp column in ticks.
        bar_timestamp_open: Name of bar open timestamp column.
        bar_timestamp_close: Name of bar close timestamp column.

    Returns:
        DataFrame with intrabar volatility columns.

    Example:
        >>> df_vol = compute_intrabar_volatility(df_ticks, df_bars)
        >>> df_bars = pd.concat([df_bars, df_vol], axis=1)
    """
    logger.info(
        "Computing intrabar volatility from %d ticks for %d bars",
        len(df_ticks),
        len(df_bars),
    )

    # Get tick data
    tick_prices = df_ticks[price_col].values.astype(np.float64)

    # Convert timestamps to int64 (nanoseconds or milliseconds)
    if pd.api.types.is_datetime64_any_dtype(df_ticks[timestamp_col]):
        tick_ts = df_ticks[timestamp_col].astype("int64").values
    else:
        tick_ts = df_ticks[timestamp_col].values.astype(np.int64)

    # Get bar boundaries
    if pd.api.types.is_datetime64_any_dtype(df_bars[bar_timestamp_open]):
        bar_open_ts = df_bars[bar_timestamp_open].astype("int64").values
        bar_close_ts = df_bars[bar_timestamp_close].astype("int64").values
    else:
        bar_open_ts = df_bars[bar_timestamp_open].values.astype(np.int64)
        bar_close_ts = df_bars[bar_timestamp_close].values.astype(np.int64)

    # Compute statistics
    intrabar_var, tick_range, realized_var, mean_return, tick_count = (
        _aggregate_tick_stats_by_bar(
            tick_prices, tick_ts, bar_open_ts, bar_close_ts, len(df_bars)
        )
    )

    # Create result DataFrame
    result = pd.DataFrame(index=df_bars.index)
    result["intrabar_variance"] = intrabar_var
    result["tick_range"] = tick_range
    result["realized_variance"] = realized_var
    result["tick_mean_return"] = mean_return
    result["tick_count"] = tick_count

    # Compute derived features
    # Intrabar volatility (std)
    result["intrabar_volatility"] = np.sqrt(intrabar_var)

    # Realized volatility from ticks
    result["tick_realized_vol"] = np.sqrt(realized_var)

    # Log statistics
    valid_var = intrabar_var[~np.isnan(intrabar_var)]
    valid_range = tick_range[~np.isnan(tick_range)]
    valid_count = tick_count[tick_count > 0]

    if len(valid_var) > 0:
        logger.info(
            "Intrabar variance stats: mean=%.2e, std=%.2e",
            np.mean(valid_var),
            np.std(valid_var),
        )

    if len(valid_range) > 0:
        logger.info(
            "Tick range stats: mean=%.6f, std=%.6f",
            np.mean(valid_range),
            np.std(valid_range),
        )

    if len(valid_count) > 0:
        logger.info(
            "Tick count stats: mean=%.1f, min=%d, max=%d",
            np.mean(valid_count),
            np.min(valid_count),
            np.max(valid_count),
        )

    return result


def compute_microstructure_features(
    df_ticks: pd.DataFrame,
    df_bars: pd.DataFrame,
    price_col: str = "price",
    timestamp_col: str = "timestamp",
    bar_timestamp_open: str = "timestamp_open",
    bar_timestamp_close: str = "timestamp_close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """Compute comprehensive microstructure volatility features.

    Computes intrabar volatility plus additional derived features:

    1. All features from compute_intrabar_volatility()
    2. range_efficiency: tick_range / ohlc_range
       - >1 means ticks show more volatility than OHLC captures
    3. volatility_of_volatility: Rolling std of intrabar_volatility
    4. tick_intensity: tick_count / bar_duration (normalized activity)

    Args:
        df_ticks: DataFrame with tick-level data.
        df_bars: DataFrame with dollar bars (must have OHLC columns).
        price_col: Name of price column in ticks.
        timestamp_col: Name of timestamp column in ticks.
        bar_timestamp_open: Name of bar open timestamp column.
        bar_timestamp_close: Name of bar close timestamp column.
        high_col: Name of high price column in bars.
        low_col: Name of low price column in bars.

    Returns:
        DataFrame with all microstructure volatility features.

    Example:
        >>> df_micro = compute_microstructure_features(df_ticks, df_bars)
        >>> df_bars = pd.concat([df_bars, df_micro], axis=1)
    """
    # Get base intrabar features
    result = compute_intrabar_volatility(
        df_ticks,
        df_bars,
        price_col=price_col,
        timestamp_col=timestamp_col,
        bar_timestamp_open=bar_timestamp_open,
        bar_timestamp_close=bar_timestamp_close,
    )

    # Compute range efficiency (tick range vs OHLC range)
    if high_col in df_bars.columns and low_col in df_bars.columns:
        ohlc_range = (df_bars[high_col] - df_bars[low_col]) / df_bars[high_col]
        ohlc_range = ohlc_range.values

        # Range efficiency: how much more granular info ticks provide
        tick_range = result["tick_range"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            range_efficiency = np.where(
                ohlc_range > 1e-10,
                tick_range / ohlc_range,
                np.nan,
            )
        result["range_efficiency"] = range_efficiency

        valid_eff = range_efficiency[~np.isnan(range_efficiency)]
        if len(valid_eff) > 0:
            logger.info(
                "Range efficiency stats: mean=%.4f (1=same, >1=ticks more volatile)",
                np.mean(valid_eff),
            )

    # Compute volatility of volatility (rolling)
    intrabar_vol = result["intrabar_volatility"].values
    vol_of_vol = _rolling_std_numba(intrabar_vol, window=20)
    result["vol_of_vol_20"] = vol_of_vol

    # Compute tick intensity (ticks per second if duration available)
    if "duration_sec" in df_bars.columns:
        duration = df_bars["duration_sec"].values.astype(np.float64)
        tick_count = result["tick_count"].values.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            tick_intensity = np.where(
                duration > 0,
                tick_count / duration,
                np.nan,
            )
        result["tick_intensity"] = tick_intensity

        valid_intensity = tick_intensity[~np.isnan(tick_intensity)]
        if len(valid_intensity) > 0:
            logger.info(
                "Tick intensity stats: mean=%.2f ticks/sec",
                np.mean(valid_intensity),
            )

    return result


@njit(cache=True)
def _rolling_std_numba(
    values: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling standard deviation (numba optimized).

    Args:
        values: Input array.
        window: Rolling window size.

    Returns:
        Array of rolling standard deviations.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        # Compute mean
        mean = 0.0
        count = 0
        for j in range(window):
            val = values[i - j]
            if not np.isnan(val):
                mean += val
                count += 1

        if count < 2:
            continue

        mean /= count

        # Compute std
        var_sum = 0.0
        for j in range(window):
            val = values[i - j]
            if not np.isnan(val):
                var_sum += (val - mean) ** 2

        result[i] = np.sqrt(var_sum / (count - 1))

    return result
