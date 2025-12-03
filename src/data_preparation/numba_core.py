"""Numba-optimized core functions for dollar bars computation.

These functions implement the core algorithms for dollar bar accumulation
with Numba JIT compilation for performance on large tick datasets.

Note: Constants like 1e-10 are hardcoded because @njit cannot access
Python module constants at compile time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "_compute_threshold_from_target_bars",
    "_compute_robust_percentile_threshold",
    "_accumulate_dollar_bars_adaptive",
    "_accumulate_dollar_bars_fixed",
]


@njit(cache=True)
def _compute_threshold_from_target_bars(
    dollar_values: NDArray[np.float64],
    target_num_bars: int,
) -> float:
    """Compute threshold using De Prado's Expected Dollar Value method.

    This is the CORRECT De Prado calibration (AFML Chapter 2, Section 2.3.2.1):
        E_0[T] = Total Dollar Volume / Target Number of Bars

    This ensures we get approximately the desired number of bars regardless
    of the distribution of tick dollar values.

    Args:
        dollar_values: Array of dollar values (price * volume) for each tick.
        target_num_bars: Target total number of bars to generate.

    Returns:
        Calibrated threshold T_0 = E[dv] for bar formation.
    """
    n = len(dollar_values)
    if n == 0 or target_num_bars <= 0:
        return 1.0

    total_dollar = 0.0
    for i in range(n):
        total_dollar += dollar_values[i]

    threshold = total_dollar / target_num_bars
    # Note: 1e-10 hardcoded because @njit cannot access Python module constants
    return max(threshold, 1e-10)


@njit(cache=True)
def _compute_robust_percentile_threshold(
    dollar_values: NDArray[np.float64],
    target_num_bars: int,
) -> tuple[float, float, float]:
    """Compute threshold with bounds for adaptive mode.

    Computes primary threshold as total dollar volume / target bars,
    then returns bounds at 50% and 200% of that threshold.

    Args:
        dollar_values: Array of dollar values (price * volume) for each tick.
        target_num_bars: Target total number of bars to generate.

    Returns:
        Tuple of (threshold, min_threshold, max_threshold) for bounded adaptation.
    """
    n = len(dollar_values)
    if n == 0 or target_num_bars <= 0:
        return 1.0, 1.0, 1.0

    total_dollar = 0.0
    for i in range(n):
        total_dollar += dollar_values[i]

    threshold = total_dollar / target_num_bars

    # Bounds: allow threshold to vary within reasonable range
    min_threshold = threshold * 0.5
    max_threshold = threshold * 2.0

    # Note: 1e-10 hardcoded because @njit cannot access Python module constants
    return max(threshold, 1e-10), max(min_threshold, 1e-10), max_threshold


@njit(cache=True)
def _accumulate_dollar_bars_adaptive(
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    initial_threshold: float,
    ema_alpha: float,
    min_threshold: float,
    max_threshold: float,
    include_incomplete_final: bool,
) -> tuple[
    NDArray[np.int64],  # bar_ids
    NDArray[np.int64],  # timestamp_open
    NDArray[np.int64],  # timestamp_close
    NDArray[np.float64],  # open
    NDArray[np.float64],  # high
    NDArray[np.float64],  # low
    NDArray[np.float64],  # close
    NDArray[np.float64],  # volume
    NDArray[np.float64],  # dollar_value
    NDArray[np.float64],  # vwap
    NDArray[np.int64],  # tick_count
    NDArray[np.float64],  # threshold_used
    int,  # num_bars
]:
    """Core numba-optimized loop for adaptive threshold (De Prado EWMA).

    Implements De Prado's adaptive dollar bar algorithm with optional bounds:
    1. Start with initial threshold T_0 (calibrated on prefix)
    2. For each tick, accumulate dollar value
    3. When cumulative >= T_k, close bar k with dollar_value D_k
    4. Update EWMA: E_k = alpha * D_k + (1 - alpha) * E_{k-1}
    5. Optional bounds: T_{k+1} = clip(E_k, min_threshold, max_threshold)

    Args:
        timestamps: Array of tick timestamps (int64 for ms precision).
        prices: Array of tick prices.
        volumes: Array of tick volumes.
        initial_threshold: Initial threshold T_0 from calibration.
        ema_alpha: Alpha for EWMA update (2 / (span + 1)).
        min_threshold: Minimum allowed threshold (prevents too many bars).
        max_threshold: Maximum allowed threshold (prevents too few bars).
        include_incomplete_final: Include incomplete final bar if True.

    Returns:
        Tuple of arrays containing bar data and the number of bars formed.
    """
    n = len(timestamps)
    if n == 0:
        empty_int = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=np.float64)
        return (
            empty_int, empty_int, empty_int,
            empty_float, empty_float, empty_float, empty_float,
            empty_float, empty_float, empty_float,
            empty_int, empty_float, 0
        )

    # Pre-allocate output arrays
    bar_ids = np.empty(n, dtype=np.int64)
    ts_open = np.empty(n, dtype=np.int64)
    ts_close = np.empty(n, dtype=np.int64)
    opens = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    lows = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    bar_volumes = np.empty(n, dtype=np.float64)
    bar_dollars = np.empty(n, dtype=np.float64)
    bar_vwaps = np.empty(n, dtype=np.float64)
    tick_counts = np.empty(n, dtype=np.int64)
    thresholds = np.empty(n, dtype=np.float64)

    # Initialize state
    bar_idx = 0
    current_threshold = initial_threshold
    ema_dollar = initial_threshold

    cum_dollar = 0.0
    cum_volume = 0.0
    cum_pv = 0.0
    bar_open = prices[0]
    bar_high = prices[0]
    bar_low = prices[0]
    bar_ts_open = timestamps[0]
    n_ticks = 0

    for i in range(n):
        price = prices[i]
        volume = volumes[i]
        ts = timestamps[i]

        dollar_value = price * volume
        cum_dollar += dollar_value
        cum_volume += volume
        cum_pv += price * volume
        n_ticks += 1

        if price > bar_high:
            bar_high = price
        if price < bar_low:
            bar_low = price

        if cum_dollar >= current_threshold:
            bar_ids[bar_idx] = bar_idx
            ts_open[bar_idx] = bar_ts_open
            ts_close[bar_idx] = ts
            opens[bar_idx] = bar_open
            highs[bar_idx] = bar_high
            lows[bar_idx] = bar_low
            closes[bar_idx] = price
            bar_volumes[bar_idx] = cum_volume
            bar_dollars[bar_idx] = cum_dollar
            bar_vwaps[bar_idx] = cum_pv / cum_volume if cum_volume > 0 else price
            tick_counts[bar_idx] = n_ticks
            thresholds[bar_idx] = current_threshold

            # Adaptive threshold update
            ema_dollar = ema_alpha * cum_dollar + (1.0 - ema_alpha) * ema_dollar
            current_threshold = ema_dollar
            if current_threshold < min_threshold:
                current_threshold = min_threshold
            if current_threshold > max_threshold:
                current_threshold = max_threshold

            bar_idx += 1

            # Reset for next bar
            cum_dollar = 0.0
            cum_volume = 0.0
            cum_pv = 0.0
            n_ticks = 0

            if i + 1 < n:
                bar_open = prices[i + 1]
                bar_high = prices[i + 1]
                bar_low = prices[i + 1]
                bar_ts_open = timestamps[i + 1]

    # Handle incomplete final bar
    if include_incomplete_final and n_ticks > 0:
        bar_ids[bar_idx] = bar_idx
        ts_open[bar_idx] = bar_ts_open
        ts_close[bar_idx] = timestamps[n - 1]
        opens[bar_idx] = bar_open
        highs[bar_idx] = bar_high
        lows[bar_idx] = bar_low
        closes[bar_idx] = prices[n - 1]
        bar_volumes[bar_idx] = cum_volume
        bar_dollars[bar_idx] = cum_dollar
        bar_vwaps[bar_idx] = cum_pv / cum_volume if cum_volume > 0 else prices[n - 1]
        tick_counts[bar_idx] = n_ticks
        thresholds[bar_idx] = current_threshold
        bar_idx += 1

    return (
        bar_ids, ts_open, ts_close, opens, highs, lows, closes,
        bar_volumes, bar_dollars, bar_vwaps, tick_counts, thresholds, bar_idx,
    )


@njit(cache=True)
def _accumulate_dollar_bars_fixed(
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    threshold: float,
    include_incomplete_final: bool,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    int,
]:
    """Core loop with FIXED threshold (non-adaptive).

    Use this when you want a constant threshold T that doesn't change.
    Simpler but doesn't adapt to market regime changes.

    Args:
        timestamps: Array of tick timestamps (int64 for ms precision).
        prices: Array of tick prices.
        volumes: Array of tick volumes.
        threshold: Fixed dollar threshold.
        include_incomplete_final: Include incomplete final bar if True.

    Returns:
        Tuple of arrays containing bar data and the number of bars formed.
    """
    n = len(timestamps)
    if n == 0:
        empty_int = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=np.float64)
        return (
            empty_int, empty_int, empty_int,
            empty_float, empty_float, empty_float, empty_float,
            empty_float, empty_float, empty_float,
            empty_int, empty_float, 0
        )

    bar_ids = np.empty(n, dtype=np.int64)
    ts_open = np.empty(n, dtype=np.int64)
    ts_close = np.empty(n, dtype=np.int64)
    opens = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    lows = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    bar_volumes = np.empty(n, dtype=np.float64)
    bar_dollars = np.empty(n, dtype=np.float64)
    bar_vwaps = np.empty(n, dtype=np.float64)
    tick_counts = np.empty(n, dtype=np.int64)
    thresholds_arr = np.empty(n, dtype=np.float64)

    bar_idx = 0
    cum_dollar = 0.0
    cum_volume = 0.0
    cum_pv = 0.0
    bar_open = prices[0]
    bar_high = prices[0]
    bar_low = prices[0]
    bar_ts_open = timestamps[0]
    n_ticks = 0

    for i in range(n):
        price = prices[i]
        volume = volumes[i]
        ts = timestamps[i]

        dollar_value = price * volume
        cum_dollar += dollar_value
        cum_volume += volume
        cum_pv += price * volume
        n_ticks += 1

        if price > bar_high:
            bar_high = price
        if price < bar_low:
            bar_low = price

        if cum_dollar >= threshold:
            bar_ids[bar_idx] = bar_idx
            ts_open[bar_idx] = bar_ts_open
            ts_close[bar_idx] = ts
            opens[bar_idx] = bar_open
            highs[bar_idx] = bar_high
            lows[bar_idx] = bar_low
            closes[bar_idx] = price
            bar_volumes[bar_idx] = cum_volume
            bar_dollars[bar_idx] = cum_dollar
            bar_vwaps[bar_idx] = cum_pv / cum_volume if cum_volume > 0 else price
            tick_counts[bar_idx] = n_ticks
            thresholds_arr[bar_idx] = threshold

            bar_idx += 1
            cum_dollar = 0.0
            cum_volume = 0.0
            cum_pv = 0.0
            n_ticks = 0

            if i + 1 < n:
                bar_open = prices[i + 1]
                bar_high = prices[i + 1]
                bar_low = prices[i + 1]
                bar_ts_open = timestamps[i + 1]

    if include_incomplete_final and n_ticks > 0:
        bar_ids[bar_idx] = bar_idx
        ts_open[bar_idx] = bar_ts_open
        ts_close[bar_idx] = timestamps[n - 1]
        opens[bar_idx] = bar_open
        highs[bar_idx] = bar_high
        lows[bar_idx] = bar_low
        closes[bar_idx] = prices[n - 1]
        bar_volumes[bar_idx] = cum_volume
        bar_dollars[bar_idx] = cum_dollar
        bar_vwaps[bar_idx] = cum_pv / cum_volume if cum_volume > 0 else prices[n - 1]
        tick_counts[bar_idx] = n_ticks
        thresholds_arr[bar_idx] = threshold
        bar_idx += 1

    return (
        bar_ids, ts_open, ts_close, opens, highs, lows, closes,
        bar_volumes, bar_dollars, bar_vwaps, tick_counts, thresholds_arr, bar_idx,
    )
