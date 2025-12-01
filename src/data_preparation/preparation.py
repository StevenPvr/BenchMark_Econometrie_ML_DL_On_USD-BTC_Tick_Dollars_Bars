"""Dollar Bars implementation following De Prado's methodology.

This module implements Dollar Bars (monetary volume bars) as described in
Marcos Lopez de Prado's "Advances in Financial Machine Learning" (Chapter 2).

Dollar Bars sample data each time a predefined monetary value (threshold T)
is exchanged, rather than at fixed time intervals. This approach:
- Produces more bars during high activity periods (volatile markets)
- Produces fewer bars during low activity periods (quiet markets)
- Improves statistical properties (closer to IID Gaussian) of returns
- Synchronizes sampling with market activity information flow

Mathematical Formulation (De Prado):
    Let each tick t have price p_t and volume v_t.
    Dollar value: dv_t = p_t * v_t
    Bar k closes at tick t when: sum(dv_i for i in [t_start, t]) >= T_k

Adaptive Threshold (De Prado EWMA method):
    T_0 = initial calibration
    After bar k with dollar_value D_k:
        E_k = alpha * D_k + (1 - alpha) * E_{k-1}  (EWMA of bar dollar values)
        T_{k+1} = E_k  (threshold adapts to market regime)

Reference:
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 2: Financial Data Structures, pp. 23-30.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np # type: ignore[import-untyped]
import pandas as pd # type: ignore[import-untyped]
from numba import njit # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.path import (
    DOLLAR_IMBALANCE_BARS_CSV,
    DOLLAR_IMBALANCE_BARS_PARQUET,
    LOG_RETURNS_PARQUET,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_dollar_bars",
    "prepare_dollar_bars",
    "generate_dollar_bars",
    "run_dollar_bars_pipeline",
    "add_log_returns_to_bars_file",
]


# =============================================================================
# PARAMETER VALIDATION
# =============================================================================


def _validate_dollar_bars_params(
    ema_span: int | None,
    threshold_bounds: tuple[float, float] | None,
    calibration_fraction: float,
) -> None:
    """Validate parameters for dollar bars computation.

    Args:
        ema_span: EMA span for adaptive threshold (must be positive if provided).
        threshold_bounds: Tuple of (min_mult, max_mult) for threshold bounds.
        calibration_fraction: Fraction of data used for calibration (must be in (0, 1]).

    Raises:
        ValueError: If any parameter is invalid.
    """
    if ema_span is not None and ema_span <= 0:
        raise ValueError(f"ema_span must be positive, got {ema_span}")

    if threshold_bounds is not None:
        if len(threshold_bounds) != 2:
            raise ValueError(
                f"threshold_bounds must be a tuple of (min_mult, max_mult), "
                f"got {len(threshold_bounds)} elements"
            )
        min_mult, max_mult = threshold_bounds
        if min_mult >= max_mult:
            raise ValueError(
                f"threshold_bounds min ({min_mult}) must be < max ({max_mult})"
            )
        if min_mult <= 0:
            raise ValueError(
                f"threshold_bounds min_mult must be positive, got {min_mult}"
            )

    if not 0 < calibration_fraction <= 1:
        raise ValueError(
            f"calibration_fraction must be in (0, 1], got {calibration_fraction}"
        )


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================


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

    # Sum all dollar values
    total_dollar = 0.0
    for i in range(n):
        total_dollar += dollar_values[i]

    # T = Total Dollar Volume / Target Bars (De Prado Expected Value method)
    threshold = total_dollar / target_num_bars
    return max(threshold, 1e-10)


@njit(cache=True)
def _compute_robust_percentile_threshold(
    dollar_values: NDArray[np.float64],
    target_num_bars: int,
    percentile: float = 50.0,
) -> tuple[float, float, float]:
    """Compute threshold with robust statistics to handle outliers.

    Uses percentile-based estimation to avoid bias from extreme ticks.

    Args:
        dollar_values: Array of dollar values (price * volume) for each tick.
        target_num_bars: Target total number of bars to generate.
        percentile: Percentile to use for robust estimation (default 50 = median).

    Returns:
        Tuple of (threshold, min_threshold, max_threshold) for bounded adaptation.
    """
    n = len(dollar_values)
    if n == 0 or target_num_bars <= 0:
        return 1.0, 1.0, 1.0

    # Calculate total for primary threshold
    total_dollar = 0.0
    for i in range(n):
        total_dollar += dollar_values[i]

    threshold = total_dollar / target_num_bars

    # Calculate bounds based on typical tick values
    # Sort a sample for percentile calculation
    sample_size = min(n, 100000)
    step = max(1, n // sample_size)
    sample = np.empty(sample_size, dtype=np.float64)
    idx = 0
    for i in range(0, n, step):
        if idx < sample_size:
            sample[idx] = dollar_values[i]
            idx += 1

    sample_slice = sample[:idx]
    sample_slice.sort()

    # Percentile-based bounds (more robust than mean)
    p25_idx = int(idx * 0.25)
    p75_idx = int(idx * 0.75)
    p25 = sample_slice[p25_idx] if p25_idx < idx else sample_slice[0]
    p75 = sample_slice[p75_idx] if p75_idx < idx else sample_slice[idx - 1]

    # Target ticks per bar
    target_ticks = n / target_num_bars

    # Bounds: allow threshold to vary within reasonable range
    # Min: 50% of calibrated threshold (more bars)
    # Max: 200% of calibrated threshold (fewer bars)
    min_threshold = threshold * 0.5
    max_threshold = threshold * 2.0

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

    # Pre-allocate output arrays (estimate max bars = n ticks)
    max_bars = n
    bar_ids = np.empty(max_bars, dtype=np.int64)
    ts_open = np.empty(max_bars, dtype=np.int64)
    ts_close = np.empty(max_bars, dtype=np.int64)
    opens = np.empty(max_bars, dtype=np.float64)
    highs = np.empty(max_bars, dtype=np.float64)
    lows = np.empty(max_bars, dtype=np.float64)
    closes = np.empty(max_bars, dtype=np.float64)
    bar_volumes = np.empty(max_bars, dtype=np.float64)
    bar_dollars = np.empty(max_bars, dtype=np.float64)
    bar_vwaps = np.empty(max_bars, dtype=np.float64)
    tick_counts = np.empty(max_bars, dtype=np.int64)
    thresholds = np.empty(max_bars, dtype=np.float64)

    # Initialize adaptive threshold state
    bar_idx = 0
    current_threshold = initial_threshold
    ema_dollar = initial_threshold  # EWMA of bar dollar values

    # Initialize bar accumulation state
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

        # Dollar value for this tick: dv_t = p_t * v_t
        dollar_value = price * volume

        # Accumulate
        cum_dollar += dollar_value
        cum_volume += volume
        cum_pv += price * volume
        n_ticks += 1

        # Update high/low
        if price > bar_high:
            bar_high = price
        if price < bar_low:
            bar_low = price

        # Check if threshold is reached: sum(dv_i) >= T_k
        if cum_dollar >= current_threshold:
            # === CLOSE BAR k ===
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

            # === ADAPTIVE THRESHOLD UPDATE (bounds optional) ===
            # E_k = alpha * D_k + (1 - alpha) * E_{k-1}
            ema_dollar = ema_alpha * cum_dollar + (1.0 - ema_alpha) * ema_dollar
            # T_{k+1} = clip(E_k, min, max) when bounds are provided
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

            # Next bar opens at next tick
            if i + 1 < n:
                bar_open = prices[i + 1]
                bar_high = prices[i + 1]
                bar_low = prices[i + 1]
                bar_ts_open = timestamps[i + 1]

    # Optionally include incomplete final bar (default False for strict De Prado)
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


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================


def compute_dollar_bars(
    df: pd.DataFrame,
    target_ticks_per_bar: int | None = None,
    target_num_bars: int | None = None,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "amount",
    threshold: float | None = None,
    ema_span: int = 100,
    adaptive: bool = False,
    threshold_bounds: tuple[float, float] | None = None,
    calibration_fraction: float = 1.0,
    include_incomplete_final: bool = False,
    exclude_calibration_prefix: bool = False,
) -> pd.DataFrame:
    """Compute Dollar Bars from tick data following De Prado's methodology.

    Dollar Bars sample the data each time a predefined monetary value T
    is exchanged. This produces a series with more regular statistical
    properties than time-based bars, better suited for ML models.

    De Prado's calibration (AFML Chapter 2):
        T = Total Dollar Volume / Target Number of Bars
        where Target Bars = n_ticks / target_ticks_per_bar

    Adaptive threshold formula (optional bounds, off by default):
        T_0 calibrated on a prefix of the data
        After bar k with dollar_value D_k:
            E_k = alpha * D_k + (1 - alpha) * E_{k-1}
            T_{k+1} = E_k  (bounds applied only if threshold_bounds provided)

    Mathematical formulation:
        dv_t = p_t * v_t  (dollar value of tick t)
        Bar k closes at tick t when: sum(dv_i for i in [t_start, t]) >= T_k

    Args:
        df: DataFrame with tick-by-tick data.
        target_ticks_per_bar: Target number of ticks per bar (RECOMMENDED).
            Typical values: 100-500 for robust OHLCV statistics.
            T is computed as: Total Dollar Volume / (n_ticks / target_ticks_per_bar)
        target_num_bars: Alternative: target number of bars (legacy).
            If both provided, target_ticks_per_bar takes precedence.
        timestamp_col: Name of timestamp column (int64 ms or datetime).
        price_col: Name of price column.
        volume_col: Name of volume column.
        threshold: Override with fixed dollar threshold T (ignores target params).
        ema_span: EMA span for adaptive threshold (default 100 bars).
        adaptive: If True, use adaptive EWMA threshold (optional).
            If False (default), use fixed threshold from calibration (standard dollar bars).
        threshold_bounds: Optional (min, max) multipliers for adaptive bounds.
            Default None = unbounded EWMA. Provide e.g. (0.5, 2.0) to clip.
        calibration_fraction: Fraction of the dataset (prefix) used to calibrate
            T_0. Default 1.0 = full sample (classic expected value).
        include_incomplete_final: If True, keeps the trailing partial bar even if
            the threshold was not hit. Default False for methodological rigor.
        exclude_calibration_prefix: If True, excludes bars generated from ticks
            in the calibration prefix. This prevents using the same data for both
            threshold calibration AND bar generation. Default False.

    Returns:
        DataFrame with dollar bars containing:
        - bar_id, timestamp_open/close, datetime_open/close
        - open, high, low, close (OHLC)
        - volume, cum_dollar_value, vwap
        - n_ticks, threshold_used, duration_sec

    Raises:
        ValueError: If required columns are missing.

    Example:
        >>> # De Prado methodology with target ticks per bar
        >>> bars = compute_dollar_bars(df_ticks, target_ticks_per_bar=100)
        >>>
        >>> # Fixed threshold override
        >>> bars = compute_dollar_bars(df_ticks, threshold=225_000)
    """
    # Validate columns
    required_cols = {timestamp_col, price_col, volume_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate parameters
    _validate_dollar_bars_params(ema_span, threshold_bounds, calibration_fraction)

    if df.empty:
        logger.warning("Empty DataFrame provided, returning empty bars")
        return _create_empty_bars_df()

    # Ensure deterministic ordering before the numba loop
    if not df[ts_col := timestamp_col].is_monotonic_increasing:
        logger.info("Sorting ticks by %s for dollar bar construction", ts_col)
        df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)

    # Convert to numpy arrays
    if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        timestamps = (df[timestamp_col].astype("int64") // 10**6).values
    else:
        timestamps = df[timestamp_col].values.astype(np.int64)

    prices = df[price_col].values.astype(np.float64)
    volumes = df[volume_col].values.astype(np.float64)

    # Data quality checks
    if np.any(prices <= 0):
        n_invalid = int(np.sum(prices <= 0))
        logger.warning(f"Found {n_invalid} non-positive prices")

    if np.any(volumes < 0):
        n_invalid = int(np.sum(volumes < 0))
        logger.warning(f"Found {n_invalid} negative volumes")

    # Compute dollar values
    dollar_values = prices * volumes
    total_dollar_volume = float(np.sum(dollar_values))

    # EMA alpha for adaptive threshold
    ema_alpha = 2.0 / (ema_span + 1.0)

    # Track calibration boundary for potential prefix exclusion
    calibration_end_timestamp: int | None = None

    # Convert target_ticks_per_bar to target_num_bars (De Prado methodology)
    n_ticks = len(df)
    if target_ticks_per_bar is not None:
        if target_ticks_per_bar <= 0:
            raise ValueError("target_ticks_per_bar must be positive")
        target_num_bars = max(1, n_ticks // target_ticks_per_bar)
        logger.info(
            f"De Prado methodology: {target_ticks_per_bar} ticks/bar → "
            f"{target_num_bars:,} target bars from {n_ticks:,} ticks"
        )

    # Determine which mode to use (priority: threshold > target_num_bars)
    if threshold is not None:
        # === MODE 1: FIXED THRESHOLD (user-specified) ===
        logger.info(f"Using FIXED threshold: {threshold:,.2f} USD")
        result = _accumulate_dollar_bars_fixed(
            timestamps, prices, volumes, threshold, include_incomplete_final
        )

    elif target_num_bars is not None:
        # === MODE 2: TARGET NUMBER OF BARS (De Prado Expected Value - RECOMMENDED) ===
        if target_num_bars <= 0:
            raise ValueError("target_num_bars must be positive when threshold is None")

        calibration_size = len(dollar_values)
        if 0 < calibration_fraction < 1.0:
            calibration_size = max(1, int(len(dollar_values) * calibration_fraction))

        target_bars_calibration = target_num_bars
        if 0 < calibration_fraction < 1.0:
            target_bars_calibration = max(
                1, int(target_num_bars * calibration_size / len(dollar_values))
            )

        calibration_values = dollar_values[:calibration_size]
        initial_threshold = float(_compute_threshold_from_target_bars(
            calibration_values, target_bars_calibration
        ))

        # Store the timestamp of the last calibration tick for potential exclusion
        if exclude_calibration_prefix and calibration_size < len(timestamps):
            calibration_end_timestamp = int(timestamps[calibration_size - 1])

        # Compute bounds for adaptive mode (None = unbounded pure EWMA)
        if threshold_bounds is None:
            min_threshold = 0.0
            max_threshold = np.inf
        else:
            bound_min_mult, bound_max_mult = threshold_bounds
            min_threshold = initial_threshold * bound_min_mult
            max_threshold = initial_threshold * bound_max_mult

        logger.info(
            f"De Prado Expected Dollar Value calibration (prefix only):\n"
            f"  Prefix ticks: {calibration_size:,} / {len(df):,} "
            f"({(calibration_size / len(df)) * 100:.1f}%)\n"
            f"  Target bars in prefix: {target_bars_calibration:,}\n"
            f"  Calibrated T_0: {initial_threshold:,.2f} USD\n"
            f"  Expected ticks/bar: {calibration_size / target_bars_calibration:.1f}"
        )

        if adaptive:
            bounds_label = (
                "unbounded EWMA (pure De Prado)"
                if threshold_bounds is None
                else f"bounded [{min_threshold:,.0f}, {max_threshold:,.0f}]"
            )
            logger.info(f"  Adaptive mode: {bounds_label}")
            result = _accumulate_dollar_bars_adaptive(
                timestamps, prices, volumes, initial_threshold, ema_alpha,
                min_threshold, max_threshold, include_incomplete_final
            )
        else:
            logger.info("  Fixed mode: threshold constant at T_0")
            result = _accumulate_dollar_bars_fixed(
                timestamps, prices, volumes, initial_threshold, include_incomplete_final
            )

    else:
        raise ValueError(
            "Either threshold, target_ticks_per_bar, or target_num_bars must be provided"
        )

    # Unpack results
    (
        bar_ids, ts_open, ts_close, opens, highs, lows, closes,
        bar_volumes, bar_dollars, bar_vwaps, tick_counts, thresholds_arr, num_bars
    ) = result

    if num_bars == 0:
        logger.warning("No bars formed (threshold too high or insufficient data)")
        return _create_empty_bars_df()

    # Trim to actual size
    bar_ids = bar_ids[:num_bars]
    ts_open = ts_open[:num_bars]
    ts_close = ts_close[:num_bars]
    opens = opens[:num_bars]
    highs = highs[:num_bars]
    lows = lows[:num_bars]
    closes = closes[:num_bars]
    bar_volumes = bar_volumes[:num_bars]
    bar_dollars = bar_dollars[:num_bars]
    bar_vwaps = bar_vwaps[:num_bars]
    tick_counts = tick_counts[:num_bars]
    thresholds_arr = thresholds_arr[:num_bars]

    # Compute derived fields
    duration_sec = (ts_close - ts_open) / 1000.0
    datetime_open = pd.to_datetime(ts_open, unit="ms", utc=True)
    datetime_close = pd.to_datetime(ts_close, unit="ms", utc=True)

    df_bars = pd.DataFrame({
        "bar_id": bar_ids,
        "timestamp_open": ts_open,
        "timestamp_close": ts_close,
        "datetime_open": datetime_open,
        "datetime_close": datetime_close,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": bar_volumes,
        "cum_dollar_value": bar_dollars,
        "vwap": bar_vwaps,
        "n_ticks": tick_counts,
        "threshold_used": thresholds_arr,
        "duration_sec": duration_sec,
    })

    # Exclude bars from calibration prefix if requested
    # This prevents using calibration data for both threshold estimation AND bar generation
    if exclude_calibration_prefix and calibration_end_timestamp is not None:
        original_count = len(df_bars)
        df_bars = df_bars[df_bars["timestamp_close"] > calibration_end_timestamp].copy()
        df_bars = df_bars.reset_index(drop=True)
        df_bars["bar_id"] = np.arange(len(df_bars))  # Re-index bar_ids
        excluded_count = original_count - len(df_bars)
        logger.info(
            f"Excluded {excluded_count} bars from calibration prefix "
            f"(timestamp <= {calibration_end_timestamp})"
        )

    if len(df_bars) == 0:
        logger.warning("No bars remaining after calibration prefix exclusion")
        return _create_empty_bars_df()

    avg_ticks = len(df) / len(df_bars) if len(df_bars) > 0 else 0
    threshold_range = f"{df_bars['threshold_used'].min():,.0f} - {df_bars['threshold_used'].max():,.0f}"
    logger.info(
        f"Generated {len(df_bars)} dollar bars from {len(df)} ticks "
        f"(avg {avg_ticks:.1f} ticks/bar, threshold range: {threshold_range})"
    )

    return pd.DataFrame(df_bars)


def _create_empty_bars_df() -> pd.DataFrame:
    """Create an empty DataFrame with the correct dollar bars schema."""
    return pd.DataFrame({
        "bar_id": pd.Series(dtype="int64"),
        "timestamp_open": pd.Series(dtype="int64"),
        "timestamp_close": pd.Series(dtype="int64"),
        "datetime_open": pd.Series(dtype="datetime64[ns, UTC]"),
        "datetime_close": pd.Series(dtype="datetime64[ns, UTC]"),
        "open": pd.Series(dtype="float64"),
        "high": pd.Series(dtype="float64"),
        "low": pd.Series(dtype="float64"),
        "close": pd.Series(dtype="float64"),
        "volume": pd.Series(dtype="float64"),
        "cum_dollar_value": pd.Series(dtype="float64"),
        "vwap": pd.Series(dtype="float64"),
        "n_ticks": pd.Series(dtype="int64"),
        "threshold_used": pd.Series(dtype="float64"),
        "duration_sec": pd.Series(dtype="float64"),
    })


def generate_dollar_bars(
    df_ticks: pd.DataFrame,
    threshold: float,
    timestamp_col: str = "date_time",
    price_col: str = "price",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Generate Dollar Bars with simplified interface (fixed threshold).

    Convenience wrapper for `compute_dollar_bars` with fixed threshold.

    Args:
        df_ticks: DataFrame with tick data.
        threshold: Fixed dollar threshold T.
        timestamp_col: Name of datetime column.
        price_col: Name of price column.
        volume_col: Name of volume column.

    Returns:
        DataFrame with OHLCV dollar bars.
    """
    return compute_dollar_bars(
        df=df_ticks,
        target_num_bars=0,  # Not used in fixed threshold mode
        threshold=threshold,
        timestamp_col=timestamp_col,
        price_col=price_col,
        volume_col=volume_col,
        adaptive=False,
    )


def prepare_dollar_bars(
    parquet_path: Path | str,
    target_ticks_per_bar: int | None = None,
    target_num_bars: int | None = None,
    output_parquet: Path | str | None = None,
    threshold: float | None = None,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "amount",
    ema_span: int = 100,
    adaptive: bool = False,
    threshold_bounds: tuple[float, float] | None = None,
    calibration_fraction: float = 1.0,
    include_incomplete_final: bool = False,
    exclude_calibration_prefix: bool = False,
) -> pd.DataFrame:
    """End-to-end pipeline for generating Dollar Bars from tick parquet.

    Args:
        parquet_path: Path to input tick data parquet file.
        target_ticks_per_bar: Target ticks per bar (RECOMMENDED, De Prado method).
        target_num_bars: Alternative: target number of bars (legacy).
        output_parquet: Path to save bars as parquet (None to skip).
        threshold: Fixed dollar threshold (overrides target params).
        timestamp_col: Name of timestamp column.
        price_col: Name of price column.
        volume_col: Name of volume column.
        ema_span: EMA span for adaptive threshold.
        adaptive: Use adaptive EWMA threshold (De Prado recommended).
        threshold_bounds: Optional bounds when adaptive=True.
        calibration_fraction: Fraction of earliest ticks used to calibrate T_0.
        include_incomplete_final: Keep trailing partial bar if True. Default False.
        exclude_calibration_prefix: Exclude bars from calibration period. Default False.

    Returns:
        DataFrame with computed dollar bars.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {parquet_path}")

    logger.info(f"Loading tick data from {parquet_path}")
    df_ticks = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df_ticks):,} ticks")

    df_bars = compute_dollar_bars(
        df=df_ticks,
        target_ticks_per_bar=target_ticks_per_bar,
        target_num_bars=target_num_bars,
        timestamp_col=timestamp_col,
        price_col=price_col,
        volume_col=volume_col,
        threshold=threshold,
        ema_span=ema_span,
        adaptive=adaptive,
        threshold_bounds=threshold_bounds,
        calibration_fraction=calibration_fraction,
        include_incomplete_final=include_incomplete_final,
        exclude_calibration_prefix=exclude_calibration_prefix,
    )

    if output_parquet is not None:
        output_parquet = Path(output_parquet)
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_bars.to_parquet(output_parquet, index=False)
        logger.info(f"Saved {len(df_bars):,} bars to {output_parquet}")

    return df_bars


@dataclass
class _BarAccumulatorState:
    """State for dollar bar accumulation across batches."""
    cum_dollar: float = 0.0
    cum_volume: float = 0.0
    cum_pv: float = 0.0  # price * volume sum for VWAP
    bar_open: float = 0.0
    bar_high: float = 0.0
    bar_low: float = 0.0
    bar_ts_open: int = 0
    n_ticks: int = 0
    bar_idx: int = 0
    current_threshold: float = 0.0
    ema_dollar: float = 0.0
    initialized: bool = False


def _process_batch_with_state(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    state: _BarAccumulatorState,
    ema_alpha: float,
    min_threshold: float,
    max_threshold: float,
    adaptive: bool,
) -> tuple[list[dict], _BarAccumulatorState]:
    """Process a batch of ticks and return completed bars + updated state.

    This function maintains state between batches so that bars spanning
    batch boundaries are handled correctly.
    """
    completed_bars: list[dict] = []
    n = len(timestamps)

    if n == 0:
        return completed_bars, state

    # Initialize state with first tick if needed
    if not state.initialized:
        state.bar_open = prices[0]
        state.bar_high = prices[0]
        state.bar_low = prices[0]
        state.bar_ts_open = int(timestamps[0])
        state.initialized = True

    for i in range(n):
        price = float(prices[i])
        volume = float(volumes[i])
        ts = int(timestamps[i])

        dollar_value = price * volume

        # Accumulate
        state.cum_dollar += dollar_value
        state.cum_volume += volume
        state.cum_pv += price * volume
        state.n_ticks += 1

        # Update high/low
        if price > state.bar_high:
            state.bar_high = price
        if price < state.bar_low:
            state.bar_low = price

        # Check threshold
        if state.cum_dollar >= state.current_threshold:
            # Close bar
            vwap = state.cum_pv / state.cum_volume if state.cum_volume > 0 else price

            completed_bars.append({
                "bar_id": state.bar_idx,
                "timestamp_open": state.bar_ts_open,
                "timestamp_close": ts,
                "open": state.bar_open,
                "high": state.bar_high,
                "low": state.bar_low,
                "close": price,
                "volume": state.cum_volume,
                "cum_dollar_value": state.cum_dollar,
                "vwap": vwap,
                "n_ticks": state.n_ticks,
                "threshold_used": state.current_threshold,
            })

            # Adaptive threshold update
            if adaptive:
                state.ema_dollar = ema_alpha * state.cum_dollar + (1.0 - ema_alpha) * state.ema_dollar
                state.current_threshold = max(min_threshold, min(max_threshold, state.ema_dollar))

            state.bar_idx += 1

            # Reset for next bar
            state.cum_dollar = 0.0
            state.cum_volume = 0.0
            state.cum_pv = 0.0
            state.n_ticks = 0

            # Next bar opens at next tick
            if i + 1 < n:
                state.bar_open = float(prices[i + 1])
                state.bar_high = float(prices[i + 1])
                state.bar_low = float(prices[i + 1])
                state.bar_ts_open = int(timestamps[i + 1])
            else:
                state.initialized = False  # Will init with first tick of next batch

    return completed_bars, state


def run_dollar_bars_pipeline_batch(
    input_parquet: Path | str,
    target_ticks_per_bar: int | None = None,
    output_parquet: Path | str | None = None,
    batch_size: int = 20_000_000,
    adaptive: bool = False,
    threshold_bounds: tuple[float, float] | None = None,
    calibration_fraction: float = 0.1,
    include_incomplete_final: bool = False,
) -> pd.DataFrame:
    """Run dollar bars pipeline with memory-efficient batch processing.

    Processes ticks in batches to limit memory usage. State is maintained
    between batches so bars spanning batch boundaries are handled correctly.

    Args:
        input_parquet: Path to cleaned tick data parquet file.
        target_ticks_per_bar: Target ticks per bar (De Prado method). Default 100.
        output_parquet: Path to save bars parquet. If None, uses default.
        batch_size: Number of ticks per batch. Default 20M.
        adaptive: Use adaptive EWMA threshold.
        threshold_bounds: Optional (min_mult, max_mult) for adaptive bounds.
        calibration_fraction: Fraction of first batch used to calibrate threshold.
        include_incomplete_final: Keep trailing partial bar if True.

    Returns:
        DataFrame with generated dollar bars.
    """
    import gc
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    input_path = Path(input_parquet)
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    # Default target ticks per bar
    if target_ticks_per_bar is None:
        target_ticks_per_bar = 100

    # Resolve output path
    if output_parquet is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_parquet = DOLLAR_IMBALANCE_BARS_PARQUET.parent / f"dollar_bars_{timestamp}.parquet"

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get total row count without loading data
    parquet_file = pq.ParquetFile(input_path)
    total_ticks = parquet_file.metadata.num_rows
    n_batches = (total_ticks + batch_size - 1) // batch_size

    logger.info("=" * 70)
    logger.info("DOLLAR BARS PIPELINE - BATCH PROCESSING")
    logger.info("=" * 70)
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Total ticks: {total_ticks:,}")
    logger.info(f"  Batch size: {batch_size:,}")
    logger.info(f"  Number of batches: {n_batches}")
    logger.info(f"  Target ticks/bar: {target_ticks_per_bar}")

    # EMA parameters
    ema_span = 100
    ema_alpha = 2.0 / (ema_span + 1.0)

    # Initialize state
    state = _BarAccumulatorState()
    parquet_writer: pq.ParquetWriter | None = None
    total_bars = 0
    schema_initialized = False

    # First pass: calibrate threshold on first batch
    logger.info("Calibrating threshold on first batch...")
    first_batch = next(parquet_file.iter_batches(batch_size=batch_size))
    df_calibration = first_batch.to_pandas()

    calibration_size = max(1, int(len(df_calibration) * calibration_fraction))
    prices_cal = df_calibration["price"].values[:calibration_size].astype(np.float64)
    volumes_cal = df_calibration["amount"].values[:calibration_size].astype(np.float64)
    dollar_values_cal = prices_cal * volumes_cal

    target_bars_cal = max(1, calibration_size // target_ticks_per_bar)
    initial_threshold = float(np.sum(dollar_values_cal)) / target_bars_cal

    state.current_threshold = initial_threshold
    state.ema_dollar = initial_threshold

    # Compute bounds
    if threshold_bounds is None:
        min_threshold = 0.0
        max_threshold = np.inf
    else:
        min_mult, max_mult = threshold_bounds
        min_threshold = initial_threshold * min_mult
        max_threshold = initial_threshold * max_mult

    logger.info(f"  Calibrated threshold: ${initial_threshold:,.2f}")
    logger.info(f"  Expected bars: ~{total_ticks // target_ticks_per_bar:,}")

    # Clean up calibration data
    del df_calibration, prices_cal, volumes_cal, dollar_values_cal
    gc.collect()

    # Process all batches
    parquet_file = pq.ParquetFile(input_path)  # Re-open for fresh iteration

    for batch_num, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size), 1):
        df_batch = batch.to_pandas()

        # Ensure sorted by timestamp
        if "timestamp" in df_batch.columns:
            df_batch = df_batch.sort_values("timestamp").reset_index(drop=True)

        # Extract arrays
        if "timestamp" in df_batch.columns:
            if pd.api.types.is_datetime64_any_dtype(df_batch["timestamp"]):
                timestamps = (df_batch["timestamp"].astype("int64") // 10**6).values
            else:
                timestamps = df_batch["timestamp"].values.astype(np.int64)
        else:
            raise ValueError("timestamp column required")

        prices = df_batch["price"].values.astype(np.float64)
        volumes = df_batch["amount"].values.astype(np.float64)

        # Process batch
        completed_bars, state = _process_batch_with_state(
            timestamps, prices, volumes, state,
            ema_alpha, min_threshold, max_threshold, adaptive
        )

        if completed_bars:
            # Convert to DataFrame
            df_bars = pd.DataFrame(completed_bars)

            # Add derived columns
            df_bars["duration_sec"] = (df_bars["timestamp_close"] - df_bars["timestamp_open"]) / 1000.0
            df_bars["datetime_open"] = pd.to_datetime(df_bars["timestamp_open"], unit="ms", utc=True)
            df_bars["datetime_close"] = pd.to_datetime(df_bars["timestamp_close"], unit="ms", utc=True)

            # Write to parquet
            table = pa.Table.from_pandas(df_bars, preserve_index=False)

            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")

            parquet_writer.write_table(table)
            total_bars += len(df_bars)

            del df_bars, completed_bars

        # Clear memory
        del df_batch, timestamps, prices, volumes, batch
        gc.collect()

        logger.info(
            f"  Batch {batch_num}/{n_batches}: {total_bars:,} bars generated"
        )

    # Handle incomplete final bar
    if include_incomplete_final and state.n_ticks > 0:
        final_bar = {
            "bar_id": state.bar_idx,
            "timestamp_open": state.bar_ts_open,
            "timestamp_close": state.bar_ts_open,  # Same as open for incomplete
            "open": state.bar_open,
            "high": state.bar_high,
            "low": state.bar_low,
            "close": state.bar_low,  # Use last known price
            "volume": state.cum_volume,
            "cum_dollar_value": state.cum_dollar,
            "vwap": state.cum_pv / state.cum_volume if state.cum_volume > 0 else state.bar_open,
            "n_ticks": state.n_ticks,
            "threshold_used": state.current_threshold,
            "duration_sec": 0.0,
            "datetime_open": pd.to_datetime(state.bar_ts_open, unit="ms", utc=True),
            "datetime_close": pd.to_datetime(state.bar_ts_open, unit="ms", utc=True),
        }
        df_final = pd.DataFrame([final_bar])
        table = pa.Table.from_pandas(df_final, preserve_index=False)
        if parquet_writer is not None:
            parquet_writer.write_table(table)
            total_bars += 1
        del df_final

    if parquet_writer is not None:
        parquet_writer.close()

    if total_bars == 0:
        raise ValueError("No bars were generated")

    logger.info("=" * 70)
    logger.info(f"Pipeline complete: {total_bars:,} dollar bars generated")
    logger.info(f"  Avg ticks/bar: {total_ticks / total_bars:.1f}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 70)

    # Return the result (read back from disk to avoid memory issues)
    return pd.read_parquet(output_path)


def run_dollar_bars_pipeline(
    target_ticks_per_bar: int | None = None,
    target_num_bars: int | None = None,
    input_parquet: Path | str | None = None,
    output_parquet: Path | str | None = None,
    threshold: float | None = None,
    adaptive: bool = False,
    threshold_bounds: tuple[float, float] | None = None,
    calibration_fraction: float = 1.0,
    include_incomplete_final: bool = False,
    exclude_calibration_prefix: bool = False,
) -> pd.DataFrame:
    """Run the complete dollar bars pipeline.

    Args:
        target_ticks_per_bar: Target ticks per bar (RECOMMENDED, De Prado method).
        target_num_bars: Alternative: target number of bars (legacy).
        input_parquet: Path to input tick data. If None, uses default.
        output_parquet: Path to save bars parquet. If None, uses default.
        threshold: Fixed dollar threshold (overrides target params).
        adaptive: Use adaptive EWMA threshold with bounds.
        threshold_bounds: Optional (min_mult, max_mult) for adaptive bounds.
        calibration_fraction: Fraction of earliest ticks used to calibrate T_0.
        include_incomplete_final: Keep trailing partial bar if True. Default False.
        exclude_calibration_prefix: Exclude bars from calibration period. Default False.

    Returns:
        DataFrame with generated dollar bars.
    """
    from src.path import RAW_DATA_DIR

    if input_parquet is None:
        # Use cleaned dataset from data_cleaning step
        from src.path import DATASET_CLEAN_PARQUET
        input_parquet = DATASET_CLEAN_PARQUET
        if not Path(input_parquet).exists():
            raise FileNotFoundError(f"No cleaned tick data found at {input_parquet}. Ensure data_cleaning has been run first.")

    if output_parquet is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_parquet = DOLLAR_IMBALANCE_BARS_PARQUET.parent / f"dollar_imbalance_bars_{timestamp}.parquet"

    logger.info("=" * 70)
    logger.info("DOLLAR BARS PIPELINE (De Prado Methodology)")
    logger.info("=" * 70)

    df_bars = prepare_dollar_bars(
        parquet_path=input_parquet,
        target_ticks_per_bar=target_ticks_per_bar,
        target_num_bars=target_num_bars,
        output_parquet=output_parquet,
        threshold=threshold,
        adaptive=adaptive,
        threshold_bounds=threshold_bounds,
        calibration_fraction=calibration_fraction,
        include_incomplete_final=include_incomplete_final,
        exclude_calibration_prefix=exclude_calibration_prefix,
    )

    logger.info("=" * 70)
    logger.info(f"Pipeline complete: {len(df_bars):,} dollar bars generated")
    logger.info("=" * 70)

    return df_bars


# =============================================================================
# LOG RETURNS PREPARATION
# =============================================================================


def add_log_returns_to_bars_file(
    bars_parquet: Path,
    price_col: str = "close",
) -> None:
    """Compute log returns (natural units) and persist inside the dollar_bars dataset."""
    if not bars_parquet.exists():
        raise FileNotFoundError(f"dollar_bars parquet not found at {bars_parquet}")

    df_bars = pd.read_parquet(bars_parquet)

    prices = df_bars[price_col]
    log_returns = np.log(prices / prices.shift(1))
    df_bars["log_return"] = log_returns
    df_bars = df_bars.dropna(subset=["log_return"])

    logger.info(
        "Log-returns stats in dollar_bars (natural) - Mean: %.6f, Std: %.6f",
        df_bars["log_return"].mean(),
        df_bars["log_return"].std(),
    )

    # Save consolidated dataset (Parquet only)
    bars_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_bars.to_parquet(bars_parquet, index=False)
    logger.info("Saved dollar_bars with log_return to: %s", bars_parquet)
