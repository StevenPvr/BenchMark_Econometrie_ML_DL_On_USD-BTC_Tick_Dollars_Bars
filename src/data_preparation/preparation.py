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
    target_num_bars: int,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "amount",
    threshold: float | None = None,
    ema_span: int = 100,
    adaptive: bool = False,
    threshold_bounds: tuple[float, float] | None = None,
    calibration_fraction: float = 1.0,
    include_incomplete_final: bool = True,
) -> pd.DataFrame:
    """Compute Dollar Bars from tick data following De Prado's methodology.

    Dollar Bars sample the data each time a predefined monetary value T
    is exchanged. This produces a series with more regular statistical
    properties than time-based bars, better suited for ML models.

    De Prado's Expected Dollar Value calibration (AFML Chapter 2):
        T = Total Dollar Volume (calibration prefix) / Target Number of Bars (prefix)

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
        target_num_bars: Target number of bars to generate (REQUIRED).
            Uses De Prado's Expected Dollar Value method for calibration.
        timestamp_col: Name of timestamp column (int64 ms or datetime).
        price_col: Name of price column.
        volume_col: Name of volume column.
        threshold: Override with fixed dollar threshold T (ignores target_num_bars).
        ema_span: EMA span for adaptive threshold (default 100 bars).
        adaptive: If True, use adaptive EWMA threshold (optional).
            If False (default), use fixed threshold from calibration (standard dollar bars).
        threshold_bounds: Optional (min, max) multipliers for adaptive bounds.
            Default None = unbounded EWMA. Provide e.g. (0.5, 2.0) to clip.
        calibration_fraction: Fraction of the dataset (prefix) used to calibrate
            T_0. Default 1.0 = full sample (classic expected value).
        include_incomplete_final: If True, keeps the trailing partial bar even if
            the threshold was not hit.

    Returns:
        DataFrame with dollar bars containing:
        - bar_id, timestamp_open/close, datetime_open/close
        - open, high, low, close (OHLC)
        - volume, cum_dollar_value, vwap
        - n_ticks, threshold_used, duration_sec

    Raises:
        ValueError: If required columns are missing.

    Example:
        >>> # De Prado Expected Value method
        >>> bars = compute_dollar_bars(df_ticks, target_num_bars=500_000)
        >>>
        >>> # Fixed threshold override
        >>> bars = compute_dollar_bars(df_ticks, target_num_bars=500_000, threshold=225_000)
    """
    # Validate columns
    required_cols = {timestamp_col, price_col, volume_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

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

    # Clamp calibration_fraction to [0, 1]
    calibration_fraction = float(calibration_fraction)
    if calibration_fraction <= 0:
        calibration_fraction = 1.0  # default to full-sample calibration
    if calibration_fraction > 1:
        calibration_fraction = 1.0

    # Determine which mode to use (priority: threshold > target_num_bars > legacy)
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
        # This should never happen with the new signature
        raise ValueError("Either threshold or target_num_bars must be provided")

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

    avg_ticks = len(df) / num_bars
    threshold_range = f"{thresholds_arr.min():,.0f} - {thresholds_arr.max():,.0f}"
    logger.info(
        f"Generated {num_bars} dollar bars from {len(df)} ticks "
        f"(avg {avg_ticks:.1f} ticks/bar, threshold range: {threshold_range})"
    )

    return df_bars


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
    target_num_bars: int,
    output_parquet: Path | str | None = None,
    threshold: float | None = None,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "amount",
    ema_span: int = 100,
    adaptive: bool = False,
    threshold_bounds: tuple[float, float] | None = None,
    calibration_fraction: float = 1.0,
    include_incomplete_final: bool = True,
) -> pd.DataFrame:
    """End-to-end pipeline for generating Dollar Bars from tick parquet.

    Args:
        parquet_path: Path to input tick data parquet file.
        output_parquet: Path to save bars as parquet (None to skip).
        target_num_bars: Target number of bars (RECOMMENDED, De Prado method).
        threshold: Fixed dollar threshold (overrides target_num_bars).
        timestamp_col: Name of timestamp column.
        price_col: Name of price column.
        volume_col: Name of volume column.
        ema_span: EMA span for adaptive threshold.
        adaptive: Use adaptive EWMA threshold (De Prado recommended).
        threshold_bounds: Optional bounds when adaptive=True.
        calibration_fraction: Fraction of earliest ticks used to calibrate T_0.
        include_incomplete_final: Keep trailing partial bar if True.

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
    )

    if output_parquet is not None:
        output_parquet = Path(output_parquet)
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_bars.to_parquet(output_parquet, index=False)
        logger.info(f"Saved {len(df_bars):,} bars to {output_parquet}")

    return df_bars


def run_dollar_bars_pipeline_batch(
    input_dir: Path | str,
    target_num_bars: int,
    output_parquet: Path | str | None = None,
    threshold: float | None = None,
    adaptive: bool = False,
    threshold_bounds: tuple[float, float] | None = None,
    calibration_fraction: float = 1.0,
    include_incomplete_final: bool = True,
) -> pd.DataFrame:
    """Run dollar bars pipeline on a directory of partitioned parquet files (memory efficient).

    Processes each file individually to avoid loading all data in memory at once.
    This is much more memory-efficient for large datasets.

    Args:
        input_dir: Directory containing partitioned parquet files
        output_parquet: Path to save consolidated bars parquet. If None, uses default.
        target_num_bars: Target number of bars (De Prado method).
        threshold: Fixed dollar threshold (overrides target_num_bars).
        adaptive: Use adaptive EWMA threshold (De Prado recommended).
        threshold_bounds: Optional bounds when adaptive=True.
        calibration_fraction: Fraction of earliest ticks used to calibrate T_0.
        include_incomplete_final: Keep trailing partial bar if True.

    Returns:
        DataFrame with consolidated dollar bars from all input files.
    """
    from pathlib import Path
    import pandas as pd
    import pyarrow as pa # type: ignore[import-untyped]
    import pyarrow.parquet as pq # type: ignore[import-untyped]

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Find all parquet files
    parquet_files = list(input_path.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {input_path}")

    logger.info(f"Processing {len(parquet_files)} partitioned files from {input_path}")

    # Resolve output path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_parquet is None:
        from src.path import DOLLAR_BARS_PARQUET
        output_parquet = DOLLAR_BARS_PARQUET.parent / f"dollar_bars_{timestamp}.parquet"

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)

    # Stream-write to Parquet to avoid holding everything in memory
    parquet_writer: pq.ParquetWriter | None = None
    total_bars = 0

    for i, file_path in enumerate(sorted(parquet_files), 1):
        logger.info(f"Processing file {i}/{len(parquet_files)}: {file_path.name}")

        try:
            # Process this file individually
            bars_df = prepare_dollar_bars(
                parquet_path=file_path,
                output_parquet=None,  # Avoid per-file saves
                target_num_bars=target_num_bars,
                threshold=threshold,
                adaptive=adaptive,
                threshold_bounds=threshold_bounds,
                calibration_fraction=calibration_fraction,
                include_incomplete_final=include_incomplete_final,
            )

            if not bars_df.empty:
                # Sort chunk locally for better ordering
                if "datetime_close" in bars_df.columns:
                    bars_df = bars_df.sort_values("datetime_close").reset_index(drop=True)

                table = pa.Table.from_pandas(bars_df, preserve_index=False)

                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_parquet, table.schema)

                parquet_writer.write_table(table)

                total_bars += len(bars_df)
                logger.info(
                    "  Generated %d bars from %d ticks (running total: %d)",
                    len(bars_df),
                    len(pd.read_parquet(file_path)),
                    total_bars,
                )
            else:
                logger.warning(f"  No bars generated from {file_path.name}")

        except Exception as e:
            logger.error(f"  Failed to process {file_path.name}: {e}")
            continue

    if parquet_writer is None or total_bars == 0:
        raise ValueError("No bars were generated from any input file")

    parquet_writer.close()

    logger.info(f"Streamed {total_bars} bars from {len(parquet_files)} files into {output_parquet}")

    # Load once to enforce global sort/unique (bar dataset is much smaller than ticks)
    consolidated_bars = pd.read_parquet(output_parquet)
    if "datetime_close" in consolidated_bars.columns:
        consolidated_bars = (
            consolidated_bars
            .sort_values("datetime_close")
            .drop_duplicates(subset=["datetime_close"])
            .reset_index(drop=True)
        )
        consolidated_bars.to_parquet(output_parquet, index=False)
        logger.info("Consolidated, sorted, and deduplicated bars saved to disk")

    logger.info(f"Saved consolidated dollar bars to: {output_parquet}")

    return consolidated_bars


def run_dollar_bars_pipeline(
    target_num_bars: int,
    input_parquet: Path | str | None = None,
    output_parquet: Path | str | None = None,
    threshold: float | None = None,
    adaptive: bool = False,
    threshold_bounds: tuple[float, float] | None = None,
    calibration_fraction: float = 1.0,
    include_incomplete_final: bool = True,
) -> pd.DataFrame:
    """Run the complete dollar bars pipeline.

    Args:
        input_parquet: Path to input tick data. If None, uses default.
        output_parquet: Path to save bars parquet. If None, uses default.
        target_num_bars: Target number of bars (De Prado method).
        threshold: Fixed dollar threshold (overrides target_num_bars).
        adaptive: Use adaptive EWMA threshold with bounds.
        threshold_bounds: Optional (min_mult, max_mult) for adaptive bounds.
        calibration_fraction: Fraction of earliest ticks used to calibrate T_0.
        include_incomplete_final: Keep trailing partial bar if True.

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
        output_parquet=output_parquet,
        target_num_bars=target_num_bars,
        threshold=threshold,
        adaptive=adaptive,
        threshold_bounds=threshold_bounds,
        calibration_fraction=calibration_fraction,
        include_incomplete_final=include_incomplete_final,
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
