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

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd # type: ignore[import-untyped]
from numba import njit # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.path import (
    DOLLAR_IMBALANCE_BARS_CSV,
    DOLLAR_IMBALANCE_BARS_PARQUET,
    LOG_RETURNS_PARQUET,
    WEIGHTED_LOG_RETURNS_SPLIT_FILE,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_dollar_bars",
    "prepare_dollar_bars",
    "generate_dollar_bars",
    "run_dollar_bars_pipeline",
    "load_train_test_data",
    "save_log_returns_split",
]


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================


@njit(cache=True)
def _compute_initial_threshold(
    dollar_values: NDArray[np.float64],
    target_ticks_per_bar: int,
    ema_span: int,
    calibration_ticks: int,
) -> float:
    """Compute initial threshold using EMA of dollar values (De Prado method).

    The threshold is calibrated to achieve approximately `target_ticks_per_bar`
    ticks per bar, based on the observed dollar flow during calibration.

    Args:
        dollar_values: Array of dollar values (price * volume) for each tick.
        target_ticks_per_bar: Target number of ticks per bar.
        ema_span: Span for EMA smoothing of dollar values.
        calibration_ticks: Number of initial ticks to use for calibration.

    Returns:
        Computed initial threshold T_0 for bar formation.
    """
    n = min(calibration_ticks, len(dollar_values))
    if n == 0:
        return 1.0

    # Compute EMA of dollar values for calibration period
    alpha = 2.0 / (ema_span + 1.0)
    ema = dollar_values[0]
    for i in range(1, n):
        ema = alpha * dollar_values[i] + (1.0 - alpha) * ema

    # Initial threshold = EMA(tick dollar value) * target_ticks_per_bar
    threshold = ema * target_ticks_per_bar
    return max(threshold, 1e-10)


@njit(cache=True)
def _accumulate_dollar_bars_adaptive(
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    initial_threshold: float,
    ema_alpha: float,
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
    """Core numba-optimized loop with ADAPTIVE threshold (De Prado method).

    Implements De Prado's adaptive dollar bar algorithm:
    1. Start with initial threshold T_0 (calibrated)
    2. For each tick, accumulate dollar value
    3. When cumulative >= T_k, close bar k with dollar_value D_k
    4. Update EWMA: E_k = alpha * D_k + (1 - alpha) * E_{k-1}
    5. Next threshold: T_{k+1} = E_k (adapts to market regime)

    This allows the threshold to automatically adjust to:
    - Increased activity during volatile periods
    - Decreased activity during quiet periods

    Args:
        timestamps: Array of tick timestamps (int64 for ms precision).
        prices: Array of tick prices.
        volumes: Array of tick volumes.
        initial_threshold: Initial threshold T_0 from calibration.
        ema_alpha: Alpha for EWMA update (2 / (span + 1)).

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

            # === ADAPTIVE THRESHOLD UPDATE (De Prado EWMA) ===
            # E_k = alpha * D_k + (1 - alpha) * E_{k-1}
            ema_dollar = ema_alpha * cum_dollar + (1.0 - ema_alpha) * ema_dollar
            # T_{k+1} = E_k (threshold adapts to market regime)
            current_threshold = max(ema_dollar, 1e-10)

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

    # Handle incomplete final bar (De Prado: include partial bars)
    if n_ticks > 0:
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

    if n_ticks > 0:
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
    threshold: float | None = None,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "amount",
    target_ticks_per_bar: int = 50,
    ema_span: int = 100,
    calibration_ticks: int = 1000,
    adaptive: bool = True,
) -> pd.DataFrame:
    """Compute Dollar Bars from tick data following De Prado's methodology.

    Dollar Bars sample the data each time a predefined monetary value T
    is exchanged. This produces a series with more regular statistical
    properties than time-based bars, better suited for ML models.

    Threshold modes:
    - **Fixed**: Provide `threshold` parameter directly (constant T)
    - **Adaptive** (De Prado recommended): Leave `threshold=None` and `adaptive=True`
      The threshold T_k updates after each bar via EWMA to adapt to market regime.

    Adaptive threshold formula (De Prado):
        T_0 = initial calibration
        After bar k with dollar_value D_k:
            E_k = alpha * D_k + (1 - alpha) * E_{k-1}
            T_{k+1} = E_k

    Mathematical formulation:
        dv_t = p_t * v_t  (dollar value of tick t)
        Bar k closes at tick t when: sum(dv_i for i in [t_start, t]) >= T_k

    Args:
        df: DataFrame with tick-by-tick data.
        threshold: Fixed dollar threshold T. If None, uses adaptive mode.
        timestamp_col: Name of timestamp column (int64 ms or datetime).
        price_col: Name of price column.
        volume_col: Name of volume column.
        target_ticks_per_bar: Target ticks per bar for initial calibration.
        ema_span: EMA span for adaptive threshold (default 100 bars).
        calibration_ticks: Number of ticks for initial threshold calibration.
        adaptive: If True and threshold=None, use adaptive EWMA threshold.
            If False, use fixed threshold from initial calibration.

    Returns:
        DataFrame with dollar bars containing:
        - bar_id, timestamp_open/close, datetime_open/close
        - open, high, low, close (OHLC)
        - volume, cum_dollar_value, vwap
        - n_ticks, threshold_used, duration_sec

    Raises:
        ValueError: If required columns are missing.

    Example:
        >>> # Adaptive threshold (De Prado recommended)
        >>> bars = compute_dollar_bars(df_ticks, target_ticks_per_bar=50)
        >>>
        >>> # Fixed threshold
        >>> bars = compute_dollar_bars(df_ticks, threshold=1_000_000)
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

    # EMA alpha for adaptive threshold
    ema_alpha = 2.0 / (ema_span + 1.0)

    # Determine which mode to use
    if threshold is not None:
        # === FIXED THRESHOLD MODE ===
        logger.info(f"Using FIXED threshold: {threshold:,.2f}")
        result = _accumulate_dollar_bars_fixed(timestamps, prices, volumes, threshold)
    else:
        # Compute initial threshold from calibration
        initial_threshold = float(_compute_initial_threshold(
            dollar_values, target_ticks_per_bar, ema_span, calibration_ticks
        ))

        if adaptive:
            # === ADAPTIVE THRESHOLD MODE (De Prado recommended) ===
            logger.info(
                f"Using ADAPTIVE threshold (De Prado EWMA): "
                f"T_0={initial_threshold:,.2f}, alpha={ema_alpha:.4f}, "
                f"calibrated on {min(calibration_ticks, len(df))} ticks"
            )
            result = _accumulate_dollar_bars_adaptive(
                timestamps, prices, volumes, initial_threshold, ema_alpha
            )
        else:
            # === FIXED THRESHOLD FROM CALIBRATION ===
            logger.info(
                f"Using FIXED threshold from calibration: {initial_threshold:,.2f} "
                f"(target {target_ticks_per_bar} ticks/bar)"
            )
            result = _accumulate_dollar_bars_fixed(
                timestamps, prices, volumes, initial_threshold
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
        threshold=threshold,
        timestamp_col=timestamp_col,
        price_col=price_col,
        volume_col=volume_col,
        adaptive=False,
    )


def prepare_dollar_bars(
    parquet_path: Path | str,
    output_parquet: Path | str | None = None,
    output_csv: Path | str | None = None,
    threshold: float | None = None,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "amount",
    target_ticks_per_bar: int = 50,
    ema_span: int = 100,
    calibration_ticks: int = 1000,
    adaptive: bool = True,
) -> pd.DataFrame:
    """End-to-end pipeline for generating Dollar Bars from tick parquet.

    Args:
        parquet_path: Path to input tick data parquet file.
        output_parquet: Path to save bars as parquet (None to skip).
        output_csv: Path to save bars as CSV (None to skip).
        threshold: Fixed dollar threshold (None for adaptive).
        timestamp_col: Name of timestamp column.
        price_col: Name of price column.
        volume_col: Name of volume column.
        target_ticks_per_bar: Target ticks per bar.
        ema_span: EMA span for adaptive threshold.
        calibration_ticks: Number of calibration ticks.
        adaptive: Use adaptive EWMA threshold (De Prado recommended).

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
        threshold=threshold,
        timestamp_col=timestamp_col,
        price_col=price_col,
        volume_col=volume_col,
        target_ticks_per_bar=target_ticks_per_bar,
        ema_span=ema_span,
        calibration_ticks=calibration_ticks,
        adaptive=adaptive,
    )

    if output_parquet is not None:
        output_parquet = Path(output_parquet)
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_bars.to_parquet(output_parquet, index=False)
        logger.info(f"Saved {len(df_bars):,} bars to {output_parquet}")

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_bars.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(df_bars):,} bars to {output_csv}")

    return df_bars


def run_dollar_bars_pipeline(
    input_parquet: Path | str | None = None,
    output_parquet: Path | str | None = None,
    output_csv: Path | str | None = None,
    threshold: float | None = None,
    target_ticks_per_bar: int = 50,
    adaptive: bool = True,
) -> pd.DataFrame:
    """Run the complete dollar bars pipeline.

    Args:
        input_parquet: Path to input tick data. If None, uses default.
        output_parquet: Path to save bars parquet. If None, uses default.
        output_csv: Path to save bars CSV. If None, uses default.
        threshold: Fixed dollar threshold. If None, uses adaptive.
        target_ticks_per_bar: Target ticks per bar.
        adaptive: Use adaptive EWMA threshold (De Prado recommended).

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
        output_parquet = DOLLAR_IMBALANCE_BARS_PARQUET

    if output_csv is None:
        output_csv = DOLLAR_IMBALANCE_BARS_CSV

    logger.info("=" * 70)
    logger.info("DOLLAR BARS PIPELINE (De Prado Methodology)")
    logger.info("=" * 70)

    df_bars = prepare_dollar_bars(
        parquet_path=input_parquet,
        output_parquet=output_parquet,
        output_csv=output_csv,
        threshold=threshold,
        target_ticks_per_bar=target_ticks_per_bar,
        adaptive=adaptive,
    )

    logger.info("=" * 70)
    logger.info(f"Pipeline complete: {len(df_bars):,} dollar bars generated")
    logger.info("=" * 70)

    return df_bars


# =============================================================================
# LOG RETURNS PREPARATION
# =============================================================================


def save_log_returns_split(
    df_bars: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute log returns from dollar bars, multiply by 100, and save.

    This function computes log returns from the close prices of dollar bars,
    scales them by 100 (to express as percentage points), and saves the result
    to both CSV and Parquet formats. All original columns are preserved.

    Args:
        df_bars: DataFrame with dollar bars containing price and datetime columns.
        price_col: Name of the price column (default: "close").

    Returns:
        DataFrame with all original columns plus scaled log returns (x100).
    """
    # Compute log returns: ln(P_t / P_{t-1})
    prices = df_bars[price_col]
    log_returns = np.log(prices / prices.shift(1))

    # Create output DataFrame with all columns
    log_returns_split_df = df_bars.copy()
    log_returns_split_df["log_return"] = log_returns.values * 100  # Scale by 100

    # Drop NaN (first row)
    log_returns_split_df = log_returns_split_df.dropna(subset=["log_return"])

    # Log statistics
    logger.info(f"Log-returns original - Mean: {log_returns.mean():.6f}, Std: {log_returns.std():.6f}")
    logger.info(f"Log-returns x100 - Mean: {log_returns_split_df['log_return'].mean():.6f}, Std: {log_returns_split_df['log_return'].std():.6f}")

    # Save to CSV and Parquet
    WEIGHTED_LOG_RETURNS_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_returns_split_df.to_csv(WEIGHTED_LOG_RETURNS_SPLIT_FILE, index=False)
    log_returns_split_df.to_parquet(WEIGHTED_LOG_RETURNS_SPLIT_FILE.with_suffix(".parquet"), index=False)
    logger.info(f"Log-returns (x100) saved: {WEIGHTED_LOG_RETURNS_SPLIT_FILE} and .parquet")

    return log_returns_split_df


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


def load_train_test_data(
    train_ratio: float = 0.8,
    file_path: Path | str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Load log returns data and split into train/test sets.

    Args:
        train_ratio: Fraction of data to use for training (default 0.8).
        file_path: Optional path to log returns file. If None, uses default.

    Returns:
        Tuple of (train_series, test_series) where each is a pandas Series
        with log returns indexed by datetime.

    Raises:
        FileNotFoundError: If the log returns file doesn't exist.
    """
    from src.utils.io import load_dataframe

    # Determine file path
    if file_path is None:
        path_to_load = LOG_RETURNS_PARQUET
    else:
        path_to_load = Path(file_path)

    # Load the prepared log returns data
    df = load_dataframe(path_to_load)

    # Ensure we have the required columns
    if "datetime_close" not in df.columns or "log_return" not in df.columns:
        raise ValueError("Log returns file must contain 'datetime_close' and 'log_return' columns")

    # Convert datetime column and set as index
    df["datetime_close"] = pd.to_datetime(df["datetime_close"])
    df = df.set_index("datetime_close").sort_index()

    # Remove any rows with NaN log returns
    df = df.dropna(subset=["log_return"])

    # Split into train/test based on chronological order
    split_idx = int(len(df) * train_ratio)
    train_series = df.iloc[:split_idx]["log_return"]
    test_series = df.iloc[split_idx:]["log_return"]

    logger.info(f"Loaded {len(df)} log returns from {path_to_load}, split into {len(train_series)} train / {len(test_series)} test samples")

    return train_series, test_series
