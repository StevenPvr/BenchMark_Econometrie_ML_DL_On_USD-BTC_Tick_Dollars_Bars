"""Dollar Bars computation API.

This module provides the main public interface for computing Dollar Bars
following De Prado's methodology from "Advances in Financial Machine Learning".
"""

from __future__ import annotations

import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from src.data_preparation.numba_core import (
    _accumulate_dollar_bars_adaptive,
    _accumulate_dollar_bars_fixed,
    _compute_threshold_from_target_bars,
)
from src.utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "compute_dollar_bars",
    "generate_dollar_bars",
    "create_empty_bars_df",
]


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


def create_empty_bars_df() -> pd.DataFrame:
    """Create an empty DataFrame with the correct dollar bars schema.

    Returns:
        Empty DataFrame with all expected columns and correct dtypes.
    """
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

    Args:
        df: DataFrame with tick-by-tick data.
        target_ticks_per_bar: Target number of ticks per bar (RECOMMENDED).
        target_num_bars: Alternative: target number of bars (legacy).
        timestamp_col: Name of timestamp column (int64 ms or datetime).
        price_col: Name of price column.
        volume_col: Name of volume column.
        threshold: Override with fixed dollar threshold T.
        ema_span: EMA span for adaptive threshold (default 100 bars).
        adaptive: If True, use adaptive EWMA threshold.
        threshold_bounds: Optional (min, max) multipliers for adaptive bounds.
        calibration_fraction: Fraction of dataset used to calibrate T_0.
        include_incomplete_final: If True, keeps trailing partial bar.
        exclude_calibration_prefix: If True, excludes calibration bars.

    Returns:
        DataFrame with dollar bars (OHLCV + metadata).

    Raises:
        ValueError: If required columns are missing or parameters invalid.
    """
    # Validate columns
    required_cols = {timestamp_col, price_col, volume_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    _validate_dollar_bars_params(ema_span, threshold_bounds, calibration_fraction)

    if df.empty:
        logger.warning("Empty DataFrame provided, returning empty bars")
        return create_empty_bars_df()

    # Ensure deterministic ordering
    if not df[timestamp_col].is_monotonic_increasing:
        logger.info("Sorting ticks by %s for dollar bar construction", timestamp_col)
        df = df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)

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
        logger.warning("Found %d non-positive prices", n_invalid)

    if np.any(volumes < 0):
        n_invalid = int(np.sum(volumes < 0))
        logger.warning("Found %d negative volumes", n_invalid)

    dollar_values = prices * volumes
    ema_alpha = 2.0 / (ema_span + 1.0)
    calibration_end_timestamp: int | None = None

    # Convert target_ticks_per_bar to target_num_bars
    n_ticks = len(df)
    if target_ticks_per_bar is not None:
        if target_ticks_per_bar <= 0:
            raise ValueError("target_ticks_per_bar must be positive")
        target_num_bars = max(1, n_ticks // target_ticks_per_bar)
        logger.info(
            "De Prado methodology: %d ticks/bar -> %d target bars from %d ticks",
            target_ticks_per_bar, target_num_bars, n_ticks
        )

    # Determine mode and compute bars
    if threshold is not None:
        logger.info("Using FIXED threshold: %.2f USD", threshold)
        result = _accumulate_dollar_bars_fixed(
            timestamps, prices, volumes, threshold, include_incomplete_final
        )

    elif target_num_bars is not None:
        if target_num_bars <= 0:
            raise ValueError("target_num_bars must be positive when threshold is None")

        calibration_size = len(dollar_values)
        if 0 < calibration_fraction < 1.0:
            calibration_size = max(1, int(len(dollar_values) * calibration_fraction))

        target_bars_cal = target_num_bars
        if 0 < calibration_fraction < 1.0:
            target_bars_cal = max(
                1, int(target_num_bars * calibration_size / len(dollar_values))
            )

        calibration_values = dollar_values[:calibration_size]
        initial_threshold = float(_compute_threshold_from_target_bars(
            calibration_values, target_bars_cal
        ))

        if exclude_calibration_prefix and calibration_size < len(timestamps):
            calibration_end_timestamp = int(timestamps[calibration_size - 1])

        if threshold_bounds is None:
            min_threshold = 0.0
            max_threshold = np.inf
        else:
            min_mult, max_mult = threshold_bounds
            min_threshold = initial_threshold * min_mult
            max_threshold = initial_threshold * max_mult

        logger.info(
            "Calibrated T_0: %.2f USD (prefix: %d/%d ticks)",
            initial_threshold, calibration_size, len(df)
        )

        if adaptive:
            result = _accumulate_dollar_bars_adaptive(
                timestamps, prices, volumes, initial_threshold, ema_alpha,
                min_threshold, max_threshold, include_incomplete_final
            )
        else:
            result = _accumulate_dollar_bars_fixed(
                timestamps, prices, volumes, initial_threshold, include_incomplete_final
            )
    else:
        raise ValueError(
            "Either threshold, target_ticks_per_bar, or target_num_bars must be provided"
        )

    # Unpack and build DataFrame
    (
        bar_ids, ts_open, ts_close, opens, highs, lows, closes,
        bar_volumes, bar_dollars, bar_vwaps, tick_counts, thresholds_arr, num_bars
    ) = result

    if num_bars == 0:
        logger.warning("No bars formed (threshold too high or insufficient data)")
        return create_empty_bars_df()

    # Trim arrays
    bar_ids = bar_ids[:num_bars]
    ts_open = ts_open[:num_bars]
    ts_close = ts_close[:num_bars]

    duration_sec = (ts_close - ts_open) / 1000.0
    datetime_open = pd.to_datetime(ts_open, unit="ms", utc=True)
    datetime_close = pd.to_datetime(ts_close, unit="ms", utc=True)

    df_bars = pd.DataFrame({
        "bar_id": bar_ids,
        "timestamp_open": ts_open,
        "timestamp_close": ts_close,
        "datetime_open": datetime_open,
        "datetime_close": datetime_close,
        "open": opens[:num_bars],
        "high": highs[:num_bars],
        "low": lows[:num_bars],
        "close": closes[:num_bars],
        "volume": bar_volumes[:num_bars],
        "cum_dollar_value": bar_dollars[:num_bars],
        "vwap": bar_vwaps[:num_bars],
        "n_ticks": tick_counts[:num_bars],
        "threshold_used": thresholds_arr[:num_bars],
        "duration_sec": duration_sec,
    })

    # Exclude calibration prefix if requested
    if exclude_calibration_prefix and calibration_end_timestamp is not None:
        original_count = len(df_bars)
        df_bars = df_bars[df_bars["timestamp_close"] > calibration_end_timestamp].copy()
        df_bars = df_bars.reset_index(drop=True)
        df_bars["bar_id"] = np.arange(len(df_bars))
        excluded = original_count - len(df_bars)
        logger.info("Excluded %d bars from calibration prefix", excluded)

    if len(df_bars) == 0:
        logger.warning("No bars remaining after calibration prefix exclusion")
        return create_empty_bars_df()

    avg_ticks = len(df) / len(df_bars)
    logger.info(
        "Generated %d dollar bars from %d ticks (avg %.1f ticks/bar)",
        len(df_bars), len(df), avg_ticks
    )

    return pd.DataFrame(df_bars)


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
        target_num_bars=0,
        threshold=threshold,
        timestamp_col=timestamp_col,
        price_col=price_col,
        volume_col=volume_col,
        adaptive=False,
    )


# Keep private alias for backward compatibility with tests
_create_empty_bars_df = create_empty_bars_df
_validate_dollar_bars_params = _validate_dollar_bars_params
