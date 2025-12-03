"""Pipeline functions for dollar bars generation.

This module provides end-to-end pipelines for generating dollar bars,
including memory-efficient batch processing for large datasets.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from src.data_preparation.dollar_bars import compute_dollar_bars
from src.path import DOLLAR_IMBALANCE_BARS_PARQUET
from src.utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "prepare_dollar_bars",
    "run_dollar_bars_pipeline",
    "run_dollar_bars_pipeline_batch",
    "add_log_returns_to_bars_file",
]


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
        adaptive: Use adaptive EWMA threshold.
        threshold_bounds: Optional bounds when adaptive=True.
        calibration_fraction: Fraction of earliest ticks used to calibrate T_0.
        include_incomplete_final: Keep trailing partial bar if True.
        exclude_calibration_prefix: Exclude bars from calibration period.

    Returns:
        DataFrame with computed dollar bars.

    Raises:
        FileNotFoundError: If input parquet does not exist.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {parquet_path}")

    logger.info("Loading tick data from %s", parquet_path)
    df_ticks = pd.read_parquet(parquet_path)
    logger.info("Loaded %d ticks", len(df_ticks))

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
        logger.info("Saved %d bars to %s", len(df_bars), output_parquet)

    return df_bars


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
    """Run the complete dollar bars pipeline with default paths.

    Args:
        target_ticks_per_bar: Target ticks per bar (RECOMMENDED).
        target_num_bars: Alternative: target number of bars (legacy).
        input_parquet: Path to input tick data. If None, uses default.
        output_parquet: Path to save bars parquet. If None, uses default.
        threshold: Fixed dollar threshold (overrides target params).
        adaptive: Use adaptive EWMA threshold.
        threshold_bounds: Optional (min_mult, max_mult) for adaptive bounds.
        calibration_fraction: Fraction of earliest ticks used to calibrate T_0.
        include_incomplete_final: Keep trailing partial bar if True.
        exclude_calibration_prefix: Exclude bars from calibration period.

    Returns:
        DataFrame with generated dollar bars.

    Raises:
        FileNotFoundError: If input data not found.
    """
    if input_parquet is None:
        from src.path import DATASET_CLEAN_PARQUET
        input_parquet = DATASET_CLEAN_PARQUET
        if not Path(input_parquet).exists():
            raise FileNotFoundError(
                f"No cleaned tick data found at {input_parquet}. "
                "Ensure data_cleaning has been run first."
            )

    if output_parquet is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_parquet = (
            DOLLAR_IMBALANCE_BARS_PARQUET.parent
            / f"dollar_imbalance_bars_{timestamp}.parquet"
        )

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
    logger.info("Pipeline complete: %d dollar bars generated", len(df_bars))
    logger.info("=" * 70)

    return df_bars


@dataclass
class _BarAccumulatorState:
    """State for dollar bar accumulation across batches."""

    cum_dollar: float = 0.0
    cum_volume: float = 0.0
    cum_pv: float = 0.0
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
    """Process a batch of ticks and return completed bars + updated state."""
    completed_bars: list[dict] = []
    n = len(timestamps)

    if n == 0:
        return completed_bars, state

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
        state.cum_dollar += dollar_value
        state.cum_volume += volume
        state.cum_pv += price * volume
        state.n_ticks += 1

        if price > state.bar_high:
            state.bar_high = price
        if price < state.bar_low:
            state.bar_low = price

        if state.cum_dollar >= state.current_threshold:
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

            if adaptive:
                state.ema_dollar = (
                    ema_alpha * state.cum_dollar
                    + (1.0 - ema_alpha) * state.ema_dollar
                )
                state.current_threshold = max(
                    min_threshold, min(max_threshold, state.ema_dollar)
                )

            state.bar_idx += 1
            state.cum_dollar = 0.0
            state.cum_volume = 0.0
            state.cum_pv = 0.0
            state.n_ticks = 0

            if i + 1 < n:
                state.bar_open = float(prices[i + 1])
                state.bar_high = float(prices[i + 1])
                state.bar_low = float(prices[i + 1])
                state.bar_ts_open = int(timestamps[i + 1])
            else:
                state.initialized = False

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
        target_ticks_per_bar: Target ticks per bar. Default 100.
        output_parquet: Path to save bars parquet. If None, uses default.
        batch_size: Number of ticks per batch. Default 20M.
        adaptive: Use adaptive EWMA threshold.
        threshold_bounds: Optional (min_mult, max_mult) for adaptive bounds.
        calibration_fraction: Fraction of first batch used to calibrate.
        include_incomplete_final: Keep trailing partial bar if True.

    Returns:
        DataFrame with generated dollar bars.

    Raises:
        FileNotFoundError: If input parquet does not exist.
        ValueError: If no bars were generated.
    """
    import gc

    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    input_path = Path(input_parquet)
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    if target_ticks_per_bar is None:
        target_ticks_per_bar = 100

    if output_parquet is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_parquet = (
            DOLLAR_IMBALANCE_BARS_PARQUET.parent / f"dollar_bars_{timestamp}.parquet"
        )

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(input_path)
    total_ticks = parquet_file.metadata.num_rows
    n_batches = (total_ticks + batch_size - 1) // batch_size

    logger.info(
        "BATCH PROCESSING: %d ticks in %d batches (target %d ticks/bar)",
        total_ticks, n_batches, target_ticks_per_bar
    )

    ema_span = 100
    ema_alpha = 2.0 / (ema_span + 1.0)

    state = _BarAccumulatorState()
    parquet_writer: pq.ParquetWriter | None = None
    total_bars = 0

    # Calibrate threshold on first batch
    first_batch = next(parquet_file.iter_batches(batch_size=batch_size))
    df_cal = first_batch.to_pandas()
    cal_size = max(1, int(len(df_cal) * calibration_fraction))
    prices_cal = df_cal["price"].values[:cal_size].astype(np.float64)
    volumes_cal = df_cal["amount"].values[:cal_size].astype(np.float64)
    dollar_cal = prices_cal * volumes_cal

    target_bars_cal = max(1, cal_size // target_ticks_per_bar)
    initial_threshold = float(np.sum(dollar_cal)) / target_bars_cal
    state.current_threshold = initial_threshold
    state.ema_dollar = initial_threshold

    min_threshold, max_threshold = (0.0, np.inf) if threshold_bounds is None else (
        initial_threshold * threshold_bounds[0], initial_threshold * threshold_bounds[1]
    )
    logger.info("Calibrated threshold: $%.2f (~%d expected bars)",
                initial_threshold, total_ticks // target_ticks_per_bar)

    del df_cal, prices_cal, volumes_cal, dollar_cal
    gc.collect()
    parquet_file = pq.ParquetFile(input_path)

    for batch_num, batch in enumerate(
        parquet_file.iter_batches(batch_size=batch_size), 1
    ):
        df_batch = batch.to_pandas()

        if "timestamp" in df_batch.columns:
            df_batch = df_batch.sort_values("timestamp").reset_index(drop=True)
            if pd.api.types.is_datetime64_any_dtype(df_batch["timestamp"]):
                timestamps = (df_batch["timestamp"].astype("int64") // 10**6).values
            else:
                timestamps = df_batch["timestamp"].values.astype(np.int64)
        else:
            raise ValueError("timestamp column required")

        prices = df_batch["price"].values.astype(np.float64)
        volumes = df_batch["amount"].values.astype(np.float64)

        completed_bars, state = _process_batch_with_state(
            timestamps, prices, volumes, state,
            ema_alpha, min_threshold, max_threshold, adaptive
        )

        if completed_bars:
            df_bars = pd.DataFrame(completed_bars)
            df_bars["duration_sec"] = (
                df_bars["timestamp_close"] - df_bars["timestamp_open"]
            ) / 1000.0
            df_bars["datetime_open"] = pd.to_datetime(
                df_bars["timestamp_open"], unit="ms", utc=True
            )
            df_bars["datetime_close"] = pd.to_datetime(
                df_bars["timestamp_close"], unit="ms", utc=True
            )

            table = pa.Table.from_pandas(df_bars, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    output_path, table.schema, compression="snappy"
                )
            parquet_writer.write_table(table)
            total_bars += len(df_bars)

            del df_bars, completed_bars

        del df_batch, timestamps, prices, volumes, batch
        gc.collect()

        logger.info("  Batch %d/%d: %d bars generated", batch_num, n_batches, total_bars)

    # Handle incomplete final bar
    if include_incomplete_final and state.n_ticks > 0:
        final_bar = {
            "bar_id": state.bar_idx,
            "timestamp_open": state.bar_ts_open,
            "timestamp_close": state.bar_ts_open,
            "open": state.bar_open,
            "high": state.bar_high,
            "low": state.bar_low,
            "close": state.bar_low,
            "volume": state.cum_volume,
            "cum_dollar_value": state.cum_dollar,
            "vwap": (
                state.cum_pv / state.cum_volume
                if state.cum_volume > 0
                else state.bar_open
            ),
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

    logger.info(
        "Pipeline complete: %d bars (avg %.1f ticks/bar) -> %s",
        total_bars, total_ticks / total_bars, output_path
    )

    return pd.read_parquet(output_path)


def add_log_returns_to_bars_file(
    bars_parquet: Path,
    price_col: str = "close",
) -> None:
    """Compute log returns and persist inside the dollar_bars dataset.

    Args:
        bars_parquet: Path to the dollar bars parquet file.
        price_col: Name of the price column to use.

    Raises:
        FileNotFoundError: If bars_parquet does not exist.
    """
    if not bars_parquet.exists():
        raise FileNotFoundError(f"dollar_bars parquet not found at {bars_parquet}")

    df_bars = pd.read_parquet(bars_parquet)

    prices = df_bars[price_col]
    log_returns = np.log(prices / prices.shift(1))
    df_bars["log_return"] = log_returns
    df_bars = df_bars.dropna(subset=["log_return"])

    logger.info(
        "Log-returns stats (natural) - Mean: %.6f, Std: %.6f",
        df_bars["log_return"].mean(),
        df_bars["log_return"].std(),
    )

    bars_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_bars.to_parquet(bars_parquet, index=False)
    logger.info("Saved dollar_bars with log_return to: %s", bars_parquet)
