"""Tick-level cleaning pipeline for crypto trades (ccxt output).

Implements robust outlier detection methods suitable for financial markets
and dollar bar construction (De Prado methodology).

All outlier detection methods are CAUSAL (use only past data) to prevent
temporal data leakage.

Outlier Detection Methods:
1. MAD (Median Absolute Deviation) - robust to fat-tailed distributions (expanding)
2. Rolling Z-score - adapts to local volatility regimes
3. Volume anomaly detection - filters dust trades and manipulation (expanding)
4. Dollar value filtering - combined price*volume anomalies (expanding)

Reference:
    Huber, P.J. (1981). Robust Statistics.
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
"""

from __future__ import annotations
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from src.constants import (
    OUTLIER_DOLLAR_VALUE_MAD_THRESHOLD,
    OUTLIER_MAD_SCALING_FACTOR,
    OUTLIER_MAD_THRESHOLD,
    OUTLIER_MIN_PERIODS,
    OUTLIER_MIN_VOLUME,
    OUTLIER_ROLLING_WINDOW,
    OUTLIER_ROLLING_ZSCORE_THRESHOLD,
    OUTLIER_VOLUME_MAD_THRESHOLD,
    SYMBOL,
)
from src.path import (
    DATASET_CLEAN_PARQUET,
    DATASET_RAW_PARQUET,
    RAW_PARTITIONS_DIR,
)
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)


@dataclass
class OutlierReport:
    """Summary of outliers detected and removed by each method."""

    total_ticks: int
    removed_mad_price: int = 0
    removed_rolling_zscore: int = 0
    removed_volume_outliers: int = 0
    removed_dollar_value: int = 0
    removed_dust_trades: int = 0
    final_ticks: int = 0

    def log_summary(self) -> None:
        """Log a summary of the outlier detection results."""
        total_removed = (
            self.removed_mad_price
            + self.removed_rolling_zscore
            + self.removed_volume_outliers
            + self.removed_dollar_value
            + self.removed_dust_trades
        )
        pct_removed = (total_removed / self.total_ticks * 100) if self.total_ticks > 0 else 0

        logger.info("=" * 60)
        logger.info("OUTLIER DETECTION SUMMARY (Causal Filters)")
        logger.info("=" * 60)
        logger.info(f"  Initial ticks:          {self.total_ticks:>12,}")
        logger.info(f"  MAD price outliers:     {self.removed_mad_price:>12,}")
        logger.info(f"  Rolling Z-score:        {self.removed_rolling_zscore:>12,}")
        logger.info(f"  Volume outliers:        {self.removed_volume_outliers:>12,}")
        logger.info(f"  Dollar value outliers:  {self.removed_dollar_value:>12,}")
        logger.info(f"  Dust trades (<min vol): {self.removed_dust_trades:>12,}")
        logger.info("-" * 60)
        logger.info(f"  Total removed:          {total_removed:>12,} ({pct_removed:.4f}%)")
        logger.info(f"  Final ticks:            {self.final_ticks:>12,}")
        logger.info("=" * 60)


def _merge_outlier_reports(
    aggregate: OutlierReport | None,
    current: OutlierReport,
) -> OutlierReport:
    """Accumulate OutlierReport metrics across partitions."""
    if aggregate is None:
        return current

    return OutlierReport(
        total_ticks=aggregate.total_ticks + current.total_ticks,
        removed_mad_price=aggregate.removed_mad_price + current.removed_mad_price,
        removed_rolling_zscore=aggregate.removed_rolling_zscore + current.removed_rolling_zscore,
        removed_volume_outliers=aggregate.removed_volume_outliers + current.removed_volume_outliers,
        removed_dollar_value=aggregate.removed_dollar_value + current.removed_dollar_value,
        removed_dust_trades=aggregate.removed_dust_trades + current.removed_dust_trades,
        final_ticks=aggregate.final_ticks + current.final_ticks,
    )


def _load_raw_trades(
    partition_dir: Path | None = None,
    output_path: Path = DATASET_RAW_PARQUET,
    use_cache: bool = False,
) -> pd.DataFrame:
    """Load raw trades from partitioned parquet files in data/raw/copie_raw.

    The merge is streamed with PyArrow to handle very large datasets efficiently:
    partitions are read with a Dataset scanner and written with a ParquetWriter,
    avoiding materializing all Arrow tables in Python before consolidation.

    Args:
        partition_dir: Directory containing part-*.parquet files. Defaults to
            RAW_PARTITIONS_DIR (data/raw/copie_raw).
        output_path: Path where the consolidated parquet should be written.

    Returns:
        Consolidated DataFrame with all raw trades.
    """
    import pyarrow.dataset as ds  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    partitions_root = Path(partition_dir) if partition_dir is not None else RAW_PARTITIONS_DIR
    if not partitions_root.is_dir():
        raise FileNotFoundError(f"Partition directory not found: {partitions_root}")

    # Use flexible pattern to catch files with trailing spaces or variations
    parquet_parts = sorted(
        p for p in partitions_root.glob("part-*")
        if ".parquet" in p.name.lower()
    )
    if not parquet_parts:
        raise ValueError(f"No parquet partition files found in {partitions_root}")

    total_size = sum(f.stat().st_size for f in parquet_parts)
    logger.info(
        "Loading %d partition(s) from %s (%.2f GB)",
        len(parquet_parts),
        partitions_root,
        total_size / (1024**3),
    )

    try:
        dataset = ds.dataset(parquet_parts, format="parquet")
    except OSError as exc:
        if "timed out" in str(exc).lower():
            raise RuntimeError(f"Parquet partitions in {partitions_root} appear to be corrupted") from exc
        raise

    latest_partition_mtime = max(f.stat().st_mtime for f in parquet_parts)
    ensure_output_dir(output_path)

    use_cached = (
        use_cache
        and output_path.exists()
        and output_path.stat().st_mtime >= latest_partition_mtime
    )
    if use_cached:
        logger.info("Using cached consolidated parquet at %s", output_path)
        df_cached = pd.read_parquet(output_path, engine="pyarrow")
        if df_cached.empty:
            raise ValueError("Raw trades dataset is empty")
        return df_cached

    # Clean any stale output artifacts (directory or file) before writing
    for stale_path in (output_path, output_path.with_suffix(".tmp.parquet")):
        if stale_path.exists():
            if stale_path.is_dir():
                shutil.rmtree(stale_path)
            else:
                stale_path.unlink()

    temp_output = output_path.with_suffix(".tmp.parquet")
    logger.info("Streaming merge to %s", output_path)

    try:
        with pq.ParquetWriter(temp_output, dataset.schema, compression="snappy") as writer:
            for batch in dataset.to_batches(batch_size=262_144, use_threads=True):
                writer.write_batch(batch)
        temp_output.replace(output_path)
    except OSError as exc:
        if "timed out" in str(exc).lower():
            raise RuntimeError(f"Parquet partitions in {partitions_root} appear to be corrupted") from exc
        raise

    df = pd.read_parquet(output_path, engine="pyarrow")
    if df.empty:
        raise ValueError("Raw trades dataset is empty")

    logger.info("Consolidation complete. Total rows: %d", len(df))
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate trades using (timestamp, id) keys when available."""
    subset: list[str] = []
    for col in ("timestamp", "id"):
        if col in df.columns:
            subset.append(col)
    if not subset:
        return df

    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        logger.info("Removed %d duplicate trades", removed)
    return df


def _filter_price_outliers(df: pd.DataFrame, max_pct_change: float = 0.05) -> pd.DataFrame:
    """Filter ticks with aberrant price changes (causes extreme log returns).

    DEPRECATED: Use _filter_outliers_robust() instead for production.
    Kept for backward compatibility.

    Removes ticks where the price change vs previous tick exceeds max_pct_change.
    This eliminates erroneous ticks that would create extreme log returns.

    Args:
        df: DataFrame with tick data (must have 'price' column).
        max_pct_change: Maximum allowed price change (default 5% = 0.05).

    Returns:
        Filtered DataFrame without aberrant price ticks.
    """
    if "price" not in df.columns or df.empty:
        return df

    # Calculate percentage change from previous tick
    pct_change = df["price"].pct_change().abs()

    # Keep first tick (NaN pct_change) and ticks within threshold
    mask_valid = pct_change.isna() | (pct_change <= max_pct_change)
    filtered = df.loc[mask_valid].copy()

    removed = len(df) - len(filtered)
    if removed > 0:
        logger.info("Filtered %d ticks with price change > %.1f%%", removed, max_pct_change * 100)

    return filtered


# =============================================================================
# ROBUST OUTLIER DETECTION METHODS (Financial Markets / Dollar Bars)
# =============================================================================


def _filter_mad_price_outliers(
    df: pd.DataFrame,
    price_col: str = "price",
    threshold: float = OUTLIER_MAD_THRESHOLD,
    min_periods: int = OUTLIER_MIN_PERIODS,
) -> tuple[pd.DataFrame, int]:
    """Filter price outliers using expanding MAD (causal, no look-ahead).

    Uses expanding window to compute statistics using only past data,
    preventing temporal data leakage. MAD-based filtering is robust to
    the fat tails typical of crypto prices.

    A tick is considered an outlier if:
        |price - expanding_median| > threshold * expanding_scaled_MAD

    Args:
        df: DataFrame with tick data.
        price_col: Name of price column.
        threshold: Number of MADs from median to consider outlier.
        min_periods: Minimum periods before applying filter (first ticks kept).

    Returns:
        Tuple of (filtered DataFrame, number of outliers removed).
    """
    if price_col not in df.columns or df.empty:
        return df, 0

    prices = df[price_col]

    # Expanding statistics (causal - only uses past data)
    expanding_median = prices.expanding(min_periods=min_periods).median()
    expanding_mad = (
        (prices - expanding_median).abs().expanding(min_periods=min_periods).median()
    )
    scaled_mad = OUTLIER_MAD_SCALING_FACTOR * expanding_mad

    # Avoid division issues - replace zero with NaN
    scaled_mad = scaled_mad.replace(0, np.nan)

    # Identify outliers: |price - median| > threshold * MAD
    # First min_periods ticks are kept (NaN in scaled_mad)
    deviation = (prices - expanding_median).abs()
    mask_valid = (deviation <= threshold * scaled_mad) | scaled_mad.isna()

    filtered = df.loc[mask_valid].reset_index(drop=True)
    removed = len(df) - len(filtered)

    if removed > 0:
        logger.info(
            "MAD filter (causal): removed %d price outliers (threshold=%.1f MADs)",
            removed,
            threshold,
        )

    return filtered, removed


def _filter_rolling_zscore_outliers(
    df: pd.DataFrame,
    price_col: str = "price",
    window: int = OUTLIER_ROLLING_WINDOW,
    threshold: float = OUTLIER_ROLLING_ZSCORE_THRESHOLD,
    min_periods: int = OUTLIER_MIN_PERIODS,
) -> tuple[pd.DataFrame, int]:
    """Filter outliers using rolling Z-score (adapts to local volatility).

    This method adapts to volatility regimes: during high volatility periods,
    larger price moves are tolerated. Uses rolling median and MAD for robustness.

    Args:
        df: DataFrame with tick data.
        price_col: Name of price column.
        window: Rolling window size for computing local statistics.
        threshold: Z-score threshold for outlier detection.
        min_periods: Minimum periods before applying filter.

    Returns:
        Tuple of (filtered DataFrame, number of outliers removed).
    """
    if price_col not in df.columns or len(df) < min_periods:
        return df, 0

    prices = df[price_col]

    # Compute log returns for stationarity
    log_returns = np.log(prices / prices.shift(1))

    # Rolling robust statistics (median and MAD-based std)
    rolling_median = log_returns.rolling(window=window, min_periods=min_periods).median()
    rolling_mad = (
        (log_returns - rolling_median)
        .abs()
        .rolling(window=window, min_periods=min_periods)
        .median()
    )
    rolling_std = OUTLIER_MAD_SCALING_FACTOR * rolling_mad

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    # Compute rolling Z-score
    zscore = (log_returns - rolling_median) / rolling_std

    # Mark as valid: first min_periods ticks OR within threshold
    mask_valid = zscore.isna() | (zscore.abs() <= threshold)

    filtered = df.loc[mask_valid].copy()
    removed = len(df) - len(filtered)

    if removed > 0:
        logger.info(
            "Rolling Z-score filter: removed %d outliers (window=%d, threshold=%.1f sigma)",
            removed,
            window,
            threshold,
        )

    return filtered, removed


def _filter_volume_outliers(
    df: pd.DataFrame,
    volume_col: str = "amount",
    threshold: float = OUTLIER_VOLUME_MAD_THRESHOLD,
    min_volume: float = OUTLIER_MIN_VOLUME,
    min_periods: int = OUTLIER_MIN_PERIODS,
    apply_mad: bool = True,
) -> tuple[pd.DataFrame, int, int]:
    """Filter volume outliers using expanding MAD (causal) and minimum threshold.

    Uses expanding window to compute statistics using only past data,
    preventing temporal data leakage.

    Removes:
    1. Dust trades (volume below minimum threshold)
    2. Extreme volume outliers (manipulation or data errors)

    Args:
        df: DataFrame with tick data.
        volume_col: Name of volume column.
        threshold: Number of MADs for outlier detection.
        min_volume: Minimum valid volume.
        min_periods: Minimum periods before applying MAD filter.
        apply_mad: If False, only dust trades are removed (no MAD outlier filter).

    Returns:
        Tuple of (filtered DataFrame, removed_outliers, removed_dust).
    """
    if volume_col not in df.columns or df.empty:
        return df, 0, 0

    volumes = df[volume_col]

    # Remove dust trades first
    mask_min_volume = volumes >= min_volume
    removed_dust = (~mask_min_volume).sum()

    # Apply MAD filter on remaining
    df_no_dust = df.loc[mask_min_volume].copy()
    if df_no_dust.empty:
        return df_no_dust, 0, removed_dust

    if not apply_mad:
        filtered = df_no_dust.reset_index(drop=True)
        removed_outliers = 0
        if removed_dust > 0:
            logger.info(
                "Volume filter (dust only): removed %d dust trades, skipped MAD filtering",
                removed_dust,
            )
        return filtered, removed_outliers, removed_dust

    volumes = df_no_dust[volume_col]

    # Expanding MAD (causal - only uses past data)
    expanding_median = volumes.expanding(min_periods=min_periods).median()
    expanding_mad = (
        (volumes - expanding_median).abs().expanding(min_periods=min_periods).median()
    )
    scaled_mad = OUTLIER_MAD_SCALING_FACTOR * expanding_mad

    # Avoid division issues - replace zero with NaN
    scaled_mad = scaled_mad.replace(0, np.nan)

    # Only filter extreme high volumes (not low - those are legitimate small trades)
    # First min_periods ticks are kept (NaN in scaled_mad)
    deviation = volumes - expanding_median
    mask_valid = (deviation <= threshold * scaled_mad) | scaled_mad.isna()

    filtered = df_no_dust.loc[mask_valid].reset_index(drop=True)
    removed_outliers = len(df_no_dust) - len(filtered)

    if removed_dust > 0 or removed_outliers > 0:
        logger.info(
            "Volume filter (causal): removed %d dust trades, %d outliers (threshold=%.1f MADs)",
            removed_dust,
            removed_outliers,
            threshold,
        )

    return filtered, removed_outliers, removed_dust


def _filter_dollar_value_outliers(
    df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "amount",
    threshold: float = OUTLIER_DOLLAR_VALUE_MAD_THRESHOLD,
    min_periods: int = OUTLIER_MIN_PERIODS,
    symbol: str = SYMBOL,
) -> tuple[pd.DataFrame, int]:
    """Filter outliers based on dollar value using expanding MAD (causal).

    Uses expanding window to compute statistics using only past data,
    preventing temporal data leakage.

    Dollar value outliers often indicate:
    - Fat-finger errors (wrong price or volume)
    - Market manipulation (wash trading)
    - Data feed glitches

    Args:
        df: DataFrame with tick data.
        price_col: Name of price column.
        volume_col: Name of volume column.
        threshold: Number of MADs for outlier detection.
        min_periods: Minimum periods before applying filter.

    Returns:
        Tuple of (filtered DataFrame, number of outliers removed).
    """
    if price_col not in df.columns or volume_col not in df.columns or df.empty:
        return df, 0

    dollar_values = _compute_dollar_notional(df, price_col, volume_col, symbol)

    # Expanding MAD (causal - only uses past data)
    expanding_median = dollar_values.expanding(min_periods=min_periods).median()
    expanding_mad = (
        (dollar_values - expanding_median)
        .abs()
        .expanding(min_periods=min_periods)
        .median()
    )
    scaled_mad = OUTLIER_MAD_SCALING_FACTOR * expanding_mad

    # Avoid division issues - replace zero with NaN
    scaled_mad = scaled_mad.replace(0, np.nan)

    # Filter extreme dollar values (both high and low can be problematic)
    # First min_periods ticks are kept (NaN in scaled_mad)
    deviation = (dollar_values - expanding_median).abs()
    mask_valid = (deviation <= threshold * scaled_mad) | scaled_mad.isna()

    filtered = df.loc[mask_valid].reset_index(drop=True)
    removed = len(df) - len(filtered)

    if removed > 0:
        logger.info(
            "Dollar value filter (causal): removed %d outliers (threshold=%.1f MADs)",
            removed,
            threshold,
        )

    return filtered, removed


def _filter_outliers_robust(
    df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "amount",
    symbol: str = SYMBOL,
) -> tuple[pd.DataFrame, OutlierReport]:
    """Apply all robust outlier detection methods in sequence.

    All methods are CAUSAL (use only past data via expanding windows)
    to prevent temporal data leakage.

    Minimalist version to reduce over-filtering:
    1. Volume dust removal only (no MAD filter on volume)
    2. MAD-based price outliers with higher threshold
    3. Rolling Z-score skipped
    4. Dollar value outliers skipped

    Returns:
        Tuple of (cleaned DataFrame, OutlierReport with statistics).
    """
    report = OutlierReport(total_ticks=len(df))

    if df.empty:
        report.final_ticks = 0
        return df, report

    price_threshold = OUTLIER_MAD_THRESHOLD * 3.0

    # 1. Volume outliers first (dust-only removal; MAD skipped)
    df, removed_vol, removed_dust = _filter_volume_outliers(
        df,
        volume_col=volume_col,
        threshold=OUTLIER_VOLUME_MAD_THRESHOLD,
        apply_mad=False,
    )
    report.removed_volume_outliers = removed_vol
    report.removed_dust_trades = removed_dust

    # 2. MAD-based price outliers (expanding, causal)
    df, removed = _filter_mad_price_outliers(
        df,
        price_col=price_col,
        threshold=price_threshold,
    )
    report.removed_mad_price = removed

    # Rolling Z-score and dollar-value filters are intentionally skipped to
    # minimize data loss.

    report.final_ticks = len(df)
    return df, report


def _drop_missing_essentials(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """Drop rows missing any essential columns."""
    before = len(df)
    df = df.dropna(subset=list(required))
    removed = before - len(df)
    if removed > 0:
        logger.info("Dropped %d rows with missing required values", removed)
    return df


def _validate_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate that specified columns exist and are numeric.

    Args:
        df: DataFrame to validate.
        columns: List of column names that must be numeric.

    Raises:
        KeyError: If a required column is not found.
        TypeError: If a column is not numeric.
    """
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric, got {df[col].dtype}")


def _strip_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove heavyweight or unused columns before saving."""
    unwanted = ["id", "info", "symbol"]
    return df.drop(columns=[c for c in unwanted if c in df.columns], errors="ignore")


def _persist_clean_dataset(df: pd.DataFrame) -> None:
    """Save cleaned dataset to parquet."""
    ensure_output_dir(DATASET_CLEAN_PARQUET)
    df.to_parquet(DATASET_CLEAN_PARQUET, index=False)
    logger.info("Saved cleaned trades to %s", DATASET_CLEAN_PARQUET)


def _list_partition_files(partition_dir: Path | None = None) -> list[Path]:
    """Return sorted parquet partitions from the raw directory.

    Uses a flexible pattern to handle files with trailing spaces or other
    minor naming variations (e.g., 'part-*.parquet ' with trailing space).
    """
    partitions_root = Path(partition_dir) if partition_dir is not None else RAW_PARTITIONS_DIR
    if not partitions_root.is_dir():
        raise FileNotFoundError(f"Partition directory not found: {partitions_root}")

    # Use flexible pattern to catch files with trailing spaces or variations
    partitions = sorted(
        p for p in partitions_root.glob("part-*")
        if ".parquet" in p.name.lower()
    )
    if not partitions:
        raise ValueError(f"No parquet partition files found in {partitions_root}")

    return partitions


def _clean_partition_dataframe(
    df: pd.DataFrame,
    use_robust_outliers: bool,
    symbol: str,
) -> tuple[pd.DataFrame, OutlierReport | None]:
    """Apply the cleaning pipeline to a single partition."""
    df = _drop_missing_essentials(df, required=("timestamp", "price", "amount"))
    _validate_numeric_columns(df, ["price", "amount"])
    df = _drop_duplicates(df)

    report: OutlierReport | None
    if use_robust_outliers:
        df, report = _filter_outliers_robust(df, symbol=symbol)
    else:
        df = _filter_price_outliers(df, max_pct_change=0.05)
        report = None

    df = _strip_unwanted_columns(df)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df, report


def clean_ticks_data(
    use_robust_outliers: bool = True,
    symbol: str | None = None,
    partition_dir: Path | None = None,
    output_path: Path | None = None,
) -> None:
    """End-to-end cleaning for tick data downloaded via ccxt.

    All outlier detection methods are CAUSAL (use only past data via expanding
    windows) to prevent temporal data leakage.

    Args:
        use_robust_outliers: If True (default), use robust MAD-based outlier
            detection suitable for financial markets and dollar bars.
            If False, use legacy simple percentage-change filter.
        symbol: Trading pair symbol (e.g., "USD/BTC"). Defaults to SYMBOL constant.
        partition_dir: Optional custom directory containing partitioned parquet files.
        output_path: Optional path for the cleaned parquet output.
    """
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    logger.info("Starting tick data cleaning")
    symbol = symbol or SYMBOL
    partitions = _list_partition_files(partition_dir=partition_dir)
    output_path = output_path or DATASET_CLEAN_PARQUET

    ensure_output_dir(output_path)

    temp_output = output_path.with_suffix(".tmp.parquet")
    for stale_path in (output_path, temp_output):
        if stale_path.exists():
            if stale_path.is_dir():
                shutil.rmtree(stale_path)
            else:
                stale_path.unlink()

    aggregated_report: OutlierReport | None = None
    writer: pq.ParquetWriter | None = None
    total_written = 0

    try:
        for idx, partition_file in enumerate(partitions, start=1):
            logger.info(
                "Cleaning partition %d/%d: %s",
                idx,
                len(partitions),
                partition_file.name,
            )
            df_partition = pd.read_parquet(partition_file, engine="pyarrow")
            if df_partition.empty:
                logger.info("Partition %s is empty, skipping", partition_file.name)
                continue

            df_cleaned, report = _clean_partition_dataframe(
                df_partition,
                use_robust_outliers=use_robust_outliers,
                symbol=symbol,
            )

            if use_robust_outliers and report is not None:
                aggregated_report = _merge_outlier_reports(aggregated_report, report)

            if df_cleaned.empty:
                logger.info("Partition %s yielded no rows after cleaning", partition_file.name)
                continue

            table = pa.Table.from_pandas(df_cleaned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_output, table.schema, compression="snappy")
            writer.write_table(table)
            total_written += len(df_cleaned)
    finally:
        if writer is not None:
            writer.close()

    if total_written == 0:
        if temp_output.exists():
            temp_output.unlink()
        raise ValueError("No data remaining after cleaning")

    temp_output.replace(output_path)

    if use_robust_outliers and aggregated_report is not None:
        aggregated_report.final_ticks = total_written
        aggregated_report.log_summary()

    logger.info("Saved cleaned trades to %s", output_path)
USD_LIKE_CODES = {"USD", "USDT", "USDC", "BUSD", "DAI"}


def _compute_dollar_notional(
    df: pd.DataFrame,
    price_col: str,
    volume_col: str,
    symbol: str,
) -> pd.Series:
    """Compute USD-like notional regardless of symbol orientation.

    - If USD-like asset is base (e.g., USD/BTC), notional is the volume column (already USD).
    - If USD-like asset is quote (e.g., BTC/USDT), notional is price * volume.
    - Fallback to price * volume for other pairs.
    """
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
    else:
        base, quote = symbol, ""

    base_upper = base.upper()
    quote_upper = quote.upper()

    if base_upper in USD_LIKE_CODES:
        return pd.Series(df[volume_col])
    if quote_upper in USD_LIKE_CODES:
        return pd.Series(df[price_col] * df[volume_col])

    return pd.Series(df[price_col] * df[volume_col])
