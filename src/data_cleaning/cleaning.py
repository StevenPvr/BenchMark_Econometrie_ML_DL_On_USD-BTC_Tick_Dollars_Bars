"""Tick-level cleaning pipeline for crypto trades (ccxt output).

Implements robust outlier detection methods suitable for financial markets
and dollar bar construction (De Prado methodology).

Outlier Detection Methods:
1. MAD (Median Absolute Deviation) - robust to fat-tailed distributions
2. Rolling Z-score - adapts to local volatility regimes
3. Flash crash/spike detection - identifies transient price anomalies
4. Volume anomaly detection - filters dust trades and manipulation
5. Dollar value filtering - combined price*volume anomalies

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
    OUTLIER_MAX_TICK_RETURN,
    OUTLIER_MIN_PERIODS,
    OUTLIER_MIN_VOLUME,
    OUTLIER_ROLLING_WINDOW,
    OUTLIER_ROLLING_ZSCORE_THRESHOLD,
    OUTLIER_SPIKE_LOOKBACK,
    OUTLIER_SPIKE_REVERSION_THRESHOLD,
    OUTLIER_VOLUME_MAD_THRESHOLD,
)
from src.path import (
    DATASET_CLEAN_PARQUET,
    DATASET_RAW_PARQUET,
)
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)


@dataclass
class OutlierReport:
    """Summary of outliers detected and removed by each method."""

    total_ticks: int
    removed_mad_price: int = 0
    removed_rolling_zscore: int = 0
    removed_flash_spikes: int = 0
    removed_volume_outliers: int = 0
    removed_dollar_value: int = 0
    removed_dust_trades: int = 0
    final_ticks: int = 0

    def log_summary(self) -> None:
        """Log a summary of the outlier detection results."""
        total_removed = (
            self.removed_mad_price
            + self.removed_rolling_zscore
            + self.removed_flash_spikes
            + self.removed_volume_outliers
            + self.removed_dollar_value
            + self.removed_dust_trades
        )
        pct_removed = (total_removed / self.total_ticks * 100) if self.total_ticks > 0 else 0

        logger.info("=" * 60)
        logger.info("OUTLIER DETECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Initial ticks:          {self.total_ticks:>12,}")
        logger.info(f"  MAD price outliers:     {self.removed_mad_price:>12,}")
        logger.info(f"  Rolling Z-score:        {self.removed_rolling_zscore:>12,}")
        logger.info(f"  Flash spikes:           {self.removed_flash_spikes:>12,}")
        logger.info(f"  Volume outliers:        {self.removed_volume_outliers:>12,}")
        logger.info(f"  Dollar value outliers:  {self.removed_dollar_value:>12,}")
        logger.info(f"  Dust trades (<min vol): {self.removed_dust_trades:>12,}")
        logger.info("-" * 60)
        logger.info(f"  Total removed:          {total_removed:>12,} ({pct_removed:.4f}%)")
        logger.info(f"  Final ticks:            {self.final_ticks:>12,}")
        logger.info("=" * 60)


def _load_raw_trades(path: Path = DATASET_RAW_PARQUET) -> pd.DataFrame:
    """Load raw trades parquet file.

    If the path is a directory (partitioned dataset), it consolidates the partitions
    iteratively into a single DataFrame, saves it as a CSV and a single Parquet file
    (replacing the directory), and returns the consolidated DataFrame.
    """
    if path.is_dir():
        logger.info("Detected partitioned dataset at %s. Consolidating...", path)
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {path}")

        # Load first partition
        logger.info("Loading partition 1/%d: %s", len(parquet_files), parquet_files[0].name)
        df = pd.read_parquet(parquet_files[0])

        # Iteratively merge remaining partitions
        for i, file_path in enumerate(parquet_files[1:], start=2):
            logger.info("Merging partition %d/%d: %s", i, len(parquet_files), file_path.name)
            df_part = pd.read_parquet(file_path)
            df = pd.concat([df, df_part], ignore_index=True)

        logger.info("Consolidation complete. Total rows: %d", len(df))

        # Replace directory with single parquet file
        # We write to a temp file first, then remove dir, then rename
        temp_parquet = path.with_suffix(".parquet.tmp")
        logger.info("Saving consolidated raw dataset to temporary file %s", temp_parquet)
        df.to_parquet(temp_parquet, index=False)

        logger.info("Removing partition directory and renaming temporary file...")
        shutil.rmtree(path)
        temp_parquet.rename(path)
        logger.info("Saved consolidated raw dataset to %s", path)

        return df

    # Normal file loading with error handling for corrupted files
    try:
        df = pd.read_parquet(path)
    except (OSError, TimeoutError) as e:
        if "Operation timed out" in str(e) or "timeout" in str(e).lower():
            raise RuntimeError(
                f"Parquet file {path} appears to be corrupted or inaccessible (timeout error). "
                f"Please remove the file and re-run data fetching to regenerate it. "
                f"Error: {e}"
            ) from e
        else:
            raise

    if df.empty:
        raise ValueError("Raw trades dataset is empty")
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


def _compute_mad(series: pd.Series) -> tuple[float, float]:
    """Compute Median Absolute Deviation (MAD) for robust outlier detection.

    MAD is more robust than standard deviation for fat-tailed distributions
    typical of financial returns. Uses the consistency factor 1.4826 to
    make it comparable to standard deviation for normal distributions.

    Reference: Huber, P.J. (1981). Robust Statistics.

    Args:
        series: Pandas Series of values.

    Returns:
        Tuple of (median, scaled_mad) where scaled_mad = 1.4826 * MAD.
    """
    median = series.median()
    mad = (series - median).abs().median()
    scaled_mad = OUTLIER_MAD_SCALING_FACTOR * mad
    return median, scaled_mad


def _filter_mad_price_outliers(
    df: pd.DataFrame,
    price_col: str = "price",
    threshold: float = OUTLIER_MAD_THRESHOLD,
) -> tuple[pd.DataFrame, int]:
    """Filter price outliers using Median Absolute Deviation (MAD).

    MAD-based filtering is robust to the fat tails typical of crypto prices.
    A tick is considered an outlier if:
        |price - median| > threshold * scaled_MAD

    Args:
        df: DataFrame with tick data.
        price_col: Name of price column.
        threshold: Number of MADs from median to consider outlier.

    Returns:
        Tuple of (filtered DataFrame, number of outliers removed).
    """
    if price_col not in df.columns or df.empty:
        return df, 0

    prices = df[price_col]
    median, scaled_mad = _compute_mad(prices)

    if scaled_mad < 1e-10:
        logger.warning("MAD is near zero, skipping MAD price filter")
        return df, 0

    # Identify outliers: |price - median| > threshold * MAD
    deviation = (prices - median).abs()
    mask_valid = deviation <= threshold * scaled_mad

    filtered = df.loc[mask_valid].copy()
    removed = len(df) - len(filtered)

    if removed > 0:
        logger.info(
            "MAD filter: removed %d price outliers (threshold=%.1f MADs, median=%.2f)",
            removed,
            threshold,
            median,
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


def _filter_flash_spikes(
    df: pd.DataFrame,
    price_col: str = "price",
    max_tick_return: float = OUTLIER_MAX_TICK_RETURN,
    lookback: int = OUTLIER_SPIKE_LOOKBACK,
    reversion_threshold: float = OUTLIER_SPIKE_REVERSION_THRESHOLD,
) -> tuple[pd.DataFrame, int]:
    """Detect and filter flash crashes/spikes that revert quickly.

    A flash spike is characterized by:
    1. Large price move exceeding max_tick_return (e.g., 15%)
    2. Significant reversion within lookback ticks

    These are typically data errors or momentary liquidity gaps.

    Args:
        df: DataFrame with tick data.
        price_col: Name of price column.
        max_tick_return: Maximum allowed log-return between ticks.
        lookback: Number of ticks to check for reversion.
        reversion_threshold: Fraction of move that must revert to classify as spike.

    Returns:
        Tuple of (filtered DataFrame, number of spikes removed).
    """
    if price_col not in df.columns or len(df) < lookback + 2:
        return df, 0

    prices = df[price_col].values
    n = len(prices)
    mask_valid = np.ones(n, dtype=bool)

    # Compute log returns
    log_returns = np.zeros(n)
    log_returns[1:] = np.log(prices[1:] / prices[:-1])

    for i in range(1, n - lookback):
        abs_return = abs(log_returns[i])

        if abs_return > max_tick_return:
            # Check for reversion in next lookback ticks
            future_prices = prices[i + 1 : i + 1 + lookback]
            pre_spike_price = prices[i - 1]
            spike_price = prices[i]

            # Calculate how much the price reverts toward pre-spike level
            spike_magnitude = spike_price - pre_spike_price
            if abs(spike_magnitude) > 1e-10:
                max_reversion = 0.0
                for future_price in future_prices:
                    reversion = (spike_price - future_price) / spike_magnitude
                    max_reversion = max(max_reversion, reversion)

                if max_reversion >= reversion_threshold:
                    # This is a flash spike - mark the spike tick as invalid
                    mask_valid[i] = False

    filtered = df.loc[mask_valid].copy()
    removed = len(df) - len(filtered)

    if removed > 0:
        logger.info(
            "Flash spike filter: removed %d spikes (max_return=%.1f%%, lookback=%d)",
            removed,
            max_tick_return * 100,
            lookback,
        )

    return filtered, removed


def _filter_volume_outliers(
    df: pd.DataFrame,
    volume_col: str = "amount",
    threshold: float = OUTLIER_VOLUME_MAD_THRESHOLD,
    min_volume: float = OUTLIER_MIN_VOLUME,
) -> tuple[pd.DataFrame, int, int]:
    """Filter volume outliers using MAD and minimum volume threshold.

    Removes:
    1. Dust trades (volume below minimum threshold)
    2. Extreme volume outliers (manipulation or data errors)

    Args:
        df: DataFrame with tick data.
        volume_col: Name of volume column.
        threshold: Number of MADs for outlier detection.
        min_volume: Minimum valid volume.

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
    df_no_dust = df.loc[mask_min_volume]
    if df_no_dust.empty:
        return df_no_dust, 0, removed_dust

    median, scaled_mad = _compute_mad(df_no_dust[volume_col])

    if scaled_mad < 1e-10:
        return df_no_dust, 0, removed_dust

    # Only filter extreme high volumes (not low - those are legitimate small trades)
    deviation = df_no_dust[volume_col] - median
    mask_valid = deviation <= threshold * scaled_mad

    filtered = df_no_dust.loc[mask_valid].copy()
    removed_outliers = len(df_no_dust) - len(filtered)

    if removed_dust > 0 or removed_outliers > 0:
        logger.info(
            "Volume filter: removed %d dust trades, %d outliers (threshold=%.1f MADs)",
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
) -> tuple[pd.DataFrame, int]:
    """Filter outliers based on dollar value (price * volume).

    Dollar value outliers often indicate:
    - Fat-finger errors (wrong price or volume)
    - Market manipulation (wash trading)
    - Data feed glitches

    Args:
        df: DataFrame with tick data.
        price_col: Name of price column.
        volume_col: Name of volume column.
        threshold: Number of MADs for outlier detection.

    Returns:
        Tuple of (filtered DataFrame, number of outliers removed).
    """
    if price_col not in df.columns or volume_col not in df.columns or df.empty:
        return df, 0

    dollar_values = df[price_col] * df[volume_col]
    median, scaled_mad = _compute_mad(dollar_values)

    if scaled_mad < 1e-10:
        return df, 0

    # Filter extreme dollar values (both high and low can be problematic)
    deviation = (dollar_values - median).abs()
    mask_valid = deviation <= threshold * scaled_mad

    filtered = df.loc[mask_valid].copy()
    removed = len(df) - len(filtered)

    if removed > 0:
        logger.info(
            "Dollar value filter: removed %d outliers (threshold=%.1f MADs, median=$%.2f)",
            removed,
            threshold,
            median,
        )

    return filtered, removed


def _filter_outliers_robust(
    df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "amount",
) -> tuple[pd.DataFrame, OutlierReport]:
    """Apply all robust outlier detection methods in sequence.

    Order of operations (designed to minimize data loss):
    1. Dust trades removal (minimum volume)
    2. MAD-based price outliers (global anomalies)
    3. Rolling Z-score (local volatility-adjusted)
    4. Flash spike detection (transient errors)
    5. Volume outliers (MAD-based)
    6. Dollar value outliers (combined anomalies)

    Args:
        df: DataFrame with tick data.
        price_col: Name of price column.
        volume_col: Name of volume column.

    Returns:
        Tuple of (cleaned DataFrame, OutlierReport with statistics).
    """
    report = OutlierReport(total_ticks=len(df))

    if df.empty:
        report.final_ticks = 0
        return df, report

    # 1. MAD-based price outliers (global)
    df, removed = _filter_mad_price_outliers(df, price_col=price_col)
    report.removed_mad_price = removed

    # 2. Rolling Z-score (local volatility-adjusted)
    df, removed = _filter_rolling_zscore_outliers(df, price_col=price_col)
    report.removed_rolling_zscore = removed

    # 3. Flash spike detection
    df, removed = _filter_flash_spikes(df, price_col=price_col)
    report.removed_flash_spikes = removed

    # 4. Volume outliers (including dust trades)
    df, removed_vol, removed_dust = _filter_volume_outliers(df, volume_col=volume_col)
    report.removed_volume_outliers = removed_vol
    report.removed_dust_trades = removed_dust

    # 5. Dollar value outliers
    df, removed = _filter_dollar_value_outliers(
        df, price_col=price_col, volume_col=volume_col
    )
    report.removed_dollar_value = removed

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


def _strip_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove heavyweight or unused columns before saving."""
    unwanted = ["id", "info", "symbol"]
    return df.drop(columns=[c for c in unwanted if c in df.columns], errors="ignore")


def _persist_clean_dataset(df: pd.DataFrame) -> None:
    """Save cleaned dataset to parquet."""
    ensure_output_dir(DATASET_CLEAN_PARQUET)
    df.to_parquet(DATASET_CLEAN_PARQUET, index=False)
    logger.info("Saved cleaned trades to %s", DATASET_CLEAN_PARQUET)


def clean_ticks_data(use_robust_outliers: bool = True) -> None:
    """End-to-end cleaning for tick data downloaded via ccxt.

    Args:
        use_robust_outliers: If True (default), use robust MAD-based outlier
            detection suitable for financial markets and dollar bars.
            If False, use legacy simple percentage-change filter.
    """
    logger.info("Starting tick data cleaning")
    df = _load_raw_trades()

    df = _drop_missing_essentials(df, required=("timestamp", "price", "amount"))
    df = _drop_duplicates(df)

    if use_robust_outliers:
        # Robust outlier detection for financial markets / dollar bars
        df, outlier_report = _filter_outliers_robust(df)
        outlier_report.log_summary()
    else:
        # Legacy simple filter (kept for backward compatibility)
        df = _filter_price_outliers(df, max_pct_change=0.05)

    df = _strip_unwanted_columns(df)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("No data remaining after cleaning")

    _persist_clean_dataset(df)

