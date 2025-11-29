"""Tick-level cleaning pipeline for crypto trades (ccxt output)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd # type: ignore[import-untyped]

from src.path import (
    DATASET_CLEAN_PARQUET,
    DATASET_RAW_PARQUET,
)
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)


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


def clean_ticks_data() -> None:
    """End-to-end cleaning for tick data downloaded via ccxt."""
    logger.info("Starting tick data cleaning")
    df = _load_raw_trades()

    df = _drop_missing_essentials(df, required=("timestamp", "price", "amount"))
    df = _drop_duplicates(df)
    df = _filter_price_outliers(df, max_pct_change=0.05)  # Filter >5% price jumps
    df = _strip_unwanted_columns(df)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("No data remaining after cleaning")

    _persist_clean_dataset(df)

