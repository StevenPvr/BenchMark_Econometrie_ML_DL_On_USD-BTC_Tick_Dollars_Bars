"""Tick-level cleaning pipeline for crypto trades (ccxt output)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd # type: ignore[import-untyped]

from src.path import DATASET_CLEAN_CSV, DATASET_CLEAN_PARQUET, DATASET_RAW_PARQUET
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)


def _load_raw_trades(path: Path = DATASET_RAW_PARQUET) -> pd.DataFrame:
    """Load raw trades parquet file."""
    df = pd.read_parquet(path)
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


def _filter_volume_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Filter obvious volume outliers using a high quantile cap."""
    if "amount" not in df.columns or df.empty:
        return df

    quantile_9999 = df["amount"].quantile(0.9999) if not df.empty else float("inf")
    mask_valid = (df["amount"] > 0) & (df["amount"] <= quantile_9999)
    filtered = df.loc[mask_valid].copy()
    removed = len(df) - len(filtered)
    if removed > 0:
        logger.info("Filtered %d outlier trades above 99.99%% quantile", removed)
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
    """Save cleaned dataset to parquet + CSV."""
    ensure_output_dir(DATASET_CLEAN_PARQUET)
    df.to_parquet(DATASET_CLEAN_PARQUET, index=False)
    df.to_csv(DATASET_CLEAN_CSV, index=False, lineterminator="\n")
    logger.info("Saved cleaned trades to %s", DATASET_CLEAN_PARQUET)
    logger.info("Saved cleaned trades to %s", DATASET_CLEAN_CSV)


def clean_ticks_data() -> None:
    """End-to-end cleaning for tick data downloaded via ccxt."""
    logger.info("Starting tick data cleaning")
    df = _load_raw_trades()

    df = _drop_missing_essentials(df, required=("timestamp", "price", "amount"))
    df = _drop_duplicates(df)
    df = _filter_volume_outliers(df)
    df = _strip_unwanted_columns(df)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("No data remaining after cleaning")

    _persist_clean_dataset(df)

