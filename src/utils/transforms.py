"""Data transformation utilities for DataFrames.

This module provides functions for:
- Ticker encoding (stable CRC32-based integer IDs)
- DataFrame filtering (by split)
- Metadata column removal
- Feature/target extraction
"""

from __future__ import annotations

from typing import Iterable, cast
import zlib

import numpy as np
import pandas as pd

from src.config_logging import get_logger

__all__ = [
    "stable_ticker_id",
    "filter_by_split",
    "remove_metadata_columns",
    "extract_features_and_target",
]


def stable_ticker_id(tickers: pd.Series | Iterable[str]) -> pd.Series:
    """Return a stable int32 identifier for each ticker symbol.

    Uses CRC32 of the UTF-8 encoded ticker string, masked to positive int32.
    Deterministic across runs and independent of dataset contents, thus
    safe to compute separately on train and test without mismatched codes.

    Args:
        tickers: Pandas Series or iterable of ticker strings.

    Returns:
        Pandas Series of dtype int32 with one id per input ticker.
    """
    if not isinstance(tickers, pd.Series):
        tickers = pd.Series(list(tickers), dtype="string")
    # Ensure string dtype for robust hashing
    s = tickers.astype("string").fillna("")
    from src.constants import TICKER_CRC32_MASK

    ids = s.map(lambda t: np.int32(zlib.crc32(bytes(t, "utf-8")) & TICKER_CRC32_MASK))
    return ids.astype("int32")


def filter_by_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter dataset by split column if present.

    Args:
        df: Input DataFrame.
        split: Split to filter ('train' or 'test').

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If no data found for split.
    """
    logger = get_logger(__name__)

    if "split" not in df.columns:
        return df

    df_filtered = cast(pd.DataFrame, df[df["split"] == split].copy())
    logger.info(f"Filtered to {split} split: {len(df_filtered)} rows")
    if df_filtered.empty:
        raise ValueError(f"No data found for split '{split}'")

    return df_filtered


def remove_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove metadata columns that should not be passed to the model.

    Removes 'date', 'split', 'ticker', and 'tickers' columns if present.
    Keeps 'ticker_id' as it's a numeric encoded feature for the model.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame without metadata columns.
    """
    columns_to_drop = [col for col in ["date", "split", "ticker", "tickers"] if col in df.columns]
    if columns_to_drop:
        return df.drop(columns=columns_to_drop)
    return df


def extract_features_and_target(
    df: pd.DataFrame, target_column: str = "log_volatility"
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target from dataset.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column. Default is 'log_volatility'.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        ValueError: If target column is missing.
    """
    if target_column not in df.columns:
        raise ValueError(f"Dataset must contain '{target_column}' column")

    X = cast(pd.DataFrame, df.drop(columns=[target_column]))
    y = cast(pd.Series, df[target_column].copy())

    return X, y


def remove_duplicates_by_columns(
    df: pd.DataFrame, subset: list[str], reset_index: bool = True
) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows based on specified columns.

    Args:
        df: DataFrame to clean.
        subset: List of column names to check for duplicates.
        reset_index: Whether to reset the index after dropping duplicates.

    Returns:
        Tuple of (cleaned DataFrame, number of duplicates removed).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 1, 2], 'b': [1, 1, 2]})
        >>> cleaned, removed = remove_duplicates_by_columns(df, ['a', 'b'])
        >>> removed
        1
    """
    if df.empty or not set(subset).issubset(df.columns):
        return df.copy(), 0

    before = len(df)
    df_dedup = df.drop_duplicates(subset=subset)

    if reset_index:
        df_dedup = df_dedup.reset_index(drop=True)

    removed = before - len(df_dedup)
    return df_dedup, removed
