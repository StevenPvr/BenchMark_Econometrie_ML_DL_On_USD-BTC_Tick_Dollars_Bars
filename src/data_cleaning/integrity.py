"""Data integrity fixing functions."""

from __future__ import annotations

import pandas as pd

from src.utils import get_logger
from src.utils.transforms import remove_duplicates_by_columns

logger = get_logger(__name__)


def _remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows based on (date, tickers) combination.

    Args:
        df: DataFrame to clean.

    Returns:
        Tuple of (cleaned DataFrame, number of duplicates removed).
    """
    cleaned_df, removed = remove_duplicates_by_columns(df, subset=["date", "tickers"])

    if removed > 0:
        logger.info("Removed %d duplicate row(s) on (date, tickers)", removed)

    return cleaned_df, int(removed)


def _get_empty_fix_counters() -> dict[str, int]:
    """Return empty fix counters dictionary.

    Returns:
        Dictionary with zeroed integrity-fix counters.
    """
    return {
        "duplicates_removed": 0,
        "missing_values_filled": 0,
        "missing_dates_filled": 0,
    }


def _fill_missing_data_with_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill missing data in critical columns with zeros.

    Args:
        df: DataFrame to clean.

    Returns:
        Tuple of (cleaned DataFrame, number of missing values filled).
    """
    if df.empty or not {"tickers", "open", "close", "volume"}.issubset(df.columns):
        return df.copy(), 0

    critical_columns = ["open", "close", "volume"]
    missing_before = df[critical_columns].isna().sum().sum()

    # Fill missing values with 0
    df_filled = df.copy()
    df_filled[critical_columns] = df_filled[critical_columns].fillna(0)

    missing_after = df_filled[critical_columns].isna().sum().sum()
    filled_count = int(missing_before - missing_after)

    if filled_count > 0:
        logger.info("Filled %d missing values with zeros in critical columns", filled_count)

    return df_filled, filled_count


def _apply_all_integrity_fixes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply all integrity fix steps sequentially.

    Args:
        df: DataFrame to clean.

    Returns:
        Tuple of (cleaned DataFrame, dictionary with fix counts).
    """
    counters: dict[str, int] = {}

    # Step 1: Remove duplicates
    df, counters["duplicates_removed"] = _remove_duplicates(df)

    # Step 2: Fill missing data in critical columns with zeros
    df, counters["missing_values_filled"] = _fill_missing_data_with_zeros(df)

    # Step 3: Preserve original date coverage (no synthetic rows)
    counters["missing_dates_filled"] = 0

    return df, counters


def apply_basic_integrity_fixes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply basic integrity fixes to the raw dataset.

    The function performs the following operations:

    - Remove duplicate rows based on (date, tickers)
    - Fill missing data in critical columns (open, close, volume) with zeros
    - Preserve original date coverage (no synthetic rows)
    - Sort by tickers, date

    Note:
        All tickers are preserved. Missing data is filled with zeros to maintain
        complete time series for all tickers in the dataset.

    Args:
        df: Raw dataset DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, dictionary with fix counts).
    """
    if df.empty:
        return df.copy(), _get_empty_fix_counters()

    df, counters = _apply_all_integrity_fixes(df)

    # Final sort
    df = df.sort_values(["tickers", "date"]).reset_index(drop=True)

    return df, counters
