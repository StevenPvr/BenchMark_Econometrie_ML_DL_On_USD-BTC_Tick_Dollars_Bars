"""Validation utilities for DataFrames, files, and parameters.

This module provides validation functions for:
- DataFrame validation (non-empty, required columns)
- File existence validation
- Split validation (train/test)
- Parameter validation (train ratio, series)
- Ticker validation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = [
    "validate_dataframe_not_empty",
    "validate_required_columns",
    "validate_ticker_id",
    "validate_file_exists",
    "validate_train_ratio",
    "validate_series",
    "has_both_splits",
]


def has_both_splits(df: pd.DataFrame, split_col: str = "split") -> bool:
    """Check if DataFrame has both train and test splits.

    Args:
        df: DataFrame with split column.
        split_col: Name of the split column. Default is 'split'.

    Returns:
        True if both train and test splits exist.
    """
    if split_col not in df.columns:
        return False
    has_train = (df[split_col] == "train").any()
    has_test = (df[split_col] == "test").any()
    return has_train and has_test


def validate_ticker_id(df: pd.DataFrame) -> None:
    """Validate that ticker_id column exists if ticker column is present.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If ticker exists but ticker_id is missing.
    """
    if "ticker" in df.columns and "ticker_id" not in df.columns:
        raise ValueError(
            "Dataset must contain a precomputed 'ticker_id' column. "
            "Re-run lightgbm data preparation pipeline to generate encoded datasets."
        )


def validate_file_exists(file_path: Path, file_name: str | None = None) -> None:
    """Validate that a file exists.

    Args:
        file_path: Path to the file to check.
        file_name: Optional name of the file for error message.
            If None, uses the file path.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not file_path.exists():
        if file_name is None:
            file_name = str(file_path)
        msg = f"{file_name} not found: {file_path}"
        raise FileNotFoundError(msg)


def validate_dataframe_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Validate that DataFrame is not empty.

    Args:
        df: DataFrame to validate.
        name: Name of the DataFrame for error messages. Default is 'DataFrame'.

    Raises:
        ValueError: If DataFrame is empty.
    """
    if df.empty:
        msg = f"{name} DataFrame is empty"
        raise ValueError(msg)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str] | list[str],
    df_name: str = "DataFrame",
) -> None:
    """Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate.
        required_columns: Set or list of required column names.
        df_name: Name of the DataFrame for error messages. Default is 'DataFrame'.

    Raises:
        KeyError: If any required column is missing.
    """
    required_set = set(required_columns)
    missing_columns = required_set - set(df.columns)
    if missing_columns:
        msg = f"Missing required columns in {df_name}: {sorted(missing_columns)}"
        raise KeyError(msg)


def validate_train_ratio(train_ratio: float) -> None:
    """Validate that train_ratio is strictly between 0 and 1.

    Ensures temporal split ratio is valid for train/test splitting operations.
    Used across data preparation and model validation pipelines.

    Args:
        train_ratio: Proportion of data for training.

    Raises:
        ValueError: If train_ratio is not between 0 and 1 (exclusive).

    Examples:
        Validate standard split:
        >>> validate_train_ratio(0.8)  # Valid

        Invalid splits raise error:
        >>> validate_train_ratio(0.0)  # Raises ValueError
        >>> validate_train_ratio(1.0)  # Raises ValueError
        >>> validate_train_ratio(1.2)  # Raises ValueError

    Usage in project:
        - Replaces src/data_preparation/utils.py:validate_train_ratio
        - Used in train/test splitting across all pipelines
    """
    if train_ratio <= 0 or train_ratio >= 1:
        msg = f"train_ratio must be between 0 and 1, got {train_ratio}"
        raise ValueError(msg)


def validate_series(series: pd.Series) -> pd.Series:
    """Return a clean Series (datetime index not enforced).

    Validates and cleans a pandas Series by removing NaN values and converting to float.
    Used across all time series modules (ARIMA, GARCH, LightGBM) for consistent
    Series validation.

    Args:
        series: Input time series.

    Returns:
        Cleaned Series with NaN values removed and converted to float.

    Raises:
        ValueError: If series is None or empty after dropna.

    Examples:
        Basic validation:
        >>> series = pd.Series([1.0, 2.0, np.nan, 3.0])
        >>> clean_series = validate_series(series)
        >>> len(clean_series)
        3

        Empty series raises error:
        >>> validate_series(pd.Series([np.nan, np.nan]))  # Raises ValueError

    Usage in project:
        - Replaces src/arima/stationnarity_check/utils.py:validate_series
        - Replaces src/arima/model_evaluation/model_evaluation.py validation functions
        - Used for all Series validation across ARIMA, GARCH modules
    """
    if series is None:
        raise ValueError("series is None")
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        raise ValueError("series is empty after dropna")
    return s
