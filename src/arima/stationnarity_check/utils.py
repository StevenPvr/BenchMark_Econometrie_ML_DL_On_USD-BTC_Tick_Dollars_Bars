"""Utility functions for stationarity checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd # type: ignore[import-untyped]

from src.utils import load_dataframe, validate_file_exists, validate_series as _validate_series


def load_series_from_csv(*, data_file: str, column: str, date_col: str = "datetime_close") -> pd.Series:
    """Load a column as Series from a CSV with a date index.

    Delegates to src.utils functions for loading and validation.

    Args:
        data_file: Path to CSV file.
        column: Column name to extract.
        date_col: Name of the date column. Defaults to "datetime_close".

    Returns:
        Series with a DatetimeIndex (parsed from date_col), sorted chronologically,
        and with NaN values removed.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If column is missing or does not return a Series.
    """
    path = Path(data_file)
    validate_file_exists(path, "Data file")

    # Load DataFrame with automatic date parsing and sorting
    df = load_dataframe(
        path,
        date_columns=[date_col],
        required_columns=[column],
        sort_by=[date_col],
    )

    # Set date as index
    df = df.set_index(date_col)

    series = df[column]
    if not isinstance(series, pd.Series):
        raise ValueError(f"Column '{column}' did not return a Series")

    # Remove duplicate timestamps, keeping the last value
    series = series[~series.index.duplicated(keep='last')]

    return series.dropna()


def validate_series(series: pd.Series) -> pd.Series:
    """Delegate to the shared series validator while keeping legacy import paths."""
    return _validate_series(series)
