"""Validation and dataset loading functions for data cleaning."""

from __future__ import annotations

import pandas as pd

from src.constants import REQUIRED_OHLCV_COLUMNS
from src.path import DATASET_FILE
from src.utils import (
    get_logger,
    validate_dataframe_not_empty,
    validate_file_exists as _validate_file_exists,
    validate_required_columns,
)

logger = get_logger(__name__)


# Use the more robust version from utils
assert_required_columns = validate_required_columns


# Use the more robust version from utils
assert_not_empty = validate_dataframe_not_empty


def validate_file_exists() -> None:
    """Validate that the dataset file exists on disk.

    Uses the project-level DATASET_FILE constant so that test code can
    monkeypatch it if needed via the src.data_cleaning.data_cleaning module.
    """
    _validate_file_exists(DATASET_FILE)


def validate_columns(raw_df: pd.DataFrame) -> None:
    """Validate that the raw dataset contains all required OHLCV columns.

    Args:
        raw_df: Raw dataset DataFrame.

    Raises:
        KeyError: If one or more required columns are missing.
    """
    assert_required_columns(raw_df, REQUIRED_OHLCV_COLUMNS)


def _has_datetime_accessor(series: pd.Series) -> bool:
    """Check if the series has datetime accessor."""
    return hasattr(series, "dt")


def _convert_to_new_york_time(series: pd.Series) -> pd.Series:
    """Convert timezone-aware datetime series to New York time."""
    return series.dt.tz_convert("America/New_York")


def _make_timezone_naive(series: pd.Series) -> pd.Series:
    """Remove timezone information from datetime series."""
    return series.dt.tz_localize(None)


def _normalize_to_midnight(series: pd.Series) -> pd.Series:
    """Normalize datetime series to midnight (remove time component)."""
    return series.dt.normalize()


def _normalize_datetime_timezone_safe(series: pd.Series) -> pd.Series:
    """Safely normalize datetime series when dt accessor is available."""
    series = _convert_to_new_york_time(series)
    series = _make_timezone_naive(series)
    return _normalize_to_midnight(series)


def _normalize_datetime_timezone_fallback(series: pd.Series) -> pd.Series:
    """Fallback normalization when dt accessor is not available (mocks/tests)."""
    # For mocks or test data, assume it's already in correct format
    series = series.tz_localize(None)  # type: ignore[call-arg]
    return series.normalize()  # type: ignore[call-arg]


def _normalize_datetime_timezone(converted: pd.Series) -> pd.Series:
    """Normalize datetime series to naive (timezone-unaware) format.

    Converts UTC-aware datetimes to New York local time (market calendar),
    then drops timezone information and normalizes to midnight. This avoids
    DST-induced shifts that would otherwise break date alignment when we
    later create business-day ranges for each ticker.

    Args:
        converted: Datetime series to normalize.

    Returns:
        Normalized datetime series.
    """
    if _has_datetime_accessor(converted):
        return _normalize_datetime_timezone_safe(converted)
    else:
        return _normalize_datetime_timezone_fallback(converted)


def convert_and_validate_dates(raw_df: pd.DataFrame) -> None:
    """Convert the ``date`` column to datetime and validate its integrity.

    The function converts the ``date`` column in-place to ``datetime64[ns]``.
    Any unparsable dates are treated as an error to avoid silently dropping
    observations.

    Args:
        raw_df: Raw dataset DataFrame with a ``date`` column.

    Raises:
        KeyError: If the ``date`` column is missing.
        ValueError: If any date cannot be parsed.
    """
    if "date" not in raw_df.columns:
        msg = "Missing required column: 'date'"
        raise KeyError(msg)

    # Convert to datetime. Handle mixed timezones by normalizing to UTC first
    # if timezone info is present, then convert to naive (timezone-unaware)
    # since financial daily data is handled in a naive calendar.
    converted = pd.to_datetime(raw_df["date"], errors="coerce", utc=True)
    converted = _normalize_datetime_timezone(converted)

    num_invalid = int(converted.isna().sum())
    if num_invalid > 0:
        msg = f"Found {num_invalid} invalid date value(s) in 'date' column"
        raise ValueError(msg)

    raw_df["date"] = converted  # type: ignore[assignment]
    logger.info(
        "Converted 'date' column to datetime. " "Period: %s → %s",
        converted.min().date(),
        converted.max().date(),
    )


def load_dataset() -> pd.DataFrame:
    """Load and validate the raw S&P 500 dataset.

    The function performs the following checks:

    1. Validates that the dataset file exists (via ``validate_file_exists``).
    2. Loads the CSV file referenced by ``DATASET_FILE``.
    3. Checks that all required OHLCV columns are present.
    4. Fails explicitly if the dataset is empty.
    5. Converts and validates the ``date`` column.

    Returns:
        Validated raw dataset DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        KeyError: If required columns are missing.
        ValueError: If the dataset is empty or dates are invalid.
    """
    validate_file_exists()

    raw_df = pd.read_csv(DATASET_FILE)

    validate_columns(raw_df)
    assert_not_empty(raw_df, name="raw dataset")

    convert_and_validate_dates(raw_df)

    return raw_df
