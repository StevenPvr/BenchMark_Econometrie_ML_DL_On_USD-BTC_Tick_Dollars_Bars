"""DateTime utilities for the project.

Provides functions for datetime normalization, parsing, date range extraction,
and date-based filtering. Used across all pipelines for consistent datetime handling.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd # type: ignore[import-untyped]

from src.config_logging import get_logger

__all__ = [
    "normalize_timestamp_to_datetime",
    "parse_date_value",
    "extract_date_range",
    "filter_by_date_range",
    "format_dates_to_string",
]


def normalize_timestamp_to_datetime(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Convert pandas Timestamp to timezone-naive datetime.

    Removes timezone information from timestamps to ensure consistent datetime
    representations across the project. Used in data fetching and processing
    pipelines where timezone-naive datetimes are required.

    Args:
        timestamp: Pandas Timestamp (may be timezone-aware or naive).

    Returns:
        Timezone-naive Timestamp object.

    Examples:
        Convert timezone-aware timestamp:
        >>> import pandas as pd
        >>> ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        >>> normalized = normalize_timestamp_to_datetime(ts)
        >>> normalized.tzinfo is None
        True

        Naive timestamps pass through:
        >>> ts_naive = pd.Timestamp("2024-01-01 12:00:00")
        >>> result = normalize_timestamp_to_datetime(ts_naive)
        >>> result == ts_naive
        True

    Usage in project:
        - Replaces src/data_fetching/download.py:_normalize_timestamp_to_datetime
        - Used in date range extraction and validation
    """
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp


def _parse_series_value(value: pd.Series, context: str, allow_none: bool) -> object | None:
    """Parse Series by extracting first element."""
    logger = get_logger(__name__)
    if value.empty:
        if not allow_none:
            logger.warning(f"{context}: Series is empty")
        return None
    return value.iloc[0]


def _parse_timestamp_value(
    value: pd.Timestamp, context: str, allow_none: bool
) -> pd.Timestamp | None:
    """Parse Timestamp value."""
    logger = get_logger(__name__)
    if value is pd.NaT:
        if not allow_none:
            logger.warning(f"{context}: Timestamp is NaT")
        return None
    return normalize_timestamp_to_datetime(value)


def _parse_scalar_value(
    value: int | float | str, context: str, allow_none: bool
) -> pd.Timestamp | None:
    """Parse string, int, or float to Timestamp."""
    logger = get_logger(__name__)
    try:
        ts_candidate = pd.Timestamp(value)
    except (ValueError, TypeError):
        if not allow_none:
            logger.warning(f"{context}: Cannot parse value '{value}'")
        return None
    if ts_candidate is pd.NaT:
        if not allow_none:
            logger.warning(f"{context}: Parsed timestamp is NaT")
        return None
    return normalize_timestamp_to_datetime(cast(pd.Timestamp, ts_candidate))


def parse_date_value(
    value: object,
    *,
    context: str = "date",
    allow_none: bool = False,
) -> pd.Timestamp | None:
    """Parse arbitrary date representations to timezone-naive Timestamps.

    Handles multiple input types: pd.Series, pd.Timestamp, datetime, str, int, float.
    Returns None for invalid inputs when allow_none=True, otherwise logs warnings.

    Args:
        value: Date value in various formats (Series, Timestamp, datetime, str, int, float).
        context: Description of the date value for error messages (e.g., "Min date", "Max date").
        allow_none: If True, return None for invalid inputs instead of logging warnings.

    Returns:
        Timezone-naive Timestamp or None if parsing fails.

    Examples:
        Parse different date types:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> parse_date_value("2024-01-01")
        Timestamp('2024-01-01 00:00:00')

        >>> parse_date_value(pd.Timestamp("2024-01-01 12:00:00", tz="UTC"))
        Timestamp('2024-01-01 12:00:00')

        >>> parse_date_value(datetime(2024, 1, 1))
        Timestamp('2024-01-01 00:00:00')

        Handle invalid input:
        >>> parse_date_value("invalid", allow_none=True) is None
        True

    Usage in project:
        - Replaces src/data_fetching/download.py:_normalize_date_value
        - Used in date range validation and extraction
        - Handles edge cases (NaT, empty Series, invalid types)
    """
    logger = get_logger(__name__)

    # Handle Series: extract first element
    if isinstance(value, pd.Series):
        value = _parse_series_value(value, context, allow_none)
        if value is None:
            return None

    # Handle Timestamp
    if isinstance(value, pd.Timestamp):
        return _parse_timestamp_value(value, context, allow_none)

    # Handle string, int, float
    if isinstance(value, (int, float, str)):
        return _parse_scalar_value(value, context, allow_none)

    # Unsupported type
    if not allow_none:
        logger.warning(f"{context}: Unsupported type {type(value)}")
    return None


def _extract_raw_min_max(df: pd.DataFrame, date_col: str) -> tuple[Any, Any]:
    """Extract raw min/max values from date column.

    Args:
        df: DataFrame with date column.
        date_col: Name of date column.

    Returns:
        Tuple of (min_candidate, max_candidate).

    Raises:
        TypeError: If min/max operations return Series instead of scalars.
    """
    if df.empty or date_col not in df.columns:
        return None, None

    dates = df[date_col]
    min_candidate = dates.min()
    max_candidate = dates.max()

    if isinstance(min_candidate, pd.Series) or isinstance(max_candidate, pd.Series):
        raise TypeError(f"{date_col}: min/max returned a Series; expected scalar date values.")

    return min_candidate, max_candidate


def _normalize_date_value(value: Any, date_col: str, value_type: str) -> str | pd.Timestamp | None:
    """Normalize a single date value to proper type.

    Args:
        value: Raw date value from DataFrame min/max.
        date_col: Name of date column (for error messages).
        value_type: Type of value ("minimum" or "maximum" for error messages).

    Returns:
        Normalized date value or None.

    Raises:
        ValueError: If string value cannot be parsed as date.
    """
    if isinstance(value, pd.Timestamp):
        return normalize_timestamp_to_datetime(value)
    elif value is pd.NaT or value is None:
        return None
    elif isinstance(value, str):
        timestamp = pd.Timestamp(value)
        if timestamp is pd.NaT:
            raise ValueError(f"{date_col}: cannot parse {value_type} date '{value}'")
        return normalize_timestamp_to_datetime(cast(pd.Timestamp, timestamp))
    else:
        return cast(str, value)


def _format_date_output(
    min_value: str | pd.Timestamp | None,
    max_value: str | pd.Timestamp | None,
    as_string: bool,
) -> tuple[str | pd.Timestamp | None, str | pd.Timestamp | None]:
    """Format date values according to output requirements.

    Args:
        min_value: Normalized minimum date value.
        max_value: Normalized maximum date value.
        as_string: Whether to return strings or Timestamps.

    Returns:
        Tuple of formatted (min_date, max_date).
    """
    if as_string:
        min_result = None if min_value is None else str(min_value)
        max_result = None if max_value is None else str(max_value)
        return min_result, max_result

    return min_value, max_value


def extract_date_range(
    df: pd.DataFrame,
    date_col: str = "date",
    *,
    as_string: bool = True,
) -> tuple[str | pd.Timestamp | None, str | pd.Timestamp | None]:
    """Extract date range (min, max) from DataFrame.

    Extracts the minimum and maximum dates from a DataFrame's date column.
    Useful for logging, reporting, and validation of temporal datasets.

    Args:
        df: DataFrame with date column.
        date_col: Name of date column. Default is 'date'.
        as_string: If True, return ISO format strings. If False, return Timestamps.

    Returns:
        Tuple of (min_date, max_date). Returns (None, None) if DataFrame is empty
        or date column is missing.

    Raises:
        TypeError: If min/max operations return unexpected types.
        ValueError: If date strings cannot be parsed.
    """
    min_candidate, max_candidate = _extract_raw_min_max(df, date_col)

    if min_candidate is None and max_candidate is None:
        return None, None

    min_value = _normalize_date_value(min_candidate, date_col, "minimum")
    max_value = _normalize_date_value(max_candidate, date_col, "maximum")

    return _format_date_output(min_value, max_value, as_string)


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    date_col: str = "date",
    raise_if_empty: bool = True,
) -> pd.DataFrame:
    """Filter DataFrame by date range.

    Generic date range filtering with proper error handling. Can be used for
    year filtering, month filtering, or any custom date range.

    Args:
        df: DataFrame with datetime index or date column.
        start_date: Start date (inclusive). Can be string or Timestamp.
        end_date: End date (inclusive). Can be string or Timestamp.
        date_col: Name of date column if not using index. Default is 'date'.
        raise_if_empty: If True, raise ValueError if filtered DataFrame is empty.

    Returns:
        Filtered DataFrame for the specified date range.

    Raises:
        ValueError: If no data available for the date range (when raise_if_empty=True).

    Examples:
        Filter by year:
        >>> df_2024 = filter_by_date_range(df, "2024-01-01", "2024-12-31")

        Filter by month:
        >>> df_jan = filter_by_date_range(df, "2024-01-01", "2024-01-31")

        Custom range:
        >>> df_q1 = filter_by_date_range(df, "2024-01-01", "2024-03-31")

        Don't raise on empty:
        >>> df_filtered = filter_by_date_range(
        ...     df, "2030-01-01", "2030-12-31",
        ...     raise_if_empty=False
        ... )

    Usage in project:
        - Replaces src/arima/data_visualisation/data_loading.py:_filter_dataframe_by_year
        - Generalizes year/month/custom date range filtering
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Check if DataFrame uses index or column for dates
    df_filtered: pd.DataFrame
    if isinstance(df.index, pd.DatetimeIndex):
        # Handle timezone-aware vs timezone-naive comparison
        if df.index.tz is not None:
            # Index is timezone-aware, make timestamps timezone-aware
            if start.tz is None:
                start = start.tz_localize(df.index.tz)
            if end.tz is None:
                end = end.tz_localize(df.index.tz)
        df_filtered = cast(pd.DataFrame, df.loc[start:end])
    else:
        # Ensure date column is datetime
        if date_col not in df.columns:
            msg = f"Date column '{date_col}' not found in DataFrame"
            raise KeyError(msg)

        df_temp = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])

        mask = (df_temp[date_col] >= start) & (df_temp[date_col] <= end)
        df_filtered = cast(pd.DataFrame, df_temp[mask])

    if raise_if_empty and df_filtered.empty:
        msg = f"No data available for date range {start.date()} to {end.date()}"
        raise ValueError(msg)

    return df_filtered


def _resolve_date_format(date_format: str | None) -> str:
    """Get date format, using default if None."""
    if date_format is None:
        from src.constants import DATE_FORMAT_DEFAULT

        return DATE_FORMAT_DEFAULT
    return date_format


def _normalize_dates_to_index(dates: pd.Series | pd.DatetimeIndex | list) -> pd.DatetimeIndex:
    """Convert various date inputs to DatetimeIndex with warning suppression."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
        # Use 'mixed' format to handle timestamps with/without microseconds consistently
        return pd.DatetimeIndex(pd.to_datetime(dates, errors="raise", format="mixed"))


def _apply_date_formatting(datetime_index: pd.DatetimeIndex, date_format: str) -> pd.Series:
    """Apply date formatting and return as Series."""
    formatted_index = datetime_index.strftime(date_format)
    return pd.Series(formatted_index, dtype=object)


def format_dates_to_string(
    dates: pd.Series | pd.DatetimeIndex | list,
    date_format: str | None = None,
) -> pd.Series:
    """Format dates to string with consistent format.

    Converts various date representations (Series, DatetimeIndex, list) to a
    pandas Series of formatted date strings. Uses a default format if none is
    provided. Useful for creating consistent date string representations for
    plotting, reporting, and file naming.

    Args:
        dates: Series, DatetimeIndex, or list of dates to format.
            Can contain datetime objects, Timestamps, or parseable date strings.
        date_format: Output format string (e.g., '%Y-%m-%d', '%d/%m/%Y').
            If None, uses DATE_FORMAT_DEFAULT from constants ('%Y-%m-%d').

    Returns:
        Series of formatted date strings with the same length as input.

    Raises:
        ValueError: If dates cannot be parsed to datetime.

    Examples:
        Format with default format ('%Y-%m-%d'):
        >>> import pandas as pd
        >>> dates = pd.Series(['2024-01-01', '2024-01-02', '2024-01-03'])
        >>> format_dates_to_string(dates)
        0    2024-01-01
        1    2024-01-02
        2    2024-01-03
        dtype: object

        Format with custom format:
        >>> format_dates_to_string(dates, '%d/%m/%Y')
        0    01/01/2024
        1    02/01/2024
        2    03/01/2024
        dtype: object

        Format DatetimeIndex:
        >>> date_index = pd.date_range('2024-01-01', periods=3)
        >>> format_dates_to_string(date_index, '%Y%m%d')
        0    20240101
        1    20240102
        2    20240103
        dtype: object

        Format list of dates:
        >>> from datetime import datetime
        >>> date_list = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        >>> format_dates_to_string(date_list)
        0    2024-01-01
        1    2024-01-02
        dtype: object

    Notes:
        - All input dates are converted to datetime before formatting
        - Timezone information is preserved in the conversion but not in output
        - Invalid dates will raise a ValueError during pd.to_datetime conversion

    Usage in project:
        - ARIMA evaluation: formatting dates for rolling predictions DataFrames
        - GARCH evaluation: formatting dates for variance forecasts
        - Plotting: creating readable date labels
        - Reporting: consistent date string formatting in logs and outputs
    """
    resolved_format = _resolve_date_format(date_format)
    datetime_index = _normalize_dates_to_index(dates)
    return _apply_date_formatting(datetime_index, resolved_format)
