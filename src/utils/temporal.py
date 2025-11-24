"""Temporal validation utilities for the project.

Provides functions for temporal split validation and look-ahead bias prevention.
Used across all pipelines to ensure proper temporal ordering in train/test splits.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.config_logging import get_logger

__all__ = [
    "validate_temporal_order_series",
    "validate_temporal_split",
    "log_split_dates",
    "compute_timeseries_split_indices",
]


def _validate_required_columns_for_split(
    df: pd.DataFrame, date_col: str, split_col: str, function_name: str
) -> None:
    """Validate that required columns exist."""
    if date_col not in df.columns:
        msg = f"{function_name}: DataFrame must contain '{date_col}' column"
        raise ValueError(msg)
    if split_col not in df.columns:
        msg = f"{function_name}: DataFrame must contain '{split_col}' column"
        raise ValueError(msg)


def _validate_split_values(df: pd.DataFrame, split_col: str, function_name: str) -> set[str]:
    """Validate split values and return actual splits."""
    valid_splits = {"train", "test"}
    actual_splits = set(df[split_col].dropna().unique())
    invalid_splits = actual_splits - valid_splits
    if invalid_splits:
        msg = (
            f"{function_name}: Invalid split values found: {invalid_splits}. "
            f"Expected only: {valid_splits}"
        )
        raise ValueError(msg)
    return actual_splits


def _check_temporal_order_for_ticker(
    ticker_df: pd.DataFrame,
    ticker: str,
    split_col: str,
    date_col: str,
    function_name: str,
) -> None:
    """Check temporal order for a single ticker."""
    train_mask = ticker_df[split_col] == "train"
    test_mask = ticker_df[split_col] == "test"

    if train_mask.sum() > 0 and test_mask.sum() > 0:
        train_dates = ticker_df[train_mask][date_col]
        test_dates = ticker_df[test_mask][date_col]

        if train_dates.max() >= test_dates.min():
            msg = (
                f"{function_name}: Look-ahead bias detected for ticker '{ticker}'. "
                f"Max train date ({train_dates.max()}) >= "
                f"Min test date ({test_dates.min()}). "
                "Train dates must be strictly before test dates."
            )
            raise ValueError(msg)


def _check_temporal_order_global(
    df: pd.DataFrame, split_col: str, date_col: str, function_name: str
) -> None:
    """Check temporal order globally for aggregated data."""
    train_mask = df[split_col] == "train"
    test_mask = df[split_col] == "test"

    if train_mask.sum() > 0 and test_mask.sum() > 0:
        train_dates = df[train_mask][date_col]
        test_dates = df[test_mask][date_col]

        if train_dates.max() >= test_dates.min():
            msg = (
                f"{function_name}: Look-ahead bias detected. "
                f"Max train date ({train_dates.max()}) >= Min test date ({test_dates.min()}). "
                "Train dates must be strictly before test dates."
            )
            raise ValueError(msg)


def validate_temporal_order_series(
    train_series: pd.Series,
    test_series: pd.Series,
    function_name: str = "validate_temporal_order_series",
) -> None:
    """Validate that test series dates are after training series dates.

    Prevents look-ahead bias by ensuring temporal order between two Series.
    This is a utility function for validating Series objects directly,
    complementing validate_temporal_split which works with DataFrames.

    Args:
        train_series: Training time series with date index
        test_series: Test time series with date index
        function_name: Name of the calling function for error messages

    Raises:
        ValueError: If test series starts before or at training series end
    """
    if len(train_series) == 0 or len(test_series) == 0:
        return

    train_max = train_series.index.max()
    test_min = test_series.index.min()

    # Already Timestamps from DatetimeIndex
    train_max_date = train_max
    test_min_date = test_min

    if pd.isna(train_max_date) or pd.isna(test_min_date):
        return

    if test_min_date <= train_max_date:
        msg = (
            f"{function_name}: Look-ahead bias detected. "
            f"Max train date ({train_max_date}) >= Min test date ({test_min_date}). "
            "Train dates must be strictly before test dates."
        )
        raise ValueError(msg)


def validate_temporal_split(
    df: pd.DataFrame,
    split_col: str = "split",
    date_col: str = "date",
    ticker_col: str | None = None,
    function_name: str = "validate_temporal_split",
) -> None:
    """Validate that train/test split maintains temporal order (no look-ahead bias).

    Checks that all train dates are strictly before all test dates.
    For ticker-level data, validates this for each ticker separately.

    Args:
        df: DataFrame with split and date columns.
        split_col: Name of the split column ('train' or 'test').
        date_col: Name of the date column.
        ticker_col: Optional name of ticker column for ticker-level validation.
        function_name: Name of the calling function for error messages.

    Raises:
        ValueError: If required columns are missing, split values are invalid,
            or temporal order is violated (look-ahead bias detected).
    """
    logger = get_logger(__name__)

    _validate_required_columns_for_split(df, date_col, split_col, function_name)

    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    actual_splits = _validate_split_values(df, split_col, function_name)

    # Check if we have both train and test
    has_train = "train" in actual_splits
    has_test = "test" in actual_splits

    if not has_train and not has_test:
        logger.warning(f"{function_name}: No train or test data found in split column")
        return

    if ticker_col is not None and ticker_col in df.columns:
        # Ticker-level validation: check temporal order for each ticker
        for ticker in df[ticker_col].unique():
            ticker_df = cast(pd.DataFrame, df[df[ticker_col] == ticker].copy())
            _check_temporal_order_for_ticker(ticker_df, ticker, split_col, date_col, function_name)
    else:
        # Aggregated data validation: check temporal order globally
        _check_temporal_order_global(df, split_col, date_col, function_name)

    logger.debug(f"{function_name}: Temporal split validation passed")


def _log_aggregated_split_dates(
    df: pd.DataFrame, split_col: str, date_col: str, function_name: str
) -> None:
    """Log split dates for aggregated data."""
    logger = get_logger(__name__)
    train_mask = df[split_col] == "train"
    test_mask = df[split_col] == "test"

    if train_mask.sum() > 0:
        train_dates = df[train_mask][date_col]
        logger.info(
            f"{function_name}: Train split - "
            f"{train_dates.min().date()} to {train_dates.max().date()} "
            f"({len(train_dates)} observations)"
        )

    if test_mask.sum() > 0:
        test_dates = df[test_mask][date_col]
        logger.info(
            f"{function_name}: Test split - "
            f"{test_dates.min().date()} to {test_dates.max().date()} "
            f"({len(test_dates)} observations)"
        )


def log_split_dates(
    df: pd.DataFrame,
    split_col: str = "split",
    date_col: str = "date",
    ticker_col: str | None = None,
    function_name: str = "log_split_dates",
) -> None:
    """Log date ranges for train and test splits.

    Provides visibility into the temporal boundaries of each split,
    which is useful for debugging and monitoring look-ahead bias prevention.
    For ticker-level data, logs aggregate summary statistics instead of per-ticker details.

    Args:
        df: DataFrame with split and date columns.
        split_col: Name of the split column ('train' or 'test').
        date_col: Name of the date column.
        ticker_col: Optional name of ticker column for ticker-level data.
        function_name: Name of the calling function for log messages.
    """
    logger = get_logger(__name__)

    if date_col not in df.columns or split_col not in df.columns:
        return

    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    if ticker_col is not None and ticker_col in df.columns:
        # For ticker-level data, include explicit ticker names for clarity
        n_tickers = df[ticker_col].nunique()
        unique_tickers = sorted(set(df[ticker_col].astype(str)))
        # Limit the number displayed to keep logs concise
        max_show = 10
        shown = ", ".join(unique_tickers[:max_show])
        suffix = "" if n_tickers <= max_show else f", +{n_tickers - max_show} more"

        train_mask = df[split_col] == "train"
        test_mask = df[split_col] == "test"

        if train_mask.sum() > 0:
            train_dates = df[train_mask][date_col]
            train_total = train_mask.sum()
            logger.info(
                f"{function_name}: {n_tickers} tickers [{shown}{suffix}] - Train: "
                f"{train_dates.min().date()} to {train_dates.max().date()} "
                f"({train_total} total observations)"
            )

        if test_mask.sum() > 0:
            test_dates = df[test_mask][date_col]
            test_total = test_mask.sum()
            logger.info(
                f"{function_name}: {n_tickers} tickers [{shown}{suffix}] - Test: "
                f"{test_dates.min().date()} to {test_dates.max().date()} "
                f"({test_total} total observations)"
            )
    else:
        # Aggregated data logging
        _log_aggregated_split_dates(df, split_col, date_col, function_name)


def compute_timeseries_split_indices(
    data: pd.DataFrame,
    train_ratio: float,
    *,
    n_splits: int = 1,
) -> tuple[list[int], list[int]]:
    """Compute train and test indices using TimeSeriesSplit.

    Performs temporal train/test splitting ensuring no look-ahead bias.
    Uses sklearn's TimeSeriesSplit to guarantee proper temporal ordering.

    Args:
        data: DataFrame with time series data (must be sorted by date).
        train_ratio: Proportion of data for training (0 < train_ratio < 1).
        n_splits: Number of splits for TimeSeriesSplit. Default is 1.

    Returns:
        Tuple of (train_indices, test_indices) as lists.

    Raises:
        ValueError: If DataFrame is too small for splitting (< 2 rows).

    Examples:
        Standard 80/20 split:
        >>> from sklearn.model_selection import TimeSeriesSplit
        >>> df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100)})
        >>> train_idx, test_idx = compute_timeseries_split_indices(df, train_ratio=0.8)
        >>> len(train_idx), len(test_idx)
        (80, 20)

        Multiple splits:
        >>> train_idx, test_idx = compute_timeseries_split_indices(
        ...     df,
        ...     train_ratio=0.8,
        ...     n_splits=5
        ... )

    Usage in project:
        - Replaces src/data_preparation/timeseriessplit.py:_compute_split_indices
        - Used in GARCH and LightGBM pipelines for temporal splitting
    """
    from sklearn.model_selection import TimeSeriesSplit

    n_total = len(data)
    if n_total < 2:
        msg = f"DataFrame must have at least 2 rows for splitting, got {n_total}"
        raise ValueError(msg)

    n_test = n_total - int(n_total * train_ratio)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=n_test)
    splits = list(tscv.split(data))
    train_indices_array, test_indices_array = splits[-1]

    # Convert numpy arrays to lists for type consistency
    train_indices = train_indices_array.tolist()
    test_indices = test_indices_array.tolist()

    return train_indices, test_indices


def is_datetime_series_monotonic_increasing(dates: pd.Series) -> bool:
    """Check if a datetime series is strictly monotonically increasing in its current order.

    This function verifies that dates are in strictly ascending order as they appear
    in the series, meaning each date is greater than the previous one.

    Args:
        dates: Series of datetime values to check in their current order.

    Returns:
        True if the series is monotonically increasing, False otherwise.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')])
        >>> is_datetime_series_monotonic_increasing(dates)
        True

        >>> dates = pd.Series([pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-01')])
        >>> is_datetime_series_monotonic_increasing(dates)
        False
    """
    if dates.empty:
        return True

    # Check for any non-positive differences in the original order
    # Strictly increasing means no backward moves (diff <= 0)
    diffs = dates.diff()
    return not (diffs <= pd.Timedelta(0)).any()
