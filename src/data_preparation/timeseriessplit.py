"""Functions for weighted log returns data preparation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from src.constants import (
    REQUIRED_COLS_SPLIT_DATA,
    REQUIRED_COLS_WEIGHTED_RETURNS,
    TIMESERIES_SPLIT_N_SPLITS,
    TRAIN_RATIO_DEFAULT,
)
from src.path import WEIGHTED_LOG_RETURNS_FILE, WEIGHTED_LOG_RETURNS_SPLIT_FILE
from src.utils import (
    compute_timeseries_split_indices,
    get_logger,
    load_and_validate_dataframe,
    load_dataframe,
    log_series_summary,
    log_split_dates,
    log_split_summary,
    save_parquet_and_csv,
    validate_dataframe_not_empty,
    validate_required_columns,
    validate_temporal_split,
    validate_train_ratio,
)

logger = get_logger(__name__)


def _load_and_clean_data(input_file: str) -> pd.DataFrame:
    """Load and clean weighted log returns data.

    Uses common loading function for consistency and reduced duplication.

    Args:
        input_file: Path to weighted log returns file (Parquet or CSV).

    Returns:
        Cleaned DataFrame sorted by date.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If DataFrame is empty after cleaning.
        KeyError: If required columns are missing.
    """
    return load_and_validate_dataframe(
        input_file=input_file,
        date_columns=["date"],
        required_columns=list(REQUIRED_COLS_WEIGHTED_RETURNS),
        sort_by=["date"],
        validate_not_empty=True,
        drop_na_subset=["weighted_log_return"],
    )


def _compute_split_indices(data: pd.DataFrame, train_ratio: float) -> tuple[list[int], list[int]]:
    """Compute train and test indices using TimeSeriesSplit.

    Delegates to src.utils.compute_timeseries_split_indices() for consistency.

    Args:
        data: DataFrame with time series data.
        train_ratio: Proportion of data for training.

    Returns:
        Tuple of (train_indices, test_indices).

    Raises:
        ValueError: If DataFrame is too small for splitting.
    """
    return compute_timeseries_split_indices(data, train_ratio, n_splits=TIMESERIES_SPLIT_N_SPLITS)


def _create_split_dataframes(
    data: pd.DataFrame,
    train_indices: list[int],
    test_indices: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train and test DataFrames with split column.

    Args:
        data: Original DataFrame.
        train_indices: Indices for training set.
        test_indices: Indices for test set.

    Returns:
        Tuple of (train_df, test_df) with split column added.
    """
    train_df = pd.DataFrame(data.iloc[train_indices])
    train_df["split"] = "train"

    test_df = pd.DataFrame(data.iloc[test_indices])
    test_df["split"] = "test"

    return train_df, test_df


def _perform_temporal_split(
    data: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform temporal split on time series data using TimeSeriesSplit.

    Uses sklearn's TimeSeriesSplit to ensure proper temporal ordering
    and prevent data leakage.

    Args:
        data: DataFrame with time series data.
        train_ratio: Proportion of data for training.

    Returns:
        Tuple of (train_df, test_df) with split column added.

    Raises:
        ValueError: If DataFrame is empty or too small for splitting.
    """
    validate_dataframe_not_empty(data, "Input data")

    train_indices, test_indices = _compute_split_indices(data, train_ratio)
    train_df, test_df = _create_split_dataframes(data, train_indices, test_indices)

    split_df = pd.concat([train_df, test_df], ignore_index=True)
    validate_temporal_split(split_df, function_name="_perform_temporal_split")
    log_split_dates(split_df, function_name="_perform_temporal_split")

    return train_df, test_df


def split_train_test(
    train_ratio: float = TRAIN_RATIO_DEFAULT,
    input_file: str | None = None,
    output_file: str | None = None,
) -> None:
    """Split time series data into train and test sets (GARCH pipeline).

    Uses TimeSeriesSplit and saves only ['date', 'weighted_log_return', 'split'].

    Args:
        train_ratio: Proportion of data for training (default: TRAIN_RATIO_DEFAULT).
        input_file: Path to weighted log returns file. If None, uses default.
        output_file: Path to save split data. If None, uses default.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If train_ratio is not between 0 and 1.
    """
    validate_train_ratio(train_ratio)

    if input_file is None:
        input_file = str(WEIGHTED_LOG_RETURNS_FILE)
    if output_file is None:
        output_file = str(WEIGHTED_LOG_RETURNS_SPLIT_FILE)

    aggregated_returns = _load_and_clean_data(input_file)
    train_df, test_df = _perform_temporal_split(aggregated_returns, train_ratio)
    split_df = cast(
        pd.DataFrame,
        pd.concat([train_df, test_df], ignore_index=True)[["date", "weighted_log_return", "split"]],
    )

    output_path = Path(output_file)
    if output_path.suffix.lower() == ".csv":
        output_path = output_path.with_suffix(".parquet")
    save_parquet_and_csv(split_df, output_path)
    log_split_summary(train_df, test_df, output_file)


def _load_split_dataframe(input_file: str) -> pd.DataFrame:
    """Load split data from the provided path.

    Delegates to src.utils.load_dataframe() for consistency.

    Args:
        input_file: Path to split data file (Parquet or CSV).

    Returns:
        DataFrame with split data.

    Raises:
        FileNotFoundError: If split data file doesn't exist.
        ValueError: If DataFrame is empty.
    """
    input_path = Path(input_file)
    try:
        split_df = load_dataframe(
            input_path,
            date_columns=["date"],
            validate_not_empty=False,
        )
    except pd.errors.EmptyDataError as err:
        # Provide a clear, standardized error for entirely empty files
        raise ValueError("Split data file is empty") from err

    validate_dataframe_not_empty(split_df, "Split data")
    return split_df


def _extract_train_test_series(
    split_data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Extract train and test series from split data.

    Args:
        split_data: DataFrame with split column.

    Returns:
        Tuple of (train_series, test_series) with date as index.

    Raises:
        ValueError: If train or test data is empty.
        KeyError: If required columns are missing.
    """
    validate_required_columns(split_data, REQUIRED_COLS_SPLIT_DATA)

    train_data = split_data[split_data["split"] == "train"].copy()
    test_data = split_data[split_data["split"] == "test"].copy()

    if train_data.empty:
        msg = "Train data is empty after splitting"
        raise ValueError(msg)
    if test_data.empty:
        msg = "Test data is empty after splitting"
        raise ValueError(msg)

    train_series = cast(pd.Series, train_data.set_index("date")["weighted_log_return"])
    test_series = cast(pd.Series, test_data.set_index("date")["weighted_log_return"])

    return train_series, test_series


def load_train_test_data(
    input_file: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Load train and test series from split data file.

    Args:
        input_file: Path to split data CSV. If None, uses default.

    Returns:
        Tuple of (train_series, test_series) with date as index.

    Raises:
        FileNotFoundError: If split data file doesn't exist.
    """
    if input_file is None:
        input_file = str(WEIGHTED_LOG_RETURNS_SPLIT_FILE)

    split_data = _load_split_dataframe(input_file)
    train_series, test_series = _extract_train_test_series(split_data)

    validate_temporal_split(split_data, function_name="load_train_test_data")
    log_split_dates(split_data, function_name="load_train_test_data")

    log_series_summary(train_series, test_series)

    return train_series, test_series
