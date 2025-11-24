"""Utility functions for the MF Tick project."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd  # type: ignore


# =============================================================================
# Logging
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)


# =============================================================================
# File I/O
# =============================================================================

def ensure_output_dir(path: Path | str) -> None:
    """Ensure the parent directory of a path exists.

    Args:
        path: File path whose parent directory should exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_file_exists(path: Path | str, description: str = "File") -> None:
    """Validate that a file exists.

    Args:
        path: Path to check.
        description: Description for error message.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def save_json_pretty(data: dict[str, Any], path: Path | str) -> None:
    """Save dictionary as formatted JSON.

    Args:
        data: Dictionary to save.
        path: Output file path.
    """
    path = Path(path)
    ensure_output_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json_data(path: Path | str) -> dict[str, Any]:
    """Load JSON file as dictionary.

    Args:
        path: Path to JSON file.

    Returns:
        Loaded dictionary.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    validate_file_exists(path, "JSON file")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_file(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Load CSV file as DataFrame.

    Args:
        path: Path to CSV file.
        **kwargs: Additional arguments for pd.read_csv.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    validate_file_exists(path, "CSV file")
    return pd.read_csv(path, **kwargs)


def load_dataframe(
    path: Path | str,
    date_columns: list[str] | None = None,
    validate_not_empty: bool = True,
) -> pd.DataFrame:
    """Load DataFrame from CSV or Parquet with optional date parsing.

    Args:
        path: Path to data file (.csv or .parquet).
        date_columns: Columns to parse as dates.
        validate_not_empty: Raise error if DataFrame is empty.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is unsupported or DataFrame is empty.
    """
    path = Path(path)
    validate_file_exists(path, "Data file")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        parse_dates = date_columns if date_columns else False
        df = pd.read_csv(path, parse_dates=parse_dates)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if validate_not_empty and df.empty:
        raise ValueError(f"Data file is empty: {path}")

    return df


# =============================================================================
# Data Validation
# =============================================================================

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    df_name: str = "DataFrame",
) -> None:
    """Validate that required columns exist in DataFrame.

    Args:
        df: DataFrame to check.
        required_columns: List of required column names.
        df_name: Name for error message.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def validate_temporal_order_series(
    train_series: pd.Series,
    test_series: pd.Series,
    function_name: str = "function",
) -> None:
    """Validate that train series ends before test series starts.

    Args:
        train_series: Training data series.
        test_series: Test data series.
        function_name: Name of calling function for error message.

    Raises:
        ValueError: If temporal order is violated.
    """
    if len(train_series) == 0 or len(test_series) == 0:
        return

    train_end = train_series.index[-1]
    test_start = test_series.index[0]

    # Ensure they are comparable (handle datetime comparisons)
    if pd.to_datetime(train_end) >= pd.to_datetime(test_start):
        raise ValueError(
            f"{function_name}: Train series must end before test series starts. "
            f"Train ends at {train_end}, test starts at {test_start}."
        )


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    date_column: str | None = None,
    raise_if_empty: bool = False,
) -> pd.DataFrame:
    """Filter DataFrame by date range.

    Args:
        df: DataFrame to filter (must have DatetimeIndex or date_column).
        start_date: Start date string (inclusive).
        end_date: End date string (inclusive).
        date_column: Column name if not using index.
        raise_if_empty: Raise error if result is empty.

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If raise_if_empty and result is empty.
    """
    df_filtered = df.copy()

    if date_column:
        dates = pd.to_datetime(df_filtered[date_column])
    else:
        dates = pd.to_datetime(df_filtered.index)

    mask = pd.Series([True] * len(df_filtered), index=df_filtered.index)

    if start_date:
        mask &= dates >= pd.to_datetime(start_date)
    if end_date:
        mask &= dates <= pd.to_datetime(end_date)

    df_filtered = pd.DataFrame(df_filtered[mask])

    if raise_if_empty and df_filtered.empty:
        raise ValueError(f"No data found in date range {start_date} to {end_date}")

    return df_filtered


# =============================================================================
# Date Formatting
# =============================================================================

def format_dates_to_string(
    dates: Sequence[Any] | pd.Index | pd.Series,
    fmt: str = "%Y-%m-%d",
) -> pd.Series:
    """Format dates to string series.

    Args:
        dates: Sequence of date-like objects.
        fmt: Date format string.

    Returns:
        Series of formatted date strings.
    """
    if isinstance(dates, pd.DatetimeIndex):
        dt_index = dates
    elif isinstance(dates, pd.Series):
        dt_index = pd.DatetimeIndex(dates)
    else:
        dt_index = pd.DatetimeIndex(list(dates))

    return pd.Series(dt_index.strftime(fmt))


# =============================================================================
# Warnings Suppression
# =============================================================================

def suppress_statsmodels_warnings() -> None:
    """Suppress common statsmodels warnings."""
    warnings.filterwarnings("ignore", message=".*frequency.*")
    warnings.filterwarnings("ignore", message=".*Maximum Likelihood.*")
    warnings.filterwarnings("ignore", message=".*No supported index is available.*", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


# =============================================================================
# Statistical Functions
# =============================================================================

def chi2_sf(statistic: float, df: int = 1) -> float:
    """Chi-squared survival function (1 - CDF).

    Args:
        statistic: Test statistic value.
        df: Degrees of freedom.

    Returns:
        Survival function value (p-value).
    """
    from scipy.stats import chi2  # type: ignore
    return float(chi2.sf(statistic, df))


def write_placeholder_file(path: Path | str) -> None:
    """Write a placeholder file with current timestamp.

    Args:
        path: Path to the placeholder file.
    """
    from datetime import datetime

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    content = f"# Placeholder file created on {timestamp}\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def validate_temporal_split(
    data: pd.DataFrame | pd.Series,
    split_column: str = "split",
    function_name: str = "function",
) -> None:
    """Validate that data has proper temporal train/test split.

    Args:
        data: DataFrame or Series with split information.
        split_column: Column name containing split labels.
        function_name: Name of calling function for error messages.

    Raises:
        ValueError: If temporal order is violated.
    """
    if isinstance(data, pd.Series):
        # If it's a Series, assume it has a split column in index or name
        if hasattr(data.index, 'get_level_values'):
            # MultiIndex
            try:
                split_values = data.index.get_level_values(split_column)
            except KeyError:
                raise ValueError(f"{function_name}: Series must have '{split_column}' in index")
        else:
            raise ValueError(f"{function_name}: Cannot validate temporal split on Series without '{split_column}' index")
    else:
        # DataFrame
        if split_column not in data.columns:
            raise ValueError(f"{function_name}: DataFrame must have '{split_column}' column")

        split_values = data[split_column]

    # Check that we have both train and test
    unique_splits = split_values.unique()
    if "train" not in unique_splits or "test" not in unique_splits:
        raise ValueError(f"{function_name}: Data must have both 'train' and 'test' splits")

    # For temporal validation, we would need datetime index
    # For now, just check that splits exist
    train_count = (split_values == "train").sum()
    test_count = (split_values == "test").sum()

    if train_count == 0 or test_count == 0:
        raise ValueError(f"{function_name}: Both train and test splits must have data")

    logger.info(f"{function_name}: Validated split - Train: {train_count}, Test: {test_count}")


# =============================================================================
# Residuals
# =============================================================================

def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute residuals (actual - predicted).

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Residuals array.
    """
    return np.asarray(y_true) - np.asarray(y_pred)


def save_parquet_and_csv(df: pd.DataFrame, parquet_path: str | Path) -> None:
    """Save DataFrame to both parquet and CSV formats.

    Args:
        df: DataFrame to save.
        parquet_path: Path for parquet file (will also create .csv version).
    """
    parquet_path = Path(parquet_path)

    # Save as parquet
    df.to_parquet(parquet_path, index=False)

    # Save as CSV (replace .parquet extension with .csv)
    csv_path = parquet_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
