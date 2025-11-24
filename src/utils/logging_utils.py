"""Logging utilities for the project.

Provides functions for logging series summaries and saving plots.
Used across all visualization and reporting modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config_logging import get_logger
from src.utils.datetime_utils import normalize_timestamp_to_datetime
from src.utils.io import ensure_output_dir

__all__ = [
    "log_series_summary",
    "log_split_summary",
    "save_plot",
]


def _log_date_range_from_index(
    series: pd.Series, label: str, logger_instance: logging.Logger
) -> None:
    """Log date range from datetime index.

    Args:
        series: Series with datetime index.
        label: Label for logging ("Train" or "Test").
        logger_instance: Logger instance to use.
    """
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        return

    try:
        min_val = series.index.min()
        max_val = series.index.max()

        if not isinstance(min_val, pd.Timestamp) or not isinstance(max_val, pd.Timestamp):
            raise TypeError(f"{label} index must provide Timestamp boundaries.")

        if min_val is pd.NaT or max_val is pd.NaT:
            raise ValueError(f"{label} index contains NaT values.")

        start_date = normalize_timestamp_to_datetime(min_val)
        end_date = normalize_timestamp_to_datetime(max_val)
        date_range = f"{start_date.date()} → {end_date.date()}"
        logger_instance.info(f"{label} period: {date_range}")

    except (AttributeError, TypeError, ValueError):
        pass  # Skip date range if formatting fails


def _log_date_range_from_column(
    df: pd.DataFrame, date_col: str, label: str, logger_instance: logging.Logger
) -> None:
    """Log date range from date column.

    Args:
        df: DataFrame with date column.
        date_col: Name of date column.
        label: Label for logging ("Train" or "Test").
        logger_instance: Logger instance to use.
    """
    if date_col not in df.columns or df.empty:
        return

    try:
        min_val = df[date_col].min()
        max_val = df[date_col].max()

        if pd.notna(min_val) and pd.notna(max_val):
            start_date = normalize_timestamp_to_datetime(pd.to_datetime(min_val))
            end_date = normalize_timestamp_to_datetime(pd.to_datetime(max_val))
            date_range = f"{start_date.date()} → {end_date.date()}"
            logger_instance.info(f"{label} period: {date_range}")

    except (AttributeError, TypeError, ValueError):
        pass  # Skip date range if formatting fails


def _log_series_statistics(series: pd.Series, label: str, logger_instance: logging.Logger) -> None:
    """Log basic statistics for a series.

    Args:
        series: Series to analyze.
        label: Label for logging.
        logger_instance: Logger instance to use.
    """
    stats = {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
    }
    logger_instance.info(
        f"{label} statistics - Mean: {stats['mean']:.6f}, "
        f"Std: {stats['std']:.6f}, "
        f"Min: {stats['min']:.6f}, "
        f"Max: {stats['max']:.6f}"
    )


def log_series_summary(
    train_series: pd.Series,
    test_series: pd.Series,
    *,
    logger_instance: logging.Logger | None = None,
) -> None:
    """Log summary statistics for train and test Series.

    Provides comprehensive logging of train/test series including:
    - Number of observations
    - Date range (if index is datetime)
    - Descriptive statistics (mean, std, min, max)

    Args:
        train_series: Training time series with datetime index.
        test_series: Test time series with datetime index.
        logger_instance: Optional logger instance. If None, uses get_logger().

    Examples:
        Log train/test summary:
        >>> log_series_summary(train_series, test_series)

        Custom logger:
        >>> import logging
        >>> custom_logger = logging.getLogger("my_module")
        >>> log_series_summary(train_series, test_series, logger_instance=custom_logger)

    Usage in project:
        - Replaces src/data_preparation/utils.py:log_series_summary
        - Used after train/test splitting to log data summaries
        - Provides visibility into temporal split characteristics
    """
    if logger_instance is None:
        logger_instance = get_logger(__name__)

    # Log train set info
    logger_instance.info(f"Train set: {len(train_series)} observations")
    _log_date_range_from_index(train_series, "Train", logger_instance)

    # Log test set info
    logger_instance.info(f"Test set: {len(test_series)} observations")
    _log_date_range_from_index(test_series, "Test", logger_instance)

    # Log train statistics
    _log_series_statistics(train_series, "Train", logger_instance)


def log_split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_file: str,
    *,
    logger_instance: logging.Logger | None = None,
) -> None:
    """Log summary of split operation for DataFrames.

    Provides comprehensive logging of train/test split including:
    - Number of observations and percentages
    - Date ranges for train/test periods
    - Output file location

    Args:
        train_df: Training DataFrame with 'date' column.
        test_df: Test DataFrame with 'date' column.
        output_file: Path where split data was saved.
        logger_instance: Optional logger instance. If None, uses get_logger().

    Examples:
        Log train/test split summary:
        >>> log_split_summary(train_df, test_df, "data/split.parquet")

        Custom logger:
        >>> import logging
        >>> custom_logger = logging.getLogger("my_module")
        >>> log_split_summary(train_df, test_df, "data/split.parquet",
        ...     logger_instance=custom_logger)

    Usage in project:
        - Used after train/test splitting to log data summaries
        - Provides visibility into split characteristics
        - Standardizes logging across data preparation modules
    """
    if logger_instance is None:
        logger_instance = get_logger(__name__)

    n_total = len(train_df) + len(test_df)
    logger_instance.info("Split complete: %d total observations", n_total)

    if n_total == 0:
        logger_instance.warning("No data to split")
        return

    train_pct = len(train_df) / n_total * 100
    test_pct = len(test_df) / n_total * 100

    logger_instance.info("Train set: %d observations (%.1f%%)", len(train_df), train_pct)
    _log_date_range_from_column(train_df, "date", "Train", logger_instance)

    logger_instance.info("Test set: %d observations (%.1f%%)", len(test_df), test_pct)
    _log_date_range_from_column(test_df, "date", "Test", logger_instance)

    logger_instance.info("Saved split data to %s", output_file)


def save_plot(
    output_path: Path | str,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
    close_after: bool = True,
) -> None:
    """Save the current matplotlib plot to file.

    Standardized plot saving with automatic directory creation and consistent formatting.
    Used across all visualization modules (ARIMA, GARCH, LightGBM).

    Args:
        output_path: Path to save the plot.
        dpi: Resolution in dots per inch. Default is 300.
        bbox_inches: Bounding box specification. Default is 'tight'.
        close_after: If True, close the plot after saving. Default is True.

    Examples:
        Basic plot saving:
        >>> import matplotlib.pyplot as plt
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> save_plot("results/plots/my_plot.png")

        High resolution without closing:
        >>> save_plot("results/plots/my_plot.png", dpi=600, close_after=False)

    Usage in project:
        - Replaces src/arima/data_visualisation/plotting.py:_save_plot
        - Standardizes plot saving across all visualization modules
        - Ensures consistent DPI and formatting
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib is required for save_plot()"
        raise ImportError(msg) from e

    path_obj = Path(output_path)
    ensure_output_dir(path_obj)

    plt.savefig(path_obj, dpi=dpi, bbox_inches=bbox_inches)  # type: ignore

    logger = get_logger(__name__)
    logger.info(f"Plot saved to {path_obj}")

    if close_after:
        plt.close()
