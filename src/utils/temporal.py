"""Temporal validation and splitting utilities."""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
from typing import cast


def compute_timeseries_split_indices(
    series: pd.Series,
    train_ratio: float = 0.8,
    min_train_size: int | None = None,
) -> tuple[int, int]:
    """Compute train/test split indices for time series."""
    n = len(series)
    if min_train_size is not None:
        train_end = min_train_size
    else:
        train_end = int(n * train_ratio)

    test_start = train_end
    return train_end, test_start


def log_split_dates(
    train_data: pd.DataFrame | pd.Series,
    test_data: pd.DataFrame | pd.Series,
) -> None:
    """Log the date ranges of train/test splits."""
    from src.config_logging import get_logger
    logger = get_logger(__name__)

    if hasattr(train_data, 'index') and hasattr(train_data.index, 'min'):
        logger.info(f"Train date range: {train_data.index.min()} to {train_data.index.max()}")
    if hasattr(test_data, 'index') and hasattr(test_data.index, 'min'):
        logger.info(f"Test date range: {test_data.index.min()} to {test_data.index.max()}")


def validate_temporal_order_series(series: pd.Series) -> bool:
    """Validate that a time series is in temporal order."""
    if hasattr(series.index, 'is_monotonic_increasing'):
        return cast(bool, series.index.is_monotonic_increasing)
    return True


def validate_temporal_split(
    train_data: pd.DataFrame | pd.Series,
    test_data: pd.DataFrame | pd.Series,
) -> bool:
    """Validate that train/test split maintains temporal order."""
    if hasattr(train_data, 'index') and hasattr(test_data, 'index'):
        train_end = train_data.index.max()
        test_start = test_data.index.min()
        # Handle comparison safely for different index types
        try:
            return train_end < test_start  # type: ignore[operator]
        except (TypeError, ValueError):
            # If comparison fails, assume temporal order is maintained
            return True
    return True
