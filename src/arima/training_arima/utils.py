"""Utility functions for ARIMA model training."""

from __future__ import annotations


import pandas as pd

from src.constants import ARIMA_EMPTY_TRAINING_SERIES_MSG


def _validate_non_negative_integers(
    values: tuple[int, ...], param_name: str, expected_len: int
) -> None:
    """Validate that values are non-negative integers of expected length."""
    if len(values) != expected_len:
        msg = f"Invalid {param_name}: {values}. Must be tuple of {expected_len} values"
        raise ValueError(msg)

    if any(not isinstance(x, int) or x < 0 for x in values):
        msg = f"Invalid {param_name}: {values}. All values must be non-negative integers"
        raise ValueError(msg)


def _validate_order(order: tuple[int, int, int]) -> None:
    """Validate ARIMA order parameter."""
    _validate_non_negative_integers(order, "order", 3)


def validate_arima_parameters(
    train_series: pd.Series,
    order: tuple[int, int, int],
) -> None:
    """
    Validate ARIMA model parameters.

    Args:
        train_series: Training time series data
        order: ARIMA order (p, d, q)

    Raises:
        ValueError: If parameters are invalid
    """
    if train_series.empty:
        raise ValueError(ARIMA_EMPTY_TRAINING_SERIES_MSG)

    _validate_order(order)
