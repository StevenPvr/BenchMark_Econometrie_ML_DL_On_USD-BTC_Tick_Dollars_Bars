"""Validation utilities for EGARCH optimization (anti-leakage and splitting).

Provides:
- Temporal anti-leakage validation helpers
- Three-phase train/validation/test splitting utilities
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constants import (
    GARCH_VALIDATION_TEST_RATIO,
    GARCH_VALIDATION_TRAIN_RATIO,
    GARCH_VALIDATION_VAL_RATIO,
)
from src.utils import get_logger

logger = get_logger(__name__)


def validate_temporal_ordering(
    train_start: int, train_end: int, val_start: int, val_end: int
) -> None:
    """Validate temporal ordering of training and validation windows.

    Ensures:
    - Training ends before validation starts
    - No overlap between windows
    - Valid window sizes

    Args:
        train_start: Training window start index.
        train_end: Training window end index (exclusive).
        val_start: Validation window start index.
        val_end: Validation window end index (exclusive).

    Raises:
        ValueError: If temporal ordering is violated.
    """
    if train_start < 0:
        msg = f"Invalid train_start: {train_start} < 0"
        raise ValueError(msg)

    if train_end <= train_start:
        msg = f"Invalid training window: train_end ({train_end}) <= train_start ({train_start})"
        raise ValueError(msg)

    if val_end <= val_start:
        msg = f"Invalid validation window: val_end ({val_end}) <= val_start ({val_start})"
        raise ValueError(msg)

    if val_start < train_end:
        msg = (
            f"Temporal leakage detected: val_start ({val_start}) < train_end ({train_end}). "
            "Validation must start at or after training end."
        )
        raise ValueError(msg)

    if val_start != train_end:
        logger.warning(
            "Gap between training and validation: train_end=%d, val_start=%d",
            train_end,
            val_start,
        )


def validate_window_bounds(
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
    data_size: int,
) -> None:
    """Validate window bounds against data size.

    Args:
        train_start: Training window start index.
        train_end: Training window end index (exclusive).
        val_start: Validation window start index.
        val_end: Validation window end index (exclusive).
        data_size: Total data size.

    Raises:
        ValueError: If windows exceed data bounds.
    """
    if train_start < 0:
        msg = f"train_start ({train_start}) < 0"
        raise ValueError(msg)

    if train_end > data_size:
        msg = f"train_end ({train_end}) exceeds data size ({data_size})"
        raise ValueError(msg)

    if val_start < 0:
        msg = f"val_start ({val_start}) < 0"
        raise ValueError(msg)

    if val_end > data_size:
        msg = f"val_end ({val_end}) exceeds data size ({data_size})"
        raise ValueError(msg)


def validate_no_test_data_used(
    residuals_train: np.ndarray,
    residuals_full: np.ndarray,
) -> None:
    """Validate that only training data is used (no test data).

    Args:
        residuals_train: Training residuals (TRAIN split only).
        residuals_full: Full residuals (TRAIN + TEST).

    Raises:
        ValueError: If residuals_train contains test data.
    """
    if residuals_train.size > residuals_full.size:
        msg = (
            f"Training residuals ({residuals_train.size}) "
            f"larger than full residuals ({residuals_full.size})"
        )
        raise ValueError(msg)

    # Check that train is a prefix of full
    if not np.array_equal(residuals_train, residuals_full[: residuals_train.size]):
        logger.warning(
            "Training residuals are not a prefix of full residuals - "
            "may indicate data leakage or data mismatch"
        )


def validate_cv_fold(
    train_window: np.ndarray,
    val_window: np.ndarray,
    min_size: int = 50,
) -> bool:
    """Validate CV fold windows meet minimum size requirements.

    Args:
        train_window: Training window data.
        val_window: Validation window data.
        min_size: Minimum window size.

    Returns:
        True if both windows are valid.
    """
    if train_window.size < min_size:
        logger.debug("Training window too small: %d < %d", train_window.size, min_size)
        return False

    if val_window.size < min_size:
        logger.debug("Validation window too small: %d < %d", val_window.size, min_size)
        return False

    return True


def assert_no_future_information(
    train_end_idx: int,
    val_start_idx: int,
    forecast_idx: int,
) -> None:
    """Assert that forecast uses no future information.

    Verifies that:
    - Training data ends before validation starts
    - Forecast index is within validation window
    - No look-ahead bias

    Args:
        train_end_idx: Last training index used (exclusive).
        val_start_idx: First validation index.
        forecast_idx: Index being forecast.

    Raises:
        AssertionError: If future information is used.
    """
    if train_end_idx > val_start_idx:
        msg = (
            f"Training uses future information: "
            f"train_end ({train_end_idx}) > val_start ({val_start_idx})"
        )
        raise AssertionError(msg)

    if forecast_idx < val_start_idx:
        msg = (
            f"Forecast index ({forecast_idx}) < val_start ({val_start_idx}). "
            "Forecasting past validation window."
        )
        raise AssertionError(msg)

    if forecast_idx < train_end_idx:
        msg = (
            f"Look-ahead bias detected: "
            f"forecast_idx ({forecast_idx}) < train_end ({train_end_idx})"
        )
        raise AssertionError(msg)


# ==================== Three-phase validation split ====================


@dataclass
class ThreePhaseDataset:
    """Container for three-phase validation split.

    Attributes:
        train: Training residuals for model fitting.
        validation: Validation residuals for hyperparameter tuning.
        test: Test residuals for final evaluation (held out).
        train_indices: Indices of training data in original array.
        val_indices: Indices of validation data in original array.
        test_indices: Indices of test data in original array.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_indices: tuple[int, int]
    val_indices: tuple[int, int]
    test_indices: tuple[int, int]

    def __post_init__(self) -> None:
        total_ratio = (
            GARCH_VALIDATION_TRAIN_RATIO + GARCH_VALIDATION_VAL_RATIO + GARCH_VALIDATION_TEST_RATIO
        )
        if not np.isclose(total_ratio, 1.0):
            msg = f"Validation ratios must sum to 1.0, got {total_ratio}"
            raise ValueError(msg)
        if self.train.size == 0:
            raise ValueError("Training split is empty")
        if self.validation.size == 0:
            raise ValueError("Validation split is empty")
        if self.test.size == 0:
            raise ValueError("Test split is empty")


def split_three_phase(residuals: np.ndarray) -> ThreePhaseDataset:
    """Split residuals into train/validation/test sets sequentially.

    Uses ratios from constants to split in temporal order.
    """
    n_total = residuals.size

    n_train = int(n_total * GARCH_VALIDATION_TRAIN_RATIO)
    n_val = int(n_total * GARCH_VALIDATION_VAL_RATIO)
    n_test = n_total - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            f"Insufficient residuals for 3-phase split: n={n_total}, "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    train_residuals = residuals[:n_train]
    val_residuals = residuals[n_train : n_train + n_val]
    test_residuals = residuals[n_train + n_val :]

    dataset = ThreePhaseDataset(
        train=train_residuals,
        validation=val_residuals,
        test=test_residuals,
        train_indices=(0, n_train),
        val_indices=(n_train, n_train + n_val),
        test_indices=(n_train + n_val, n_total),
    )

    logger.info(
        "3-phase split: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        n_train,
        100 * GARCH_VALIDATION_TRAIN_RATIO,
        n_val,
        100 * GARCH_VALIDATION_VAL_RATIO,
        n_test,
        100 * GARCH_VALIDATION_TEST_RATIO,
    )

    return dataset


def combine_train_validation(dataset: ThreePhaseDataset) -> np.ndarray:
    """Combine train and validation sets for final training."""
    combined = np.concatenate([dataset.train, dataset.validation])
    logger.debug(
        "Combined train+val: %d observations (%d train + %d val)",
        combined.size,
        dataset.train.size,
        dataset.validation.size,
    )
    return combined


__all__ = [
    "validate_temporal_ordering",
    "validate_window_bounds",
    "validate_no_test_data_used",
    "validate_cv_fold",
    "assert_no_future_information",
    # Three-phase split
    "ThreePhaseDataset",
    "split_three_phase",
    "combine_train_validation",
]
