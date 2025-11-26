"""Walk-forward cross-validation with purge for time series.

This module implements temporal cross-validation with:
- Purge gap between train and test to prevent data leakage
- Expanding or rolling window strategies
- Compatible with any model implementing the BaseModel interface
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol

import numpy as np
import pandas as pd # type: ignore[import-untyped]

from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default walk-forward parameters
DEFAULT_N_SPLITS: int = 5
DEFAULT_PURGE_GAP: int = 1
DEFAULT_MIN_TRAIN_SIZE: int = 100
DEFAULT_WINDOW_TYPE: str = "expanding"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward cross-validation.

    Attributes:
        n_splits: Number of CV splits (folds).
        purge_gap: Number of samples to purge between train and test.
        min_train_size: Minimum training set size.
        window_type: "expanding" or "rolling".
        rolling_window_size: Fixed window size for rolling (required if rolling).
    """

    n_splits: int = DEFAULT_N_SPLITS
    purge_gap: int = DEFAULT_PURGE_GAP
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE
    window_type: str = DEFAULT_WINDOW_TYPE
    rolling_window_size: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {self.purge_gap}")
        if self.min_train_size < 10:
            raise ValueError(f"min_train_size must be >= 10, got {self.min_train_size}")
        if self.window_type not in ("expanding", "rolling"):
            raise ValueError(f"window_type must be 'expanding' or 'rolling', got {self.window_type}")
        if self.window_type == "rolling" and self.rolling_window_size is None:
            raise ValueError("rolling_window_size required when window_type='rolling'")
        if self.rolling_window_size is not None and self.rolling_window_size < self.min_train_size:
            raise ValueError(
                f"rolling_window_size ({self.rolling_window_size}) must be >= "
                f"min_train_size ({self.min_train_size})"
            )


# ============================================================================
# PROTOCOL FOR MODELS
# ============================================================================


class FittableModel(Protocol):
    """Protocol for models that can be fitted and make predictions."""

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "FittableModel":
        """Fit the model on training data."""
        ...

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        ...

    def set_params(self, **params: Any) -> "FittableModel":
        """Set model hyperparameters."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Get model hyperparameters."""
        ...


# ============================================================================
# WALK-FORWARD CV SPLITTER
# ============================================================================


@dataclass
class CVFold:
    """Represents a single cross-validation fold.

    Attributes:
        fold_idx: Index of the fold (0-based).
        train_indices: Array of training indices.
        test_indices: Array of test indices.
        train_start: First training index.
        train_end: Last training index (exclusive).
        test_start: First test index.
        test_end: Last test index (exclusive).
    """

    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class WalkForwardSplitter:
    """Walk-forward cross-validation splitter with purge gap.

    This splitter implements temporal CV where:
    - Training data always comes before test data
    - A purge gap removes samples between train and test to prevent leakage
    - Supports both expanding and rolling windows

    Example:
        With n_splits=3, purge_gap=2, and 100 samples:
        - Fold 1: train=[0:28], purge=[28:30], test=[30:50]
        - Fold 2: train=[0:48], purge=[48:50], test=[50:75]
        - Fold 3: train=[0:73], purge=[73:75], test=[75:100]
    """

    def __init__(self, config: WalkForwardConfig) -> None:
        """Initialize the splitter.

        Args:
            config: Walk-forward configuration.
        """
        self.config = config

    def split(self, n_samples: int) -> Iterator[CVFold]:
        """Generate train/test indices for each fold.

        Args:
            n_samples: Total number of samples.

        Yields:
            CVFold objects containing train and test indices.

        Raises:
            ValueError: If insufficient data for requested splits.
        """
        self._validate_data_size(n_samples)

        # Calculate test size per fold
        # Reserve space for purge gaps
        total_purge = self.config.purge_gap * self.config.n_splits
        available_for_test = n_samples - self.config.min_train_size - total_purge
        test_size_per_fold = available_for_test // self.config.n_splits

        if test_size_per_fold < 1:
            raise ValueError(
                f"Insufficient data for {self.config.n_splits} splits with "
                f"purge_gap={self.config.purge_gap} and min_train_size={self.config.min_train_size}. "
                f"Need at least {self.config.min_train_size + total_purge + self.config.n_splits} samples."
            )

        # Generate folds
        test_start = self.config.min_train_size + self.config.purge_gap
        for fold_idx in range(self.config.n_splits):
            # Calculate test end
            if fold_idx == self.config.n_splits - 1:
                test_end = n_samples  # Last fold gets remaining samples
            else:
                test_end = test_start + test_size_per_fold

            # Calculate train bounds
            train_end = test_start - self.config.purge_gap

            if self.config.window_type == "rolling" and self.config.rolling_window_size is not None:
                train_start = max(0, train_end - self.config.rolling_window_size)
            else:
                train_start = 0

            # Create fold
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield CVFold(
                fold_idx=fold_idx,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            # Move to next fold
            test_start = test_end + self.config.purge_gap

    def _validate_data_size(self, n_samples: int) -> None:
        """Validate that we have enough data for the requested configuration."""
        min_required = (
            self.config.min_train_size
            + self.config.purge_gap * self.config.n_splits
            + self.config.n_splits  # At least 1 test sample per fold
        )
        if n_samples < min_required:
            raise ValueError(
                f"Need at least {min_required} samples for {self.config.n_splits} splits "
                f"with purge_gap={self.config.purge_gap}, got {n_samples}"
            )


# ============================================================================
# METRICS
# ============================================================================


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """QLIKE loss for variance forecasting.

    QLIKE = mean(y_true / y_pred + log(y_pred))

    Args:
        y_true: Actual values (must be positive for variance).
        y_pred: Predicted values (must be positive for variance).
        epsilon: Small value to avoid division by zero.

    Returns:
        QLIKE loss value.
    """
    y_pred_safe = np.maximum(y_pred, epsilon)
    return float(np.mean(y_true / y_pred_safe + np.log(y_pred_safe)))


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "qlike": qlike,
}


# ============================================================================
# WALK-FORWARD CROSS-VALIDATION
# ============================================================================


@dataclass
class CVResult:
    """Result from a single CV fold.

    Attributes:
        fold_idx: Index of the fold.
        metric_value: Evaluation metric value.
        train_size: Number of training samples.
        test_size: Number of test samples.
        predictions: Model predictions on test set.
        actuals: Actual test values.
    """

    fold_idx: int
    metric_value: float
    train_size: int
    test_size: int
    predictions: np.ndarray | None = None
    actuals: np.ndarray | None = None


@dataclass
class WalkForwardResult:
    """Result from walk-forward cross-validation.

    Attributes:
        mean_score: Mean metric value across folds.
        std_score: Standard deviation of metric values.
        fold_results: List of individual fold results.
        metric_name: Name of the metric used.
        config: Configuration used for CV.
    """

    mean_score: float
    std_score: float
    fold_results: list[CVResult]
    metric_name: str
    config: WalkForwardConfig


def walk_forward_cv(
    model: FittableModel,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    config: WalkForwardConfig,
    metric: str = "mse",
    return_predictions: bool = False,
    verbose: bool = False,
    fit_kwargs: dict[str, Any] | None = None,
) -> WalkForwardResult:
    """Perform walk-forward cross-validation with purge.

    Args:
        model: Model implementing FittableModel protocol.
        X: Features array or DataFrame.
        y: Target array or Series.
        config: Walk-forward configuration.
        metric: Metric to use ("mse", "rmse", "mae", "qlike").
        return_predictions: If True, store predictions in results.
        verbose: If True, log progress.
        fit_kwargs: Additional kwargs for model.fit().

    Returns:
        WalkForwardResult with mean score and fold details.

    Raises:
        ValueError: If metric is unknown or data is invalid.
    """
    if metric not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(METRIC_FUNCTIONS.keys())}")

    metric_fn = METRIC_FUNCTIONS[metric]
    fit_kwargs = fit_kwargs or {}

    # Convert to numpy arrays
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()

    if len(X_arr) != len(y_arr):
        raise ValueError(f"X and y must have same length: {len(X_arr)} != {len(y_arr)}")

    splitter = WalkForwardSplitter(config)
    fold_results: list[CVResult] = []

    for fold in splitter.split(len(X_arr)):
        # Extract train/test data
        X_train = X_arr[fold.train_indices]
        y_train = y_arr[fold.train_indices]
        X_test = X_arr[fold.test_indices]
        y_test = y_arr[fold.test_indices]

        if verbose:
            logger.info(
                "Fold %d: train=[%d:%d] (%d), purge=%d, test=[%d:%d] (%d)",
                fold.fold_idx + 1,
                fold.train_start,
                fold.train_end,
                len(X_train),
                config.purge_gap,
                fold.test_start,
                fold.test_end,
                len(X_test),
            )

        # Fit model
        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except Exception as e:
            logger.warning("Fold %d fit failed: %s", fold.fold_idx + 1, e)
            fold_results.append(
                CVResult(
                    fold_idx=fold.fold_idx,
                    metric_value=float("inf"),
                    train_size=len(X_train),
                    test_size=len(X_test),
                )
            )
            continue

        # Predict and evaluate
        try:
            predictions = model.predict(X_test)
            metric_value = metric_fn(y_test, predictions)
        except Exception as e:
            logger.warning("Fold %d prediction failed: %s", fold.fold_idx + 1, e)
            fold_results.append(
                CVResult(
                    fold_idx=fold.fold_idx,
                    metric_value=float("inf"),
                    train_size=len(X_train),
                    test_size=len(X_test),
                )
            )
            continue

        fold_results.append(
            CVResult(
                fold_idx=fold.fold_idx,
                metric_value=metric_value,
                train_size=len(X_train),
                test_size=len(X_test),
                predictions=predictions if return_predictions else None,
                actuals=y_test if return_predictions else None,
            )
        )

        if verbose:
            logger.info("Fold %d: %s=%.6f", fold.fold_idx + 1, metric, metric_value)

    # Compute summary statistics
    valid_scores = [r.metric_value for r in fold_results if np.isfinite(r.metric_value)]

    if not valid_scores:
        logger.warning("No valid folds completed")
        return WalkForwardResult(
            mean_score=float("inf"),
            std_score=float("nan"),
            fold_results=fold_results,
            metric_name=metric,
            config=config,
        )

    mean_score = float(np.mean(valid_scores))
    std_score = float(np.std(valid_scores)) if len(valid_scores) > 1 else 0.0

    if verbose:
        logger.info(
            "CV complete: %s=%.6f (+/- %.6f) over %d folds",
            metric,
            mean_score,
            std_score,
            len(valid_scores),
        )

    return WalkForwardResult(
        mean_score=mean_score,
        std_score=std_score,
        fold_results=fold_results,
        metric_name=metric,
        config=config,
    )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_cv_config(
    n_splits: int = DEFAULT_N_SPLITS,
    purge_gap: int = DEFAULT_PURGE_GAP,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    window_type: str = DEFAULT_WINDOW_TYPE,
    rolling_window_size: int | None = None,
) -> WalkForwardConfig:
    """Create a walk-forward configuration.

    Args:
        n_splits: Number of CV splits.
        purge_gap: Samples to purge between train and test.
        min_train_size: Minimum training size.
        window_type: "expanding" or "rolling".
        rolling_window_size: Window size for rolling.

    Returns:
        WalkForwardConfig instance.
    """
    return WalkForwardConfig(
        n_splits=n_splits,
        purge_gap=purge_gap,
        min_train_size=min_train_size,
        window_type=window_type,
        rolling_window_size=rolling_window_size,
    )
