"""Scalers for feature normalization with train/test split handling.

This module provides proper scaling that:
1. Fits ONLY on training data (no data leakage)
2. Transforms both train and test with the same parameters
3. Saves scalers for later use in inference

Available scalers:
- StandardScaler: Z-score normalization (mean=0, std=1)
- MinMaxScaler: Min-max normalization to [-1, 1] range

CRITICAL: Never fit on test data - this causes data leakage!

Reference:
    Sklearn preprocessing documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "StandardScalerCustom",
    "MinMaxScalerCustom",
    "ScalerManager",
    "fit_and_transform_features",
]


# =============================================================================
# CUSTOM SCALERS (Numba-compatible, saveable)
# =============================================================================


@dataclass
class StandardScalerCustom:
    """Standard scaler (z-score normalization).

    z = (x - mean) / std

    Attributes:
        mean_: Mean values per column (fit on train).
        std_: Standard deviation per column (fit on train).
        columns_: Column names.
    """

    mean_: NDArray[np.float64] | None = None
    std_: NDArray[np.float64] | None = None
    columns_: list[str] | None = None

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "StandardScalerCustom":
        """Fit scaler on training data.

        Args:
            df: Training DataFrame.
            columns: Columns to scale.

        Returns:
            Self for chaining.
        """
        self.columns_ = columns
        data = df[columns].values.astype(np.float64)

        # Compute mean and std, ignoring NaN
        self.mean_ = np.nanmean(data, axis=0)
        self.std_ = np.nanstd(data, axis=0, ddof=1)

        # Avoid division by zero
        self.std_ = np.where(self.std_ < 1e-10, 1.0, self.std_)

        logger.info(
            "StandardScaler fit on %d columns, %d rows",
            len(columns),
            len(df),
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame with scaled columns.
        """
        if self.mean_ is None or self.std_ is None or self.columns_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        result = df.copy()

        for i, col in enumerate(self.columns_):
            if col in df.columns:
                values = df[col].values.astype(np.float64)
                # Handle NaN properly - only transform non-NaN values
                mask = ~np.isnan(values)
                scaled = np.full_like(values, np.nan)
                if mask.any():
                    scaled[mask] = (values[mask] - self.mean_[i]) / self.std_[i]
                result[col] = scaled

        return result

    def fit_transform(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame.
            columns: Columns to scale.

        Returns:
            Transformed DataFrame.
        """
        self.fit(df, columns)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform to original scale.

        Args:
            df: Scaled DataFrame.

        Returns:
            DataFrame in original scale.
        """
        if self.mean_ is None or self.std_ is None or self.columns_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        result = df.copy()

        for i, col in enumerate(self.columns_):
            if col in df.columns:
                result[col] = df[col].values * self.std_[i] + self.mean_[i]

        return result

    def save(self, path: str | Path) -> None:
        """Save scaler to file.

        Args:
            path: Output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved StandardScaler to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "StandardScalerCustom":
        """Load scaler from file.

        Args:
            path: Input path.

        Returns:
            Loaded scaler.
        """
        scaler = joblib.load(path)
        logger.info("Loaded StandardScaler from %s", path)
        return scaler


@dataclass
class MinMaxScalerCustom:
    """Min-max scaler to [-1, 1] range.

    x_scaled = 2 * (x - min) / (max - min) - 1

    Attributes:
        min_: Minimum values per column (fit on train).
        max_: Maximum values per column (fit on train).
        columns_: Column names.
    """

    min_: NDArray[np.float64] | None = None
    max_: NDArray[np.float64] | None = None
    columns_: list[str] | None = None

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "MinMaxScalerCustom":
        """Fit scaler on training data.

        Args:
            df: Training DataFrame.
            columns: Columns to scale.

        Returns:
            Self for chaining.
        """
        self.columns_ = columns
        data = df[columns].values.astype(np.float64)

        # Compute min and max, ignoring NaN
        self.min_ = np.nanmin(data, axis=0)
        self.max_ = np.nanmax(data, axis=0)

        # Avoid division by zero
        range_ = self.max_ - self.min_
        range_ = np.where(range_ < 1e-10, 1.0, range_)
        self.max_ = self.min_ + range_

        logger.info(
            "MinMaxScaler fit on %d columns, %d rows",
            len(columns),
            len(df),
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame with scaled columns in [-1, 1].
        """
        if self.min_ is None or self.max_ is None or self.columns_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        result = df.copy()

        for i, col in enumerate(self.columns_):
            if col in df.columns:
                values = df[col].values.astype(np.float64)
                # Handle NaN properly - only transform non-NaN values
                mask = ~np.isnan(values)
                scaled = np.full_like(values, np.nan)
                if mask.any():
                    range_ = self.max_[i] - self.min_[i]
                    scaled[mask] = 2 * (values[mask] - self.min_[i]) / range_ - 1
                    # Clip to handle values outside train range
                    scaled[mask] = np.clip(scaled[mask], -1, 1)
                result[col] = scaled

        return result

    def fit_transform(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame.
            columns: Columns to scale.

        Returns:
            Transformed DataFrame.
        """
        self.fit(df, columns)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform to original scale.

        Args:
            df: Scaled DataFrame.

        Returns:
            DataFrame in original scale.
        """
        if self.min_ is None or self.max_ is None or self.columns_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        result = df.copy()

        for i, col in enumerate(self.columns_):
            if col in df.columns:
                range_ = self.max_[i] - self.min_[i]
                result[col] = (df[col].values + 1) / 2 * range_ + self.min_[i]

        return result

    def save(self, path: str | Path) -> None:
        """Save scaler to file.

        Args:
            path: Output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved MinMaxScaler to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxScalerCustom":
        """Load scaler from file.

        Args:
            path: Input path.

        Returns:
            Loaded scaler.
        """
        scaler = joblib.load(path)
        logger.info("Loaded MinMaxScaler from %s", path)
        return scaler


# =============================================================================
# SCALER MANAGER
# =============================================================================


@dataclass
class ScalerManager:
    """Manager for multiple scalers with train/test handling.

    Handles proper scaling workflow:
    1. Split data into train/test
    2. Fit scalers on train only
    3. Transform both train and test
    4. Save scalers for inference
    """

    zscore_scaler: StandardScalerCustom | None = None
    minmax_scaler: MinMaxScalerCustom | None = None

    def save_all(self, scalers_dir: str | Path) -> None:
        """Save all scalers.

        Args:
            scalers_dir: Directory for scaler files.
        """
        scalers_dir = Path(scalers_dir)
        scalers_dir.mkdir(parents=True, exist_ok=True)

        if self.zscore_scaler is not None:
            self.zscore_scaler.save(scalers_dir / "zscore_scaler.joblib")

        if self.minmax_scaler is not None:
            self.minmax_scaler.save(scalers_dir / "minmax_scaler.joblib")

    def load_all(self, scalers_dir: str | Path) -> "ScalerManager":
        """Load all scalers.

        Args:
            scalers_dir: Directory with scaler files.

        Returns:
            Self for chaining.
        """
        scalers_dir = Path(scalers_dir)

        zscore_path = scalers_dir / "zscore_scaler.joblib"
        if zscore_path.exists():
            self.zscore_scaler = StandardScalerCustom.load(zscore_path)

        minmax_path = scalers_dir / "minmax_scaler.joblib"
        if minmax_path.exists():
            self.minmax_scaler = MinMaxScalerCustom.load(minmax_path)

        return self


# =============================================================================
# HIGH-LEVEL API
# =============================================================================


def fit_and_transform_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    columns_to_scale: list[str],
    scaler_type: str = "zscore",
    scaler_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScalerCustom | MinMaxScalerCustom]:
    """Fit scaler on train, transform both train and test.

    CRITICAL: This is the correct way to scale features:
    - Fit ONLY on training data
    - Transform both train and test with same parameters
    - No data leakage from test to train

    Args:
        df_train: Training DataFrame.
        df_test: Test DataFrame.
        columns_to_scale: Columns to scale.
        scaler_type: "zscore" or "minmax".
        scaler_path: Optional path to save scaler.

    Returns:
        Tuple of (scaled_train, scaled_test, scaler).

    Example:
        >>> train_scaled, test_scaled, scaler = fit_and_transform_features(
        ...     df_train, df_test, feature_cols, scaler_type="minmax"
        ... )
    """
    # Select scaler type
    if scaler_type == "zscore":
        scaler: StandardScalerCustom | MinMaxScalerCustom = StandardScalerCustom()
    elif scaler_type == "minmax":
        scaler = MinMaxScalerCustom()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    # Fit on train only
    logger.info(
        "Fitting %s scaler on TRAIN data (%d rows, %d columns)",
        scaler_type,
        len(df_train),
        len(columns_to_scale),
    )
    scaler.fit(df_train, columns_to_scale)

    # Transform train
    df_train_scaled = scaler.transform(df_train)
    logger.info("Transformed TRAIN data")

    # Transform test
    df_test_scaled = scaler.transform(df_test)
    logger.info("Transformed TEST data")

    # Save scaler if path provided
    if scaler_path is not None:
        scaler.save(scaler_path)

    return df_train_scaled, df_test_scaled, scaler


def get_columns_to_scale(
    df: pd.DataFrame,
    exclude_target: bool = True,
    include_log_return_lags: bool = False,
) -> list[str]:
    """Get list of columns to scale.

    Args:
        df: DataFrame with features.
        exclude_target: If True, exclude log_return (target).
        include_log_return_lags: If True, include log_return_lag* columns.

    Returns:
        List of column names to scale.
    """
    # Patterns to always exclude
    # Note: Use suffixes for sin/cos to avoid matching "bars_since"
    exclude_exact_suffixes = ["_sin", "_cos"]
    exclude_patterns = [
        "timestamp",
        "datetime",
        "date",
        "crash_",
        "vol_regime_",
        "cross_ma_",
    ]

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    cols_to_scale = []
    for col in numeric_cols:
        col_lower = col.lower()

        # Check if should exclude
        exclude = False

        # Check exact suffixes (e.g., _sin, _cos)
        for suffix in exclude_exact_suffixes:
            if col_lower.endswith(suffix):
                exclude = True
                break

        # Check substring patterns
        if not exclude:
            for pattern in exclude_patterns:
                if pattern in col_lower:
                    exclude = True
                    break

        # Handle log_return specially
        if col == "log_return" and exclude_target:
            exclude = True
        elif "log_return_lag" in col_lower and not include_log_return_lags:
            exclude = True

        if not exclude:
            cols_to_scale.append(col)

    return cols_to_scale
