"""Core scaler classes for feature normalization.

This module provides the scaler classes for feature normalization:
- StandardScalerCustom: Z-score normalization (mean=0, std=1)
- MinMaxScalerCustom: Min-max normalization to [-1, 1] range

Both scalers support:
- Incremental fitting via partial_fit() for large datasets
- Saving/loading with joblib
- Proper handling of NaN and inf values

CRITICAL: Never fit on test data - this causes data leakage!

Reference:
    Sklearn preprocessing documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from numpy.typing import NDArray

from src.constants import SCALER_MIN_RANGE
from src.utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "StandardScalerCustom",
    "MinMaxScalerCustom",
]


@dataclass
class StandardScalerCustom:
    """Standard scaler (z-score normalization): z = (x - mean) / std."""

    mean_: NDArray[np.float64] | None = None
    std_: NDArray[np.float64] | None = None
    columns_: list[str] | None = None
    _n_samples_seen_: NDArray[np.int64] | None = None
    _var_: NDArray[np.float64] | None = None

    def partial_fit(self, df: pd.DataFrame, columns: list[str] | None = None) -> "StandardScalerCustom":
        """Incrementally fit using Welford's algorithm."""
        if columns is not None:
            self.columns_ = columns
        elif self.columns_ is None:
            raise ValueError("columns must be provided on first call to partial_fit")

        # Build mapping of column name to index in self.columns_
        col_to_idx = {c: i for i, c in enumerate(self.columns_)}

        # Only use columns that exist in this batch
        available_cols = [c for c in self.columns_ if c in df.columns]
        if not available_cols:
            return self

        data = df[available_cols].values.astype(np.float64)
        # Replace inf with nan
        data = np.where(np.isinf(data), np.nan, data)

        n_cols = len(self.columns_)

        # Initialize on first batch
        if self.mean_ is None:
            self.mean_ = np.zeros(n_cols, dtype=np.float64)
            self._var_ = np.zeros(n_cols, dtype=np.float64)
            self._n_samples_seen_ = np.zeros(n_cols, dtype=np.int64)

        # Type assertions for type checker
        assert self.mean_ is not None
        assert self._var_ is not None
        assert self._n_samples_seen_ is not None

        # Welford's online algorithm for each available column
        for data_idx, col_name in enumerate(available_cols):
            col_idx = col_to_idx[col_name]  # Index in self.columns_
            col_data = data[:, data_idx]
            valid_mask = ~np.isnan(col_data)
            col_valid = col_data[valid_mask]

            if len(col_valid) == 0:
                continue

            for x in col_valid:
                self._n_samples_seen_[col_idx] += 1
                n = self._n_samples_seen_[col_idx]
                delta = x - self.mean_[col_idx]
                self.mean_[col_idx] += delta / n
                delta2 = x - self.mean_[col_idx]
                self._var_[col_idx] += delta * delta2

        return self

    def finalize_fit(self) -> "StandardScalerCustom":
        """Finalize incremental fitting by computing std from variance."""
        if self._var_ is None or self._n_samples_seen_ is None:
            raise ValueError("No data has been fitted. Call partial_fit() first.")

        total_samples = self._n_samples_seen_.sum()
        if total_samples == 0:
            raise ValueError("No valid samples found. Call partial_fit() first.")

        # Compute std from variance (per column)
        self.std_ = np.zeros_like(self._var_)
        for i in range(len(self._var_)):
            n = self._n_samples_seen_[i]
            if n > 1:
                self.std_[i] = np.sqrt(self._var_[i] / (n - 1))
            else:
                self.std_[i] = 1.0

        # Avoid division by zero
        self.std_ = np.where(
            cast(NDArray[np.float64], self.std_) < SCALER_MIN_RANGE,
            1.0,
            cast(NDArray[np.float64], self.std_),
        )

        # Handle NaN
        self.mean_ = np.where(
            np.isnan(cast(NDArray[np.float64], self.mean_)),
            0.0,
            cast(NDArray[np.float64], self.mean_),
        )

        logger.info(
            "StandardScaler finalized: %d columns, %d total samples",
            len(self.columns_) if self.columns_ else 0,
            total_samples,
        )

        return self

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "StandardScalerCustom":
        """Fit scaler on training data."""
        self.columns_ = columns
        data = df[columns].values.astype(np.float64)

        # Replace inf with nan before computing statistics
        data = np.where(np.isinf(data), np.nan, data)

        # Compute mean and std, ignoring NaN
        self.mean_ = np.nanmean(data, axis=0)
        self.std_ = np.nanstd(data, axis=0, ddof=1)

        # Avoid division by zero
        self.std_ = np.where(
            cast(NDArray[np.float64], self.std_) < SCALER_MIN_RANGE,
            1.0,
            cast(NDArray[np.float64], self.std_),
        )

        # Handle case where mean is NaN (all values were NaN/inf)
        self.mean_ = np.where(
            np.isnan(cast(NDArray[np.float64], self.mean_)),
            0.0,
            cast(NDArray[np.float64], self.mean_),
        )

        logger.info(
            "StandardScaler fit on %d columns, %d rows",
            len(columns),
            len(df),
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        if self.mean_ is None or self.std_ is None or self.columns_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        result = df.copy()

        for i, col in enumerate(self.columns_):
            if col in df.columns:
                values = df[col].values.astype(np.float64)
                # Handle NaN and inf properly - only transform finite values
                mask = np.isfinite(values)
                scaled = np.full_like(values, np.nan)
                if mask.any():
                    scaled[mask] = (values[mask] - self.mean_[i]) / self.std_[i]
                result[col] = scaled
        return result

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, columns)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform to original scale."""
        if self.mean_ is None or self.std_ is None or self.columns_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        result = df.copy()
        for i, col in enumerate(self.columns_):
            if col in df.columns:
                result[col] = df[col].values * self.std_[i] + self.mean_[i]
        return result

    def save(self, path: str | Path) -> None:
        """Save scaler to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved StandardScaler to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "StandardScalerCustom":
        """Load scaler from file."""
        scaler = joblib.load(path)
        logger.info("Loaded StandardScaler from %s", path)
        return scaler


@dataclass
class MinMaxScalerCustom:
    """Min-max scaler to [-1, 1] range: x_scaled = 2 * (x - min) / (max - min) - 1."""

    min_: NDArray[np.float64] | None = None
    max_: NDArray[np.float64] | None = None
    columns_: list[str] | None = None

    def partial_fit(self, df: pd.DataFrame, columns: list[str] | None = None) -> "MinMaxScalerCustom":
        """Incrementally fit scaler, updating min/max with each batch."""
        if columns is not None:
            self.columns_ = columns
        elif self.columns_ is None:
            raise ValueError("columns must be provided on first call to partial_fit")

        # Build mapping of column name to index in self.columns_
        col_to_idx = {c: i for i, c in enumerate(self.columns_)}

        # Only use columns that exist in this batch
        available_cols = [c for c in self.columns_ if c in df.columns]
        if not available_cols:
            return self

        data = df[available_cols].values.astype(np.float64)
        # Replace inf with nan
        data = np.where(np.isinf(data), np.nan, data)

        n_cols = len(self.columns_)

        # Initialize on first batch
        if self.min_ is None:
            self.min_ = np.full(n_cols, np.inf, dtype=np.float64)
            self.max_ = np.full(n_cols, -np.inf, dtype=np.float64)

        # Update min/max for available columns
        for data_idx, col_name in enumerate(available_cols):
            col_idx = col_to_idx[col_name]
            col_data = data[:, data_idx]
            valid_mask = ~np.isnan(col_data)
            col_valid = col_data[valid_mask]

            if len(col_valid) == 0:
                continue

            batch_min = np.min(col_valid)
            batch_max = np.max(col_valid)
            cast(NDArray[np.float64], self.min_)[col_idx] = min(
                cast(NDArray[np.float64], self.min_)[col_idx], batch_min
            )
            cast(NDArray[np.float64], self.max_)[col_idx] = max(
                cast(NDArray[np.float64], self.max_)[col_idx], batch_max
            )

        return self

    def finalize_fit(self) -> "MinMaxScalerCustom":
        """Finalize incremental fitting."""
        if self.min_ is None or self.max_ is None:
            raise ValueError("No data has been fitted. Call partial_fit() first.")

        # Handle columns that never received data (still at inf/-inf)
        # Also handle NaN (all values were NaN/inf)
        self.min_ = np.where(np.isinf(self.min_) | np.isnan(self.min_), 0.0, self.min_)
        self.max_ = np.where(np.isinf(self.max_) | np.isnan(self.max_), 1.0, self.max_)

        # Avoid division by zero
        range_ = self.max_ - self.min_
        range_ = np.where(range_ < SCALER_MIN_RANGE, 1.0, range_)
        self.max_ = self.min_ + range_

        logger.info(
            "MinMaxScaler finalized: %d columns",
            len(self.columns_) if self.columns_ else 0,
        )

        return self

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "MinMaxScalerCustom":
        """Fit scaler on training data."""
        self.columns_ = columns
        data = df[columns].values.astype(np.float64)

        # Replace inf with nan before computing statistics
        data = np.where(np.isinf(data), np.nan, data)

        # Compute min and max, ignoring NaN
        self.min_ = np.nanmin(data, axis=0)
        self.max_ = np.nanmax(data, axis=0)

        # Handle case where min/max is NaN (all values were NaN/inf)
        self.min_ = np.where(
            np.isnan(cast(NDArray[np.float64], self.min_)),
            0.0,
            cast(NDArray[np.float64], self.min_),
        )
        self.max_ = np.where(
            np.isnan(cast(NDArray[np.float64], self.max_)),
            1.0,
            cast(NDArray[np.float64], self.max_),
        )

        # Avoid division by zero
        range_ = cast(NDArray[np.float64], self.max_) - cast(NDArray[np.float64], self.min_)
        range_ = np.where(range_ < SCALER_MIN_RANGE, 1.0, range_)
        self.max_ = cast(NDArray[np.float64], self.min_) + range_

        logger.info(
            "MinMaxScaler fit on %d columns, %d rows",
            len(columns),
            len(df),
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        if self.min_ is None or self.max_ is None or self.columns_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        result = df.copy()

        for i, col in enumerate(self.columns_):
            if col in df.columns:
                values = df[col].values.astype(np.float64)
                # Handle NaN and inf properly - only transform finite values
                mask = np.isfinite(values)
                scaled = np.full_like(values, np.nan)
                if mask.any():
                    range_ = self.max_[i] - self.min_[i]
                    scaled[mask] = 2 * (values[mask] - self.min_[i]) / range_ - 1
                    # Clip to handle values outside train range
                    scaled[mask] = np.clip(scaled[mask], -1, 1)
                result[col] = scaled
        return result

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, columns)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform to original scale."""
        if self.min_ is None or self.max_ is None or self.columns_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        result = df.copy()
        for i, col in enumerate(self.columns_):
            if col in df.columns:
                range_ = self.max_[i] - self.min_[i]
                result[col] = (cast(NDArray[np.float64], df[col].values) + 1) / 2 * range_ + self.min_[i]
        return result

    def save(self, path: str | Path) -> None:
        """Save scaler to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved MinMaxScaler to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxScalerCustom":
        """Load scaler from file."""
        scaler = joblib.load(path)
        logger.info("Loaded MinMaxScaler from %s", path)
        return scaler
