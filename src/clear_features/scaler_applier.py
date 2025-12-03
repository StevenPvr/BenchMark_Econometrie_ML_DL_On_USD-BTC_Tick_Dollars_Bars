"""Scaler fitting and application for clear_features module.

Scalers are fit AFTER PCA transformation on TRAIN data only to avoid data leakage.
This ensures scalers are fitted on the actual columns that will be in the final dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.clear_features.config import (
    CLEAR_FEATURES_DIR,
    META_COLUMNS,
    TARGET_COLUMN,
)
from src.config_logging import get_logger
from src.features.scalers import MinMaxScalerCustom, StandardScalerCustom

logger = get_logger(__name__)

# Default paths for scalers (in clear_features directory, not features)
ZSCORE_SCALER_PATH = CLEAR_FEATURES_DIR / "zscore_scaler.joblib"
MINMAX_SCALER_PATH = CLEAR_FEATURES_DIR / "minmax_scaler.joblib"


class ScalerFitter:
    """Incrementally fits scalers on PCA-transformed data.

    Uses Welford's online algorithm for incremental mean/variance computation.
    Fits on TRAIN data only to prevent data leakage.
    """

    def __init__(self) -> None:
        """Initialize scaler fitter with empty state."""
        # Z-score (StandardScaler) stats - Welford's algorithm
        self._zscore_n: int = 0
        self._zscore_mean: np.ndarray | None = None
        self._zscore_m2: np.ndarray | None = None  # Sum of squared differences
        self._zscore_columns: list[str] | None = None

        # Min-max stats
        self._minmax_min: np.ndarray | None = None
        self._minmax_max: np.ndarray | None = None
        self._minmax_columns: list[str] | None = None

        self._is_finalized = False

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get feature columns to scale (exclude meta and target).

        Args:
            df: DataFrame to extract columns from.

        Returns:
            List of feature column names to scale.
        """
        excluded = set(META_COLUMNS) | {TARGET_COLUMN}
        return [
            c for c in df.columns
            if c not in excluded
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

    def partial_fit(self, df_train_batch: pd.DataFrame) -> None:
        """Incrementally fit scalers on a batch of TRAIN data.

        Args:
            df_train_batch: Batch of training data (already PCA-transformed).

        Raises:
            RuntimeError: If finalize() was already called.
        """
        if self._is_finalized:
            raise RuntimeError("Cannot partial_fit after finalize()")

        if len(df_train_batch) == 0:
            return

        feature_cols = self._get_feature_columns(df_train_batch)
        if not feature_cols:
            return

        # Initialize on first batch
        if self._zscore_columns is None:
            self._zscore_columns = feature_cols
            self._minmax_columns = feature_cols
            n_features = len(feature_cols)
            self._zscore_mean = np.zeros(n_features)
            self._zscore_m2 = np.zeros(n_features)
            self._minmax_min = np.full(n_features, np.inf)
            self._minmax_max = np.full(n_features, -np.inf)

        # Get data as numpy array
        X = df_train_batch[feature_cols].values.astype(np.float64)

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Update z-score stats (Welford's online algorithm)
        for row in X:
            self._zscore_n += 1
            delta = row - self._zscore_mean
            self._zscore_mean += delta / self._zscore_n
            delta2 = row - self._zscore_mean
            self._zscore_m2 += delta * delta2

        # Update min-max stats
        batch_min = np.min(X, axis=0)
        batch_max = np.max(X, axis=0)
        self._minmax_min = np.minimum(cast(np.ndarray, self._minmax_min), batch_min)
        self._minmax_max = np.maximum(cast(np.ndarray, self._minmax_max), batch_max)

    def finalize(self) -> tuple[StandardScalerCustom, MinMaxScalerCustom]:
        """Finalize scaler fitting and return fitted scalers.

        Returns:
            Tuple of (zscore_scaler, minmax_scaler).

        Raises:
            RuntimeError: If no data was fitted.
        """
        if self._zscore_columns is None:
            raise RuntimeError("No data was fitted. Call partial_fit() first.")

        # Compute final std from M2
        variance = cast(np.ndarray, self._zscore_m2) / max(self._zscore_n - 1, 1)
        std = np.sqrt(variance)
        # Avoid division by zero
        std = np.where(std < 1e-10, 1.0, std)

        # Create z-score scaler
        zscore_scaler = StandardScalerCustom()
        zscore_scaler.columns_ = self._zscore_columns
        zscore_scaler.mean_ = self._zscore_mean
        zscore_scaler.std_ = std

        # Create min-max scaler
        minmax_scaler = MinMaxScalerCustom()
        minmax_scaler.columns_ = self._minmax_columns
        minmax_scaler.min_ = self._minmax_min
        minmax_scaler.max_ = self._minmax_max
        # Handle case where min == max
        range_vals = cast(np.ndarray, self._minmax_max) - cast(np.ndarray, self._minmax_min)
        range_vals = np.where(range_vals < 1e-10, 1.0, range_vals)
        # Store range for transform (MinMaxScalerCustom uses min_ and max_ internally)

        self._is_finalized = True

        logger.info(
            "Finalized scalers: %d columns, %d samples",
            len(self._zscore_columns),
            self._zscore_n,
        )

        return zscore_scaler, minmax_scaler


class ScalerApplier:
    """Applies fitted scalers to datasets.

    Handles loading, saving, and applying z-score and min-max scalers.
    """

    def __init__(self) -> None:
        """Initialize scaler applier with no scalers loaded."""
        self._zscore_scaler: StandardScalerCustom | None = None
        self._minmax_scaler: MinMaxScalerCustom | None = None

    def set_scalers(
        self,
        zscore_scaler: StandardScalerCustom,
        minmax_scaler: MinMaxScalerCustom,
    ) -> None:
        """Set scalers directly (after fitting).

        Args:
            zscore_scaler: Fitted z-score scaler.
            minmax_scaler: Fitted min-max scaler.
        """
        self._zscore_scaler = zscore_scaler
        self._minmax_scaler = minmax_scaler

    def load_scalers(
        self,
        zscore_path: Path = ZSCORE_SCALER_PATH,
        minmax_path: Path = MINMAX_SCALER_PATH,
    ) -> None:
        """Load pre-fitted scalers from disk.

        Args:
            zscore_path: Path to z-score scaler file.
            minmax_path: Path to min-max scaler file.
        """
        if zscore_path.exists():
            self._zscore_scaler = StandardScalerCustom.load(zscore_path)
            logger.info("Loaded z-score scaler from %s", zscore_path)
        else:
            logger.warning("Z-score scaler not found: %s", zscore_path)

        if minmax_path.exists():
            self._minmax_scaler = MinMaxScalerCustom.load(minmax_path)
            logger.info("Loaded min-max scaler from %s", minmax_path)
        else:
            logger.warning("Min-max scaler not found: %s", minmax_path)

    def save_scalers(
        self,
        zscore_path: Path = ZSCORE_SCALER_PATH,
        minmax_path: Path = MINMAX_SCALER_PATH,
    ) -> None:
        """Save scalers to disk.

        Args:
            zscore_path: Path to save z-score scaler.
            minmax_path: Path to save min-max scaler.
        """
        zscore_path.parent.mkdir(parents=True, exist_ok=True)

        if self._zscore_scaler is not None:
            self._zscore_scaler.save(zscore_path)
            logger.info("Saved z-score scaler to %s", zscore_path)

        if self._minmax_scaler is not None:
            self._minmax_scaler.save(minmax_path)
            logger.info("Saved min-max scaler to %s", minmax_path)

    def apply_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization.

        Args:
            df: DataFrame to scale.

        Returns:
            Scaled DataFrame.
        """
        if self._zscore_scaler is None:
            logger.warning("No z-score scaler available, returning unchanged")
            return df

        if self._zscore_scaler.columns_ is None:
            logger.warning("Z-score scaler has no columns, returning unchanged")
            return df

        cols_to_scale = [c for c in self._zscore_scaler.columns_ if c in df.columns]

        if not cols_to_scale:
            logger.warning("No columns to z-score scale")
            return df

        result = self._zscore_scaler.transform(df)

        logger.info("Applied z-score scaling to %d columns", len(cols_to_scale))

        return result

    def apply_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max normalization.

        Args:
            df: DataFrame to scale.

        Returns:
            Scaled DataFrame.
        """
        if self._minmax_scaler is None:
            logger.warning("No min-max scaler available, returning unchanged")
            return df

        if self._minmax_scaler.columns_ is None:
            logger.warning("Min-max scaler has no columns, returning unchanged")
            return df

        cols_to_scale = [c for c in self._minmax_scaler.columns_ if c in df.columns]

        if not cols_to_scale:
            logger.warning("No columns to min-max scale")
            return df

        result = self._minmax_scaler.transform(df)

        logger.info("Applied min-max scaling to %d columns", len(cols_to_scale))

        return result
