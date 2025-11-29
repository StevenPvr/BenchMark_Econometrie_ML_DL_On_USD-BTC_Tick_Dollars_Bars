"""Apply pre-fitted scalers from features module."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from src.clear_features.config import (
    SCALER_CONFIG,
    META_COLUMNS,
    TARGET_COLUMN,
)
from src.features.scalers import StandardScalerCustom, MinMaxScalerCustom

logger = logging.getLogger(__name__)


class ScalerApplier:
    """Applies pre-fitted scalers to datasets."""

    def __init__(
        self,
        zscore_scaler_path: Path | None = None,
        minmax_scaler_path: Path | None = None,
    ):
        self.zscore_scaler_path = zscore_scaler_path or SCALER_CONFIG["zscore_scaler_path"]
        self.minmax_scaler_path = minmax_scaler_path or SCALER_CONFIG["minmax_scaler_path"]

        self._zscore_scaler: StandardScalerCustom | None = None
        self._minmax_scaler: MinMaxScalerCustom | None = None

    def load_scalers(self) -> None:
        """Load pre-fitted scalers from disk."""
        if self.zscore_scaler_path.exists():
            self._zscore_scaler = StandardScalerCustom.load(self.zscore_scaler_path)
            logger.info("Loaded z-score scaler from %s", self.zscore_scaler_path)
        else:
            logger.warning("Z-score scaler not found: %s", self.zscore_scaler_path)

        if self.minmax_scaler_path.exists():
            self._minmax_scaler = MinMaxScalerCustom.load(self.minmax_scaler_path)
            logger.info("Loaded min-max scaler from %s", self.minmax_scaler_path)
        else:
            logger.warning("Min-max scaler not found: %s", self.minmax_scaler_path)

    def _update_scaler_columns(
        self,
        scaler: StandardScalerCustom | MinMaxScalerCustom,
        df: pd.DataFrame,
    ) -> list[str]:
        """Get columns to scale that exist in both scaler and dataframe.

        After PCA reduction, some original columns are removed and new PCA columns added.
        We need to update the scaling to only apply to columns that still exist.

        Returns:
            List of columns that can be scaled.
        """
        if scaler.columns_ is None:
            return []

        # Find columns that exist in both scaler and dataframe
        existing_cols = [c for c in scaler.columns_ if c in df.columns]

        # New columns from PCA (not in original scaler) - skip these
        new_cols = [
            c for c in df.columns
            if c not in META_COLUMNS
            and c != TARGET_COLUMN
            and c not in scaler.columns_
        ]

        if new_cols:
            logger.info(
                "Skipping %d new columns not in original scaler (e.g., PCA components): %s",
                len(new_cols),
                new_cols[:5],
            )

        return existing_cols

    def apply_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization using pre-fitted scaler."""
        if self._zscore_scaler is None:
            self.load_scalers()

        if self._zscore_scaler is None:
            logger.warning("No z-score scaler available, returning unchanged")
            return df

        # Get columns that exist in both
        cols_to_scale = self._update_scaler_columns(self._zscore_scaler, df)

        if not cols_to_scale:
            logger.warning("No columns to z-score scale")
            return df

        # Create temporary scaler with only existing columns
        temp_scaler = StandardScalerCustom()
        temp_scaler.columns_ = cols_to_scale

        # Get indices of these columns in original scaler
        if self._zscore_scaler.columns_ is None:
            raise ValueError("Z-score scaler columns_ is None")
        if self._zscore_scaler.mean_ is None or self._zscore_scaler.std_ is None:
            raise ValueError("Z-score scaler not fitted (mean_ or std_ is None)")
        original_cols = list(self._zscore_scaler.columns_)
        indices = [original_cols.index(c) for c in cols_to_scale]

        temp_scaler.mean_ = self._zscore_scaler.mean_[indices]
        temp_scaler.std_ = self._zscore_scaler.std_[indices]

        # Transform
        result = temp_scaler.transform(df)

        logger.info("Applied z-score scaling to %d columns", len(cols_to_scale))

        return result

    def apply_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max normalization using pre-fitted scaler."""
        if self._minmax_scaler is None:
            self.load_scalers()

        if self._minmax_scaler is None:
            logger.warning("No min-max scaler available, returning unchanged")
            return df

        # Get columns that exist in both
        cols_to_scale = self._update_scaler_columns(self._minmax_scaler, df)

        if not cols_to_scale:
            logger.warning("No columns to min-max scale")
            return df

        # Create temporary scaler with only existing columns
        temp_scaler = MinMaxScalerCustom()
        temp_scaler.columns_ = cols_to_scale

        # Get indices of these columns in original scaler
        if self._minmax_scaler.columns_ is None:
            raise ValueError("Min-max scaler columns_ is None")
        if self._minmax_scaler.min_ is None or self._minmax_scaler.max_ is None:
            raise ValueError("Min-max scaler not fitted (min_ or max_ is None)")
        original_cols = list(self._minmax_scaler.columns_)
        indices = [original_cols.index(c) for c in cols_to_scale]

        temp_scaler.min_ = self._minmax_scaler.min_[indices]
        temp_scaler.max_ = self._minmax_scaler.max_[indices]

        # Transform
        result = temp_scaler.transform(df)

        logger.info("Applied min-max scaling to %d columns", len(cols_to_scale))

        return result
