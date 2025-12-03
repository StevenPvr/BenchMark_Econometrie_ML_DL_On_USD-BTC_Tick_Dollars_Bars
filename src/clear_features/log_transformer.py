"""Log transformation for non-stationary features.

Applies log transformations to features identified as non-stationary
by the stationarity analysis module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.clear_features.config import (
    LOG_TRANSFORM_ARTIFACTS_DIR,
    LOG_TRANSFORM_CONFIG,
    META_COLUMNS,
    STATIONARITY_RESULTS_FILE,
    TARGET_COLUMN,
)
from src.config_logging import get_logger

logger = get_logger(__name__)


@dataclass
class LogTransformResult:
    """Result of log transformation analysis.

    Attributes:
        features_transformed: List of features that were transformed.
        features_skipped: List of features that were skipped.
        transform_params: Dict mapping feature name to transform parameters.
    """

    features_transformed: list[str] = field(default_factory=list)
    features_skipped: list[str] = field(default_factory=list)
    transform_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "n_transformed": len(self.features_transformed),
            "n_skipped": len(self.features_skipped),
            "features_transformed": self.features_transformed,
            "features_skipped": self.features_skipped,
            "transform_params": self.transform_params,
        }


class LogTransformer:
    """Applies log transformation to non-stationary features.

    Uses stationarity test results to identify features that need transformation.
    Supports multiple transform types: log, log1p, signed_log1p.
    """

    def __init__(
        self,
        stationarity_file: Path = STATIONARITY_RESULTS_FILE,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize log transformer.

        Args:
            stationarity_file: Path to stationarity test results JSON.
            config: Log transform configuration dict.
        """
        self.stationarity_file = stationarity_file
        self.config = config or LOG_TRANSFORM_CONFIG

        self._stationarity_data: dict[str, Any] | None = None
        self._non_stationary_features: list[str] = []
        self._transform_params: dict[str, dict[str, Any]] = {}
        self._result: LogTransformResult | None = None

    def load_stationarity_results(self) -> None:
        """Load stationarity test results from analyse_features module."""
        logger.info("Loading stationarity results from %s", self.stationarity_file)

        if not self.stationarity_file.exists():
            logger.warning(
                "Stationarity results file not found: %s", self.stationarity_file
            )
            self._stationarity_data = {"all_results": []}
            return

        with open(self.stationarity_file) as f:
            self._stationarity_data = json.load(f)

        logger.info("Loaded stationarity results")

    def identify_non_stationary_features(self, df: pd.DataFrame) -> list[str]:
        """Identify features that need log transformation based on stationarity tests.

        Args:
            df: DataFrame containing features to check.

        Returns:
            List of feature names identified as non-stationary.
        """
        if self._stationarity_data is None:
            self.load_stationarity_results()

        if self._stationarity_data is None:
            logger.error("Failed to load stationarity data")
            self._stationarity_data = {"all_results": []}

        non_stationary_conclusions = self.config["non_stationary_conclusions"]
        all_results = self._stationarity_data.get("all_results", [])

        self._non_stationary_features = []

        for item in all_results:
            feature = item.get("feature", "")
            conclusion = item.get("stationarity_conclusion", "")

            if feature in df.columns and conclusion in non_stationary_conclusions:
                if feature not in META_COLUMNS and feature != TARGET_COLUMN:
                    self._non_stationary_features.append(feature)

        logger.info(
            "Identified %d non-stationary features for log transform: %s",
            len(self._non_stationary_features),
            self._non_stationary_features,
        )

        return self._non_stationary_features

    def _determine_transform_type(
        self, min_val: float, min_threshold: float, use_log1p: bool
    ) -> tuple[str, float]:
        """Determine transformation type based on value range.

        Args:
            min_val: Minimum value in the feature.
            min_threshold: Threshold for near-zero values.
            use_log1p: Whether to prefer log1p over log.

        Returns:
            Tuple of (transform_type, shift_value).
        """
        if min_val < 0:
            return "signed_log1p", 0.0
        elif min_val < min_threshold:
            shift = abs(min_val) + min_threshold if min_val <= 0 else 0.0
            return "log1p", shift
        else:
            transform_type = "log" if not use_log1p else "log1p"
            return transform_type, 0.0

    def fit(self, df: pd.DataFrame) -> LogTransformResult:
        """Fit log transformer - compute parameters on training data.

        Args:
            df: DataFrame containing features to fit.

        Returns:
            LogTransformResult with transformation details.
        """
        if not self._non_stationary_features:
            self.identify_non_stationary_features(df)

        self._result = LogTransformResult()
        self._transform_params = {}

        use_log1p = bool(self.config.get("use_log1p", True))
        min_threshold = float(self.config.get("min_value_threshold", 1e-10))  # type: ignore

        for feature in self._non_stationary_features:
            if feature not in df.columns:
                self._result.features_skipped.append(feature)
                continue

            values = df[feature].values
            valid_values = values[~np.isnan(values)]

            if len(valid_values) == 0:
                self._result.features_skipped.append(feature)
                continue

            min_val = float(np.min(valid_values))
            max_val = float(np.max(valid_values))

            transform_type, shift = self._determine_transform_type(
                min_val, min_threshold, use_log1p
            )

            self._transform_params[feature] = {
                "transform_type": transform_type,
                "shift": shift,
                "original_min": min_val,
                "original_max": max_val,
            }

            self._result.features_transformed.append(feature)
            self._result.transform_params[feature] = self._transform_params[feature]

        logger.info(
            "Log transformer fit: %d features to transform, %d skipped",
            len(self._result.features_transformed),
            len(self._result.features_skipped),
        )

        return self._result

    def _apply_transform(
        self, values: np.ndarray, transform_type: str, shift: float
    ) -> np.ndarray:
        """Apply log transformation to values.

        Args:
            values: Array of values to transform.
            transform_type: Type of transformation (log, log1p, signed_log1p).
            shift: Shift value to add before transformation.

        Returns:
            Transformed values.
        """
        if transform_type == "signed_log1p":
            return np.sign(values) * np.log1p(np.abs(values))
        elif transform_type == "log1p":
            return np.log1p(values + shift)
        else:
            return np.log(values + shift + 1e-10)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to features.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.

        Raises:
            RuntimeError: If fit() was not called first.
        """
        if not self._transform_params:
            raise RuntimeError("Must call fit() before transform()")

        result_df = df.copy()

        for feature, params in self._transform_params.items():
            if feature not in result_df.columns:
                continue

            values = result_df[feature].values.astype(np.float64)
            transformed = self._apply_transform(
                values, params["transform_type"], params["shift"]
            )
            result_df[feature] = transformed

        return result_df

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, LogTransformResult]:
        """Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform.

        Returns:
            Tuple of (transformed DataFrame, LogTransformResult).
        """
        result = self.fit(df)
        df_transformed = self.transform(df)
        return df_transformed, result

    def save_artifacts(self, output_dir: Path = LOG_TRANSFORM_ARTIFACTS_DIR) -> None:
        """Save transformation artifacts to disk.

        Args:
            output_dir: Directory to save artifacts to.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        params_file = output_dir / "log_transform_params.joblib"
        joblib.dump(self._transform_params, params_file)
        logger.info("Saved log transform params to %s", params_file)

        if self._result is not None:
            summary_file = output_dir / "log_transform_summary.json"
            with open(summary_file, "w") as f:
                json.dump(self._result.to_dict(), f, indent=2)
            logger.info("Saved summary to %s", summary_file)

    def load_artifacts(self, input_dir: Path = LOG_TRANSFORM_ARTIFACTS_DIR) -> None:
        """Load previously saved artifacts from disk.

        Args:
            input_dir: Directory containing saved artifacts.
        """
        params_file = input_dir / "log_transform_params.joblib"
        self._transform_params = joblib.load(params_file)
        logger.info("Loaded log transform params from %s", params_file)
