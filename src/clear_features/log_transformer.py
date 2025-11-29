"""Log transformation for non-stationary features."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.clear_features.config import (
    STATIONARITY_RESULTS_FILE,
    LOG_TRANSFORM_ARTIFACTS_DIR,
    LOG_TRANSFORM_CONFIG,
    META_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


@dataclass
class LogTransformResult:
    """Result of log transformation analysis."""

    features_transformed: list[str] = field(default_factory=list)
    features_skipped: list[str] = field(default_factory=list)
    transform_params: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_transformed": len(self.features_transformed),
            "n_skipped": len(self.features_skipped),
            "features_transformed": self.features_transformed,
            "features_skipped": self.features_skipped,
            "transform_params": self.transform_params,
        }


class LogTransformer:
    """Applies log transformation to non-stationary features."""

    def __init__(
        self,
        stationarity_file: Path = STATIONARITY_RESULTS_FILE,
        config: dict | None = None,
    ):
        self.stationarity_file = stationarity_file
        self.config = config or LOG_TRANSFORM_CONFIG

        self._stationarity_data: dict | None = None
        self._non_stationary_features: list[str] = []
        self._transform_params: dict[str, dict] = {}
        self._result: LogTransformResult | None = None

    def load_stationarity_results(self) -> None:
        """Load stationarity test results from analyse_features."""
        logger.info("Loading stationarity results from %s", self.stationarity_file)

        if not self.stationarity_file.exists():
            logger.warning("Stationarity results file not found: %s", self.stationarity_file)
            self._stationarity_data = {"all_results": []}
            return

        with open(self.stationarity_file) as f:
            self._stationarity_data = json.load(f)

        logger.info("Loaded stationarity results")

    def identify_non_stationary_features(self, df: pd.DataFrame) -> list[str]:
        """Identify features that need log transformation based on stationarity tests."""
        if self._stationarity_data is None:
            self.load_stationarity_results()

        non_stationary_conclusions = self.config["non_stationary_conclusions"]

        # Get features from stationarity results
        all_results = self._stationarity_data.get("all_results", [])

        self._non_stationary_features = []

        for item in all_results:
            feature = item.get("feature", "")
            conclusion = item.get("stationarity_conclusion", "")

            # Check if feature exists in dataframe and is non-stationary
            if feature in df.columns and conclusion in non_stationary_conclusions:
                # Skip metadata and target columns
                if feature not in META_COLUMNS and feature != TARGET_COLUMN:
                    self._non_stationary_features.append(feature)

        logger.info(
            "Identified %d non-stationary features for log transform: %s",
            len(self._non_stationary_features),
            self._non_stationary_features,
        )

        return self._non_stationary_features

    def fit(self, df: pd.DataFrame) -> LogTransformResult:
        """Fit log transformer - compute parameters on training data."""
        if not self._non_stationary_features:
            self.identify_non_stationary_features(df)

        self._result = LogTransformResult()
        self._transform_params = {}

        use_log1p = self.config.get("use_log1p", True)
        min_threshold = self.config.get("min_value_threshold", 1e-10)

        for feature in self._non_stationary_features:
            if feature not in df.columns:
                self._result.features_skipped.append(feature)
                continue

            values = df[feature].values
            valid_values = values[~np.isnan(values)]

            if len(valid_values) == 0:
                self._result.features_skipped.append(feature)
                continue

            # Compute parameters for transformation
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)

            # Determine transformation strategy based on value range
            if min_val < 0:
                # Values include negatives - use signed log transform
                # log_transform = sign(x) * log1p(|x|)
                transform_type = "signed_log1p"
                shift = 0
            elif min_val < min_threshold:
                # Values include zeros or near-zeros - use log1p with shift
                transform_type = "log1p"
                shift = abs(min_val) + min_threshold if min_val <= 0 else 0
            else:
                # All positive values - use standard log
                transform_type = "log" if not use_log1p else "log1p"
                shift = 0

            self._transform_params[feature] = {
                "transform_type": transform_type,
                "shift": float(shift),
                "original_min": float(min_val),
                "original_max": float(max_val),
            }

            self._result.features_transformed.append(feature)
            self._result.transform_params[feature] = self._transform_params[feature]

        logger.info(
            "Log transformer fit: %d features to transform, %d skipped",
            len(self._result.features_transformed),
            len(self._result.features_skipped),
        )

        return self._result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to features."""
        if not self._transform_params:
            raise RuntimeError("Must call fit() before transform()")

        result_df = df.copy()

        for feature, params in self._transform_params.items():
            if feature not in result_df.columns:
                continue

            values = result_df[feature].values.astype(np.float64)
            transform_type = params["transform_type"]
            shift = params["shift"]

            if transform_type == "signed_log1p":
                # sign(x) * log1p(|x|)
                transformed = np.sign(values) * np.log1p(np.abs(values))
            elif transform_type == "log1p":
                # log1p(x + shift)
                transformed = np.log1p(values + shift)
            else:  # "log"
                # log(x + shift)
                transformed = np.log(values + shift + 1e-10)

            result_df[feature] = transformed

        return result_df

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, LogTransformResult]:
        """Fit and transform in one step."""
        result = self.fit(df)
        df_transformed = self.transform(df)
        return df_transformed, result

    def save_artifacts(self, output_dir: Path = LOG_TRANSFORM_ARTIFACTS_DIR) -> None:
        """Save transformation artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save transform params
        params_file = output_dir / "log_transform_params.joblib"
        joblib.dump(self._transform_params, params_file)
        logger.info("Saved log transform params to %s", params_file)

        # Save summary as JSON
        if self._result is not None:
            summary_file = output_dir / "log_transform_summary.json"
            with open(summary_file, "w") as f:
                json.dump(self._result.to_dict(), f, indent=2)
            logger.info("Saved summary to %s", summary_file)

    def load_artifacts(self, input_dir: Path = LOG_TRANSFORM_ARTIFACTS_DIR) -> None:
        """Load previously saved artifacts."""
        params_file = input_dir / "log_transform_params.joblib"
        self._transform_params = joblib.load(params_file)
        logger.info("Loaded log transform params from %s", params_file)
