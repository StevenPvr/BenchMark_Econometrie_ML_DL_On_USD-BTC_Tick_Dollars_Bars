"""PCA-based feature reduction by feature group.

Applies PCA to each feature group defined in feature_categories.json.
Fits on training data only to avoid data leakage, then transforms all data.

Note: Data is assumed to already be normalized/standardized before PCA.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from src.clear_features.config import (
    FEATURE_CATEGORIES_FILE,
    PCA_ARTIFACTS_DIR,
    PCA_CONFIG,
    META_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def _to_python_type(val: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.integer, np.int64, np.int32)):  # type: ignore
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):  # type: ignore
        return float(val) if not np.isnan(val) else None
    elif isinstance(val, np.ndarray):
        return [_to_python_type(v) for v in val]
    elif isinstance(val, dict):
        return {k: _to_python_type(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_to_python_type(v) for v in val]
    return val


@dataclass
class GroupPCAResult:
    """Result of PCA applied to a single feature group."""

    group_name: str
    category: str
    lag: str
    original_features: list[str]
    n_original: int
    n_components: int
    explained_variance_ratio: list[float]
    cumulative_variance: list[float]
    component_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return _to_python_type({
            "group_name": self.group_name,
            "category": self.category,
            "lag": self.lag,
            "original_features": self.original_features,
            "n_original": self.n_original,
            "n_components": self.n_components,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_variance": self.cumulative_variance,
            "component_names": self.component_names,
        })


@dataclass
class PCAReductionSummary:
    """Summary of all PCA reductions applied."""

    groups_processed: list[GroupPCAResult] = field(default_factory=list)
    groups_skipped: list[str] = field(default_factory=list)
    features_removed: list[str] = field(default_factory=list)
    features_added: list[str] = field(default_factory=list)
    features_kept: list[str] = field(default_factory=list)
    original_n_features: int = 0
    final_n_features: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return _to_python_type({
            "n_groups_processed": len(self.groups_processed),
            "n_groups_skipped": len(self.groups_skipped),
            "groups_skipped": self.groups_skipped,
            "original_n_features": self.original_n_features,
            "final_n_features": self.final_n_features,
            "n_features_removed": len(self.features_removed),
            "n_features_added": len(self.features_added),
            "n_features_kept": len(self.features_kept),
            "features_removed": self.features_removed,
            "features_added": self.features_added,
            "features_kept": self.features_kept,
            "groups": [g.to_dict() for g in self.groups_processed],
        })


class GroupPCAReducer:
    """Applies PCA to feature groups defined in feature_categories.json.

    Fits on training data only (split == 'train') to avoid data leakage.
    """

    def __init__(
        self,
        categories_file: Path = FEATURE_CATEGORIES_FILE,
        config: dict | None = None,
    ):
        """
        Initialize PCA reducer.

        Args:
            categories_file: Path to feature_categories.json
            config: PCA configuration (variance threshold, etc.)
        """
        self.config = config or PCA_CONFIG
        self.categories_file = categories_file

        self._feature_groups: list[dict] = []
        self._pca_models: dict[str, PCA] = {}
        self._imputation_medians: dict[str, np.ndarray] = {}  # Store medians from fit() for transform()
        self._summary: PCAReductionSummary | None = None
        self._is_fitted = False

        # Load feature categories
        self._load_feature_categories()

    def _load_feature_categories(self) -> None:
        """Load feature groups from feature_categories.json."""
        if not self.categories_file.exists():
            raise FileNotFoundError(f"Feature categories file not found: {self.categories_file}")

        with open(self.categories_file) as f:
            data = json.load(f)

        self._feature_groups = data.get("groups", [])
        logger.info(f"Loaded {len(self._feature_groups)} feature groups from {self.categories_file}")

    def _determine_n_components(
        self, pca: PCA, variance_threshold: float
    ) -> int:
        """Determine optimal number of components based on variance threshold."""
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, variance_threshold)) + 1
        max_components = len(pca.explained_variance_ratio_)
        return min(n_components, max_components)

    def fit(self, df: pd.DataFrame) -> PCAReductionSummary:
        """Fit PCA models on training data only.

        Args:
            df: DataFrame with features (must have 'split' column)

        Returns:
            PCAReductionSummary with details of the reduction
        """
        # Extract training data only
        if "split" not in df.columns:
            raise ValueError("DataFrame must have 'split' column to identify training data")

        df_train = df[df["split"] == "train"].copy()
        logger.info(f"Fitting PCA on training data: {len(df_train)} rows")

        # Get all feature columns (excluding metadata and target)
        excluded_cols = set(META_COLUMNS) | {TARGET_COLUMN}
        all_features = [c for c in df.columns if c not in excluded_cols]

        self._summary = PCAReductionSummary()
        self._summary.original_n_features = len(all_features)

        features_in_groups = set()
        variance_threshold_val = self.config.get("variance_explained_threshold", 0.90)
        variance_threshold = float(variance_threshold_val) if variance_threshold_val is not None else 0.90

        for group in self._feature_groups:
            group_name = group["name"]
            category = group.get("category", "unknown")
            lag = group.get("lag", "unknown")
            features = group.get("features", [])

            # Skip pca_clusters group (old PCA features to be removed)
            if category == "pca_clusters":
                logger.info(f"Skipping group '{group_name}' (pca_clusters category - will be removed)")
                self._summary.groups_skipped.append(group_name)
                # Mark these features for removal
                available_features = [f for f in features if f in df_train.columns]
                self._summary.features_removed.extend(available_features)
                features_in_groups.update(available_features)
                continue

            # Filter to features that exist in the dataframe
            available_features = [f for f in features if f in df_train.columns]

            if len(available_features) < 2:
                # Keep single features as-is (no PCA needed)
                if len(available_features) == 1:
                    logger.debug(f"Group '{group_name}': Single feature, keeping as-is")
                    self._summary.features_kept.extend(available_features)
                    features_in_groups.update(available_features)
                else:
                    logger.warning(f"Group '{group_name}': No features available, skipping")
                self._summary.groups_skipped.append(group_name)
                continue

            features_in_groups.update(available_features)

            # Get training data for this group
            X_train = cast(np.ndarray[Any, np.dtype[np.float64]], np.asarray(df_train[available_features].values, dtype=np.float64))

            # Handle inf values first (replace with NaN, then impute)
            inf_mask = np.isinf(X_train)
            if inf_mask.any():
                n_inf = inf_mask.sum()
                logger.warning(f"Group '{group_name}': Found {n_inf} inf values, replacing with NaN for imputation")
                X_train[inf_mask] = np.nan

            # Handle NaN values (use median imputation from training data)
            # Store medians for use in transform() to prevent data leakage
            col_medians = np.nanmedian(X_train, axis=0)
            # Handle case where all values are NaN
            col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
            self._imputation_medians[group_name] = col_medians

            if np.isnan(X_train).any():
                logger.warning(f"Group '{group_name}': Found NaN values, using median imputation")
                nan_mask = np.isnan(X_train)
                X_train[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

            # Final check for any remaining non-finite values
            if not np.isfinite(X_train).all():
                logger.error(f"Group '{group_name}': Still has non-finite values after imputation, skipping")
                self._summary.groups_skipped.append(group_name)
                continue

            # Fit PCA on all components first to analyze variance
            n_features = X_train.shape[1]
            pca_full = PCA(n_components=min(n_features, X_train.shape[0]))
            pca_full.fit(X_train)

            # Determine number of components for 90% variance
            n_components = self._determine_n_components(pca_full, variance_threshold)

            # Ensure at least 1 component
            n_components = max(1, n_components)

            # Fit final PCA model with determined components
            pca_final = PCA(n_components=n_components)
            pca_final.fit(X_train)

            # Store model
            self._pca_models[group_name] = pca_final

            # Create component names
            component_names = [
                f"pca_{group_name}_c{i}" for i in range(n_components)
            ]

            # Calculate cumulative variance
            cumulative_variance = list(np.cumsum(pca_final.explained_variance_ratio_))

            # Create result
            result = GroupPCAResult(
                group_name=group_name,
                category=category,
                lag=lag,
                original_features=available_features,
                n_original=len(available_features),
                n_components=n_components,
                explained_variance_ratio=[float(v) for v in pca_final.explained_variance_ratio_],
                cumulative_variance=[float(v) for v in cumulative_variance],
                component_names=component_names,
            )

            self._summary.groups_processed.append(result)
            self._summary.features_removed.extend(available_features)
            self._summary.features_added.extend(component_names)

            logger.info(
                f"Group '{group_name}': {len(available_features)} -> {n_components} components "
                f"(variance: {cumulative_variance[-1]:.2%})"
            )

        # Track features that were kept (not in any processed group)
        self._summary.features_kept.extend([
            f for f in all_features if f not in features_in_groups
        ])

        self._summary.final_n_features = (
            len(self._summary.features_kept) + len(self._summary.features_added)
        )

        self._is_fitted = True

        logger.info(
            f"PCA Summary: {self._summary.original_n_features} -> "
            f"{self._summary.final_n_features} features "
            f"({len(self._summary.groups_processed)} groups processed)"
        )

        return self._summary

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe using fitted PCA models.

        Args:
            df: DataFrame to transform (can be train, test, or full data)

        Returns:
            Transformed DataFrame with PCA components replacing original features
        """
        if not self._is_fitted or self._summary is None:
            raise RuntimeError("Must call fit() before transform()")

        # Collect columns to drop and new PCA arrays to add
        columns_to_drop: set[str] = set()
        pca_arrays: list[np.ndarray] = []
        pca_col_names: list[str] = []

        # Pre-compute column set for fast lookup
        df_columns = set(df.columns)

        # Process each fitted PCA group
        for group_result in self._summary.groups_processed:
            group_name = group_result.group_name
            features = group_result.original_features
            component_names = group_result.component_names

            pca = self._pca_models.get(group_name)
            if pca is None:
                continue

            # Check if features exist (some may have been already removed)
            available_features = [f for f in features if f in df_columns]
            if not available_features:
                continue

            # Get data directly from df (not result_df)
            X = df[available_features].values.astype(np.float64)

            # Handle inf values first (replace with NaN, then impute)
            inf_mask = np.isinf(X)
            if inf_mask.any():
                X[inf_mask] = np.nan

            # Handle NaN values using medians from fit() to prevent data leakage
            if np.isnan(X).any():
                # Use stored medians from training data (computed during fit)
                col_medians = self._imputation_medians.get(group_name)
                if col_medians is None:
                    # Fallback if no medians stored (should not happen in normal flow)
                    logger.warning(f"Group '{group_name}': No stored medians, computing from current batch")
                    col_medians = np.nanmedian(X, axis=0)
                    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
                nan_mask = np.isnan(X)
                X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

            # Apply PCA transform
            X_pca = pca.transform(X)

            # Mark original features for removal
            columns_to_drop.update(available_features)

            # Store PCA results for batch concatenation
            pca_arrays.append(X_pca)
            pca_col_names.extend(component_names)

        # Also remove old pca_cluster features (from previous PCA)
        old_pca_cols = [c for c in df_columns if c.startswith("pca_cluster")]
        if old_pca_cols:
            logger.info(f"Removing {len(old_pca_cols)} old pca_cluster features")
            columns_to_drop.update(old_pca_cols)

        # Build result DataFrame in one operation
        # Keep columns that are not being dropped
        keep_cols = [c for c in df.columns if c not in columns_to_drop]
        result_df = df[keep_cols].copy()

        # Add all PCA components at once
        if pca_arrays:
            pca_combined = np.hstack(pca_arrays)
            pca_df = pd.DataFrame(
                pca_combined,
                columns=pca_col_names,
                index=df.index,
            )
            result_df = pd.concat([result_df, pca_df], axis=1)

        return result_df

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, PCAReductionSummary]:
        """Fit and transform in one step.

        Args:
            df: DataFrame with features (must have 'split' column)

        Returns:
            Tuple of (transformed DataFrame, summary)
        """
        summary = self.fit(df)
        df_transformed = self.transform(df)
        return df_transformed, summary

    def save_artifacts(self, output_dir: Path = PCA_ARTIFACTS_DIR) -> None:
        """Save PCA models and summary."""
        if self._summary is None:
            raise RuntimeError("No summary available. Must call fit() before save_artifacts()")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PCA models and imputation medians (for data leakage prevention)
        models_file = output_dir / "pca_models.joblib"
        joblib.dump({
            "pca_models": self._pca_models,
            "imputation_medians": self._imputation_medians,
        }, models_file)
        logger.info(f"Saved PCA models to {models_file}")

        # Save summary as JSON
        summary_file = output_dir / "pca_summary.json"
        with open(summary_file, "w") as f:
            json.dump(self._summary.to_dict(), f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

    def load_artifacts(self, input_dir: Path = PCA_ARTIFACTS_DIR) -> None:
        """Load previously saved PCA models and imputation medians."""
        models_file = input_dir / "pca_models.joblib"
        data = joblib.load(models_file)
        self._pca_models = data["pca_models"]
        # Load imputation medians (for data leakage prevention in transform)
        self._imputation_medians = data.get("imputation_medians", {})

        # Load summary
        summary_file = input_dir / "pca_summary.json"
        with open(summary_file) as f:
            summary_dict = json.load(f)

        # Reconstruct summary
        self._summary = PCAReductionSummary(
            features_removed=summary_dict["features_removed"],
            features_added=summary_dict["features_added"],
            features_kept=summary_dict.get("features_kept", []),
            groups_skipped=summary_dict.get("groups_skipped", []),
            original_n_features=summary_dict["original_n_features"],
            final_n_features=summary_dict["final_n_features"],
        )

        # Reconstruct group results
        for g_dict in summary_dict.get("groups", []):
            result = GroupPCAResult(
                group_name=g_dict["group_name"],
                category=g_dict.get("category", "unknown"),
                lag=g_dict.get("lag", "unknown"),
                original_features=g_dict["original_features"],
                n_original=g_dict["n_original"],
                n_components=g_dict["n_components"],
                explained_variance_ratio=g_dict["explained_variance_ratio"],
                cumulative_variance=g_dict["cumulative_variance"],
                component_names=g_dict["component_names"],
            )
            self._summary.groups_processed.append(result)

        self._is_fitted = True
        logger.info(f"Loaded PCA artifacts from {input_dir}")


# Alias for backward compatibility
WeightedPCAReducer = GroupPCAReducer
