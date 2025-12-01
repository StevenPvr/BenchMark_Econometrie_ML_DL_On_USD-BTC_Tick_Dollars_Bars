"""PCA-based feature reduction by feature group.

Applies PCA to each feature group defined in feature_categories.json.
Fits on training data only to avoid data leakage, then transforms all data.

Note: Data is assumed to already be normalized/standardized before PCA.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.decomposition import PCA, IncrementalPCA  # type: ignore[import-untyped]

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
        self._standardization_params: dict[str, dict[str, np.ndarray]] = {}  # Store mean/std for standardization before PCA
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

            # STANDARDIZE before PCA (critical for proper variance analysis)
            # PCA is sensitive to scale - features with larger variance dominate without standardization
            X_mean = np.mean(X_train, axis=0)
            X_std = np.std(X_train, axis=0, ddof=1)
            # Avoid division by zero for constant features
            X_std = np.where(X_std < 1e-10, 1.0, X_std)
            # Store standardization params for transform()
            self._standardization_params[group_name] = {
                "mean": X_mean,
                "std": X_std,
            }
            # Apply standardization
            X_standardized = (X_train - X_mean) / X_std

            # Fit PCA on STANDARDIZED data to analyze variance
            n_features = X_standardized.shape[1]
            pca_full = PCA(n_components=min(n_features, X_standardized.shape[0]))
            pca_full.fit(X_standardized)

            # Determine number of components for 90% variance
            n_components = self._determine_n_components(pca_full, variance_threshold)

            # Ensure at least 1 component
            n_components = max(1, n_components)

            # Fit final PCA model with determined components on STANDARDIZED data
            pca_final = PCA(n_components=n_components)
            pca_final.fit(X_standardized)

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
                    # This should never happen if fit() was called properly
                    raise RuntimeError(
                        f"Group '{group_name}': No stored medians found. "
                        "This indicates the PCA was not properly fitted. "
                        "Call fit() before transform()."
                    )
                nan_mask = np.isnan(X)
                X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

            # Apply STANDARDIZATION before PCA transform (using params from fit)
            std_params = self._standardization_params.get(group_name)
            if std_params is None:
                raise RuntimeError(
                    f"Group '{group_name}': No standardization parameters found. "
                    "This indicates the PCA was not properly fitted. "
                    "Call fit() before transform()."
                )
            X_standardized = (X - std_params["mean"]) / std_params["std"]

            # Apply PCA transform on standardized data
            X_pca = pca.transform(X_standardized)

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
                columns=pd.Index(pca_col_names),
                index=df.index,
            )
            result_df = pd.concat([result_df, pca_df], axis=1)

        return cast(pd.DataFrame, result_df)

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

        # Save PCA models, imputation medians, and standardization params (for data leakage prevention)
        models_file = output_dir / "pca_models.joblib"
        joblib.dump({
            "pca_models": self._pca_models,
            "imputation_medians": self._imputation_medians,
            "standardization_params": self._standardization_params,
        }, models_file)
        logger.info(f"Saved PCA models to {models_file}")

        # Save summary as JSON
        summary_file = output_dir / "pca_summary.json"
        with open(summary_file, "w") as f:
            json.dump(self._summary.to_dict(), f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

    def load_artifacts(self, input_dir: Path = PCA_ARTIFACTS_DIR) -> None:
        """Load previously saved PCA models, imputation medians, and standardization params."""
        models_file = input_dir / "pca_models.joblib"
        data = joblib.load(models_file)
        self._pca_models = data["pca_models"]
        # Load imputation medians (for data leakage prevention in transform)
        self._imputation_medians = data.get("imputation_medians", {})
        # Load standardization params (for proper PCA transform)
        self._standardization_params = data.get("standardization_params", {})

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


class IncrementalGroupPCAReducer:
    """Incremental PCA reducer for batch processing.

    Uses IncrementalPCA to fit PCA models batch-by-batch, then determines
    optimal number of components based on variance threshold.

    Workflow:
    1. partial_fit() - Call repeatedly with batches of TRAIN data
    2. finalize_fit() - Compute optimal n_components and prepare for transform
    3. transform() - Transform batches (can be train or test)
    """

    def __init__(
        self,
        categories_file: Path = FEATURE_CATEGORIES_FILE,
        config: dict | None = None,
    ):
        """Initialize incremental PCA reducer."""
        self.config = config or PCA_CONFIG
        self.categories_file = categories_file

        self._feature_groups: list[dict] = []
        # IncrementalPCA models for fitting
        self._ipca_models: dict[str, IncrementalPCA] = {}
        # Final PCA-like objects with truncated components
        self._final_n_components: dict[str, int] = {}
        # Imputation medians (computed incrementally)
        self._imputation_sums: dict[str, np.ndarray] = {}
        self._imputation_counts: dict[str, np.ndarray] = {}
        self._imputation_medians: dict[str, np.ndarray] = {}
        # Standardization params (computed incrementally via Welford's algorithm)
        self._std_n: dict[str, int] = {}
        self._std_mean: dict[str, np.ndarray] = {}
        self._std_m2: dict[str, np.ndarray] = {}  # Sum of squared differences
        self._standardization_params: dict[str, dict[str, np.ndarray]] = {}
        # Group metadata
        self._group_features: dict[str, list[str]] = {}
        self._groups_to_skip: set[str] = set()
        self._features_in_groups: set[str] = set()

        self._summary: PCAReductionSummary | None = None
        self._is_fitted = False
        self._fit_finalized = False
        self._n_batches_seen = 0

        self._load_feature_categories()

    def _load_feature_categories(self) -> None:
        """Load feature groups from feature_categories.json."""
        if not self.categories_file.exists():
            raise FileNotFoundError(f"Feature categories file not found: {self.categories_file}")

        with open(self.categories_file) as f:
            data = json.load(f)

        self._feature_groups = data.get("groups", [])
        logger.info(f"Loaded {len(self._feature_groups)} feature groups")

    def partial_fit(self, df_batch: pd.DataFrame) -> None:
        """Incrementally fit PCA on a batch of TRAIN data.

        Call this repeatedly with batches of training data.
        Only pass rows where split == 'train'.

        Args:
            df_batch: Batch of training data (must be train only!)
        """
        if self._fit_finalized:
            raise RuntimeError("Cannot partial_fit after finalize_fit()")

        if len(df_batch) == 0:
            return

        self._n_batches_seen += 1

        for group in self._feature_groups:
            group_name = group["name"]
            category = group.get("category", "unknown")
            features = group.get("features", [])

            # Skip pca_clusters group
            if category == "pca_clusters":
                self._groups_to_skip.add(group_name)
                continue

            # Filter to features that exist
            available_features = [f for f in features if f in df_batch.columns]

            if len(available_features) < 2:
                self._groups_to_skip.add(group_name)
                continue

            # Store group features on first batch
            if group_name not in self._group_features:
                self._group_features[group_name] = available_features
                self._features_in_groups.update(available_features)

            # Get data
            X = df_batch[available_features].values.astype(np.float64)

            # Handle inf -> nan
            X = np.where(np.isinf(X), np.nan, X)

            # Update imputation statistics (running median approximation using mean)
            # For true incremental median we'd need reservoir sampling, use mean as proxy
            if group_name not in self._imputation_sums:
                self._imputation_sums[group_name] = np.zeros(len(available_features))
                self._imputation_counts[group_name] = np.zeros(len(available_features))

            valid_mask = ~np.isnan(X)
            self._imputation_sums[group_name] += np.nansum(X, axis=0)
            self._imputation_counts[group_name] += valid_mask.sum(axis=0)

            # Impute NaN with current running mean for this batch
            col_means = np.divide(
                self._imputation_sums[group_name],
                self._imputation_counts[group_name],
                out=np.zeros_like(self._imputation_sums[group_name]),
                where=self._imputation_counts[group_name] > 0,
            )
            nan_mask = np.isnan(X)
            if nan_mask.any():
                X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

            # Update standardization stats (Welford's online algorithm)
            if group_name not in self._std_n:
                self._std_n[group_name] = 0
                self._std_mean[group_name] = np.zeros(len(available_features))
                self._std_m2[group_name] = np.zeros(len(available_features))

            for row in X:
                self._std_n[group_name] += 1
                delta = row - self._std_mean[group_name]
                self._std_mean[group_name] += delta / self._std_n[group_name]
                delta2 = row - self._std_mean[group_name]
                self._std_m2[group_name] += delta * delta2

            # Compute current standardization params
            n = self._std_n[group_name]
            mean = self._std_mean[group_name]
            std = np.sqrt(self._std_m2[group_name] / max(n - 1, 1))
            std = np.where(std < 1e-10, 1.0, std)

            # Standardize before IncrementalPCA
            X_std = (X - mean) / std

            # Initialize or update IncrementalPCA
            if group_name not in self._ipca_models:
                n_features = X_std.shape[1]
                # IncrementalPCA needs n_components <= min(n_samples, n_features)
                n_components = min(n_features, len(df_batch))
                self._ipca_models[group_name] = IncrementalPCA(n_components=n_components)

            try:
                self._ipca_models[group_name].partial_fit(X_std)
            except Exception as e:
                logger.warning(f"Group '{group_name}': partial_fit failed: {e}")
                self._groups_to_skip.add(group_name)

        if self._n_batches_seen % 5 == 0:
            logger.info(f"  Partial fit batch {self._n_batches_seen} complete")

    def finalize_fit(self) -> PCAReductionSummary:
        """Finalize the incremental fit and determine optimal n_components.

        Must be called after all partial_fit() calls and before transform().

        Returns:
            PCAReductionSummary with details
        """
        if self._fit_finalized:
            return self._summary  # type: ignore

        variance_threshold = float(cast(float, self.config.get("variance_explained_threshold", 0.90)))

        self._summary = PCAReductionSummary()

        # Compute all features count
        all_features_seen = set()
        for features in self._group_features.values():
            all_features_seen.update(features)
        self._summary.original_n_features = len(all_features_seen)

        for group_name, ipca in self._ipca_models.items():
            if group_name in self._groups_to_skip:
                continue

            features = self._group_features.get(group_name, [])
            if not features:
                continue

            # Finalize imputation medians (using mean as approximation)
            counts = self._imputation_counts.get(group_name, np.ones(len(features)))
            sums = self._imputation_sums.get(group_name, np.zeros(len(features)))
            self._imputation_medians[group_name] = np.divide(
                sums, counts, out=np.zeros_like(sums), where=counts > 0
            )

            # Finalize standardization params
            n = self._std_n.get(group_name, 1)
            mean = self._std_mean.get(group_name, np.zeros(len(features)))
            m2 = self._std_m2.get(group_name, np.zeros(len(features)))
            std = np.sqrt(m2 / max(n - 1, 1))
            std = np.where(std < 1e-10, 1.0, std)
            self._standardization_params[group_name] = {"mean": mean, "std": std}

            # Determine optimal n_components based on variance
            explained_var = cast(np.ndarray, ipca.explained_variance_ratio_)
            cumulative = np.cumsum(explained_var)
            n_components = int(np.searchsorted(cumulative, variance_threshold)) + 1
            n_components = min(n_components, len(explained_var))
            n_components = max(1, n_components)

            self._final_n_components[group_name] = n_components

            # Create component names
            component_names = [f"pca_{group_name}_c{i}" for i in range(n_components)]

            # Get group metadata
            group_meta = next((g for g in self._feature_groups if g["name"] == group_name), {})

            result = GroupPCAResult(
                group_name=group_name,
                category=group_meta.get("category", "unknown"),
                lag=group_meta.get("lag", "unknown"),
                original_features=features,
                n_original=len(features),
                n_components=n_components,
                explained_variance_ratio=[float(v) for v in explained_var[:n_components]],
                cumulative_variance=[float(v) for v in cumulative[:n_components]],
                component_names=component_names,
            )

            self._summary.groups_processed.append(result)
            self._summary.features_removed.extend(features)
            self._summary.features_added.extend(component_names)

            logger.info(
                f"Group '{group_name}': {len(features)} -> {n_components} components "
                f"(variance: {cumulative[n_components-1]:.2%})"
            )

        # Track skipped groups
        self._summary.groups_skipped = list(self._groups_to_skip)

        self._summary.final_n_features = (
            len(self._summary.features_kept) + len(self._summary.features_added)
        )

        self._is_fitted = True
        self._fit_finalized = True

        logger.info(
            f"PCA Summary: {len(self._ipca_models)} groups, "
            f"{len(self._summary.features_added)} components total"
        )

        return self._summary

    def transform(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        """Transform a batch using fitted PCA models.

        Args:
            df_batch: Batch to transform (can be train or test)

        Returns:
            Transformed DataFrame with PCA components
        """
        if not self._fit_finalized:
            raise RuntimeError("Must call finalize_fit() before transform()")

        columns_to_drop: set[str] = set()
        pca_arrays: list[np.ndarray] = []
        pca_col_names: list[str] = []

        df_columns = set(df_batch.columns)

        for group_result in self._summary.groups_processed:  # type: ignore
            group_name = group_result.group_name
            features = group_result.original_features
            n_components = self._final_n_components.get(group_name, 0)

            if n_components == 0:
                continue

            ipca = self._ipca_models.get(group_name)
            if ipca is None:
                continue

            available_features = [f for f in features if f in df_columns]
            if not available_features:
                continue

            X = df_batch[available_features].values.astype(np.float64)

            # Handle inf -> nan
            X = np.where(np.isinf(X), np.nan, X)

            # Impute NaN using stored medians
            medians = self._imputation_medians.get(group_name)
            if medians is not None and np.isnan(X).any():
                nan_mask = np.isnan(X)
                X[nan_mask] = np.take(medians, np.where(nan_mask)[1])

            # NOTE: No standardization during transform - external scaling (zscore/minmax)
            # is applied after PCA in the pipeline. Internal standardization was only
            # used during partial_fit() to compute proper PCA components.

            # Transform and truncate to n_components
            X_pca = ipca.transform(X)[:, :n_components]

            columns_to_drop.update(available_features)
            pca_arrays.append(X_pca)
            pca_col_names.extend(group_result.component_names)

        # Remove old pca_cluster features
        old_pca_cols = [c for c in df_columns if c.startswith("pca_cluster")]
        columns_to_drop.update(old_pca_cols)

        # Build result
        keep_cols = [c for c in df_batch.columns if c not in columns_to_drop]
        result_df = df_batch[keep_cols].copy()

        if pca_arrays:
            pca_combined = np.hstack(pca_arrays)
            pca_df = pd.DataFrame(
                pca_combined,
                columns=pd.Index(pca_col_names),
                index=df_batch.index,
            )
            result_df = pd.concat([result_df, pca_df], axis=1)

        return cast(pd.DataFrame, result_df)

    def save_artifacts(self, output_dir: Path = PCA_ARTIFACTS_DIR) -> None:
        """Save PCA models and summary."""
        if not self._fit_finalized or self._summary is None:
            raise RuntimeError("Must call finalize_fit() before save_artifacts()")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save all artifacts
        models_file = output_dir / "pca_models.joblib"
        joblib.dump({
            "ipca_models": self._ipca_models,
            "final_n_components": self._final_n_components,
            "imputation_medians": self._imputation_medians,
            "standardization_params": self._standardization_params,
            "group_features": self._group_features,
            "is_incremental": True,
        }, models_file)
        logger.info(f"Saved incremental PCA models to {models_file}")

        # Save summary
        summary_file = output_dir / "pca_summary.json"
        with open(summary_file, "w") as f:
            json.dump(self._summary.to_dict(), f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

    def load_artifacts(self, input_dir: Path = PCA_ARTIFACTS_DIR) -> None:
        """Load previously saved artifacts."""
        models_file = input_dir / "pca_models.joblib"
        data = joblib.load(models_file)

        # Handle both old (GroupPCAReducer) and new (IncrementalGroupPCAReducer) formats
        if data.get("is_incremental", False):
            self._ipca_models = data["ipca_models"]
            self._final_n_components = data["final_n_components"]
        else:
            # Load old format - convert PCA to IncrementalPCA-like usage
            # The old format stores regular PCA models
            old_pca_models = data.get("pca_models", {})
            for name, pca in old_pca_models.items():
                # Create a wrapper that mimics IncrementalPCA
                self._ipca_models[name] = pca  # PCA.transform works the same
                self._final_n_components[name] = pca.n_components_

        self._imputation_medians = data.get("imputation_medians", {})
        self._standardization_params = data.get("standardization_params", {})
        self._group_features = data.get("group_features", {})

        # Load summary
        summary_file = input_dir / "pca_summary.json"
        with open(summary_file) as f:
            summary_dict = json.load(f)

        self._summary = PCAReductionSummary(
            features_removed=summary_dict["features_removed"],
            features_added=summary_dict["features_added"],
            features_kept=summary_dict.get("features_kept", []),
            groups_skipped=summary_dict.get("groups_skipped", []),
            original_n_features=summary_dict["original_n_features"],
            final_n_features=summary_dict["final_n_features"],
        )

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
        self._fit_finalized = True
        logger.info(f"Loaded PCA artifacts from {input_dir}")
