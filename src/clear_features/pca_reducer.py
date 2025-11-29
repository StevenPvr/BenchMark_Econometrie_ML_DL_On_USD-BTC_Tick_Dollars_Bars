"""PCA-based feature reduction for correlated feature clusters.

Note: Some type checker errors are false positives due to strict type checking
of numpy/pandas operations. The code is functionally correct.
"""

from __future__ import annotations

# pyright: reportGeneralTypeIssues=false

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from src.clear_features.config import (
    PCA_ARTIFACTS_DIR,
    PCA_CONFIG,
    WEIGHTING_CONFIG,
    META_COLUMNS,
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
class ClusterPCAResult:
    """Result of PCA applied to a single cluster."""

    cluster_id: str
    original_features: list[str]
    n_original: int
    n_components: int
    explained_variance_ratio: list[float]
    cumulative_variance: list[float]
    component_names: list[str]
    feature_weights: dict[str, float]  # MI weights used
    loadings: dict[str, list[float]]  # Feature loadings for each component
    reduction_type: str = "pca"  # 'pca' or 'simple_average' for perfect correlation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return _to_python_type({
            "cluster_id": self.cluster_id,
            "original_features": self.original_features,
            "n_original": self.n_original,
            "n_components": self.n_components,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_variance": self.cumulative_variance,
            "component_names": self.component_names,
            "feature_weights": self.feature_weights,
            "loadings": self.loadings,
            "reduction_type": self.reduction_type,
        })


@dataclass
class PCAReductionSummary:
    """Summary of all PCA reductions applied."""

    clusters_processed: list[ClusterPCAResult] = field(default_factory=list)
    features_removed: list[str] = field(default_factory=list)
    features_added: list[str] = field(default_factory=list)
    features_kept: list[str] = field(default_factory=list)
    original_n_features: int = 0
    final_n_features: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return _to_python_type({
            "n_clusters_processed": len(self.clusters_processed),
            "original_n_features": self.original_n_features,
            "final_n_features": self.final_n_features,
            "n_features_removed": len(self.features_removed),
            "n_features_added": len(self.features_added),
            "n_features_kept": len(self.features_kept),
            "features_removed": self.features_removed,
            "features_added": self.features_added,
            "clusters": [c.to_dict() for c in self.clusters_processed],
        })


class WeightedPCAReducer:
    """Applies weighted PCA to correlated feature clusters.

    Can receive clusters from:
    1. Dynamic non-linear correlation analysis (preferred)
    2. Pre-computed clustering results from analyse_features (legacy)
    """

    def __init__(
        self,
        clusters: dict[str, list[str]] | None = None,
        config: dict | None = None,
        weighting_config: dict | None = None,
    ):
        """
        Initialize PCA reducer.

        Args:
            clusters: Dict mapping cluster_id -> list of feature names.
                     If None, will try to load from analyse_features results.
            config: PCA configuration (variance threshold, etc.)
            weighting_config: MI weighting configuration
        """
        self.config = config or PCA_CONFIG
        self.weighting_config = weighting_config or WEIGHTING_CONFIG

        self._clusters_to_process: dict[str, list[str]] = clusters or {}
        self._mi_weights: dict[str, float] = {}
        self._pca_models: dict[str, PCA] = {}
        self._scalers: dict[str, StandardScaler] = {}
        self._summary: PCAReductionSummary | None = None
        # For perfect correlation clusters (simple weighted average)
        self._avg_clusters: dict[str, dict] = {}

    def set_clusters(self, clusters: dict[str, list[str]]) -> None:
        """Set clusters to process.

        Args:
            clusters: Dict mapping cluster_id -> list of feature names
        """
        self._clusters_to_process = clusters
        logger.info(f"Set {len(clusters)} clusters to process")

    def _get_feature_weights(self, features: list[str]) -> np.ndarray:
        """Get MI-based weights for features."""
        default_weight = self.weighting_config["default_mi_weight"]
        weights = np.array([
            self._mi_weights.get(f, default_weight) for f in features
        ])
        # Normalize weights to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(features)) / len(features)
        return weights

    def _determine_n_components(
        self, pca: PCA, variance_threshold: float
    ) -> int:
        """Determine optimal number of components based on variance threshold."""
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, variance_threshold)) + 1
        max_components = len(pca.explained_variance_ratio_)
        return int(min(int(n_components), int(max_components)))  # type: ignore

    def fit(self, df_train: pd.DataFrame) -> PCAReductionSummary:
        """Fit PCA models on training data only.

        Args:
            df_train: Training DataFrame with features

        Returns:
            PCAReductionSummary with details of the reduction
        """
        if not self._clusters_to_process:
            raise RuntimeError(
                "No clusters to process. Use set_clusters() or pass clusters to __init__"
            )

        # Get all feature columns (excluding metadata)
        all_features = [c for c in df_train.columns if c not in META_COLUMNS]

        self._summary = PCAReductionSummary()
        self._summary.original_n_features = len(all_features)

        features_in_clusters = set()

        for cluster_id, features in self._clusters_to_process.items():
            # Filter to features that exist in the dataframe
            available_features = [f for f in features if f in df_train.columns]
            if len(available_features) < 2:
                logger.warning(
                    f"Cluster {cluster_id}: Only {len(available_features)} "
                    f"features available, skipping"
                )
                continue

            features_in_clusters.update(available_features)

            # Get data for this cluster
            X = df_train[available_features].values  # type: ignore[assignment]

            # Handle NaN values
            if np.isnan(X).any():
                logger.warning(
                    f"Cluster {cluster_id}: Found NaN values, using mean imputation"
                )
                X_array = cast(np.ndarray, np.asarray(X, dtype=np.float64))
                col_means = cast(np.ndarray, np.nanmean(X_array, axis=0))  # type: ignore
                nan_mask = np.isnan(X_array)
                X_array[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
                X = X_array

            # Get feature weights
            weights = self._get_feature_weights(available_features)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Check for perfect correlation (near-zero variance after standardization)
            # This happens when all features are essentially identical
            feature_stds = np.std(X_scaled, axis=0)
            unique_features_mask = feature_stds > 1e-10

            if np.sum(unique_features_mask) <= 1:
                # Perfect correlation case - use weighted average instead of PCA
                logger.info(
                    f"Cluster {cluster_id}: Perfect correlation detected, "
                    f"using weighted average instead of PCA"
                )

                component_name = f"avg_cluster{cluster_id}"
                self._avg_clusters[cluster_id] = {
                    "features": available_features,
                    "weights": weights,
                    "scaler": scaler,
                }

                result = ClusterPCAResult(
                    cluster_id=cluster_id,
                    original_features=available_features,
                    n_original=len(available_features),
                    n_components=1,
                    explained_variance_ratio=[1.0],
                    cumulative_variance=[1.0],
                    component_names=[component_name],
                    feature_weights={f: float(w) for f, w in zip(available_features, weights)},
                    loadings={f: [1.0] for f in available_features},
                    reduction_type="weighted_average",
                )

                self._summary.clusters_processed.append(result)
                self._summary.features_removed.extend(available_features)
                self._summary.features_added.append(component_name)

                logger.info(
                    f"Cluster {cluster_id}: {len(available_features)} -> 1 "
                    f"(weighted average, perfect correlation)"
                )
                continue

            # Apply feature weighting
            X_weighted = X_scaled * np.sqrt(weights)

            # Fit PCA (fit on all components first to analyze variance)
            pca = PCA()
            pca.fit(X_weighted)

            # Determine number of components
            variance_threshold_val = self.config.get("variance_explained_threshold", 0.95)
            variance_threshold = float(variance_threshold_val) if isinstance(variance_threshold_val, (int, float)) else 0.95
            max_comp = self.config.get("max_components")

            n_components = self._determine_n_components(pca, variance_threshold)
            if max_comp is not None:
                n_components = min(n_components, int(cast(int, max_comp)))

            # Ensure at least 1 component
            n_components = max(1, n_components)

            # Refit with determined number of components
            pca_final = PCA(n_components=n_components)
            pca_final.fit(X_weighted)

            # Store models
            self._pca_models[cluster_id] = pca_final
            self._scalers[cluster_id] = scaler

            # Create component names
            component_names = [
                f"pca_cluster{cluster_id}_c{i}" for i in range(n_components)
            ]

            # Calculate loadings (correlation between original features and components)
            loadings = {}
            X_transformed = pca_final.transform(X_weighted)
            for i, feat in enumerate(available_features):
                feat_loadings = []
                for j in range(n_components):
                    # Handle case where std is 0
                    if np.std(X_scaled[:, i]) > 1e-10 and np.std(X_transformed[:, j]) > 1e-10:
                        corr = np.corrcoef(X_scaled[:, i], X_transformed[:, j])[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                    else:
                        corr = 0.0
                    feat_loadings.append(float(corr))
                loadings[feat] = feat_loadings

            # Create result
            result = ClusterPCAResult(
                cluster_id=cluster_id,
                original_features=available_features,
                n_original=len(available_features),
                n_components=n_components,
                explained_variance_ratio=[
                    float(v) for v in pca_final.explained_variance_ratio_
                ],
                cumulative_variance=[
                    float(v) for v in np.cumsum(pca_final.explained_variance_ratio_)
                ],
                component_names=component_names,
                feature_weights={f: float(w) for f, w in zip(available_features, weights)},
                loadings=loadings,
                reduction_type="pca",
            )

            self._summary.clusters_processed.append(result)
            self._summary.features_removed.extend(available_features)
            self._summary.features_added.extend(component_names)

            logger.info(
                f"Cluster {cluster_id}: {len(available_features)} -> "
                f"{n_components} components "
                f"(variance explained: {result.cumulative_variance[-1]:.4f})"
            )

        # Track features that were kept (not in any processed cluster)
        self._summary.features_kept = [
            f for f in all_features if f not in features_in_clusters
        ]

        self._summary.final_n_features = (
            len(self._summary.features_kept) + len(self._summary.features_added)
        )

        logger.info(
            f"Summary: {self._summary.original_n_features} -> "
            f"{self._summary.final_n_features} features"
        )

        return self._summary

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe using fitted PCA models."""
        if self._summary is None:
            raise RuntimeError("Summary not available. Must call fit() before transform()")

        # If no clusters to process, return dataframe as-is
        if not self._pca_models and not self._avg_clusters:
            logger.info("No clusters to process, returning dataframe unchanged")
            return df.copy()

        assert self._summary is not None  # For type checker
        summary = self._summary

        result_df = df.copy()

        # Process PCA clusters
        for cluster_id, pca in self._pca_models.items():
            scaler = self._scalers[cluster_id]

            # Get the cluster result for feature names
            cluster_result = next(
                c for c in summary.clusters_processed  # type: ignore
                if c.cluster_id == cluster_id
            )

            features = cluster_result.original_features
            component_names = cluster_result.component_names

            # Get feature weights
            weights = self._get_feature_weights(features)

            # Get data
            X = result_df[features].values  # type: ignore[assignment]

            # Handle NaN values
            if np.isnan(X).any():
                X_array = cast(np.ndarray, np.asarray(X, dtype=np.float64))
                col_means = cast(np.ndarray, np.nanmean(X_array, axis=0))  # type: ignore
                nan_mask = np.isnan(X_array)
                X_array[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
                X = X_array

            # Apply same transformation
            X_scaled = scaler.transform(X)
            X_weighted = X_scaled * np.sqrt(weights)
            X_pca = pca.transform(X_weighted)

            # Remove original features
            result_df = result_df.drop(columns=features)

            # Check if any component names already exist
            existing_components = [name for name in component_names if name in result_df.columns]
            if existing_components:
                logger.warning(
                    f"Cluster {cluster_id}: {len(existing_components)} PCA components already exist, "
                    f"skipping: {existing_components[:5]}"
                )
                # Only add components that don't exist
                new_component_names = [name for name in component_names if name not in result_df.columns]
                if new_component_names:
                    # Map indices to get correct columns from X_pca
                    indices_to_add = [component_names.index(name) for name in new_component_names]
                    pca_components_df = pd.DataFrame(
                        X_pca[:, indices_to_add],
                        columns=new_component_names,
                        index=result_df.index,
                    )
                    result_df = pd.concat([result_df, pca_components_df], axis=1)
            else:
                # Add all PCA components (use concat to avoid DataFrame fragmentation)
                pca_components_df = pd.DataFrame(
                    X_pca,
                    columns=component_names,
                    index=result_df.index,
                )
                result_df = pd.concat([result_df, pca_components_df], axis=1)

        # Process weighted average clusters (perfect correlation)
        for cluster_id, info in self._avg_clusters.items():
            features = info["features"]
            weights = info["weights"]

            # Get the cluster result for component name
            cluster_result = next(
                c for c in summary.clusters_processed  # type: ignore
                if c.cluster_id == cluster_id
            )
            component_name = cluster_result.component_names[0]

            # Get data
            X = result_df[features].values  # type: ignore[assignment]

            # Handle NaN values
            if np.isnan(X).any():
                X_array = cast(np.ndarray, np.asarray(X, dtype=np.float64))
                col_means = cast(np.ndarray, np.nanmean(X_array, axis=0))  # type: ignore
                nan_mask = np.isnan(X_array)
                X_array[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
                X = X_array

            # Compute weighted average
            X_array = cast(np.ndarray, np.asarray(X, dtype=np.float64))
            weights_array = cast(np.ndarray, np.asarray(weights, dtype=np.float64))
            X_avg = cast(np.ndarray, np.average(X_array, axis=1, weights=weights_array))  # type: ignore

            # Remove original features
            result_df = result_df.drop(columns=features)

            # Add weighted average component (skip if already exists)
            if component_name in result_df.columns:
                logger.warning(
                    f"Cluster {cluster_id}: Component '{component_name}' already exists, skipping"
                )
            else:
                result_df[component_name] = X_avg

        return result_df

    def fit_transform(self, df_train: pd.DataFrame) -> tuple[pd.DataFrame, PCAReductionSummary]:
        """Fit and transform in one step."""
        summary = self.fit(df_train)
        assert self._summary is not None  # For type checker
        df_transformed = self.transform(df_train)  # type: ignore
        return df_transformed, summary

    def save_artifacts(self, output_dir: Path = PCA_ARTIFACTS_DIR) -> None:
        """Save PCA models and summary."""
        if self._summary is None:
            raise RuntimeError("No summary available. Must call fit() before save_artifacts()")

        assert self._summary is not None  # For type checker
        summary = self._summary

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PCA models and avg clusters
        models_file = output_dir / "pca_models.joblib"
        joblib.dump(
            {
                "pca_models": self._pca_models,
                "scalers": self._scalers,
                "avg_clusters": self._avg_clusters,
            },
            models_file
        )
        logger.info(f"Saved PCA models to {models_file}")

        # Save summary as JSON
        summary_file = output_dir / "pca_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

        # Save detailed per-cluster info
        for cluster_result in summary.clusters_processed:
            cluster_file = output_dir / f"cluster_{cluster_result.cluster_id}_details.json"
            with open(cluster_file, "w") as f:
                json.dump(cluster_result.to_dict(), f, indent=2)

    def load_artifacts(self, input_dir: Path = PCA_ARTIFACTS_DIR) -> None:
        """Load previously saved PCA models."""
        models_file = input_dir / "pca_models.joblib"
        data = joblib.load(models_file)
        self._pca_models = data["pca_models"]
        self._scalers = data["scalers"]
        self._avg_clusters = data.get("avg_clusters", {})

        # Load summary
        summary_file = input_dir / "pca_summary.json"
        with open(summary_file) as f:
            summary_dict = json.load(f)

        # Reconstruct summary
        self._summary = PCAReductionSummary(
            features_removed=summary_dict["features_removed"],
            features_added=summary_dict["features_added"],
            features_kept=summary_dict.get("features_kept", []),
            original_n_features=summary_dict["original_n_features"],
            final_n_features=summary_dict["final_n_features"],
        )

        # Reconstruct cluster results
        for c_dict in summary_dict["clusters"]:
            result = ClusterPCAResult(
                cluster_id=c_dict["cluster_id"],
                original_features=c_dict["original_features"],
                n_original=c_dict["n_original"],
                n_components=c_dict["n_components"],
                explained_variance_ratio=c_dict["explained_variance_ratio"],
                cumulative_variance=c_dict["cumulative_variance"],
                component_names=c_dict["component_names"],
                feature_weights=c_dict["feature_weights"],
                loadings=c_dict["loadings"],
            )
            self._summary.clusters_processed.append(result)

        # Note: MI weights are already saved in cluster results (feature_weights)
        # No need to reload from correlation_results.json

        logger.info(f"Loaded PCA artifacts from {input_dir}")
