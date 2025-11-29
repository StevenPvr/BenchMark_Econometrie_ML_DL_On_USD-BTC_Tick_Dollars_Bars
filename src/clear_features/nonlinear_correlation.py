"""Non-linear correlation detection for feature clustering.

Uses Spearman correlation (rank-based, captures monotonic relationships)
to identify feature groups that should be merged via PCA.

Finance-appropriate thresholds:
- 0.7: Moderate correlation (common threshold)
- 0.8: High correlation (conservative, recommended for finance)
- 0.9: Very high correlation (aggressive reduction)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]

from src.clear_features.config import (
    CLEAR_FEATURES_DIR,
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


# Finance-appropriate correlation thresholds
CORRELATION_THRESHOLDS = {
    "moderate": 0.7,
    "high": 0.8,  # Recommended for finance
    "very_high": 0.9,
}

DEFAULT_THRESHOLD = 0.7  # Conservative for financial data


@dataclass
class CorrelationCluster:
    """A cluster of correlated features."""

    cluster_id: int
    features: list[str]
    avg_correlation: float
    min_correlation: float
    max_correlation: float

    def to_dict(self) -> dict[str, Any]:
        return _to_python_type({
            "cluster_id": self.cluster_id,
            "features": self.features,
            "n_features": len(self.features),
            "avg_correlation": self.avg_correlation,
            "min_correlation": self.min_correlation,
            "max_correlation": self.max_correlation,
        })


@dataclass
class CorrelationAnalysisResult:
    """Result of non-linear correlation analysis."""

    correlation_matrix: pd.DataFrame | None = None
    clusters: list[CorrelationCluster] = field(default_factory=list)
    unclustered_features: list[str] = field(default_factory=list)
    threshold_used: float = DEFAULT_THRESHOLD
    method: str = "spearman"

    def to_dict(self) -> dict[str, Any]:
        return _to_python_type({
            "method": self.method,
            "threshold": self.threshold_used,
            "n_clusters": len(self.clusters),
            "n_clustered_features": sum(len(c.features) for c in self.clusters),
            "n_unclustered_features": len(self.unclustered_features),
            "clusters": [c.to_dict() for c in self.clusters],
        })


class NonLinearCorrelationAnalyzer:
    """Analyzes non-linear correlations and identifies feature clusters."""

    def __init__(
        self,
        method: str = "spearman",
        threshold: float = DEFAULT_THRESHOLD,
        min_cluster_size: int = 2,
    ):
        """
        Initialize analyzer.

        Args:
            method: Correlation method ('spearman' or 'kendall')
            threshold: Correlation threshold for clustering (0.7-0.9)
            min_cluster_size: Minimum features in a cluster
        """
        self.method = method
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size

        self._corr_matrix: pd.DataFrame | None = None
        self._result: CorrelationAnalysisResult | None = None

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get numeric feature columns excluding metadata and target."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [
            c for c in numeric_cols
            if c not in META_COLUMNS and c != TARGET_COLUMN
        ]
        return feature_cols

    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        sample_size: int | None = 50000,
    ) -> pd.DataFrame:
        """
        Compute non-linear correlation matrix.

        Args:
            df: DataFrame with features
            sample_size: Sample size for faster computation (None = use all)

        Returns:
            Correlation matrix (absolute values)
        """
        feature_cols = self._get_feature_columns(df)
        logger.info(f"Computing {self.method} correlation for {len(feature_cols)} features")

        # Sample for faster computation if needed
        if sample_size and len(df) > sample_size:
            df_sample = cast(pd.DataFrame, df.loc[:, feature_cols].sample(n=sample_size, random_state=42))
            logger.info(f"Using sample of {sample_size} rows")
        else:
            df_sample = cast(pd.DataFrame, df.loc[:, feature_cols])

        # Compute correlation matrix
        if self.method == "spearman":
            corr_matrix = df_sample.corr(method="spearman")
        elif self.method == "kendall":
            corr_matrix = df_sample.corr(method="kendall")
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Use absolute correlation (we care about strength, not direction)
        self._corr_matrix = corr_matrix.abs()

        logger.info(f"Correlation matrix computed: {self._corr_matrix.shape}")

        return self._corr_matrix

    def identify_clusters(
        self,
        corr_matrix: pd.DataFrame | None = None,
    ) -> list[CorrelationCluster]:
        """
        Identify clusters of correlated features using hierarchical clustering.

        Args:
            corr_matrix: Pre-computed correlation matrix (optional)

        Returns:
            List of correlation clusters
        """
        if corr_matrix is not None:
            self._corr_matrix = corr_matrix
        elif self._corr_matrix is None:
            raise ValueError("No correlation matrix. Call compute_correlation_matrix first.")

        corr = self._corr_matrix

        # Convert correlation to distance (1 - correlation)
        # Handle NaN by setting them to max distance
        corr_filled = corr.fillna(0)
        distance_matrix = 1 - corr_filled.values

        # Ensure diagonal is 0
        np.fill_diagonal(distance_matrix, 0)

        # Hierarchical clustering
        # condensed distance matrix (upper triangle)
        condensed_dist = squareform(distance_matrix, checks=False)

        # Linkage (average linkage works well for correlation)
        linkage_matrix = linkage(condensed_dist, method="average")

        # Cut tree at distance = 1 - threshold
        # (correlation >= threshold means distance <= 1 - threshold)
        distance_threshold = 1 - self.threshold
        cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")

        # Group features by cluster
        features = corr.columns.tolist()
        cluster_dict: dict[int, list[str]] = {}

        for feat, label in zip(features, cluster_labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(feat)

        # Create cluster objects (only for clusters with min_cluster_size features)
        clusters = []
        unclustered = []

        for label, feat_list in cluster_dict.items():
            if len(feat_list) >= self.min_cluster_size:
                # Compute cluster statistics
                sub_corr = corr.loc[feat_list, feat_list].values
                # Get upper triangle (excluding diagonal)
                upper_idx = np.triu_indices_from(sub_corr, k=1)
                corr_values = sub_corr[upper_idx]

                if len(corr_values) > 0:
                    cluster = CorrelationCluster(
                        cluster_id=label,
                        features=feat_list,
                        avg_correlation=float(np.nanmean(corr_values)),
                        min_correlation=float(np.nanmin(corr_values)),
                        max_correlation=float(np.nanmax(corr_values)),
                    )
                    clusters.append(cluster)
                else:
                    unclustered.extend(feat_list)
            else:
                unclustered.extend(feat_list)

        # Sort clusters by size (largest first)
        clusters.sort(key=lambda c: len(c.features), reverse=True)

        logger.info(
            f"Found {len(clusters)} clusters with {sum(len(c.features) for c in clusters)} features"
        )
        logger.info(f"Unclustered features: {len(unclustered)}")

        return clusters

    def analyze(
        self,
        df: pd.DataFrame,
        sample_size: int | None = 50000,
    ) -> CorrelationAnalysisResult:
        """
        Full analysis: compute correlation and identify clusters.

        Args:
            df: DataFrame with features
            sample_size: Sample size for correlation computation

        Returns:
            CorrelationAnalysisResult with all findings
        """
        # Compute correlation
        corr_matrix = self.compute_correlation_matrix(df, sample_size)

        # Identify clusters
        clusters = self.identify_clusters(corr_matrix)

        # Get unclustered features
        clustered_features = set()
        for cluster in clusters:
            clustered_features.update(cluster.features)

        all_features = set(self._get_feature_columns(df))
        unclustered = list(all_features - clustered_features)

        self._result = CorrelationAnalysisResult(
            correlation_matrix=corr_matrix,
            clusters=clusters,
            unclustered_features=unclustered,
            threshold_used=self.threshold,
            method=self.method,
        )

        return self._result

    def save_results(self, output_dir: Path = CLEAR_FEATURES_DIR / "correlation_analysis") -> None:
        """Save analysis results."""
        if self._result is None:
            raise RuntimeError("No results to save. Call analyze() first.")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary JSON
        summary_file = output_dir / "correlation_clusters.json"
        with open(summary_file, "w") as f:
            json.dump(self._result.to_dict(), f, indent=2)
        logger.info(f"Saved cluster summary to {summary_file}")

        # Save correlation matrix
        if self._result.correlation_matrix is not None:
            corr_file = output_dir / "correlation_matrix.parquet"
            self._result.correlation_matrix.to_parquet(corr_file)
            logger.info(f"Saved correlation matrix to {corr_file}")

    def get_clusters_for_pca(self) -> dict[str, list[str]]:
        """
        Get clusters formatted for PCA reducer.

        Returns:
            Dict mapping cluster_id to list of features
        """
        if self._result is None:
            raise RuntimeError("No results. Call analyze() first.")

        return {
            str(c.cluster_id): c.features
            for c in self._result.clusters
        }
