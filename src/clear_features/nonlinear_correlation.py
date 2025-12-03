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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.clear_features.config import (
    CLEAR_FEATURES_DIR,
    META_COLUMNS,
    TARGET_COLUMN,
)
from src.config_logging import get_logger
from src.constants import (
    CLEAR_FEATURES_CORRELATION_DEFAULT_THRESHOLD,
    CLEAR_FEATURES_CORRELATION_MIN_CLUSTER_SIZE,
    CLEAR_FEATURES_CORRELATION_SAMPLE_SIZE,
)

logger = get_logger(__name__)


def _to_python_type(val: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization.

    Args:
        val: Value to convert (can be numpy or native type).

    Returns:
        Python native type equivalent.
    """
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val) if not np.isnan(val) else None
    if isinstance(val, np.ndarray):
        return [_to_python_type(v) for v in val]
    if isinstance(val, dict):
        return {k: _to_python_type(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_python_type(v) for v in val]
    return val


@dataclass
class CorrelationCluster:
    """A cluster of correlated features.

    Attributes:
        cluster_id: Unique identifier for the cluster.
        features: List of feature names in the cluster.
        avg_correlation: Average pairwise correlation.
        min_correlation: Minimum pairwise correlation.
        max_correlation: Maximum pairwise correlation.
    """

    cluster_id: int
    features: list[str]
    avg_correlation: float
    min_correlation: float
    max_correlation: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the cluster.
        """
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
    """Result of non-linear correlation analysis.

    Attributes:
        correlation_matrix: Pairwise correlation matrix (absolute values).
        clusters: List of identified feature clusters.
        unclustered_features: Features not assigned to any cluster.
        threshold_used: Correlation threshold used for clustering.
        method: Correlation method used (spearman or kendall).
    """

    correlation_matrix: pd.DataFrame | None = None
    clusters: list[CorrelationCluster] = field(default_factory=list)
    unclustered_features: list[str] = field(default_factory=list)
    threshold_used: float = CLEAR_FEATURES_CORRELATION_DEFAULT_THRESHOLD
    method: str = "spearman"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the analysis result.
        """
        return _to_python_type({
            "method": self.method,
            "threshold": self.threshold_used,
            "n_clusters": len(self.clusters),
            "n_clustered_features": sum(len(c.features) for c in self.clusters),
            "n_unclustered_features": len(self.unclustered_features),
            "clusters": [c.to_dict() for c in self.clusters],
        })


class NonLinearCorrelationAnalyzer:
    """Analyzes non-linear correlations and identifies feature clusters.

    Uses Spearman or Kendall correlation with hierarchical clustering
    to identify groups of highly correlated features.
    """

    def __init__(
        self,
        method: str = "spearman",
        threshold: float = CLEAR_FEATURES_CORRELATION_DEFAULT_THRESHOLD,
        min_cluster_size: int = CLEAR_FEATURES_CORRELATION_MIN_CLUSTER_SIZE,
    ) -> None:
        """Initialize analyzer.

        Args:
            method: Correlation method ('spearman' or 'kendall').
            threshold: Correlation threshold for clustering (0.7-0.9).
            min_cluster_size: Minimum features in a cluster.
        """
        self.method = method
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size

        self._corr_matrix: pd.DataFrame | None = None
        self._result: CorrelationAnalysisResult | None = None

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get numeric feature columns excluding metadata and target.

        Args:
            df: DataFrame to extract columns from.

        Returns:
            List of feature column names.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [
            c for c in numeric_cols
            if c not in META_COLUMNS and c != TARGET_COLUMN
        ]

    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        sample_size: int | None = CLEAR_FEATURES_CORRELATION_SAMPLE_SIZE,
    ) -> pd.DataFrame:
        """Compute non-linear correlation matrix.

        Args:
            df: DataFrame with features.
            sample_size: Sample size for faster computation (None = use all).

        Returns:
            Correlation matrix (absolute values).

        Raises:
            ValueError: If unknown correlation method.
        """
        feature_cols = self._get_feature_columns(df)
        logger.info(
            "Computing %s correlation for %d features", self.method, len(feature_cols)
        )

        if sample_size and len(df) > sample_size:
            df_sample = cast(
                pd.DataFrame,
                df.loc[:, feature_cols].sample(n=sample_size, random_state=42),
            )
            logger.info("Using sample of %d rows", sample_size)
        else:
            df_sample = cast(pd.DataFrame, df.loc[:, feature_cols])

        if self.method == "spearman":
            corr_matrix = df_sample.corr(method="spearman")
        elif self.method == "kendall":
            corr_matrix = df_sample.corr(method="kendall")
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._corr_matrix = corr_matrix.abs()

        logger.info("Correlation matrix computed: %s", self._corr_matrix.shape)

        return self._corr_matrix

    def _compute_cluster_statistics(
        self, feat_list: list[str], corr: pd.DataFrame
    ) -> tuple[float, float, float] | None:
        """Compute correlation statistics for a cluster.

        Args:
            feat_list: List of features in the cluster.
            corr: Correlation matrix.

        Returns:
            Tuple of (avg, min, max) correlation or None if no values.
        """
        sub_corr = corr.loc[feat_list, feat_list].values
        upper_idx = np.triu_indices_from(sub_corr, k=1)
        corr_values = sub_corr[upper_idx]

        if len(corr_values) == 0:
            return None

        return (
            float(np.nanmean(corr_values)),
            float(np.nanmin(corr_values)),
            float(np.nanmax(corr_values)),
        )

    def identify_clusters(
        self,
        corr_matrix: pd.DataFrame | None = None,
    ) -> list[CorrelationCluster]:
        """Identify clusters of correlated features using hierarchical clustering.

        Args:
            corr_matrix: Pre-computed correlation matrix (optional).

        Returns:
            List of correlation clusters.

        Raises:
            ValueError: If no correlation matrix available.
        """
        if corr_matrix is not None:
            self._corr_matrix = corr_matrix
        elif self._corr_matrix is None:
            raise ValueError(
                "No correlation matrix. Call compute_correlation_matrix first."
            )

        corr = self._corr_matrix
        corr_filled = corr.fillna(0)
        distance_matrix = 1 - corr_filled.values

        np.fill_diagonal(distance_matrix, 0)

        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method="average")

        distance_threshold = 1 - self.threshold
        cluster_labels = fcluster(
            linkage_matrix, t=distance_threshold, criterion="distance"
        )

        features = corr.columns.tolist()
        cluster_dict: dict[int, list[str]] = {}

        for feat, label in zip(features, cluster_labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(feat)

        clusters = []
        unclustered = []

        for label, feat_list in cluster_dict.items():
            if len(feat_list) >= self.min_cluster_size:
                stats = self._compute_cluster_statistics(feat_list, corr)
                if stats is not None:
                    avg_corr, min_corr, max_corr = stats
                    cluster = CorrelationCluster(
                        cluster_id=label,
                        features=feat_list,
                        avg_correlation=avg_corr,
                        min_correlation=min_corr,
                        max_correlation=max_corr,
                    )
                    clusters.append(cluster)
                else:
                    unclustered.extend(feat_list)
            else:
                unclustered.extend(feat_list)

        clusters.sort(key=lambda c: len(c.features), reverse=True)

        logger.info(
            "Found %d clusters with %d features",
            len(clusters),
            sum(len(c.features) for c in clusters),
        )
        logger.info("Unclustered features: %d", len(unclustered))

        return clusters

    def analyze(
        self,
        df: pd.DataFrame,
        sample_size: int | None = CLEAR_FEATURES_CORRELATION_SAMPLE_SIZE,
    ) -> CorrelationAnalysisResult:
        """Full analysis: compute correlation and identify clusters.

        Args:
            df: DataFrame with features.
            sample_size: Sample size for correlation computation.

        Returns:
            CorrelationAnalysisResult with all findings.
        """
        corr_matrix = self.compute_correlation_matrix(df, sample_size)
        clusters = self.identify_clusters(corr_matrix)

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

    def save_results(
        self, output_dir: Path = CLEAR_FEATURES_DIR / "correlation_analysis"
    ) -> None:
        """Save analysis results.

        Args:
            output_dir: Directory to save results to.

        Raises:
            RuntimeError: If analyze() was not called first.
        """
        if self._result is None:
            raise RuntimeError("No results to save. Call analyze() first.")

        output_dir.mkdir(parents=True, exist_ok=True)

        summary_file = output_dir / "correlation_clusters.json"
        with open(summary_file, "w") as f:
            json.dump(self._result.to_dict(), f, indent=2)
        logger.info("Saved cluster summary to %s", summary_file)

        if self._result.correlation_matrix is not None:
            corr_file = output_dir / "correlation_matrix.parquet"
            self._result.correlation_matrix.to_parquet(corr_file)
            logger.info("Saved correlation matrix to %s", corr_file)

    def get_clusters_for_pca(self) -> dict[str, list[str]]:
        """Get clusters formatted for PCA reducer.

        Returns:
            Dict mapping cluster_id to list of features.

        Raises:
            RuntimeError: If analyze() was not called first.
        """
        if self._result is None:
            raise RuntimeError("No results. Call analyze() first.")

        return {str(c.cluster_id): c.features for c in self._result.clusters}
