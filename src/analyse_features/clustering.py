"""Feature clustering analysis with rigorous quality metrics.

This module groups similar features to understand feature families:

1. Hierarchical Clustering:
   - Based on correlation distance (1 - |corr|)
   - Produces dendrogram for visualization
   - Automatic optimal k selection via silhouette analysis

2. Clustering Quality Metrics:
   - Silhouette Score: Measures cluster cohesion and separation [-1, 1]
   - Davies-Bouldin Index: Lower is better (ratio of within/between cluster distances)
   - Calinski-Harabasz Index: Higher is better (variance ratio criterion)

3. Outlier Detection:
   - Z-score based detection in embedding space
   - Isolation of anomalous features for investigation

4. t-SNE and UMAP Embeddings:
   - For visualization only (not for clustering)
   - Outlier flagging in embedding space

Use cases:
- Identify redundant feature groups
- Understand feature relationships
- Guide feature selection
- Detect anomalous features
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.cluster import hierarchy  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]
from scipy.stats import zscore  # type: ignore[import-untyped]
from sklearn.manifold import TSNE  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from src.analyse_features.config import (
    CLUSTER_LINKAGE,
    CLUSTERING_RESULTS_JSON,
    TARGET_COLUMN,
    TSNE_N_ITER,
    TSNE_PERPLEXITY,
    TSNE_RANDOM_STATE,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    ensure_directories,
)
from src.analyse_features.utils.json_utils import save_json
from src.analyse_features.utils.parallel import get_n_jobs
from src.analyse_features.utils.plotting import plot_dendrogram, plot_embedding
from src.config_logging import get_logger

# Optional imports
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False

logger = get_logger(__name__)

# Clustering constants
MIN_CLUSTERS: int = 2
MAX_CLUSTERS: int = 15
OUTLIER_ZSCORE_THRESHOLD: float = 3.0


def compute_correlation_distance_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute correlation-based distance matrix for features.

    Distance = 1 - |correlation|, so perfectly correlated features
    have distance 0, uncorrelated features have distance 1.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.

    Returns:
        Tuple of (distance_matrix, correlation_matrix).
    """
    logger.info("Computing correlation distance matrix...")

    # Extract feature matrix
    X = df[feature_columns].values

    # Remove NaN rows
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_clean, rowvar=False)

    # Handle NaN in correlation (constant features)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Distance = 1 - |correlation|
    distance_matrix = 1 - np.abs(corr_matrix)

    # Ensure diagonal is exactly 0
    np.fill_diagonal(distance_matrix, 0)

    # Ensure symmetry
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    logger.info("Distance matrix shape: %s", distance_matrix.shape)

    return distance_matrix, corr_matrix


def diagnose_data_quality(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Diagnose data quality issues before clustering.

    Checks for:
    - Constant features (zero variance)
    - Near-constant features (very low variance)
    - Features with many NaN values
    - Features with extreme values

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.

    Returns:
        Dictionary with quality diagnostics.
    """
    logger.info("Diagnosing data quality for %d features...", len(feature_columns))

    X = df[feature_columns].to_numpy()
    diagnostics: dict[str, Any] = {
        "n_features": len(feature_columns),
        "n_samples": len(df),
        "issues": [],
    }

    # Check for constant features
    variances = np.nanvar(X, axis=0)
    constant_mask = variances == 0
    constant_features = [f for f, is_const in zip(feature_columns, constant_mask) if is_const]
    if constant_features:
        diagnostics["constant_features"] = constant_features
        diagnostics["issues"].append(f"{len(constant_features)} constant features detected")
        logger.warning("Found %d constant features: %s", len(constant_features), constant_features[:5])

    # Check for near-constant features (variance < 1e-10)
    near_constant_mask = (variances > 0) & (variances < 1e-10)
    near_constant_features = [f for f, is_nc in zip(feature_columns, near_constant_mask) if is_nc]
    if near_constant_features:
        diagnostics["near_constant_features"] = near_constant_features
        diagnostics["issues"].append(f"{len(near_constant_features)} near-constant features")
        logger.warning("Found %d near-constant features", len(near_constant_features))

    # Check for features with many NaN
    nan_fractions = np.mean(np.isnan(X), axis=0)
    high_nan_mask = nan_fractions > 0.1
    high_nan_features = [
        {"feature": f, "nan_fraction": float(frac)}
        for f, frac, has_nan in zip(feature_columns, nan_fractions, high_nan_mask)
        if has_nan
    ]
    if high_nan_features:
        diagnostics["high_nan_features"] = high_nan_features
        diagnostics["issues"].append(f"{len(high_nan_features)} features with >10% NaN")

    # Check for extreme values (|z-score| > 10)
    with np.errstate(invalid="ignore"):
        z_scores = np.abs(zscore(X, axis=0, nan_policy="omit"))
    extreme_mask = np.nanmax(z_scores, axis=0) > 10
    extreme_features = [f for f, is_extreme in zip(feature_columns, extreme_mask) if is_extreme]
    if extreme_features:
        diagnostics["extreme_value_features"] = extreme_features
        diagnostics["issues"].append(f"{len(extreme_features)} features with extreme values (|z|>10)")
        logger.warning("Found %d features with extreme values", len(extreme_features))

    diagnostics["n_issues"] = len(diagnostics["issues"])
    if diagnostics["n_issues"] == 0:
        logger.info("Data quality check passed - no issues detected")
    else:
        logger.warning("Data quality issues detected: %s", diagnostics["issues"])

    return diagnostics


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute clustering quality metrics.

    Args:
        X: Feature matrix (n_features, n_samples) - transposed.
        labels: Cluster labels for each feature.

    Returns:
        Dictionary with quality metrics.
    """
    n_clusters = len(np.unique(labels))

    if n_clusters < 2:
        return {
            "silhouette_score": np.nan,
            "davies_bouldin_score": np.nan,
            "calinski_harabasz_score": np.nan,
            "n_clusters": n_clusters,
        }

    try:
        sil_score = float(silhouette_score(X, labels))
    except Exception:
        sil_score = np.nan

    try:
        db_score = float(davies_bouldin_score(X, labels))
    except Exception:
        db_score = np.nan

    try:
        ch_score = float(calinski_harabasz_score(X, labels))
    except Exception:
        ch_score = np.nan

    return {
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "calinski_harabasz_score": ch_score,
        "n_clusters": n_clusters,
    }


def find_optimal_k(
    linkage_matrix: np.ndarray,
    X: np.ndarray,
    feature_columns: list[str],
    min_k: int = MIN_CLUSTERS,
    max_k: int = MAX_CLUSTERS,
) -> dict[str, Any]:
    """Find optimal number of clusters using silhouette analysis.

    Tests multiple values of k and selects the one with highest silhouette score.

    Args:
        linkage_matrix: Hierarchical clustering linkage matrix.
        X: Feature matrix (n_features, n_samples).
        feature_columns: Feature names.
        min_k: Minimum number of clusters to test.
        max_k: Maximum number of clusters to test.

    Returns:
        Dictionary with optimal k and metrics for all tested values.
    """
    logger.info("Finding optimal k in range [%d, %d]...", min_k, max_k)

    n_features = len(feature_columns)
    max_k = min(max_k, n_features - 1)  # Can't have more clusters than features

    if max_k < min_k:
        logger.warning("Cannot compute optimal k: max_k=%d < min_k=%d", max_k, min_k)
        return {
            "optimal_k": min_k,
            "metrics_by_k": {},
            "silhouette_scores": [],
            "davies_bouldin_scores": [],
        }

    metrics_by_k: dict[int, dict[str, float]] = {}
    silhouette_scores: list[float] = []
    davies_bouldin_scores: list[float] = []

    for k in range(min_k, max_k + 1):
        labels = hierarchy.fcluster(linkage_matrix, k, criterion="maxclust")
        metrics = compute_clustering_metrics(X, labels)
        metrics_by_k[k] = metrics

        sil = metrics["silhouette_score"]
        db = metrics["davies_bouldin_score"]
        silhouette_scores.append(sil if not np.isnan(sil) else -1)
        davies_bouldin_scores.append(db if not np.isnan(db) else float("inf"))

        logger.debug(
            "k=%d: silhouette=%.4f, davies_bouldin=%.4f, calinski_harabasz=%.1f",
            k,
            metrics["silhouette_score"],
            metrics["davies_bouldin_score"],
            metrics["calinski_harabasz_score"],
        )

    # Find optimal k by silhouette score (highest is best)
    valid_scores = [s for s in silhouette_scores if s > -1]
    if valid_scores:
        best_idx = silhouette_scores.index(max(valid_scores))
        optimal_k = min_k + best_idx
    else:
        optimal_k = min_k

    logger.info(
        "Optimal k=%d with silhouette=%.4f",
        optimal_k,
        metrics_by_k[optimal_k]["silhouette_score"],
    )

    return {
        "optimal_k": optimal_k,
        "metrics_by_k": metrics_by_k,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": davies_bouldin_scores,
        "k_range": list(range(min_k, max_k + 1)),
    }


def compute_silhouette_per_sample(
    X: np.ndarray,
    labels: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compute silhouette score for each feature.

    Features with negative silhouette are poorly clustered.

    Args:
        X: Feature matrix (n_features, n_samples).
        labels: Cluster labels.
        feature_columns: Feature names.

    Returns:
        DataFrame with silhouette score per feature.
    """
    try:
        sample_silhouettes = silhouette_samples(X, labels)
    except Exception as e:
        logger.warning("Could not compute per-sample silhouette: %s", e)
        return pd.DataFrame({
            "feature": feature_columns,
            "silhouette": np.nan,
            "cluster": labels,
        })

    return pd.DataFrame({
        "feature": feature_columns,
        "silhouette": sample_silhouettes,
        "cluster": labels,
    })


def hierarchical_clustering(
    distance_matrix: np.ndarray,
    feature_columns: list[str],
    linkage_method: str = CLUSTER_LINKAGE,
) -> dict[str, Any]:
    """Perform hierarchical clustering on features.

    Args:
        distance_matrix: Pairwise distance matrix.
        feature_columns: Feature names.
        linkage_method: Linkage method ('ward', 'complete', 'average', 'single').

    Returns:
        Dictionary with clustering results.
    """
    logger.info("Performing hierarchical clustering (linkage=%s)...", linkage_method)

    # Convert to condensed form for scipy
    condensed = squareform(distance_matrix, checks=False)

    # Handle any NaN/inf
    condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=0.0)

    # Compute linkage
    if linkage_method == "ward":
        # Ward requires Euclidean distance, not correlation distance
        corr_values = 1 - distance_matrix
        corr_values = np.clip(corr_values, -1, 1)
        euclidean_distance = np.sqrt(2 * (1 - corr_values))
        np.fill_diagonal(euclidean_distance, 0)
        euclidean_condensed = squareform(euclidean_distance, checks=False)
        euclidean_condensed = np.nan_to_num(euclidean_condensed, nan=1.0, posinf=1.0, neginf=0.0)
        linkage_matrix = hierarchy.linkage(euclidean_condensed, method="ward")
    else:
        linkage_matrix = hierarchy.linkage(condensed, method=linkage_method)

    # Compute dendrogram (for visualization data)
    dendro = hierarchy.dendrogram(
        linkage_matrix,
        labels=feature_columns,
        no_plot=True,
    )

    result = {
        "linkage_matrix": linkage_matrix,
        "dendrogram": dendro,
        "feature_order": dendro["ivl"],
        "leaves": dendro["leaves"],
    }

    logger.info("Hierarchical clustering complete")

    return result


def cut_dendrogram(
    linkage_matrix: np.ndarray,
    feature_columns: list[str],
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
) -> pd.DataFrame:
    """Cut dendrogram to form flat clusters.

    Args:
        linkage_matrix: Linkage matrix from hierarchical_clustering.
        feature_columns: Feature names.
        n_clusters: Number of clusters (mutually exclusive with distance_threshold).
        distance_threshold: Distance threshold for cutting.

    Returns:
        DataFrame with cluster assignments.
    """
    if n_clusters is not None:
        labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    elif distance_threshold is not None:
        labels = hierarchy.fcluster(linkage_matrix, distance_threshold, criterion="distance")
    else:
        labels = hierarchy.fcluster(linkage_matrix, 1.0, criterion="inconsistent")

    result_df = pd.DataFrame({
        "feature": feature_columns,
        "cluster": labels,
    })

    result_df = result_df.sort_values("cluster").reset_index(drop=True)

    n_unique = len(result_df["cluster"].unique())
    logger.info("Cut dendrogram into %d clusters", n_unique)

    return result_df


def detect_embedding_outliers(
    embedding_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    threshold: float = OUTLIER_ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    """Detect outliers in embedding space using z-score.

    Args:
        embedding_df: DataFrame with embedding coordinates.
        x_col: Column name for x coordinate.
        y_col: Column name for y coordinate.
        threshold: Z-score threshold for outlier detection.

    Returns:
        DataFrame with outlier flags and distances.
    """
    df = embedding_df.copy()

    # Skip if all NaN
    if bool(df[x_col].isna().all()) or bool(df[y_col].isna().all()):
        df["is_outlier"] = False
        df["distance_from_center"] = np.nan
        df["z_score"] = np.nan
        return df

    # Compute center
    center_x = df[x_col].mean()
    center_y = df[y_col].mean()

    # Compute Euclidean distance from center
    df["distance_from_center"] = np.sqrt((df[x_col] - center_x) ** 2 + (df[y_col] - center_y) ** 2)

    # Z-score of distances
    distances = df["distance_from_center"].values
    with np.errstate(invalid="ignore"):
        z_scores = zscore(distances, nan_policy="omit")
    df["z_score"] = z_scores

    # Flag outliers
    df["is_outlier"] = np.abs(df["z_score"]) > threshold

    n_outliers = df["is_outlier"].sum()
    if n_outliers > 0:
        outlier_features = df[df["is_outlier"]]["feature"].tolist()
        logger.warning(
            "Detected %d outliers in embedding (z>%.1f): %s",
            n_outliers,
            threshold,
            outlier_features,
        )

    return df


def compute_tsne_embedding(
    df: pd.DataFrame,
    feature_columns: list[str],
    perplexity: int = TSNE_PERPLEXITY,
    n_iter: int = TSNE_N_ITER,
    random_state: int = TSNE_RANDOM_STATE,
) -> pd.DataFrame:
    """Compute t-SNE embedding for features.

    Note: t-SNE operates on observations, but we want to embed features.
    We transpose the data to treat features as observations.

    Args:
        df: DataFrame with features.
        feature_columns: Features to embed.
        perplexity: t-SNE perplexity parameter.
        n_iter: Number of iterations.
        random_state: Random seed.

    Returns:
        DataFrame with 2D t-SNE coordinates for each feature.
    """
    logger.info("Computing t-SNE embedding (perplexity=%d)...", perplexity)

    # Extract and transpose (features become rows)
    X = df[feature_columns].values
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    # Transpose: each feature is now an observation
    X_T = X_clean.T  # Shape: (n_features, n_samples)

    # Adjust perplexity if needed
    n_features = X_T.shape[0]
    effective_perplexity = int(min(perplexity, max(1, n_features - 1)))

    if effective_perplexity != perplexity:
        logger.warning(
            "Adjusted perplexity from %d to %d (n_features=%d)",
            perplexity,
            effective_perplexity,
            n_features,
        )

    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )

    embedding = tsne.fit_transform(X_T)

    result_df = pd.DataFrame({
        "feature": feature_columns,
        "tsne_1": embedding[:, 0],
        "tsne_2": embedding[:, 1],
    })

    logger.info("t-SNE embedding complete")

    return result_df


def compute_umap_embedding(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    n_components: int = UMAP_N_COMPONENTS,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Compute UMAP embedding for features.

    Args:
        df: DataFrame with features.
        feature_columns: Features to embed.
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance between points.
        n_components: Output dimensionality.
        n_jobs: Number of parallel jobs.

    Returns:
        DataFrame with UMAP coordinates for each feature.
    """
    if not UMAP_AVAILABLE:
        logger.warning("umap-learn not installed, skipping UMAP embedding")
        return pd.DataFrame({
            "feature": feature_columns,
            "umap_1": np.nan,
            "umap_2": np.nan,
        })

    logger.info("Computing UMAP embedding (n_neighbors=%d)...", n_neighbors)

    n_jobs = get_n_jobs(n_jobs)

    # Extract and transpose
    X = df[feature_columns].values
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]
    X_T = X_clean.T

    # Adjust n_neighbors if needed
    n_features = X_T.shape[0]

    if n_features < 4:
        logger.warning(
            "UMAP requires at least 4 features, but only %d provided. Skipping UMAP embedding.",
            n_features,
        )
        return pd.DataFrame({
            "feature": feature_columns,
            "umap_1": np.nan,
            "umap_2": np.nan,
        })

    effective_n_neighbors = min(n_neighbors, max(2, n_features - 1))

    if effective_n_neighbors != n_neighbors:
        logger.warning(
            "Adjusted n_neighbors from %d to %d (n_features=%d)",
            n_neighbors,
            effective_n_neighbors,
            n_features,
        )

    # Compute UMAP
    assert umap is not None, "UMAP should be available at this point"
    umap_module = cast(Any, umap)
    reducer = umap_module.UMAP(
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="correlation",
        n_jobs=n_jobs,
        random_state=42,
    )

    embedding = reducer.fit_transform(X_T)

    result_df = pd.DataFrame({
        "feature": feature_columns,
        "umap_1": embedding[:, 0],
        "umap_2": embedding[:, 1] if n_components >= 2 else np.nan,
    })

    logger.info("UMAP embedding complete")

    return result_df


def identify_feature_families(
    cluster_df: pd.DataFrame,
    corr_matrix: np.ndarray,
    feature_columns: list[str],
    silhouette_df: pd.DataFrame | None = None,
) -> dict[int, dict[str, Any]]:
    """Identify and describe feature families from clusters.

    Args:
        cluster_df: DataFrame with cluster assignments.
        corr_matrix: Correlation matrix.
        feature_columns: Feature names.
        silhouette_df: Optional DataFrame with per-feature silhouette scores.

    Returns:
        Dictionary with family descriptions.
    """
    families = {}

    for cluster_id in sorted(cluster_df["cluster"].unique()):
        features_in_cluster = cluster_df[cluster_df["cluster"] == cluster_id]["feature"].tolist()

        # Get indices
        indices = [feature_columns.index(f) for f in features_in_cluster]

        # Average intra-cluster correlation
        if len(indices) > 1:
            sub_corr = corr_matrix[np.ix_(indices, indices)]
            triu_idx = np.triu_indices(len(indices), k=1)
            avg_corr = float(np.mean(np.abs(sub_corr[triu_idx])))
            min_corr = float(np.min(np.abs(sub_corr[triu_idx])))
            max_corr = float(np.max(np.abs(sub_corr[triu_idx])))
        else:
            avg_corr = 1.0
            min_corr = 1.0
            max_corr = 1.0

        # Average silhouette for cluster
        avg_silhouette = np.nan
        if silhouette_df is not None:
            cluster_sil = silhouette_df[silhouette_df["cluster"] == cluster_id]["silhouette"]
            avg_silhouette = float(cluster_sil.mean()) if len(cluster_sil) > 0 else np.nan

        families[int(cluster_id)] = {
            "n_features": len(features_in_cluster),
            "features": features_in_cluster,
            "avg_intra_correlation": avg_corr,
            "min_intra_correlation": min_corr,
            "max_intra_correlation": max_corr,
            "avg_silhouette": avg_silhouette,
        }

    return families


def run_clustering_analysis(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_clusters: int | None = None,  # None = auto-select optimal k
    min_clusters: int = MIN_CLUSTERS,
    max_clusters: int = MAX_CLUSTERS,
    compute_tsne: bool = True,
    compute_umap: bool = True,
    detect_outliers: bool = True,
    save_results: bool = True,
) -> dict[str, Any]:
    """Run complete feature clustering analysis with quality metrics.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.
        n_clusters: Fixed number of clusters (None = auto-select via silhouette).
        min_clusters: Minimum clusters for auto-selection.
        max_clusters: Maximum clusters for auto-selection.
        compute_tsne: Whether to compute t-SNE embedding.
        compute_umap: Whether to compute UMAP embedding.
        detect_outliers: Whether to detect outliers in embeddings.
        save_results: Whether to save results.

    Returns:
        Dictionary with all clustering results.
    """
    ensure_directories()

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if TARGET_COLUMN in feature_columns:
            feature_columns = [c for c in feature_columns if c != TARGET_COLUMN]

    logger.info("=" * 60)
    logger.info("FEATURE CLUSTERING ANALYSIS")
    logger.info("=" * 60)
    logger.info("Analyzing %d features", len(feature_columns))

    results: dict[str, Any] = {}

    # 0. Data quality diagnostics
    logger.info("-" * 40)
    logger.info("STEP 0: Data Quality Diagnostics")
    logger.info("-" * 40)

    quality_diagnostics = diagnose_data_quality(df, feature_columns)
    results["data_quality"] = quality_diagnostics

    # 1. Compute distance matrix
    logger.info("-" * 40)
    logger.info("STEP 1: Distance Matrix")
    logger.info("-" * 40)

    distance_matrix, corr_matrix = compute_correlation_distance_matrix(df, feature_columns)
    results["distance_matrix"] = distance_matrix
    results["correlation_matrix"] = corr_matrix

    # Prepare feature matrix for metrics (transpose)
    X = df[feature_columns].values
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]
    X_T = X_clean.T  # Shape: (n_features, n_samples)

    # 2. Hierarchical clustering
    logger.info("-" * 40)
    logger.info("STEP 2: Hierarchical Clustering")
    logger.info("-" * 40)

    hier_result = hierarchical_clustering(distance_matrix, feature_columns)
    results["hierarchical"] = hier_result

    # 3. Optimal k selection or use fixed k
    logger.info("-" * 40)
    logger.info("STEP 3: Optimal K Selection")
    logger.info("-" * 40)

    if n_clusters is None:
        optimal_k_result = find_optimal_k(
            hier_result["linkage_matrix"],
            X_T,
            feature_columns,
            min_k=min_clusters,
            max_k=max_clusters,
        )
        results["optimal_k_analysis"] = optimal_k_result
        n_clusters = optimal_k_result["optimal_k"]
        logger.info("Auto-selected k=%d based on silhouette analysis", n_clusters)
    else:
        logger.info("Using fixed k=%d", n_clusters)
        # Still compute metrics for the fixed k
        labels = hierarchy.fcluster(hier_result["linkage_matrix"], n_clusters, criterion="maxclust")
        metrics = compute_clustering_metrics(X_T, labels)
        results["fixed_k_metrics"] = metrics

    # Cut dendrogram with optimal/fixed k
    cluster_df = cut_dendrogram(
        hier_result["linkage_matrix"],
        feature_columns,
        n_clusters=n_clusters,
    )
    results["clusters"] = cluster_df

    # 4. Compute per-feature silhouette scores
    logger.info("-" * 40)
    logger.info("STEP 4: Per-Feature Silhouette Scores")
    logger.info("-" * 40)

    labels = cluster_df.set_index("feature").loc[feature_columns]["cluster"].values
    silhouette_df = compute_silhouette_per_sample(X_T, labels, feature_columns)
    results["silhouette_per_feature"] = silhouette_df

    # Log poorly clustered features (negative silhouette)
    poor_features = silhouette_df[silhouette_df["silhouette"] < 0]
    if len(poor_features) > 0:
        logger.warning(
            "%d features have negative silhouette (poorly clustered):",
            len(poor_features),
        )
        for _, row in poor_features.iterrows():
            logger.warning("  %s: silhouette=%.3f (cluster %d)", row["feature"], row["silhouette"], row["cluster"])

    # 5. Feature families with enriched metrics
    logger.info("-" * 40)
    logger.info("STEP 5: Feature Families")
    logger.info("-" * 40)

    families = identify_feature_families(cluster_df, corr_matrix, feature_columns, silhouette_df)
    results["families"] = families

    for cluster_id, info in sorted(families.items()):
        logger.info(
            "Cluster %d: %d features (avg_corr=%.3f, avg_sil=%.3f) - %s",
            cluster_id,
            info["n_features"],
            info["avg_intra_correlation"],
            info["avg_silhouette"] if not np.isnan(info["avg_silhouette"]) else 0,
            info["features"][:3],
        )

    # 6. t-SNE embedding
    if compute_tsne:
        logger.info("-" * 40)
        logger.info("STEP 6: t-SNE Embedding")
        logger.info("-" * 40)

        tsne_df = compute_tsne_embedding(df, feature_columns)

        if detect_outliers:
            tsne_df = detect_embedding_outliers(tsne_df, "tsne_1", "tsne_2")
            tsne_outliers = tsne_df[tsne_df["is_outlier"]]
            if len(tsne_outliers) > 0:
                results["tsne_outliers"] = tsne_outliers["feature"].tolist()

        results["tsne"] = tsne_df

    # 7. UMAP embedding
    if compute_umap:
        logger.info("-" * 40)
        logger.info("STEP 7: UMAP Embedding")
        logger.info("-" * 40)

        umap_df = compute_umap_embedding(df, feature_columns)

        if detect_outliers and not bool(umap_df["umap_1"].isna().all()):
            umap_df = detect_embedding_outliers(umap_df, "umap_1", "umap_2")
            umap_outliers = umap_df[umap_df["is_outlier"]]
            if len(umap_outliers) > 0:
                results["umap_outliers"] = umap_outliers["feature"].tolist()

        results["umap"] = umap_df

    # 8. Summary statistics
    logger.info("-" * 40)
    logger.info("STEP 8: Summary")
    logger.info("-" * 40)

    final_metrics = compute_clustering_metrics(X_T, labels)
    results["final_metrics"] = final_metrics

    logger.info("Final clustering metrics:")
    logger.info("  Silhouette Score: %.4f (range: -1 to 1, higher is better)", final_metrics["silhouette_score"])
    logger.info("  Davies-Bouldin Index: %.4f (lower is better)", final_metrics["davies_bouldin_score"])
    logger.info("  Calinski-Harabasz Index: %.1f (higher is better)", final_metrics["calinski_harabasz_score"])

    # Interpretation
    sil = final_metrics["silhouette_score"]
    if not np.isnan(sil):
        if sil > 0.5:
            interpretation = "Strong cluster structure"
        elif sil > 0.25:
            interpretation = "Reasonable cluster structure"
        elif sil > 0:
            interpretation = "Weak cluster structure - features form a continuum"
        else:
            interpretation = "No cluster structure detected - clusters are artificial"
        logger.info("  Interpretation: %s", interpretation)
        results["cluster_interpretation"] = interpretation

    # Save results
    if save_results:
        json_data: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "analysis": "clustering",
            "n_features": len(feature_columns),
            "n_clusters": n_clusters,
            "data_quality": {
                "n_issues": quality_diagnostics["n_issues"],
                "issues": quality_diagnostics["issues"],
            },
            "metrics": {
                "silhouette_score": final_metrics["silhouette_score"],
                "davies_bouldin_score": final_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": final_metrics["calinski_harabasz_score"],
            },
        }

        if "cluster_interpretation" in results:
            json_data["interpretation"] = results["cluster_interpretation"]

        if "optimal_k_analysis" in results:
            opt_k = results["optimal_k_analysis"]
            json_data["optimal_k_analysis"] = {
                "optimal_k": opt_k["optimal_k"],
                "k_range": opt_k["k_range"],
                "silhouette_by_k": {
                    k: float(m["silhouette_score"]) for k, m in opt_k["metrics_by_k"].items()
                },
            }

        if "clusters" in results:
            clusters_df = results["clusters"]
            json_data["cluster_assignments"] = clusters_df.to_dict(orient="records")

            # Generate dendrogram plot
            if "hierarchical" in results:
                hier_result = results["hierarchical"]
                linkage_matrix = hier_result["linkage_matrix"]
                dist_matrix_for_plot: np.ndarray | None = results.get("distance_matrix")

                plot_dendrogram(
                    linkage_matrix,
                    feature_columns,
                    title="Feature Hierarchical Clustering",
                    filename="feature_dendrogram",
                    distance_matrix=dist_matrix_for_plot,
                )

        if "families" in results:
            # Convert numpy types to Python types for JSON
            json_families = {}
            for k, v in results["families"].items():
                json_families[str(k)] = {
                    "n_features": v["n_features"],
                    "features": v["features"],
                    "avg_intra_correlation": float(v["avg_intra_correlation"]),
                    "avg_silhouette": float(v["avg_silhouette"]) if not np.isnan(v["avg_silhouette"]) else None,
                }
            json_data["families"] = json_families

        if "tsne_outliers" in results:
            json_data["tsne_outliers"] = results["tsne_outliers"]

        if "umap_outliers" in results:
            json_data["umap_outliers"] = results["umap_outliers"]

        if "tsne" in results:
            json_data["tsne_computed"] = True
            if "clusters" in results:
                tsne_df = results["tsne"].merge(results["clusters"], on="feature")
                plot_embedding(
                    tsne_df,
                    x_col="tsne_1",
                    y_col="tsne_2",
                    label_col="feature",
                    color_col="cluster",
                    title="t-SNE Feature Embedding",
                    filename="tsne_embedding",
                )

        if "umap" in results:
            umap_series = results["umap"]["umap_1"]
            all_na = bool(umap_series.isna().all())
            json_data["umap_computed"] = not all_na
            if json_data["umap_computed"] and "clusters" in results:
                umap_df = results["umap"].merge(results["clusters"], on="feature")
                plot_embedding(
                    umap_df,
                    x_col="umap_1",
                    y_col="umap_2",
                    label_col="feature",
                    color_col="cluster",
                    title="UMAP Feature Embedding",
                    filename="umap_embedding",
                )

        save_json(json_data, CLUSTERING_RESULTS_JSON)

    logger.info("=" * 60)
    logger.info("CLUSTERING ANALYSIS COMPLETE")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    from src.config_logging import setup_logging
    from src.path import DATASET_FEATURES_PARQUET

    setup_logging()

    logger.info("Loading features from %s", DATASET_FEATURES_PARQUET)
    df = pd.read_parquet(DATASET_FEATURES_PARQUET)

    # Filter to train split only
    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
        assert isinstance(train_df, pd.DataFrame), "Expected DataFrame after filtering"
        df = train_df.copy()
        df = df.drop(columns=["split"])

    # Run with auto-selection of optimal k
    results = run_clustering_analysis(df, n_clusters=None)

    logger.info("Cluster distribution:\n%s", results["clusters"]["cluster"].value_counts().to_string())
