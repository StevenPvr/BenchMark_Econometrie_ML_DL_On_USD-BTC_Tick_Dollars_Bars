"""Feature clustering analysis.

This module groups similar features to understand feature families:

1. Hierarchical Clustering:
   - Based on correlation distance (1 - |corr|)
   - Produces dendrogram for visualization
   - Can cut at different levels for flat clusters

2. t-SNE (t-distributed Stochastic Neighbor Embedding):
   - Non-linear dimensionality reduction
   - Preserves local structure
   - Good for visualization

3. UMAP (Uniform Manifold Approximation and Projection):
   - Faster than t-SNE
   - Preserves both local and global structure
   - Better for larger datasets

Use cases:
- Identify redundant feature groups
- Understand feature relationships
- Guide feature selection

Performance optimizations:
- Pre-compute correlation matrix once
- UMAP with n_jobs=-1 for parallelization
- OpenTSNE for faster t-SNE (if available)
"""

from __future__ import annotations


from pathlib import Path
import sys

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.cluster import hierarchy  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]
from sklearn.manifold import TSNE  # type: ignore[import-untyped]

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
        # Ward requires Euclidean distance, we'll use 'average' as fallback
        # or convert correlation distance to Euclidean-like
        linkage_matrix = hierarchy.linkage(condensed, method="average")
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
        "feature_order": dendro["ivl"],  # Ordered feature names
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
        # Default: use inconsistency method
        labels = hierarchy.fcluster(linkage_matrix, 1.0, criterion="inconsistent")

    result_df = pd.DataFrame({
        "feature": feature_columns,
        "cluster": labels,
    })

    result_df = result_df.sort_values("cluster").reset_index(drop=True)

    n_unique = len(result_df["cluster"].unique())
    logger.info("Cut dendrogram into %d clusters", n_unique)

    return result_df


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
    effective_perplexity = min(perplexity, max(5, n_features // 4))

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
) -> dict[int, dict[str, Any]]:
    """Identify and describe feature families from clusters.

    Args:
        cluster_df: DataFrame with cluster assignments.
        corr_matrix: Correlation matrix.
        feature_columns: Feature names.

    Returns:
        Dictionary with family descriptions.
    """
    families = {}

    for cluster_id in cluster_df["cluster"].unique():
        features_in_cluster = cluster_df[
            cluster_df["cluster"] == cluster_id
        ]["feature"].tolist()

        # Get indices
        indices = [feature_columns.index(f) for f in features_in_cluster]

        # Average intra-cluster correlation
        if len(indices) > 1:
            sub_corr = corr_matrix[np.ix_(indices, indices)]
            # Upper triangle excluding diagonal
            triu_idx = np.triu_indices(len(indices), k=1)
            avg_corr = np.mean(np.abs(sub_corr[triu_idx]))
        else:
            avg_corr = 1.0

        families[cluster_id] = {
            "n_features": len(features_in_cluster),
            "features": features_in_cluster,
            "avg_intra_correlation": avg_corr,
        }

    return families


def run_clustering_analysis(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_clusters: int = 10,
    compute_tsne: bool = True,
    compute_umap: bool = True,
    save_results: bool = True,
) -> dict[str, Any]:
    """Run complete feature clustering analysis.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.
        n_clusters: Number of clusters for flat clustering.
        compute_tsne: Whether to compute t-SNE embedding.
        compute_umap: Whether to compute UMAP embedding.
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

    # 1. Compute distance matrix
    logger.info("-" * 40)
    logger.info("STEP 1: Distance Matrix")
    logger.info("-" * 40)

    distance_matrix, corr_matrix = compute_correlation_distance_matrix(df, feature_columns)
    results["distance_matrix"] = distance_matrix
    results["correlation_matrix"] = corr_matrix

    # 2. Hierarchical clustering
    logger.info("-" * 40)
    logger.info("STEP 2: Hierarchical Clustering")
    logger.info("-" * 40)

    hier_result = hierarchical_clustering(distance_matrix, feature_columns)
    results["hierarchical"] = hier_result

    # Cut dendrogram
    cluster_df = cut_dendrogram(
        hier_result["linkage_matrix"],
        feature_columns,
        n_clusters=n_clusters,
    )
    results["clusters"] = cluster_df

    # 3. Feature families
    logger.info("-" * 40)
    logger.info("STEP 3: Feature Families")
    logger.info("-" * 40)

    families = identify_feature_families(cluster_df, corr_matrix, feature_columns)
    results["families"] = families

    for cluster_id, info in sorted(families.items()):
        logger.info(
            "Cluster %d: %d features (avg_corr=%.3f) - %s",
            cluster_id,
            info["n_features"],
            info["avg_intra_correlation"],
            info["features"][:5],  # First 5
        )

    # 4. t-SNE embedding
    if compute_tsne:
        logger.info("-" * 40)
        logger.info("STEP 4: t-SNE Embedding")
        logger.info("-" * 40)

        tsne_df = compute_tsne_embedding(df, feature_columns)
        results["tsne"] = tsne_df

    # 5. UMAP embedding
    if compute_umap:
        logger.info("-" * 40)
        logger.info("STEP 5: UMAP Embedding")
        logger.info("-" * 40)

        umap_df = compute_umap_embedding(df, feature_columns)
        results["umap"] = umap_df

    # Save results
    if save_results:
        json_data: dict[str, Any] = cast(
            dict[str, Any],
            {
                "timestamp": datetime.now().isoformat(),
                "analysis": "clustering",
            },
        )

        if "clusters" in results:
            clusters_df = results["clusters"]
            json_data["n_clusters"] = int(clusters_df["cluster"].nunique())
            json_data["cluster_assignments"] = clusters_df.to_dict(orient="records")

            # Generate dendrogram plot
            if "hierarchical" in results:
                hier_result = results["hierarchical"]
                linkage_matrix = hier_result["linkage_matrix"]
                
                # Use distance matrix for plotly (it computes linkage internally)
                dist_matrix_for_plot: np.ndarray | None = results.get("distance_matrix")
                
                plot_dendrogram(
                    linkage_matrix,
                    feature_columns,
                    title="Feature Hierarchical Clustering",
                    filename="feature_dendrogram",
                    distance_matrix=dist_matrix_for_plot,
                )

        if "families" in results:
            json_data["families"] = results["families"]

        if "tsne" in results:
            json_data["tsne_computed"] = True
            # Generate t-SNE plot
            if "clusters" in results:
                tsne_df = results["tsne"].merge(
                    results["clusters"],
                    on="feature",
                )
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
            # Generate UMAP plot
            if json_data["umap_computed"] and "clusters" in results:
                umap_df = results["umap"].merge(
                    results["clusters"],
                    on="feature",
                )
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

    results = run_clustering_analysis(df, n_clusters=15)

    logger.info("Cluster distribution:\n%s", results["clusters"]["cluster"].value_counts().to_string())
