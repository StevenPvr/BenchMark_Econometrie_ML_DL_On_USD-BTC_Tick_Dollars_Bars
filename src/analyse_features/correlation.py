"""Non-linear correlation analysis for features.

This module computes various correlation measures that capture both linear
and non-linear dependencies between features:

1. Spearman correlation: Rank-based, captures monotonic relationships
2. Mutual Information: Information-theoretic measure of shared information

Performance optimizations:
- Spearman: scipy.stats.spearmanr (vectorized C implementation)
- MI: sklearn with n_jobs=-1 for parallelization

References:
- Kraskov, A., et al. (2004). "Estimating mutual information". Physical Review E.
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

import warnings
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]
from sklearn.feature_selection import mutual_info_regression  # type: ignore[import-untyped]

from src.analyse_features.config import (
    CORRELATION_RESULTS_JSON,
    MI_N_NEIGHBORS,
    MI_RANDOM_STATE,
    SPEARMAN_HIGH_CORR_THRESHOLD,
    TARGET_COLUMN,
    ensure_directories,
)
from src.analyse_features.utils.json_utils import save_json
from src.analyse_features.utils.parallel import parallel_apply, get_n_jobs
from src.analyse_features.utils.plotting import plot_correlation_heatmap
from src.config_logging import get_logger

logger = get_logger(__name__)


def compute_spearman_matrix(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute Spearman correlation matrix for all features.

    Spearman correlation measures monotonic relationships using ranks.
    It's robust to outliers and captures non-linear monotonic dependencies.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze (default: all numeric).

    Returns:
        DataFrame with Spearman correlation matrix (n_features x n_features).
    """
    logger.info("Computing Spearman correlation matrix...")

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target if present (we analyze feature-feature correlations)
    if TARGET_COLUMN in feature_columns:
        feature_columns = [c for c in feature_columns if c != TARGET_COLUMN]

    n_features = len(feature_columns)
    logger.info("Analyzing %d features", n_features)

    # Extract feature matrix
    X = cast(np.ndarray, df[feature_columns].values)

    # scipy.stats.spearmanr is vectorized for matrix input
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        if n_features >= 3:
            # For 3+ features, returns correlation matrix
            corr_matrix, _ = stats.spearmanr(X, nan_policy="omit")
            corr_matrix = cast(np.ndarray, corr_matrix)
        else:
            # For 1-2 features, compute pairwise correlations
            corr_matrix = np.eye(n_features)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if n_features == 1:
                        # Single feature case
                        corr_val = 1.0
                    else:
                        # Two features case
                        # spearmanr returns (correlation, pvalue) tuple or SpearmanrResult
                        # Both support unpacking
                        result = stats.spearmanr(X[:, i], X[:, j], nan_policy="omit")
                        result_tuple = cast(tuple[float, float], result)
                        corr_val, _ = result_tuple
                        corr_val = float(corr_val)
                    corr_matrix[i, j] = corr_val
                    corr_matrix[j, i] = corr_val

    # Create DataFrame
    result = pd.DataFrame(corr_matrix)
    result.index = feature_columns
    result.columns = feature_columns

    # Log high correlations
    high_corr_count = (
        (np.abs(result.values) > SPEARMAN_HIGH_CORR_THRESHOLD)
        & (np.abs(result.values) < 1.0)
    ).sum() // 2  # Divide by 2 for symmetric matrix

    logger.info(
        "Spearman matrix computed. High correlations (>%.2f): %d pairs",
        SPEARMAN_HIGH_CORR_THRESHOLD,
        high_corr_count,
    )

    return result


def compute_mutual_information(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    n_neighbors: int = MI_N_NEIGHBORS,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Compute mutual information between features and target.

    MI measures the amount of information shared between variables.
    Uses k-nearest neighbors estimator (Kraskov et al., 2004).

    Args:
        df: DataFrame with features and target.
        target_column: Name of target variable.
        feature_columns: Features to analyze.
        n_neighbors: Number of neighbors for MI estimation.
        n_jobs: Number of parallel jobs.

    Returns:
        DataFrame with MI scores for each feature.
    """
    logger.info("Computing mutual information with target '%s'...", target_column)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [c for c in feature_columns if c != target_column]

    n_jobs = get_n_jobs(n_jobs)
    n_features = len(feature_columns)

    logger.info(
        "Analyzing %d features (n_neighbors=%d, n_jobs=%d)",
        n_features,
        n_neighbors,
        n_jobs,
    )

    # Prepare data
    X = df[feature_columns].values
    y = df[target_column].values

    # Handle NaN by dropping rows
    mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    logger.info("Using %d samples (dropped %d with NaN)", len(y_clean), len(y) - len(y_clean))

    # Compute MI using sklearn (parallelized)
    mi_scores = mutual_info_regression(
        X_clean,
        y_clean,
        n_neighbors=n_neighbors,
        random_state=MI_RANDOM_STATE,
        n_jobs=n_jobs,
    )

    # Create result DataFrame
    result = pd.DataFrame({
        "feature": feature_columns,
        "mutual_information": mi_scores,
    }).sort_values("mutual_information", ascending=False)

    result = result.reset_index(drop=True)

    # Log top features
    top_5 = result.head(5)
    logger.info("Top 5 features by MI:\n%s", top_5.to_string(index=False))

    return result


def compute_feature_feature_mi(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_neighbors: int = MI_N_NEIGHBORS,
    n_jobs: int | None = None,
    top_n_pairs: int = 50,
) -> pd.DataFrame:
    """Compute mutual information between all feature pairs.

    Note: This is computationally expensive for many features.
    We parallelize and return only top pairs.

    Args:
        df: DataFrame with features.
        feature_columns: Features to analyze.
        n_neighbors: Number of neighbors for MI estimation.
        n_jobs: Number of parallel jobs.
        top_n_pairs: Number of top pairs to return.

    Returns:
        DataFrame with top feature-feature MI pairs.
    """
    logger.info("Computing feature-feature mutual information...")

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if TARGET_COLUMN in feature_columns:
            feature_columns = [c for c in feature_columns if c != TARGET_COLUMN]

    n_jobs = get_n_jobs(n_jobs)
    n_features = len(feature_columns)

    # Create pairs (upper triangle only)
    pairs = [
        (i, j, feature_columns[i], feature_columns[j])
        for i in range(n_features)
        for j in range(i + 1, n_features)
    ]

    logger.info(
        "Computing MI for %d feature pairs (n_jobs=%d)",
        len(pairs),
        n_jobs,
    )

    # Prepare data dict for efficiency
    data = {col: cast(np.ndarray, df[col].dropna().values) for col in feature_columns}

    def compute_mi_pair(i, j, col_i, col_j):
        x = data[col_i]
        y = data[col_j]

        # Align lengths
        min_len = min(len(x), len(y))
        x = np.asarray(x[:min_len]).reshape(-1, 1)
        y = np.asarray(y[:min_len])

        if min_len < 10:
            return (col_i, col_j, np.nan)

        try:
            mi = mutual_info_regression(
                x, y,
                n_neighbors=n_neighbors,
                random_state=MI_RANDOM_STATE,
            )[0]
        except Exception:
            mi = np.nan

        return (col_i, col_j, mi)

    results = parallel_apply(
        lambda args: compute_mi_pair(*args),
        pairs,
        n_jobs=n_jobs,
    )

    # Create DataFrame and sort
    result = pd.DataFrame(results)
    result.columns = ["feature_1", "feature_2", "mutual_information"]
    result = result.dropna()
    result = result.sort_values("mutual_information", ascending=False)
    result = result.head(top_n_pairs).reset_index(drop=True)

    logger.info("Top 5 feature pairs by MI:\n%s", result.head(5).to_string(index=False))

    return result


def run_correlation_analysis(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    compute_dcor: bool = True,
    save_results: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run complete correlation analysis pipeline.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.
        compute_dcor: Whether to compute distance correlation (deprecated, ignored).
        save_results: Whether to save results to parquet.

    Returns:
        Dictionary with all correlation results.
    """
    ensure_directories()

    results = {}

    # 1. Spearman correlation matrix
    logger.info("=" * 60)
    logger.info("STEP 1: Spearman Correlation Matrix")
    logger.info("=" * 60)

    spearman_df = compute_spearman_matrix(df, feature_columns)
    results["spearman"] = spearman_df

    # 2. Mutual information with target
    logger.info("=" * 60)
    logger.info("STEP 2: Mutual Information with Target")
    logger.info("=" * 60)

    if TARGET_COLUMN in df.columns:
        mi_df = compute_mutual_information(df, TARGET_COLUMN, feature_columns)
        results["mutual_information"] = mi_df
    else:
        logger.warning("Target column '%s' not found, skipping MI analysis", TARGET_COLUMN)

    # Save results
    if save_results:
        json_data: dict[str, Any] = cast(
            dict[str, Any],
            {
                "timestamp": datetime.now().isoformat(),
                "analysis": "correlation",
            },
        )

        # Spearman summary (top correlations, not full matrix)
        if "spearman" in results:
            spearman = results["spearman"]
            n_features = len(spearman)

            # Get top correlated pairs
            corr_pairs = []
            for i in range(len(spearman)):
                for j in range(i + 1, len(spearman)):
                    val = spearman.iloc[i, j]
                    if abs(val) > 0.5:  # Only significant correlations
                        corr_pairs.append({
                            "feature_1": spearman.index[i],
                            "feature_2": spearman.columns[j],
                            "correlation": float(val),
                        })

            corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            json_data["spearman"] = {
                "n_features": n_features,
                "high_correlation_pairs": corr_pairs[:100],  # Top 100
                "n_high_corr_pairs": len(corr_pairs),
            }

            # Generate plot
            plot_correlation_heatmap(
                spearman,
                title="Spearman Correlation Matrix",
                filename="spearman_correlation",
            )

        # Mutual information
        if "mutual_information" in results:
            mi_df = results["mutual_information"]
            json_data["mutual_information"] = {
                "top_features": mi_df.head(50).to_dict(orient="records"),
            }

        save_json(json_data, CORRELATION_RESULTS_JSON)

    logger.info("=" * 60)
    logger.info("CORRELATION ANALYSIS COMPLETE")
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
        df = df[df["split"] == "train"].copy()
        df = df.drop(columns=["split"])

    results = run_correlation_analysis(cast(pd.DataFrame, df), compute_dcor=False)

    logger.info("Results keys: %s", list(results.keys()))
