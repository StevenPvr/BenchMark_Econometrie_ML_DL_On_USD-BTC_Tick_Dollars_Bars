"""Feature-target relationship analysis.

This module analyzes relationships between features and the target variable:

1. Correlation Analysis:
   - Pearson: Linear relationships
   - Spearman: Monotonic relationships (rank-based)

2. Mutual Information:
   - Non-linear dependency measure
   - MI = 0 iff X and Y are independent

3. Statistical Tests:
   - F-test for regression
   - Significance testing

Performance optimizations:
- Vectorized correlation computation
- Parallel MI calculation with sklearn
- Efficient sampling for large datasets

Use cases:
- Feature importance ranking
- Feature selection guidance
- Understanding predictive power
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]
from sklearn.feature_selection import f_regression, mutual_info_regression  # type: ignore[import-untyped]

from src.analyse_features.config import (
    MI_N_NEIGHBORS,
    MI_RANDOM_STATE,
    SCATTER_SAMPLE_SIZE,
    TARGET_COLUMN,
    TARGET_RESULTS_JSON,
    TOP_N_FEATURES,
    ensure_directories,
)
from src.analyse_features.utils.json_utils import save_json
from src.analyse_features.utils.parallel import get_n_jobs
from src.analyse_features.utils.plotting import plot_target_correlations
from src.config_logging import get_logger

logger = get_logger(__name__)


def compute_target_correlations(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations with target.

    Args:
        df: DataFrame with features and target.
        target_column: Name of target variable.
        feature_columns: Features to analyze.

    Returns:
        DataFrame with correlation metrics for each feature.
    """
    logger.info("Computing feature-target correlations...")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [c for c in feature_columns if c != target_column]

    y = df[target_column].values
    results = []

    for col in feature_columns:
        x = df[col].values

        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 10:
            results.append({
                "feature": col,
                "pearson_corr": np.nan,
                "pearson_pvalue": np.nan,
                "spearman_corr": np.nan,
                "spearman_pvalue": np.nan,
            })
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
            pearson_r = cast(float, pearson_r)
            pearson_p = cast(float, pearson_p)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
            spearman_r = cast(float, spearman_r)
            spearman_p = cast(float, spearman_p)

        results.append({
            "feature": col,
            "pearson_corr": pearson_r,
            "pearson_pvalue": pearson_p,
            "spearman_corr": spearman_r,
            "spearman_pvalue": spearman_p,
            "abs_pearson": abs(pearson_r),
            "abs_spearman": abs(spearman_r),
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values("abs_spearman", ascending=False).reset_index(drop=True)

    logger.info("Computed correlations for %d features", len(result_df))

    return result_df


def compute_target_mutual_information(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    n_neighbors: int = MI_N_NEIGHBORS,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Compute mutual information between features and target.

    Args:
        df: DataFrame with features and target.
        target_column: Name of target variable.
        feature_columns: Features to analyze.
        n_neighbors: Number of neighbors for MI estimation.
        n_jobs: Number of parallel jobs.

    Returns:
        DataFrame with MI scores for each feature.
    """
    logger.info("Computing feature-target mutual information...")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [c for c in feature_columns if c != target_column]

    n_jobs = get_n_jobs(n_jobs)

    X = df[feature_columns].values
    y = df[target_column].values

    mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    logger.info("Computing MI for %d features (n_jobs=%d)", len(feature_columns), n_jobs)

    mi_scores = mutual_info_regression(
        X_clean,
        y_clean,
        n_neighbors=n_neighbors,
        random_state=MI_RANDOM_STATE,
        n_jobs=n_jobs,
    )

    result_df = pd.DataFrame({
        "feature": feature_columns,
        "mutual_information": mi_scores,
    })

    result_df = result_df.sort_values("mutual_information", ascending=False).reset_index(drop=True)

    return result_df


def compute_f_scores(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute F-scores for feature-target relationships.

    Uses univariate F-test for regression (linear relationship).

    Args:
        df: DataFrame with features and target.
        target_column: Name of target variable.
        feature_columns: Features to analyze.

    Returns:
        DataFrame with F-scores and p-values.
    """
    logger.info("Computing F-scores for feature-target relationships...")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [c for c in feature_columns if c != target_column]

    X = df[feature_columns].values
    y = df[target_column].values

    mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    f_scores, p_values = f_regression(X_clean, y_clean)

    result_df = pd.DataFrame({
        "feature": feature_columns,
        "f_score": f_scores,
        "f_pvalue": p_values,
    })

    result_df = result_df.sort_values("f_score", ascending=False).reset_index(drop=True)

    return result_df


def compute_combined_target_metrics(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Compute all target relationship metrics combined.

    Args:
        df: DataFrame with features and target.
        target_column: Name of target variable.
        feature_columns: Features to analyze.
        n_jobs: Number of parallel jobs.

    Returns:
        DataFrame with all metrics merged.
    """
    logger.info("Computing combined target metrics...")

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [c for c in feature_columns if c != target_column]

    corr_df = compute_target_correlations(df, target_column, feature_columns)
    mi_df = compute_target_mutual_information(df, target_column, feature_columns, n_jobs=n_jobs)
    f_df = compute_f_scores(df, target_column, feature_columns)

    result_df = corr_df.merge(mi_df, on="feature")
    result_df = result_df.merge(f_df, on="feature")

    result_df["rank_pearson"] = result_df["abs_pearson"].rank(ascending=False)
    result_df["rank_spearman"] = result_df["abs_spearman"].rank(ascending=False)
    result_df["rank_mi"] = result_df["mutual_information"].rank(ascending=False)
    result_df["rank_f"] = result_df["f_score"].rank(ascending=False)

    result_df["avg_rank"] = (
        result_df["rank_pearson"]
        + result_df["rank_spearman"]
        + result_df["rank_mi"]
        + result_df["rank_f"]
    ) / 4

    result_df = result_df.sort_values("avg_rank").reset_index(drop=True)

    return result_df


def get_top_features(
    target_metrics_df: pd.DataFrame,
    n: int = TOP_N_FEATURES,
    by: str = "avg_rank",
) -> list[str]:
    """Get top N features by specified metric.

    Args:
        target_metrics_df: DataFrame from compute_combined_target_metrics.
        n: Number of top features.
        by: Metric to sort by.

    Returns:
        List of top feature names.
    """
    if by == "avg_rank":
        sorted_df = target_metrics_df.sort_values(by, ascending=True)
    else:
        sorted_df = target_metrics_df.sort_values(by, ascending=False)

    return sorted_df.head(n)["feature"].tolist()


def compute_feature_target_scatter_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
    sample_size: int = SCATTER_SAMPLE_SIZE,
) -> dict[str, pd.DataFrame]:
    """Prepare scatter plot data for feature-target relationships.

    Samples data for efficient visualization.

    Args:
        df: DataFrame with features and target.
        feature_columns: Features to include.
        target_column: Target column name.
        sample_size: Maximum samples per feature.

    Returns:
        Dictionary mapping feature names to scatter data.
    """
    logger.info("Preparing scatter data for %d features", len(feature_columns))

    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    scatter_data = {}

    for col in feature_columns:
        data = pd.DataFrame({
            "x": df_sample[col].values,
            "y": df_sample[target_column].values,
        }).dropna()

        scatter_data[col] = data

    return scatter_data


def analyze_nonlinearity(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Analyze non-linearity of feature-target relationships.

    Compares linear (Pearson) vs non-linear (Spearman, MI) metrics.
    Large differences indicate non-linear relationships.

    Args:
        df: DataFrame with features and target.
        target_column: Target column name.
        feature_columns: Features to analyze.

    Returns:
        DataFrame with non-linearity analysis.
    """
    logger.info("Analyzing non-linearity in feature-target relationships...")

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [c for c in feature_columns if c != target_column]

    metrics_df = compute_combined_target_metrics(df, target_column, feature_columns)

    metrics_df["pearson_spearman_diff"] = (
        metrics_df["abs_spearman"] - metrics_df["abs_pearson"]
    )

    mi_max = metrics_df["mutual_information"].max()
    if mi_max > 0:
        metrics_df["mi_normalized"] = metrics_df["mutual_information"] / mi_max
    else:
        metrics_df["mi_normalized"] = 0

    metrics_df["nonlinearity_score"] = (
        metrics_df["pearson_spearman_diff"].abs()
        + (metrics_df["mi_normalized"] - metrics_df["abs_pearson"]).clip(lower=0)
    )

    metrics_df = metrics_df.sort_values("nonlinearity_score", ascending=False)

    return metrics_df


def run_target_analysis(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    """Run complete target relationship analysis.

    Args:
        df: DataFrame with features and target.
        target_column: Target column name.
        feature_columns: Features to analyze.
        save_results: Whether to save results.

    Returns:
        Dictionary with all analysis results.
    """
    ensure_directories()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    logger.info("=" * 60)
    logger.info("TARGET RELATIONSHIP ANALYSIS")
    logger.info("=" * 60)
    logger.info("Target column: %s", target_column)

    results: dict[str, Any] = {}

    # 1. Combined metrics
    logger.info("-" * 40)
    logger.info("STEP 1: Computing Combined Metrics")
    logger.info("-" * 40)

    combined_df = compute_combined_target_metrics(df, target_column, feature_columns)
    results["combined_metrics"] = combined_df

    top_features = get_top_features(combined_df, n=10)
    logger.info("Top 10 features by average rank: %s", top_features)

    # 2. Non-linearity analysis
    logger.info("-" * 40)
    logger.info("STEP 2: Non-linearity Analysis")
    logger.info("-" * 40)

    nonlin_df = analyze_nonlinearity(df, target_column, feature_columns)
    results["nonlinearity"] = nonlin_df

    most_nonlinear = nonlin_df.head(10)["feature"].tolist()
    logger.info("Most non-linear relationships: %s", most_nonlinear)

    # 3. Summary statistics
    logger.info("-" * 40)
    logger.info("STEP 3: Summary Statistics")
    logger.info("-" * 40)

    summary = {
        "n_features": len(combined_df),
        "avg_abs_pearson": combined_df["abs_pearson"].mean(),
        "avg_abs_spearman": combined_df["abs_spearman"].mean(),
        "avg_mi": combined_df["mutual_information"].mean(),
        "max_abs_pearson": combined_df["abs_pearson"].max(),
        "max_abs_spearman": combined_df["abs_spearman"].max(),
        "max_mi": combined_df["mutual_information"].max(),
        "top_10_features": top_features,
        "most_nonlinear_features": most_nonlinear,
    }

    results["summary"] = summary

    logger.info("Average |Pearson|: %.4f", summary["avg_abs_pearson"])
    logger.info("Average |Spearman|: %.4f", summary["avg_abs_spearman"])
    logger.info("Average MI: %.4f", summary["avg_mi"])

    if save_results:
        json_data: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "analysis": "target_relationship",
        }

        if "combined_metrics" in results:
            metrics_df = results["combined_metrics"]
            json_data["top_features"] = metrics_df.head(50).to_dict(orient="records")
            json_data["feature_rankings"] = {
                "by_pearson": metrics_df.nsmallest(30, "rank_pearson")["feature"].tolist(),
                "by_spearman": metrics_df.nsmallest(30, "rank_spearman")["feature"].tolist(),
                "by_mi": metrics_df.nsmallest(30, "rank_mi")["feature"].tolist(),
            }

            plot_target_correlations(metrics_df)

        if "summary" in results:
            json_data["summary"] = results["summary"]

        save_json(json_data, TARGET_RESULTS_JSON)

    logger.info("=" * 60)
    logger.info("TARGET ANALYSIS COMPLETE")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    from src.config_logging import setup_logging
    from src.path import DATASET_FEATURES_PARQUET

    setup_logging()

    logger.info("Loading features from %s", DATASET_FEATURES_PARQUET)
    df = pd.read_parquet(DATASET_FEATURES_PARQUET)

    if "split" in df.columns:
        df = df[df["split"] == "train"].copy()
        df = df.drop(columns=["split"])

    results = run_target_analysis(cast(pd.DataFrame, df))

    logger.info("Top features:\n%s", results["combined_metrics"].head(20).to_string())
