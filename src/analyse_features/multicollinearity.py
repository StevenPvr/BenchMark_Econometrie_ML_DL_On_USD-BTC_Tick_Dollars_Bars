"""Multicollinearity analysis for features.

This module detects multicollinearity using:

1. VIF (Variance Inflation Factor):
   - VIF_i = 1 / (1 - R²_i) where R²_i is from regressing X_i on all other X_j
   - VIF = 1: No correlation
   - VIF > 5: Moderate multicollinearity
   - VIF > 10: High multicollinearity (problematic for linear models)

2. Condition Number:
   - κ(X) = σ_max / σ_min (ratio of largest to smallest singular value)
   - κ < 30: Low multicollinearity
   - 30 < κ < 100: Moderate
   - κ > 100: Severe multicollinearity

3. Eigenvalue Analysis:
   - Small eigenvalues indicate near-singularity
   - Eigenvalues near 0 cause numerical instability

Performance optimizations:
- VIF via correlation matrix inverse (O(n³) once vs O(n⁴) naive)
- Parallel VIF computation when using regression method
- Eigenvalue decomposition via LAPACK

References:
- Belsley, D. A., et al. (1980). "Regression Diagnostics".
- O'Brien, R. M. (2007). "A Caution Regarding Rules of Thumb for VIF".
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

import warnings  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Any, cast  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # type: ignore[import-untyped]  # noqa: E402
from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore[import-untyped]  # noqa: E402

# Suppress warnings for singular matrices and numerical issues
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=".*divide by zero.*")

from src.analyse_features.config import (
    CONDITION_NUMBER_HIGH,
    CONDITION_NUMBER_MODERATE,
    MULTICOLLINEARITY_RESULTS_JSON,
    TARGET_COLUMN,
    VIF_CRITICAL_THRESHOLD,
    VIF_HIGH_THRESHOLD,
    ensure_directories,
)
from src.analyse_features.utils.json_utils import save_json
from src.analyse_features.utils.parallel import parallel_map, get_n_jobs
from src.analyse_features.utils.plotting import plot_vif_scores
from src.config_logging import get_logger

logger = get_logger(__name__)


def compute_vif_correlation_method(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compute VIF using correlation matrix inverse (fast method).

    VIF_i = diag(R^-1)_i where R is the correlation matrix.

    This is O(n³) for matrix inversion, much faster than the O(n⁴)
    regression-based method for many features.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.

    Returns:
        DataFrame with VIF scores.
    """
    logger.info("Computing VIF via correlation matrix inverse...")

    # Extract feature matrix
    X = df[feature_columns].values

    # Remove rows with NaN
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    logger.info("Using %d samples (dropped %d with NaN)", X_clean.shape[0], X.shape[0] - X_clean.shape[0])

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_clean, rowvar=False)

    # Add small regularization for numerical stability
    corr_matrix += np.eye(len(feature_columns)) * 1e-8

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Invert correlation matrix
            corr_inv = np.linalg.inv(corr_matrix)

            # VIF = diagonal of inverse
            vif_values = np.diag(corr_inv)

    except np.linalg.LinAlgError:
        logger.warning("Correlation matrix is singular, using pseudo-inverse")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr_inv = np.linalg.pinv(corr_matrix)
            vif_values = np.diag(corr_inv)

    # Create result DataFrame
    result = pd.DataFrame({
        "feature": feature_columns,
        "vif": vif_values,
    })

    # Add interpretation
    result["interpretation"] = result["vif"].apply(_interpret_vif)

    # Sort by VIF descending
    result = result.sort_values("vif", ascending=False).reset_index(drop=True)

    return result


def compute_vif_regression_method(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Compute VIF using regression method (statsmodels).

    This is the traditional method: for each feature, regress it on all
    others and compute VIF = 1 / (1 - R²).

    Slower but more robust for ill-conditioned matrices.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.
        n_jobs: Number of parallel jobs.

    Returns:
        DataFrame with VIF scores.
    """
    logger.info("Computing VIF via regression method...")

    n_jobs = get_n_jobs(n_jobs)
    n_features = len(feature_columns)

    # Prepare data matrix
    X = df[feature_columns].values

    # Remove NaN rows
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    logger.info(
        "Computing VIF for %d features (n_jobs=%d)",
        n_features,
        n_jobs,
    )

    # Compute VIF for each feature in parallel
    def compute_single_vif(idx: int) -> tuple[str, float]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vif = variance_inflation_factor(X_clean, idx)
        except Exception:
            vif = np.inf
        return (feature_columns[idx], vif)

    results = parallel_map(
        compute_single_vif,
        range(n_features),
        n_jobs=n_jobs,
    )

    # Create DataFrame
    result = pd.DataFrame(results)
    result.columns = ["feature", "vif"]
    result["interpretation"] = result["vif"].apply(_interpret_vif)
    result = result.sort_values("vif", ascending=False).reset_index(drop=True)

    return result


def _interpret_vif(vif: float) -> str:
    """Interpret VIF value.

    Args:
        vif: VIF value.

    Returns:
        Interpretation string.
    """
    if np.isnan(vif) or np.isinf(vif):
        return "infinite (perfect collinearity)"
    elif vif > VIF_CRITICAL_THRESHOLD:
        return "critical"
    elif vif > VIF_HIGH_THRESHOLD:
        return "high"
    elif vif > 5.0:
        return "moderate"
    else:
        return "low"


def compute_condition_number(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Compute condition number of the feature matrix.

    The condition number κ(X) = σ_max / σ_min measures how sensitive
    the matrix is to small perturbations.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.

    Returns:
        Dictionary with condition number analysis.
    """
    logger.info("Computing condition number...")

    # Extract and standardize feature matrix
    X = df[feature_columns].values

    # Remove NaN rows
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    # Standardize (important for condition number)
    X_std = (X_clean - X_clean.mean(axis=0)) / (X_clean.std(axis=0) + 1e-10)

    # Compute singular values
    _, singular_values, _ = np.linalg.svd(X_std, full_matrices=False)

    # Condition number
    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]

    if sigma_min > 1e-12:
        condition_number = sigma_max / sigma_min
    else:
        condition_number = np.inf

    # Interpretation
    if condition_number < CONDITION_NUMBER_MODERATE:
        interpretation = "low_multicollinearity"
    elif condition_number < CONDITION_NUMBER_HIGH:
        interpretation = "moderate_multicollinearity"
    else:
        interpretation = "severe_multicollinearity"

    result = {
        "condition_number": condition_number,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "interpretation": interpretation,
        "n_features": len(feature_columns),
        "n_samples": X_clean.shape[0],
        "singular_values": singular_values.tolist(),
    }

    logger.info(
        "Condition number: %.2f (%s)",
        condition_number,
        interpretation,
    )

    return result


def compute_eigenvalue_analysis(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Analyze eigenvalues of the correlation matrix.

    Small eigenvalues indicate near-singularity and potential
    multicollinearity issues.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.

    Returns:
        DataFrame with eigenvalue analysis.
    """
    logger.info("Computing eigenvalue analysis...")

    # Extract feature matrix
    X = df[feature_columns].values

    # Remove NaN rows
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_clean, rowvar=False)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute condition indices
    max_eigenvalue = eigenvalues[0]
    condition_indices = np.sqrt(max_eigenvalue / (eigenvalues + 1e-12))

    # Create result DataFrame
    result = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(eigenvalues))],
        "eigenvalue": eigenvalues,
        "variance_explained": eigenvalues / eigenvalues.sum(),
        "cumulative_variance": np.cumsum(eigenvalues) / eigenvalues.sum(),
        "condition_index": condition_indices,
    })

    # Identify problematic components (condition index > 30)
    n_problematic = (result["condition_index"] > 30).sum()
    logger.info(
        "Eigenvalue analysis complete. Problematic components (CI > 30): %d",
        n_problematic,
    )

    return result


def identify_collinear_pairs(
    df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Identify highly correlated feature pairs.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.
        threshold: Correlation threshold for "high" correlation.

    Returns:
        DataFrame with highly correlated pairs.
    """
    logger.info("Identifying collinear pairs (threshold=%.2f)...", threshold)

    # Extract feature matrix
    X = df[feature_columns].values

    # Remove NaN rows
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_clean, rowvar=False)

    # Find pairs above threshold
    pairs = []
    n = len(feature_columns)

    for i in range(n):
        for j in range(i + 1, n):
            corr = abs(corr_matrix[i, j])
            if corr > threshold:
                pairs.append({
                    "feature_1": feature_columns[i],
                    "feature_2": feature_columns[j],
                    "correlation": corr_matrix[i, j],
                    "abs_correlation": corr,
                })

    result = pd.DataFrame(pairs)

    if not result.empty:
        result = result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    logger.info("Found %d highly correlated pairs", len(result))

    return result


def suggest_features_to_drop(
    vif_df: pd.DataFrame,
    corr_pairs_df: pd.DataFrame,
    vif_threshold: float = VIF_HIGH_THRESHOLD,
) -> list[str]:
    """Suggest features to drop based on VIF and correlation analysis.

    Strategy:
    1. For each highly correlated pair, suggest dropping the one with higher VIF
    2. For features with VIF > threshold, suggest dropping

    Args:
        vif_df: DataFrame with VIF scores.
        corr_pairs_df: DataFrame with correlated pairs.
        vif_threshold: VIF threshold for suggestion.

    Returns:
        List of feature names to consider dropping.
    """
    to_drop = set()

    # Create VIF lookup
    vif_lookup = dict(zip(vif_df["feature"], vif_df["vif"]))

    # For each correlated pair, suggest dropping higher VIF feature
    for _, row in corr_pairs_df.iterrows():
        f1, f2 = row["feature_1"], row["feature_2"]
        vif1 = vif_lookup.get(f1, 0)
        vif2 = vif_lookup.get(f2, 0)

        if vif1 > vif2:
            to_drop.add(f1)
        else:
            to_drop.add(f2)

    # Add features with VIF > threshold
    high_vif = vif_df[vif_df["vif"] > vif_threshold]["feature"].tolist()
    to_drop.update(high_vif)

    return sorted(list(to_drop))


def run_multicollinearity_analysis(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    method: str = "correlation",
    save_results: bool = True,
) -> dict[str, Any]:
    """Run complete multicollinearity analysis pipeline.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.
        method: VIF method ('correlation' or 'regression').
        save_results: Whether to save results to parquet.

    Returns:
        Dictionary with all analysis results.
    """
    ensure_directories()

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if TARGET_COLUMN in feature_columns:
            feature_columns = [c for c in feature_columns if c != TARGET_COLUMN]

    logger.info("=" * 60)
    logger.info("MULTICOLLINEARITY ANALYSIS")
    logger.info("=" * 60)
    logger.info("Analyzing %d features", len(feature_columns))

    results = {}

    # 1. VIF Analysis
    logger.info("-" * 40)
    logger.info("STEP 1: VIF Analysis")
    logger.info("-" * 40)

    if method == "correlation":
        vif_df = compute_vif_correlation_method(df, feature_columns)
    else:
        vif_df = compute_vif_regression_method(df, feature_columns)

    results["vif"] = vif_df

    # Log VIF summary
    vif_summary = vif_df["interpretation"].value_counts()
    logger.info("VIF summary:\n%s", vif_summary.to_string())

    # 2. Condition Number
    logger.info("-" * 40)
    logger.info("STEP 2: Condition Number")
    logger.info("-" * 40)

    cond_result = compute_condition_number(df, feature_columns)
    results["condition_number"] = cond_result

    # 3. Eigenvalue Analysis
    logger.info("-" * 40)
    logger.info("STEP 3: Eigenvalue Analysis")
    logger.info("-" * 40)

    eigen_df = compute_eigenvalue_analysis(df, feature_columns)
    results["eigenvalues"] = eigen_df

    # 4. Collinear Pairs
    logger.info("-" * 40)
    logger.info("STEP 4: Collinear Pairs")
    logger.info("-" * 40)

    pairs_df = identify_collinear_pairs(df, feature_columns)
    results["collinear_pairs"] = pairs_df

    # 5. Suggestions
    logger.info("-" * 40)
    logger.info("STEP 5: Feature Drop Suggestions")
    logger.info("-" * 40)

    suggestions = suggest_features_to_drop(vif_df, pairs_df)
    results["drop_suggestions"] = suggestions

    if suggestions:
        logger.warning(
            "Suggested features to drop (%d): %s",
            len(suggestions),
            suggestions[:10],  # First 10
        )

    # Save results
    if save_results:
        json_data: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "analysis": "multicollinearity",
        }

        if "vif" in results:
            vif_df = results["vif"]
            json_data["vif"] = {
                "high_vif_features": vif_df[vif_df["vif"] > 10]["feature"].tolist(),
                "critical_vif_features": vif_df[vif_df["vif"] > 100]["feature"].tolist(),
                "all_scores": vif_df.to_dict(orient="records"),
            }

            # Generate plot
            plot_vif_scores(vif_df)

        if "condition_number" in results:
            json_data["condition_number"] = results["condition_number"]

        if "collinear_pairs" in results:
            pairs_df = results["collinear_pairs"]
            if not pairs_df.empty:
                json_data["collinear_pairs"] = pairs_df.head(50).to_dict(orient="records")

        if "drop_suggestions" in results:
            json_data["drop_suggestions"] = results["drop_suggestions"]

        save_json(json_data, MULTICOLLINEARITY_RESULTS_JSON)

    logger.info("=" * 60)
    logger.info("MULTICOLLINEARITY ANALYSIS COMPLETE")
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

    results = run_multicollinearity_analysis(cast(pd.DataFrame, df))

    logger.info("VIF top 10:\n%s", results["vif"].head(10).to_string())
