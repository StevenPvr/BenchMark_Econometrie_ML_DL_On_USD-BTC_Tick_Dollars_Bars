"""Temporal analysis for features.

This module analyzes time-dependent properties of features:

1. ACF/PACF (Autocorrelation Function / Partial ACF):
   - Measures serial correlation within each feature
   - PACF shows direct correlation at lag k (controlling for 1..k-1)
   - Important for understanding feature persistence

2. Rolling Correlations:
   - Feature-target correlation over time windows
   - Detects regime changes and non-stationarity
   - Identifies features with time-varying predictive power

3. Temporal Stability:
   - How stable are feature distributions over time?
   - Rolling mean, variance, skewness
   - Detects concept drift

Performance optimizations:
- Numba-accelerated rolling correlations
- Vectorized ACF/PACF computation
- Parallel processing across features

Use cases:
- Detect features with decaying predictive power
- Identify regime-dependent features
- Guide walk-forward validation design
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
from statsmodels.tsa.stattools import acf, pacf, grangercausalitytests  # type: ignore[import-untyped]  # noqa: E402

# Suppress statsmodels warnings for singular matrices and division by zero
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", module="statsmodels.regression.linear_model")
warnings.filterwarnings("ignore", message=".*Matrix is singular.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
# Suppress numpy/scipy RuntimeWarnings for expected edge cases
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN axis encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Precision loss occurred.*")

from src.analyse_features.config import (  # noqa: E402
    ACF_LAGS,
    N_TEMPORAL_PERIODS,
    ROLLING_WINDOWS,
    TEMPORAL_RESULTS_JSON,
    TARGET_COLUMN,
    ensure_directories,
)
from src.analyse_features.utils.json_utils import save_json  # noqa: E402
from src.analyse_features.utils.parallel import parallel_map, get_n_jobs  # noqa: E402
from src.analyse_features.utils.numba_funcs import fast_rolling_correlation  # noqa: E402
from src.config_logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def compute_acf_single(
    series: np.ndarray,
    nlags: int = ACF_LAGS,
) -> dict[str, Any]:
    """Compute ACF and PACF for a single time series.

    Args:
        series: Time series values.
        nlags: Number of lags to compute.

    Returns:
        Dictionary with 'acf' and 'pacf' arrays.
    """
    # Remove NaN
    series_clean = series[~np.isnan(series)]

    if len(series_clean) < nlags + 10:
        return {
            "acf": np.full(nlags + 1, np.nan),
            "pacf": np.full(nlags + 1, np.nan),
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acf_values = acf(series_clean, nlags=nlags, fft=True)
            pacf_values = pacf(series_clean, nlags=nlags, method="ywm")
    except Exception:
        return {
            "acf": np.full(nlags + 1, np.nan),
            "pacf": np.full(nlags + 1, np.nan),
        }

    return {
        "acf": acf_values,
        "pacf": pacf_values,
    }


def compute_acf_all_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    nlags: int = ACF_LAGS,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Compute ACF and PACF for all features.

    Args:
        df: DataFrame with features.
        feature_columns: Features to analyze.
        nlags: Number of lags.
        n_jobs: Number of parallel jobs.

    Returns:
        DataFrame with ACF/PACF values for each feature.
    """
    logger.info("Computing ACF/PACF for %d features (nlags=%d)...", len(feature_columns), nlags)

    n_jobs = get_n_jobs(n_jobs)
    data = {col: cast(np.ndarray, df[col].values) for col in feature_columns}

    def compute_for_feature(col: str) -> dict[str, Any]:
        result = compute_acf_single(data[col], nlags)
        return {
            "feature": col,
            "acf": result["acf"],
            "pacf": result["pacf"],
        }

    results = parallel_map(compute_for_feature, feature_columns, n_jobs=n_jobs)

    # Create summary DataFrame
    summary_data = []
    for r in results:
        acf_vals = r["acf"]
        pacf_vals = r["pacf"]

        summary_data.append({
            "feature": r["feature"],
            "acf_lag1": acf_vals[1] if len(acf_vals) > 1 else np.nan,
            "acf_lag5": acf_vals[5] if len(acf_vals) > 5 else np.nan,
            "acf_lag10": acf_vals[10] if len(acf_vals) > 10 else np.nan,
            "acf_lag20": acf_vals[20] if len(acf_vals) > 20 else np.nan,
            "pacf_lag1": pacf_vals[1] if len(pacf_vals) > 1 else np.nan,
            "pacf_lag5": pacf_vals[5] if len(pacf_vals) > 5 else np.nan,
            "acf_decay_rate": _compute_decay_rate(acf_vals),
            "persistence": _compute_persistence(acf_vals),
        })

    result_df = pd.DataFrame(summary_data)

    # Sort by persistence (high persistence = slow decay = long memory)
    result_df = result_df.sort_values("persistence", ascending=False).reset_index(drop=True)

    return result_df


def _compute_decay_rate(acf_values: np.ndarray) -> float:
    """Compute ACF decay rate (how fast autocorrelation decays).

    Args:
        acf_values: ACF values starting from lag 0.

    Returns:
        Decay rate (higher = faster decay).
    """
    if len(acf_values) < 5 or np.all(np.isnan(acf_values)):
        return np.nan

    # Find first lag where |ACF| < 0.1
    abs_acf = np.abs(acf_values[1:])  # Skip lag 0
    below_threshold = np.where(abs_acf < 0.1)[0]

    if len(below_threshold) > 0:
        return 1.0 / (below_threshold[0] + 1)
    else:
        return 0.0  # Slow decay


def _compute_persistence(acf_values: np.ndarray) -> float:
    """Compute persistence score from ACF.

    Sum of absolute ACF values = measure of total autocorrelation.

    Args:
        acf_values: ACF values.

    Returns:
        Persistence score.
    """
    if len(acf_values) < 2 or np.all(np.isnan(acf_values)):
        return np.nan

    return np.nansum(np.abs(acf_values[1:]))  # Skip lag 0


def compute_rolling_target_correlation(
    df: pd.DataFrame,
    feature_column: str,
    target_column: str = TARGET_COLUMN,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling correlation between feature and target.

    Args:
        df: DataFrame with feature and target.
        feature_column: Feature column name.
        target_column: Target column name.
        windows: Rolling window sizes.

    Returns:
        DataFrame with rolling correlation for each window.
    """
    windows = windows or ROLLING_WINDOWS

    x = df[feature_column].values.astype(np.float64)
    y = df[target_column].values.astype(np.float64)

    result = pd.DataFrame({"index": range(len(df))})

    for window in windows:
        col_name = f"corr_w{window}"

        # Use Numba-accelerated rolling correlation
        corr_values = fast_rolling_correlation(
            np.ascontiguousarray(x),
            np.ascontiguousarray(y),
            window,
        )

        result[col_name] = corr_values

    return result


def compute_rolling_correlations_all(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
    windows: list[int] | None = None,
    n_jobs: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute rolling target correlations for all features.

    Args:
        df: DataFrame with features and target.
        feature_columns: Features to analyze.
        target_column: Target column name.
        windows: Rolling window sizes.
        n_jobs: Number of parallel jobs.

    Returns:
        Dictionary mapping feature names to rolling correlation DataFrames.
    """
    logger.info("Computing rolling correlations for %d features...", len(feature_columns))

    windows = windows or ROLLING_WINDOWS
    n_jobs = get_n_jobs(n_jobs)

    def compute_single(col: str) -> tuple[str, pd.DataFrame]:
        result = compute_rolling_target_correlation(
            df, col, target_column, windows
        )
        return (col, result)

    results = parallel_map(compute_single, feature_columns, n_jobs=n_jobs)

    return dict(results)


def compute_temporal_stability(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_periods: int = N_TEMPORAL_PERIODS,
) -> pd.DataFrame:
    """Analyze temporal stability of features.

    Splits data into periods and computes statistics for each.

    Args:
        df: DataFrame with features.
        feature_columns: Features to analyze.
        n_periods: Number of temporal periods.

    Returns:
        DataFrame with stability metrics.
    """
    logger.info("Analyzing temporal stability (%d periods)...", n_periods)

    n_rows = len(df)
    period_size = n_rows // n_periods

    results = []

    for col in feature_columns:
        x = df[col].values

        means = []
        stds = []
        skews = []

        for i in range(n_periods):
            start = i * period_size
            end = start + period_size if i < n_periods - 1 else n_rows
            period_data = x[start:end]
            period_data = period_data[~np.isnan(period_data)]

            if len(period_data) > 10:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    means.append(np.mean(period_data))
                    stds.append(np.std(period_data))
                    from scipy.stats import skew  # type: ignore[import-untyped]
                    skews.append(skew(period_data))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                skews.append(np.nan)

        # Stability = low coefficient of variation across periods
        mean_cv = np.nanstd(means) / (np.nanmean(np.abs(means)) + 1e-10)
        std_cv = np.nanstd(stds) / (np.nanmean(stds) + 1e-10)

        # Compute mean_trend safely handling NaN values
        means_arr = np.array(means)
        valid_mask = ~np.isnan(means_arr)
        if valid_mask.sum() >= 2:
            x_valid = np.arange(len(means_arr))[valid_mask]
            y_valid = means_arr[valid_mask]
            mean_trend = np.polyfit(x_valid, y_valid, 1)[0]
        else:
            mean_trend = np.nan

        results.append({
            "feature": col,
            "mean_stability": 1.0 / (1.0 + mean_cv),
            "std_stability": 1.0 / (1.0 + std_cv),
            "mean_range": np.nanmax(means) - np.nanmin(means),
            "std_range": np.nanmax(stds) - np.nanmin(stds),
            "mean_trend": mean_trend,
        })

    result_df = pd.DataFrame(results)

    # Combined stability score
    result_df["overall_stability"] = (
        result_df["mean_stability"] + result_df["std_stability"]
    ) / 2

    result_df = result_df.sort_values("overall_stability", ascending=False).reset_index(drop=True)

    return result_df


def compute_correlation_over_time(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
    n_periods: int = N_TEMPORAL_PERIODS,
) -> pd.DataFrame:
    """Compute feature-target correlation for each time period.

    Args:
        df: DataFrame with features and target.
        feature_columns: Features to analyze.
        target_column: Target column name.
        n_periods: Number of periods.

    Returns:
        DataFrame with correlation per period for each feature.
    """
    logger.info("Computing correlation over %d time periods...", n_periods)

    n_rows = len(df)
    period_size = n_rows // n_periods

    y = df[target_column].values

    all_results = []

    for col in feature_columns:
        x = df[col].values

        period_corrs = []

        for i in range(n_periods):
            start = i * period_size
            end = start + period_size if i < n_periods - 1 else n_rows

            x_period = x[start:end]
            y_period = y[start:end]

            # Remove NaN
            mask = ~(np.isnan(x_period) | np.isnan(y_period))
            x_clean = x_period[mask]
            y_clean = y_period[mask]

            if len(x_clean) > 10:
                corr = np.corrcoef(x_clean, y_clean)[0, 1]
            else:
                corr = np.nan

            period_corrs.append(corr)

        # Compute statistics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            all_results.append({
                "feature": col,
                "mean_corr": np.nanmean(period_corrs),
                "std_corr": np.nanstd(period_corrs),
                "min_corr": np.nanmin(period_corrs),
                "max_corr": np.nanmax(period_corrs),
                "corr_range": np.nanmax(period_corrs) - np.nanmin(period_corrs),
                "corr_consistency": 1.0 / (1.0 + np.nanstd(period_corrs)),
            })

    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values("corr_consistency", ascending=False).reset_index(drop=True)

    return result_df


def summarize_rolling_correlations(
    rolling_corrs: dict[str, pd.DataFrame],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Summarize rolling correlation results.

    Args:
        rolling_corrs: Dictionary from compute_rolling_correlations_all.
        windows: Window sizes used.

    Returns:
        DataFrame with summary statistics.
    """
    windows = windows or ROLLING_WINDOWS

    results = []

    for feature, corr_df in rolling_corrs.items():
        row: dict[str, Any] = {"feature": feature}

        for window in windows:
            col_name = f"corr_w{window}"
            if col_name in corr_df.columns:
                values = np.asarray(corr_df[col_name].dropna().values)
                if len(values) > 0:
                    row[f"mean_corr_w{window}"] = np.mean(values)
                    row[f"std_corr_w{window}"] = np.std(values)
                    row[f"min_corr_w{window}"] = np.min(values)
                    row[f"max_corr_w{window}"] = np.max(values)

        results.append(row)

    return pd.DataFrame(results)


def compute_granger_causality(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
    max_lag: int = 5,
) -> pd.DataFrame:
    """Test Granger causality from features to target.

    Granger causality tests whether past values of X help predict Y
    beyond what past values of Y alone can predict.

    Note: Granger causality is correlation-based and does not imply
    true causation. Both X and Y should be stationary for valid results.

    Args:
        df: DataFrame with features and target.
        feature_columns: Features to test.
        target_column: Target column name.
        max_lag: Maximum lag to test (tests lags 1 to max_lag).

    Returns:
        DataFrame with Granger causality results for each feature.
    """
    logger.info("Computing Granger causality tests (max_lag=%d)...", max_lag)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    y = df[target_column].values
    results = []

    for col in feature_columns:
        x = df[col].values

        # Align data removing NaN from both
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        # Need enough data points for reliable test
        if len(x_clean) < max_lag * 20:
            results.append({
                "feature": col,
                "granger_pvalue": np.nan,
                "best_lag": np.nan,
                "granger_significant": None,
            })
            continue

        # grangercausalitytests expects [y, x] format (testing if x causes y)
        data_matrix = np.column_stack([y_clean, x_clean])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_result = grangercausalitytests(data_matrix, maxlag=max_lag, verbose=False)

            # Extract F-test p-values for each lag
            pvalues = []
            for lag in range(1, max_lag + 1):
                if lag in gc_result:
                    # gc_result[lag] is a tuple: (test_results_dict, ols_results_tuple)
                    test_dict = gc_result[lag][0]
                    if "ssr_ftest" in test_dict:
                        pvalues.append(test_dict["ssr_ftest"][1])  # p-value
                    else:
                        pvalues.append(np.nan)
                else:
                    pvalues.append(np.nan)

            if pvalues and not all(np.isnan(pvalues)):
                # Find best lag (lowest p-value)
                valid_pvalues = [(i + 1, p) for i, p in enumerate(pvalues) if not np.isnan(p)]
                if valid_pvalues:
                    best_lag, best_pvalue = min(valid_pvalues, key=lambda x: x[1])
                else:
                    best_lag, best_pvalue = np.nan, np.nan
            else:
                best_lag, best_pvalue = np.nan, np.nan

            results.append({
                "feature": col,
                "granger_pvalue": float(best_pvalue) if not np.isnan(best_pvalue) else np.nan,
                "best_lag": int(best_lag) if not np.isnan(best_lag) else np.nan,
                "granger_significant": best_pvalue < 0.05 if not np.isnan(best_pvalue) else None,
            })

        except Exception as e:
            logger.debug("Granger test failed for %s: %s", col, e)
            results.append({
                "feature": col,
                "granger_pvalue": np.nan,
                "best_lag": np.nan,
                "granger_significant": None,
            })

    result_df = pd.DataFrame(results)

    # Sort by p-value (most significant first)
    result_df = result_df.sort_values("granger_pvalue", ascending=True).reset_index(drop=True)

    # Log summary
    if "granger_significant" in result_df.columns:
        n_significant = result_df["granger_significant"].sum()
        logger.info("Features with significant Granger causality: %d", n_significant)

    return result_df


def run_temporal_analysis(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
    save_results: bool = True,
) -> dict[str, Any]:
    """Run complete temporal analysis pipeline.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.
        target_column: Target column name.
        save_results: Whether to save results.

    Returns:
        Dictionary with all temporal analysis results.
    """
    ensure_directories()

    if feature_columns is None:
        feature_columns = cast(list[str], df.select_dtypes(include=[np.number]).columns.tolist())
        if target_column in feature_columns:
            feature_columns = [c for c in feature_columns if c != target_column]

    logger.info("=" * 60)
    logger.info("TEMPORAL ANALYSIS")
    logger.info("=" * 60)
    logger.info("Analyzing %d features", len(feature_columns))

    results = {}

    # 1. ACF/PACF Analysis
    logger.info("-" * 40)
    logger.info("STEP 1: ACF/PACF Analysis")
    logger.info("-" * 40)

    acf_df = compute_acf_all_features(df, feature_columns)
    results["acf_summary"] = acf_df

    # Most persistent features
    most_persistent = acf_df.head(10)["feature"].tolist()
    logger.info("Most persistent features: %s", most_persistent)

    # 2. Temporal Stability
    logger.info("-" * 40)
    logger.info("STEP 2: Temporal Stability")
    logger.info("-" * 40)

    stability_df = compute_temporal_stability(df, feature_columns)
    results["stability"] = stability_df

    # Most stable features
    most_stable = stability_df.head(10)["feature"].tolist()
    logger.info("Most stable features: %s", most_stable)

    # 3. Correlation Over Time
    logger.info("-" * 40)
    logger.info("STEP 3: Correlation Over Time")
    logger.info("-" * 40)

    if target_column in df.columns:
        corr_time_df = compute_correlation_over_time(df, feature_columns, target_column)
        results["correlation_over_time"] = corr_time_df

        # Most consistent correlations
        most_consistent = corr_time_df.head(10)["feature"].tolist()
        logger.info("Most consistent correlations: %s", most_consistent)

    # 4. Rolling Correlations (sample of features)
    logger.info("-" * 40)
    logger.info("STEP 4: Rolling Correlations (sample)")
    logger.info("-" * 40)

    if target_column in df.columns:
        # Just compute for top 20 features to save time
        top_features = feature_columns[:20]
        rolling_corrs = compute_rolling_correlations_all(df, top_features, target_column)
        results["rolling_correlations"] = rolling_corrs

        # Summarize
        rolling_summary = summarize_rolling_correlations(rolling_corrs)
        results["rolling_summary"] = rolling_summary

    # Save results
    if save_results:
        json_data: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "analysis": "temporal",
        }

        if "acf_summary" in results:
            acf_df = results["acf_summary"]
            json_data["most_persistent"] = acf_df.head(30).to_dict(orient="records")

        if "stability" in results:
            stab_df = results["stability"]
            json_data["most_stable"] = stab_df.head(30).to_dict(orient="records")

        if "correlation_over_time" in results:
            corr_df = results["correlation_over_time"]
            json_data["most_consistent"] = corr_df.head(30).to_dict(orient="records")

        save_json(json_data, TEMPORAL_RESULTS_JSON)

    logger.info("=" * 60)
    logger.info("TEMPORAL ANALYSIS COMPLETE")
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

    results = run_temporal_analysis(cast(pd.DataFrame, df))

    logger.info("ACF summary:\n%s", results["acf_summary"].head(10).to_string())
