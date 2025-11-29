"""Stationarity analysis for features.

This module tests each feature for stationarity using:

1. ADF (Augmented Dickey-Fuller): H0 = unit root (non-stationary)
   - p-value < alpha => reject H0 => stationary

2. KPSS (Kwiatkowski-Phillips-Schmidt-Shin): H0 = stationary
   - p-value < alpha => reject H0 => non-stationary

Interpretation:
- ADF rejects, KPSS fails to reject => Stationary
- ADF fails to reject, KPSS rejects => Non-stationary (unit root)
- Both reject => Difference-stationary (trend-stationary after differencing)
- Neither rejects => Uncertain (may need more data)

Performance: Processes features sequentially with incremental saves to avoid
memory issues. Results are cached and resumed if interrupted.

References:
- Dickey, D. A., & Fuller, W. A. (1979). "Distribution of the estimators for
  autoregressive time series with a unit root". JASA.
- Kwiatkowski, D., et al. (1992). "Testing the null hypothesis of stationarity
  against the alternative of a unit root". J. Econometrics.
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

import gc
import json
import warnings
from datetime import datetime
from typing import Any, Literal, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

# Maximum parallel jobs per batch to limit memory usage
MAX_PARALLEL_JOBS = 12

# Sample size for stationarity tests (50k observations is sufficient)
STATIONARITY_SAMPLE_SIZE = 50_000

from src.analyse_features.config import (
    ADF_MAX_LAGS,
    ADF_REGRESSION,
    KPSS_REGRESSION,
    STATIONARITY_ALPHA,
    STATIONARITY_RESULTS_JSON,
    TARGET_COLUMN,
    ensure_directories,
)
from src.analyse_features.utils.json_utils import save_json, convert_to_serializable
from src.analyse_features.utils.plotting import plot_stationarity_summary
from src.config_logging import get_logger

logger = get_logger(__name__)

# Cache file for incremental saves
STATIONARITY_CACHE_JSON = STATIONARITY_RESULTS_JSON.parent / "stationarity_cache.json"


def _test_adf(
    series: np.ndarray,
    regression: Literal['c', 'ct', 'n'] = ADF_REGRESSION,
    maxlag: int | None = ADF_MAX_LAGS,
) -> dict[str, Any]:
    """Run ADF test on a single series.

    Args:
        series: Time series (1D array).
        regression: Regression type ('c', 'ct', 'n').
        maxlag: Maximum lag for test (None = auto).

    Returns:
        Dictionary with test results.
    """
    # Remove NaN
    series_clean = series[~np.isnan(series)]

    if len(series_clean) < 20:
        return {
            "adf_statistic": np.nan,
            "adf_pvalue": np.nan,
            "adf_lags_used": np.nan,
            "adf_nobs": len(series_clean),
            "adf_reject_h0": None,
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(
                series_clean,
                maxlag=maxlag,
                regression=regression,
                autolag="AIC",
            )

        return {
            "adf_statistic": result[0],
            "adf_pvalue": result[1],
            "adf_lags_used": result[2],
            "adf_nobs": result[3],
            "adf_reject_h0": result[1] < STATIONARITY_ALPHA,
        }
    except Exception as e:
        logger.debug("ADF test failed: %s", e)
        return {
            "adf_statistic": np.nan,
            "adf_pvalue": np.nan,
            "adf_lags_used": np.nan,
            "adf_nobs": len(series_clean),
            "adf_reject_h0": None,
        }


def _test_kpss(
    series: np.ndarray,
    regression: Literal['c', 'ct'] = KPSS_REGRESSION,
) -> dict[str, Any]:
    """Run KPSS test on a single series.

    Args:
        series: Time series (1D array).
        regression: Regression type ('c', 'ct').

    Returns:
        Dictionary with test results.
    """
    # Remove NaN
    series_clean = series[~np.isnan(series)]

    if len(series_clean) < 20:
        return {
            "kpss_statistic": np.nan,
            "kpss_pvalue": np.nan,
            "kpss_lags_used": np.nan,
            "kpss_reject_h0": None,
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(series_clean, regression=regression, nlags="auto")

        # KPSS p-value can be outside [0.01, 0.1] bounds
        # statsmodels returns 0.01 or 0.1 as bounds
        return {
            "kpss_statistic": result[0],
            "kpss_pvalue": result[1],
            "kpss_lags_used": result[2],
            "kpss_reject_h0": result[1] < STATIONARITY_ALPHA,
        }
    except Exception as e:
        logger.debug("KPSS test failed: %s", e)
        return {
            "kpss_statistic": np.nan,
            "kpss_pvalue": np.nan,
            "kpss_lags_used": np.nan,
            "kpss_reject_h0": None,
        }


def test_stationarity_single(
    series: np.ndarray,
    feature_name: str,
) -> dict[str, Any]:
    """Test stationarity of a single feature using ADF and KPSS.

    Args:
        series: Feature values (1D array).
        feature_name: Name of the feature.

    Returns:
        Dictionary with all test results.
    """
    result = {"feature": feature_name}

    # ADF test
    adf_result = _test_adf(series)
    result.update(adf_result)

    # KPSS test
    kpss_result = _test_kpss(series)
    result.update(kpss_result)

    # Combined interpretation
    adf_reject = adf_result["adf_reject_h0"]
    kpss_reject = kpss_result["kpss_reject_h0"]

    if adf_reject is None or kpss_reject is None:
        result["stationarity_conclusion"] = "insufficient_data"
    elif adf_reject and not kpss_reject:
        result["stationarity_conclusion"] = "stationary"
    elif not adf_reject and kpss_reject:
        result["stationarity_conclusion"] = "non_stationary"
    elif adf_reject and kpss_reject:
        result["stationarity_conclusion"] = "trend_stationary"
    else:
        result["stationarity_conclusion"] = "uncertain"

    return result


def _load_cache() -> dict[str, dict[str, Any]]:
    """Load cached results from previous runs.

    Returns:
        Dictionary mapping feature names to their results.
    """
    if STATIONARITY_CACHE_JSON.exists():
        try:
            with open(STATIONARITY_CACHE_JSON, encoding="utf-8") as f:
                cache_data = json.load(f)
            logger.info("Loaded %d cached results from previous run", len(cache_data))
            return cache_data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load cache: %s", e)
    return {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    """Save cache to disk.

    Args:
        cache: Dictionary mapping feature names to their results.
    """
    STATIONARITY_CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    serializable_cache = convert_to_serializable(cache)
    with open(STATIONARITY_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(serializable_cache, f, indent=2, ensure_ascii=False)


def _clear_cache() -> None:
    """Remove cache file after successful completion."""
    if STATIONARITY_CACHE_JSON.exists():
        STATIONARITY_CACHE_JSON.unlink()
        logger.info("Cleared stationarity cache file")


def _process_feature_batch(
    df: pd.DataFrame,
    batch_cols: list[str],
    n_jobs: int,
) -> list[dict[str, Any]]:
    """Process a batch of features in parallel.

    Args:
        df: DataFrame with features.
        batch_cols: List of column names to process.
        n_jobs: Number of parallel jobs.

    Returns:
        List of test results for each feature.
    """
    def _test_single(col: str) -> dict[str, Any]:
        series = cast(np.ndarray, df[col].values)
        return test_stationarity_single(series, col)

    batch_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(_test_single)(col) for col in batch_cols
    )
    return cast(list[dict[str, Any]], batch_results)


def test_stationarity_all(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_jobs: int = MAX_PARALLEL_JOBS,
    sample_size: int = STATIONARITY_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Test stationarity for all features in parallel batches.

    Samples data to reduce memory and computation time. 50k observations
    is sufficient for reliable stationarity tests.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to test (default: all numeric).
        n_jobs: Number of parallel jobs per batch (default: 12).
        sample_size: Number of rows to sample (default: 50,000).

    Returns:
        DataFrame with test results for each feature.
    """
    logger.info("Testing stationarity (n_jobs=%d, sample_size=%d)...", n_jobs, sample_size)

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target
        if TARGET_COLUMN in feature_columns:
            feature_columns = [c for c in feature_columns if c != TARGET_COLUMN]

    n_features = len(feature_columns)
    logger.info("Testing %d features", n_features)

    # Load cache from previous runs
    cache = _load_cache()
    results: list[dict[str, Any]] = []

    # Separate cached and uncached features
    uncached_cols = [col for col in feature_columns if col not in cache]
    cached_cols = [col for col in feature_columns if col in cache]

    # Add cached results
    for col in cached_cols:
        results.append(cache[col])

    if cached_cols:
        logger.info("Resuming: %d/%d features already cached", len(cached_cols), n_features)

    if not uncached_cols:
        logger.info("All features already cached, skipping computation")
    else:
        # Sample data ONCE to reduce memory (keep temporal order)
        n_rows = len(df)
        if n_rows > sample_size:
            # Systematic sampling to preserve time series structure
            step = n_rows // sample_size
            indices = np.arange(0, n_rows, step)[:sample_size]
            df_sampled = cast(pd.DataFrame, df.iloc[indices][uncached_cols].copy())
            logger.info("Sampled %d/%d rows (step=%d)", len(df_sampled), n_rows, step)
        else:
            df_sampled = cast(pd.DataFrame, df[uncached_cols].copy())
            logger.info("Using all %d rows (below sample threshold)", n_rows)

        # Clear original df reference for memory
        del df
        gc.collect()

        # Process uncached features in batches
        n_batches = (len(uncached_cols) + n_jobs - 1) // n_jobs

        pbar = tqdm(range(n_batches), desc="Stationarity", unit="batch")
        for batch_idx in pbar:
            start_idx = batch_idx * n_jobs
            end_idx = min(start_idx + n_jobs, len(uncached_cols))
            batch_cols = uncached_cols[start_idx:end_idx]

            pbar.set_postfix_str(f"{start_idx+1}-{end_idx}/{len(uncached_cols)}")

            # Process batch in parallel
            batch_results = _process_feature_batch(df_sampled, batch_cols, n_jobs)

            # Add results and update cache
            for col, result in zip(batch_cols, batch_results):
                results.append(result)
                cache[col] = result

            # Save cache after each batch
            _save_cache(cache)

            # Clear memory after batch
            gc.collect()

        pbar.close()

        # Clear sampled df
        del df_sampled
        gc.collect()

    logger.info("Completed all %d features", n_features)

    # Create DataFrame
    result_df = pd.DataFrame(results)

    # Sort by conclusion
    conclusion_order = {
        "stationary": 0,
        "trend_stationary": 1,
        "uncertain": 2,
        "non_stationary": 3,
        "insufficient_data": 4,
    }
    result_df["_sort_key"] = result_df["stationarity_conclusion"].replace(conclusion_order)
    result_df = result_df.sort_values("_sort_key").drop(columns=["_sort_key"])
    result_df = result_df.reset_index(drop=True)

    # Log summary
    summary = result_df["stationarity_conclusion"].value_counts()
    logger.info("Stationarity summary:\n%s", summary.to_string())

    return result_df


def summarize_stationarity(result_df: pd.DataFrame) -> dict[str, Any]:
    """Generate summary statistics from stationarity tests.

    Args:
        result_df: DataFrame from test_stationarity_all.

    Returns:
        Dictionary with summary statistics.
    """
    n_total = len(result_df)

    conclusion_counts = result_df["stationarity_conclusion"].value_counts().to_dict()

    summary = {
        "total_features": n_total,
        "stationary_count": conclusion_counts.get("stationary", 0),
        "non_stationary_count": conclusion_counts.get("non_stationary", 0),
        "trend_stationary_count": conclusion_counts.get("trend_stationary", 0),
        "uncertain_count": conclusion_counts.get("uncertain", 0),
        "insufficient_data_count": conclusion_counts.get("insufficient_data", 0),
        "stationary_pct": 100 * conclusion_counts.get("stationary", 0) / n_total,
        "non_stationary_pct": 100 * conclusion_counts.get("non_stationary", 0) / n_total,
    }

    # Add lists of features by category
    summary["stationary_features"] = result_df[
        result_df["stationarity_conclusion"] == "stationary"
    ]["feature"].tolist()

    summary["non_stationary_features"] = result_df[
        result_df["stationarity_conclusion"] == "non_stationary"
    ]["feature"].tolist()

    return summary


def get_non_stationary_features(result_df: pd.DataFrame) -> list[str]:
    """Get list of non-stationary features.

    Args:
        result_df: DataFrame from test_stationarity_all.

    Returns:
        List of non-stationary feature names.
    """
    return result_df[
        result_df["stationarity_conclusion"] == "non_stationary"
    ]["feature"].tolist()


def get_stationary_features(result_df: pd.DataFrame) -> list[str]:
    """Get list of stationary features.

    Args:
        result_df: DataFrame from test_stationarity_all.

    Returns:
        List of stationary feature names.
    """
    return result_df[
        result_df["stationarity_conclusion"] == "stationary"
    ]["feature"].tolist()


def run_stationarity_analysis(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    save_results: bool = True,
    clear_cache_on_success: bool = True,
) -> pd.DataFrame:
    """Run complete stationarity analysis pipeline.

    Processes features sequentially to avoid memory issues. Results are
    cached incrementally and can be resumed if interrupted.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to analyze.
        save_results: Whether to save results to JSON.
        clear_cache_on_success: Whether to clear cache after completion.

    Returns:
        DataFrame with stationarity test results.
    """
    ensure_directories()

    logger.info("=" * 60)
    logger.info("STATIONARITY ANALYSIS")
    logger.info("=" * 60)

    # Run tests (sequential with incremental saves)
    result_df = test_stationarity_all(df, feature_columns)

    # Generate summary
    summary = summarize_stationarity(result_df)

    logger.info("=" * 60)
    logger.info("STATIONARITY SUMMARY")
    logger.info("=" * 60)
    logger.info("Total features: %d", summary["total_features"])
    logger.info("Stationary: %d (%.1f%%)", summary["stationary_count"], summary["stationary_pct"])
    logger.info("Non-stationary: %d (%.1f%%)", summary["non_stationary_count"], summary["non_stationary_pct"])
    logger.info("Trend-stationary: %d", summary["trend_stationary_count"])
    logger.info("Uncertain: %d", summary["uncertain_count"])

    if summary["non_stationary_features"]:
        logger.warning(
            "Non-stationary features found: %s",
            summary["non_stationary_features"][:10],  # First 10
        )

    # Save results
    if save_results:
        summary_counts = result_df["stationarity_conclusion"].value_counts().to_dict()

        json_data = {
            "timestamp": datetime.now().isoformat(),
            "analysis": "stationarity",
            "summary": summary_counts,
            "stationary_features": result_df[
                result_df["stationarity_conclusion"] == "stationary"
            ]["feature"].tolist(),
            "non_stationary_features": result_df[
                result_df["stationarity_conclusion"] == "non_stationary"
            ]["feature"].tolist(),
            "all_results": result_df.to_dict(orient="records"),
        }

        save_json(json_data, STATIONARITY_RESULTS_JSON)

        # Generate plot
        plot_stationarity_summary(result_df)

    # Clear cache after successful completion
    if clear_cache_on_success:
        _clear_cache()

    # Final memory cleanup
    gc.collect()

    return result_df


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

    result = run_stationarity_analysis(cast(pd.DataFrame, df))

    logger.info("Results shape: %s", result.shape)
