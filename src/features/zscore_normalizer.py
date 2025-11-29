"""Rolling z-score normalization for all features.

This module provides rolling z-score normalization, which is essential
for linear models (Ridge, Lasso, OLS) that are sensitive to feature scales.

Z-score transformation:
    z_t = (x_t - μ_t) / σ_t

Where μ_t and σ_t are computed on a rolling window ending at t (no lookahead).

Benefits:
    - Standardizes all features to same scale
    - Removes trend/level effects
    - Makes coefficients comparable
    - Improves convergence for gradient-based optimization

Reference:
    Hastie, T., Tibshirani, R., & Friedman, J. (2009).
    The Elements of Statistical Learning. Springer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.config_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_rolling_zscore",
    "compute_all_features_zscore",
    "save_zscore_features",
]


# =============================================================================
# NUMBA-OPTIMIZED ROLLING Z-SCORE
# =============================================================================


@njit(cache=True)
def _rolling_zscore(
    values: NDArray[np.float64],
    window: int,
    min_periods: int,
) -> NDArray[np.float64]:
    """Compute rolling z-score (numba optimized).

    z_t = (x_t - mean(x_{t-window+1:t})) / std(x_{t-window+1:t})

    Args:
        values: Input array.
        window: Rolling window size.
        min_periods: Minimum observations required.

    Returns:
        Array of z-scores.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < min_periods:
        return result

    for i in range(min_periods - 1, n):
        # Determine window bounds
        start = max(0, i - window + 1)

        # Compute mean
        mean = 0.0
        count = 0
        for j in range(start, i + 1):
            if not np.isnan(values[j]):
                mean += values[j]
                count += 1

        if count < min_periods:
            continue

        mean /= count

        # Compute std
        var_sum = 0.0
        for j in range(start, i + 1):
            if not np.isnan(values[j]):
                var_sum += (values[j] - mean) ** 2

        if count > 1:
            std = np.sqrt(var_sum / (count - 1))
        else:
            std = 0.0

        # Z-score
        if std > 1e-10:
            result[i] = (values[i] - mean) / std
        else:
            # Zero std means constant value, z-score is 0
            result[i] = 0.0

    return result


@njit(cache=True)
def _rolling_zscore_expanding(
    values: NDArray[np.float64],
    min_periods: int,
) -> NDArray[np.float64]:
    """Compute expanding z-score (numba optimized).

    Uses all data from start up to current point (no fixed window).

    Args:
        values: Input array.
        min_periods: Minimum observations required.

    Returns:
        Array of z-scores.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < min_periods:
        return result

    # Running statistics
    running_sum = 0.0
    running_sum_sq = 0.0
    count = 0

    for i in range(n):
        if not np.isnan(values[i]):
            running_sum += values[i]
            running_sum_sq += values[i] ** 2
            count += 1

        if count < min_periods:
            continue

        mean = running_sum / count

        if count > 1:
            # Variance = E[X²] - E[X]² (with Bessel correction)
            var = (running_sum_sq - running_sum * running_sum / count) / (count - 1)
            std = np.sqrt(max(var, 0.0))
        else:
            std = 0.0

        if std > 1e-10:
            result[i] = (values[i] - mean) / std
        else:
            result[i] = 0.0

    return result


# =============================================================================
# PUBLIC API
# =============================================================================


def compute_rolling_zscore(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    window: int = 100,
    min_periods: int | None = None,
    expanding: bool = False,
    suffix: str = "_zscore",
) -> pd.DataFrame:
    """Compute rolling z-score for specified columns.

    Args:
        df: Input DataFrame.
        columns: Columns to normalize (default: all numeric columns).
        window: Rolling window size (ignored if expanding=True).
        min_periods: Minimum observations (default: window // 2).
        expanding: If True, use expanding window instead of rolling.
        suffix: Suffix for z-scored column names.

    Returns:
        DataFrame with z-scored columns.

    Example:
        >>> df_zscore = compute_rolling_zscore(df_bars, window=100)
    """
    if min_periods is None:
        min_periods = max(window // 2, 10)

    # Select columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = pd.DataFrame(index=df.index)

    for col in columns:
        values = df[col].values.astype(np.float64)

        if expanding:
            zscore = _rolling_zscore_expanding(values, min_periods)
        else:
            zscore = _rolling_zscore(values, window, min_periods)

        result[f"{col}{suffix}"] = zscore

    logger.info(
        "Computed rolling z-score for %d columns (window=%s, expanding=%s)",
        len(columns),
        "expanding" if expanding else window,
        expanding,
    )

    return result


def compute_all_features_zscore(
    df_features: pd.DataFrame,
    window: int = 100,
    min_periods: int | None = None,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute rolling z-score for all feature columns.

    This function:
    1. Identifies all numeric feature columns
    2. Excludes specified columns (e.g., timestamps, identifiers)
    3. Applies rolling z-score normalization
    4. Returns a DataFrame with both original and z-scored features

    Args:
        df_features: DataFrame with all computed features.
        window: Rolling window size for z-score calculation.
        min_periods: Minimum observations required.
        exclude_columns: Columns to exclude from normalization.

    Returns:
        DataFrame with original features + z-scored features.

    Example:
        >>> df_all = compute_all_features_zscore(df_features, window=100)
    """
    if exclude_columns is None:
        exclude_columns = []

    # Default columns to exclude (timestamps, identifiers, etc.)
    default_exclude = [
        "timestamp",
        "timestamp_open",
        "timestamp_close",
        "date",
        "datetime",
        "bar_id",
        "index",
    ]

    # Also exclude columns that are already z-scores or dummies
    for col in df_features.columns:
        col_lower = col.lower()
        if (
            "_zscore" in col_lower
            or col_lower.startswith("crash_")
            or col_lower.startswith("vol_regime_")
            or col_lower.endswith("_sin")
            or col_lower.endswith("_cos")
        ):
            default_exclude.append(col)

    all_exclude = set(exclude_columns + default_exclude)

    # Select numeric columns to normalize
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_zscore = [c for c in numeric_cols if c not in all_exclude]

    logger.info(
        "Z-scoring %d columns (excluding %d)",
        len(cols_to_zscore),
        len(all_exclude),
    )

    # Compute z-scores
    df_zscore = compute_rolling_zscore(
        df_features,
        columns=cols_to_zscore,
        window=window,
        min_periods=min_periods,
        suffix="_zscore",
    )

    # Combine original + z-scored
    result = pd.concat([df_features, df_zscore], axis=1)

    # Log statistics
    zscore_cols = [c for c in result.columns if c.endswith("_zscore")]
    if zscore_cols:
        # Check for any extreme values
        extreme_count = 0
        for col in zscore_cols:
            valid = result[col].dropna()
            if len(valid) > 0:
                extreme_count += (np.abs(valid) > 3).sum()

        logger.info(
            "Z-score features: %d columns, %.2f%% extreme values (|z|>3)",
            len(zscore_cols),
            100 * extreme_count / (len(result) * len(zscore_cols))
            if len(zscore_cols) > 0
            else 0,
        )

    return result


def save_zscore_features(
    df_features: pd.DataFrame,
    output_path: str | Path,
    window: int = 100,
    min_periods: int | None = None,
    exclude_columns: list[str] | None = None,
    save_original: bool = False,
) -> pd.DataFrame:
    """Compute z-score features and save to CSV and Parquet.

    This function computes z-scored features and saves them to separate files,
    optimized for linear models (Ridge, Lasso, OLS).

    Args:
        df_features: DataFrame with all computed features.
        output_path: Base path for output files (without extension).
        window: Rolling window size for z-score calculation.
        min_periods: Minimum observations required.
        exclude_columns: Columns to exclude from normalization.
        save_original: If True, include original features in output.

    Returns:
        DataFrame with z-scored features.

    Example:
        >>> df_zscore = save_zscore_features(
        ...     df_features,
        ...     "data/features/features_zscore",
        ...     window=100,
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute z-scores
    if save_original:
        df_result = compute_all_features_zscore(
            df_features,
            window=window,
            min_periods=min_periods,
            exclude_columns=exclude_columns,
        )
    else:
        # Only z-score columns
        if exclude_columns is None:
            exclude_columns = []

        default_exclude = [
            "timestamp",
            "timestamp_open",
            "timestamp_close",
            "date",
            "datetime",
            "bar_id",
            "index",
        ]

        for col in df_features.columns:
            col_lower = col.lower()
            if (
                "_zscore" in col_lower
                or col_lower.startswith("crash_")
                or col_lower.startswith("vol_regime_")
                or col_lower.endswith("_sin")
                or col_lower.endswith("_cos")
            ):
                default_exclude.append(col)

        all_exclude = set(exclude_columns + default_exclude)
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_zscore = [c for c in numeric_cols if c not in all_exclude]

        df_zscore = compute_rolling_zscore(
            df_features,
            columns=cols_to_zscore,
            window=window,
            min_periods=min_periods,
            suffix="_zscore",
        )

        # Keep timestamp columns for reference
        timestamp_cols = [c for c in df_features.columns if "timestamp" in c.lower()]
        if timestamp_cols:
            df_result = pd.concat(
                [df_features[timestamp_cols], df_zscore], axis=1
            )
        else:
            df_result = df_zscore

    # Save to CSV - commented out, using Parquet only
    # csv_path = output_path.with_suffix(".csv")
    # df_result.to_csv(csv_path, index=False)
    # logger.info("Saved z-score features to CSV: %s", csv_path)

    # Save to Parquet
    parquet_path = output_path.with_suffix(".parquet")
    df_result.to_parquet(parquet_path, index=False)
    logger.info("Saved z-score features to Parquet: %s", parquet_path)

    # Log summary
    logger.info(
        "Z-score features saved: %d rows, %d columns, window=%d",
        len(df_result),
        len(df_result.columns),
        window,
    )

    return df_result
