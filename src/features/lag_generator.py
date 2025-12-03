"""Intelligent lag generation for features.

This module generates lagged features with differentiated lag structures
based on variable type:

1. Fast variables (returns, order flow, volume, activity):
   - Lags: 1, 5, 10, 15, 25, 50 bars
   - Captures short-term dynamics and immediate dependencies

2. Slow variables (volatility, range, toxicity):
   - Lags: 1, 5, 10, 15, 25, 50 bars
   - Already persistent, fewer lags needed

3. Already aggregated/rolling features:
   - Lags: 1, 5, 10, 15, 25, 50 bars
   - Already contain compressed memory

4. Time/calendar features:
   - No lags (deterministic at each bar)

5. Fractional diff:
   - No lags (memory already integrated)

Reference:
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 5: Feature Importance.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.utils import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "classify_feature_type",
    "get_lags_for_feature",
    "compute_lagged_features",
    "generate_all_lags",
]


# =============================================================================
# FEATURE CLASSIFICATION
# =============================================================================

# Pattern definitions for feature classification
FAST_PATTERNS = [
    r"^log_return$",
    r"^return$",
    r"^r_t$",
    r"volume_imbalance$",
    r"^oi$",
    r"order_imbalance$",
    r"^volume$",
    r"^volume_btc$",
    r"^volume_usd$",
    r"^tick_count$",
    r"tick_intensity",
    r"^n_ticks$",
    r"activity",
    r"^buy_volume$",
    r"^sell_volume$",
    r"signed_volume",
    r"tick_mean_return",
]

SLOW_PATTERNS = [
    r"parkinson",
    r"garman_klass",
    r"rogers_satchell",
    r"yang_zhang",
    r"range_ratio$",
    r"body_ratio$",
    r"intrabar_var",
    r"intrabar_vol",
    r"tick_range$",
    r"realized_var",
    r"tick_realized",
    r"vpin",
    r"kyle_lambda",
    r"entropy",
    r"range_efficiency",
    r"vol_of_vol",
]

AGGREGATED_PATTERNS = [
    r"cum_return_\d+",
    r"realized_vol_\d+",
    r"local_sharpe_\d+",
    r"ma_\d+",
    r"price_zscore_\d+",
    r"cross_ma_",
    r"max_return_\d+",
    r"min_return_\d+",
    r"volatility_\d+",
    r"_roll_",
    r"_sum_\d+",
    r"_mean_\d+",
    r"bars_since_",
]

NO_LAG_PATTERNS = [
    r"_sin$",
    r"_cos$",
    r"hour_",
    r"day_",
    r"month_",
    r"frac_diff",
    r"^log_price$",
    r"timestamp",
    r"datetime",
    r"^date$",
    r"crash_\d+",
    r"vol_regime_",
    r"vol_zscore$",
    r"drawdown$",
    r"drawup$",
    r"rolling_max$",
    r"rolling_min$",
    r"return_streak$",
]


def classify_feature_type(col_name: str) -> str:
    """Classify a feature into its lag category.

    Categories:
        - 'fast': Returns, order flow, volume, activity
        - 'slow': Volatility, range, toxicity indicators
        - 'aggregated': Already rolling/aggregated features
        - 'no_lag': Time, calendar, frac diff, regimes

    Args:
        col_name: Column name to classify.

    Returns:
        Feature type: 'fast', 'slow', 'aggregated', or 'no_lag'.

    Example:
        >>> classify_feature_type("log_return")
        'fast'
        >>> classify_feature_type("parkinson_vol_10")
        'slow'
        >>> classify_feature_type("cum_return_5")
        'aggregated'
        >>> classify_feature_type("hour_sin")
        'no_lag'
    """
    col_lower = col_name.lower()

    # Check no-lag patterns first (highest priority)
    for pattern in NO_LAG_PATTERNS:
        if re.search(pattern, col_lower):
            return "no_lag"

    # Check aggregated patterns
    for pattern in AGGREGATED_PATTERNS:
        if re.search(pattern, col_lower):
            return "aggregated"

    # Check slow patterns
    for pattern in SLOW_PATTERNS:
        if re.search(pattern, col_lower):
            return "slow"

    # Check fast patterns
    for pattern in FAST_PATTERNS:
        if re.search(pattern, col_lower):
            return "fast"

    # Default: treat as slow (conservative)
    return "slow"


def get_lags_for_feature(feature_type: str) -> list[int]:
    """Get appropriate lags for a feature type.

    Args:
        feature_type: One of 'fast', 'slow', 'aggregated', 'no_lag'.

    Returns:
        List of lag values.

    Example:
        >>> get_lags_for_feature('fast')
        [1, 5, 10, 15, 25, 50]
        >>> get_lags_for_feature('slow')
        [1, 5, 10, 15, 25, 50]
        >>> get_lags_for_feature('aggregated')
        [1, 5, 10, 15, 25, 50]
        >>> get_lags_for_feature('no_lag')
        []
    """
    lag_schemes = {
        "fast": [1, 5, 10, 15, 25, 50],
        "slow": [1, 5, 10, 15, 25, 50],
        "aggregated": [1, 5, 10, 15, 25, 50],
        "no_lag": [],
    }
    return lag_schemes.get(feature_type, [1, 5, 10, 15, 25, 50])


# =============================================================================
# LAG COMPUTATION
# =============================================================================


@njit(cache=True)
def _shift_array(
    values: NDArray[np.float64],
    lag: int,
) -> NDArray[np.float64]:
    """Shift array by lag periods (numba optimized).

    Args:
        values: Input array.
        lag: Number of periods to shift (positive = backward).

    Returns:
        Shifted array with NaN for unavailable values.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if lag >= n or lag < 0:
        return result

    for i in range(lag, n):
        result[i] = values[i - lag]

    return result


def compute_lagged_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
    suffix_format: str = "_lag{lag}",
) -> pd.DataFrame:
    """Compute lagged features for specified columns.

    Args:
        df: Input DataFrame.
        columns: Columns to lag.
        lags: List of lag values.
        suffix_format: Format string for lag suffix.

    Returns:
        DataFrame with lagged features.

    Example:
        >>> df_lags = compute_lagged_features(df, ["log_return"], [1, 2, 3])
    """
    # Build all lagged columns in a dictionary to avoid DataFrame fragmentation
    lagged_data: dict[str, np.ndarray] = {}

    for col in columns:
        values = df[col].values.astype(np.float64)

        for lag in lags:
            lagged = _shift_array(values, lag)
            col_name = f"{col}{suffix_format.format(lag=lag)}"
            lagged_data[col_name] = lagged

    # Create DataFrame in one operation to avoid fragmentation
    result = pd.DataFrame(lagged_data, index=df.index)

    return result


def generate_all_lags(
    df_features: pd.DataFrame,
    fast_lags: list[int] | None = None,
    slow_lags: list[int] | None = None,
    aggregated_lags: list[int] | None = None,
    exclude_columns: list[str] | None = None,
    include_original: bool = True,
) -> pd.DataFrame:
    """Generate lagged features with intelligent lag structure.

    Automatically classifies features and applies appropriate lags:
    - Fast variables: lags 1, 5, 10, 15, 25, 50
    - Slow variables: lags 1, 5, 10, 15, 25, 50
    - Aggregated: lags 1, 5, 10, 15, 25, 50
    - Time/calendar/frac_diff: no lags

    Args:
        df_features: DataFrame with all features.
        fast_lags: Override lags for fast variables.
        slow_lags: Override lags for slow variables.
        aggregated_lags: Override lags for aggregated variables.
        exclude_columns: Columns to exclude from lagging.
        include_original: If True, include original (non-lagged) features.

    Returns:
        DataFrame with original and lagged features.

    Example:
        >>> df_with_lags = generate_all_lags(df_features)
    """
    if fast_lags is None:
        fast_lags = [1, 5, 10, 15, 25, 50]
    if slow_lags is None:
        slow_lags = [1, 5, 10, 15, 25, 50]
    if aggregated_lags is None:
        aggregated_lags = [1, 5, 10, 15, 25, 50]
    if exclude_columns is None:
        exclude_columns = []

    # Classify all columns
    classification: dict[str, str] = {}
    for col in df_features.columns:
        if col in exclude_columns:
            classification[col] = "excluded"
        else:
            classification[col] = classify_feature_type(col)

    # Count by type
    type_counts: dict[str, int] = {}
    for col, ftype in classification.items():
        type_counts[ftype] = type_counts.get(ftype, 0) + 1

    logger.info(
        "Feature classification: fast=%d, slow=%d, aggregated=%d, no_lag=%d, excluded=%d",
        type_counts.get("fast", 0),
        type_counts.get("slow", 0),
        type_counts.get("aggregated", 0),
        type_counts.get("no_lag", 0),
        type_counts.get("excluded", 0),
    )

    # Group columns by type
    fast_cols = [c for c, t in classification.items() if t == "fast"]
    slow_cols = [c for c, t in classification.items() if t == "slow"]
    agg_cols = [c for c, t in classification.items() if t == "aggregated"]
    no_lag_cols = [c for c, t in classification.items() if t == "no_lag"]

    # Generate lags
    lag_dfs = []

    if include_original:
        lag_dfs.append(df_features)

    # Fast variables
    if fast_cols and fast_lags:
        df_fast_lags = compute_lagged_features(df_features, fast_cols, fast_lags)
        lag_dfs.append(df_fast_lags)
        logger.info(
            "Generated %d lagged features for %d fast variables (lags: %s)",
            len(df_fast_lags.columns),
            len(fast_cols),
            fast_lags,
        )

    # Slow variables
    if slow_cols and slow_lags:
        df_slow_lags = compute_lagged_features(df_features, slow_cols, slow_lags)
        lag_dfs.append(df_slow_lags)
        logger.info(
            "Generated %d lagged features for %d slow variables (lags: %s)",
            len(df_slow_lags.columns),
            len(slow_cols),
            slow_lags,
        )

    # Aggregated variables
    if agg_cols and aggregated_lags:
        df_agg_lags = compute_lagged_features(df_features, agg_cols, aggregated_lags)
        lag_dfs.append(df_agg_lags)
        logger.info(
            "Generated %d lagged features for %d aggregated variables (lags: %s)",
            len(df_agg_lags.columns),
            len(agg_cols),
            aggregated_lags,
        )

    # No lag variables - just include as-is (already in original if include_original)
    if no_lag_cols:
        logger.info(
            "Skipped lagging for %d no-lag variables (time, calendar, frac_diff)",
            len(no_lag_cols),
        )

    # Combine all
    if lag_dfs:
        result = pd.concat(lag_dfs, axis=1)
        # Remove duplicate columns (if include_original=True)
        result = result.loc[:, ~result.columns.duplicated()]
    else:
        result = df_features.copy()

    logger.info(
        "Total features after lagging: %d (original: %d)",
        len(result.columns),
        len(df_features.columns),
    )

    return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def summarize_lag_structure(df_features: pd.DataFrame) -> pd.DataFrame:
    """Summarize the lag structure that would be applied to features.

    Args:
        df_features: DataFrame with features.

    Returns:
        Summary DataFrame with columns and their lag assignments.

    Example:
        >>> summary = summarize_lag_structure(df_features)
        >>> print(summary)
    """
    rows = []
    for col in df_features.columns:
        ftype = classify_feature_type(col)
        lags = get_lags_for_feature(ftype)
        rows.append({
            "column": col,
            "type": ftype,
            "lags": str(lags) if lags else "none",
            "n_lagged_features": len(lags),
        })

    summary = pd.DataFrame(rows)

    # Log summary
    total_features = len(summary)
    total_lagged = summary["n_lagged_features"].sum()
    logger.info(
        "Lag structure summary: %d original features -> %d lagged features",
        total_features,
        total_lagged,
    )

    return summary
