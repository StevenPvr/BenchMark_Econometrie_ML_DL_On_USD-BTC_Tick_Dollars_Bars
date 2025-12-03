"""Data transformation and cleaning functions for feature pipeline.

This module provides utility functions for the feature engineering pipeline:
- NaN handling (interpolation, dropping)
- Train/test splitting
- Target alignment
- Timestamp processing
- Column scaling selection
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from src.constants import TRAIN_RATIO_DEFAULT
from src.utils import get_logger

logger = get_logger(__name__)

# Train/test split ratio (from constants)
TRAIN_RATIO = TRAIN_RATIO_DEFAULT

__all__ = [
    "interpolate_sporadic_nan",
    "drop_empty_columns",
    "drop_initial_nan_rows",
    "shift_target_to_future_return",
    "split_train_test",
    "get_columns_to_scale",
    "compute_timestamp_features",
    "drop_timestamp_columns",
    "TRAIN_RATIO",
]


def interpolate_sporadic_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate sporadic NaN values using forward-fill (respects causality).

    Some features produce sporadic NaN values due to edge cases:
    - body_ratio: NaN when high == low (zero range bar)
    - sampen: NaN when no similar patterns found

    We use forward-fill (ffill) to propagate the last valid value,
    which respects temporal causality (no future information leakage).

    Args:
        df: DataFrame with potential sporadic NaN values.

    Returns:
        DataFrame with NaN values interpolated.
    """
    df_result = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    nan_before = df_result[numeric_cols].isna().sum().sum()

    if nan_before == 0:
        logger.info("No NaN values to interpolate")
        return df_result

    df_result[numeric_cols] = df_result[numeric_cols].ffill()

    nan_after_ffill = df_result[numeric_cols].isna().sum().sum()
    if nan_after_ffill > 0:
        df_result[numeric_cols] = df_result[numeric_cols].bfill()

    nan_after = df_result[numeric_cols].isna().sum().sum()

    logger.info(
        "Interpolated %d sporadic NaN values (forward-fill). Remaining: %d",
        nan_before - nan_after,
        nan_after,
    )

    cols_with_nan = df[numeric_cols].columns[df[numeric_cols].isna().any()].tolist()
    if cols_with_nan:
        logger.info("Columns with interpolated NaN: %s", cols_with_nan)

    return df_result


def drop_empty_columns(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop columns that are entirely empty or have too many NaN values.

    Args:
        df: DataFrame to clean.
        threshold: Maximum ratio of NaN allowed (default 0.95 = drop if >95% NaN).

    Returns:
        DataFrame with problematic columns removed.
    """
    initial_cols = len(df.columns)
    n_rows = len(df)

    nan_ratio = df.isna().sum() / n_rows

    empty_cols = nan_ratio[nan_ratio == 1.0].index.tolist()
    high_nan_cols = nan_ratio[(nan_ratio > threshold) & (nan_ratio < 1.0)].index.tolist()
    cols_to_drop = empty_cols + high_nan_cols

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        if empty_cols:
            logger.warning(
                "Dropped %d entirely empty columns: %s",
                len(empty_cols),
                empty_cols[:10] if len(empty_cols) > 10 else empty_cols
            )
        if high_nan_cols:
            logger.warning(
                "Dropped %d columns with >%.0f%% NaN: %s",
                len(high_nan_cols),
                threshold * 100,
                high_nan_cols[:10] if len(high_nan_cols) > 10 else high_nan_cols
            )
        logger.info(
            "Columns after cleanup: %d (dropped %d)",
            len(df.columns),
            initial_cols - len(df.columns)
        )

    return df


def drop_initial_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop initial rows with NaN values due to lags and rolling windows.

    Args:
        df: DataFrame with potential NaN rows at the start.

    Returns:
        DataFrame with initial NaN rows removed.
    """
    df = drop_empty_columns(df)
    initial_rows = len(df)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        logger.warning("No numeric columns found after cleanup")
        return df

    valid_rows = df[numeric_cols].dropna()

    if valid_rows.empty:
        nan_threshold = 0.05
        nan_ratio = df[numeric_cols].isna().sum(axis=1) / len(numeric_cols)
        valid_mask = nan_ratio <= nan_threshold

        if valid_mask.any():
            first_valid_idx = df[valid_mask].index[0]
            df_clean = df.loc[first_valid_idx:].copy()
            logger.warning(
                "No rows without NaN found. Using threshold approach "
                "(keeping rows with <=%.0f%% NaN). Columns with most NaN: %s",
                nan_threshold * 100,
                df[numeric_cols].isna().sum().nlargest(5).to_dict()
            )
        else:
            nan_counts = df[numeric_cols].isna().sum(axis=1)
            first_valid_idx = nan_counts.idxmin()
            df_clean = df.loc[first_valid_idx:].copy()
            logger.warning(
                "Using minimum NaN approach. Min NaN count: %d/%d columns",
                nan_counts.min(),
                len(numeric_cols)
            )
    else:
        first_valid_idx = valid_rows.index[0]
        df_clean = df.loc[first_valid_idx:].copy()

    dropped_rows = initial_rows - len(df_clean)

    logger.info(
        "Dropped %d initial NaN rows (lags/rolling windows). Remaining: %d rows",
        dropped_rows,
        len(df_clean),
    )

    return df_clean


def shift_target_to_future_return(
    df: pd.DataFrame,
    target_col: str = "log_return",
) -> pd.DataFrame:
    """Align target to predict the next-bar return and prevent leakage.

    Shifts the target column by -1 so that features at time t predict
    return at time t+1. Drops the final row created by the shift.

    Args:
        df: DataFrame with target column.
        target_col: Name of target column.

    Returns:
        DataFrame with shifted target.

    Raises:
        ValueError: If target column not found.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    df_shifted = df.copy()
    original_len = len(df_shifted)

    df_shifted[target_col] = df_shifted[target_col].shift(-1)
    df_shifted = df_shifted.dropna(subset=[target_col])

    dropped = original_len - len(df_shifted)
    logger.info(
        "Shifted %s by -1 to predict next return (dropped %d row%s)",
        target_col,
        dropped,
        "" if dropped == 1 else "s",
    )

    return df_shifted


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split data into train and test sets.

    Args:
        df: Full DataFrame.
        train_ratio: Proportion for training.

    Returns:
        Tuple of (df_train, df_test, split_index).
    """
    n = len(df)
    split_idx = int(n * train_ratio)

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    logger.info(
        "Train/test split: %d train (%.1f%%), %d test (%.1f%%)",
        len(df_train),
        100 * train_ratio,
        len(df_test),
        100 * (1 - train_ratio),
    )

    return df_train, df_test, split_idx


def get_columns_to_scale(
    df: pd.DataFrame,
    exclude_target: bool = True,
    include_log_return_lags: bool = False,
) -> list[str]:
    """Get columns to scale.

    Args:
        df: DataFrame with features.
        exclude_target: Exclude log_return (target).
        include_log_return_lags: Include log_return_lag* columns.

    Returns:
        List of column names to scale.
    """
    exclude_exact_suffixes = ["_sin", "_cos"]
    exclude_patterns = [
        "timestamp",
        "datetime",
        "date",
        "crash_",
        "vol_regime_",
        "cross_ma_",
        "frac_diff",
    ]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cols_to_scale = []

    for col in numeric_cols:
        col_lower = col.lower()
        exclude = False

        for suffix in exclude_exact_suffixes:
            if col_lower.endswith(suffix):
                exclude = True
                break

        if not exclude:
            for pattern in exclude_patterns:
                if pattern in col_lower:
                    exclude = True
                    break

        if col == "log_return" and exclude_target:
            exclude = True
        elif "log_return_lag" in col_lower:
            if not include_log_return_lags:
                exclude = True
            else:
                exclude = False

        if not exclude:
            cols_to_scale.append(col)

    return cols_to_scale


def compute_timestamp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features from timestamps before dropping them.

    Features computed:
    - bar_duration_ts: Duration of bar (timestamp_close - timestamp_open)
    - time_gap_bars: Time gap from previous bar close to current bar open

    Args:
        df: DataFrame with timestamp columns.

    Returns:
        DataFrame with additional timestamp-derived features.
    """
    result = df.copy()

    if "timestamp_open" in df.columns and "timestamp_close" in df.columns:
        result["bar_duration_ts"] = (
            df["timestamp_close"] - df["timestamp_open"]
        ) / 1000.0

        result["time_gap_bars"] = (
            df["timestamp_open"] - df["timestamp_close"].shift(1)
        ) / 1000.0

        result["time_gap_bars"] = result["time_gap_bars"].fillna(0)

        logger.info(
            "Computed timestamp features: bar_duration_ts (mean=%.2fs), "
            "time_gap_bars (mean=%.2fs)",
            result["bar_duration_ts"].mean(),
            result["time_gap_bars"].mean(),
        )

    return result


def drop_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop timestamp columns that should not be in final dataset.

    Args:
        df: DataFrame with timestamp columns.

    Returns:
        DataFrame without timestamp columns.
    """
    cols_to_drop = [
        "timestamp_open",
        "timestamp_close",
        "datetime_open",
    ]

    existing_cols = [c for c in cols_to_drop if c in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)
        logger.info("Dropped timestamp columns: %s", existing_cols)

    return df
