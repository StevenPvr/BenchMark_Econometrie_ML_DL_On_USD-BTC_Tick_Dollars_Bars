"""Main pipeline for feature generation.

This module orchestrates the complete feature engineering pipeline:

1. Load input: log_returns_split.parquet
2. Compute all features (microstructure, volatility, momentum, etc.)
3. Apply intelligent lag structure
4. Split into train/test
5. Save raw (unscaled) outputs:
   - dataset_features.parquet: Raw features for tree-based ML (XGBoost, etc.)
   - dataset_features_linear.parquet: Copy for linear models (scaled in clear_features)
   - dataset_features_lstm.parquet: Copy for LSTM (scaled in clear_features)

NOTE: Scaler fitting moved to clear_features module (after PCA transformation).

Usage:
    python -m src.features.main
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import cast
import gc
import numpy as np
import pyarrow as pa # type: ignore[import-untyped]
import pyarrow.parquet as pq # type: ignore[import-untyped]


# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

NUMBA_CACHE_DIR = os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(NUMBA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from src.config_logging import get_logger, setup_logging
from src.features.entropy import (
    compute_approximate_entropy,
    compute_sample_entropy,
    compute_shannon_entropy,
)
from src.features.technical_indicators import compute_all_technical_indicators
from src.features.fractional_diff import compute_frac_diff_features
from src.features.kyle_lambda import compute_kyle_lambda
from src.features.lag_generator import generate_all_lags
from src.features.momentum import compute_cumulative_returns, compute_recent_extremes
from src.features.range_volatility import (
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
    compute_range_ratios,
    compute_rogers_satchell_volatility,
    compute_yang_zhang_volatility,
)
from src.features.realized_volatility import (
    compute_return_volatility_ratio,
    compute_realized_volatility,
    compute_realized_skewness,
    compute_realized_kurtosis,
)
from src.features.jump_detection import compute_all_jump_features
from src.features.scalers import (
    MinMaxScalerCustom,
    StandardScalerCustom,
)
from src.features.temporal_acceleration import (
    compute_temporal_acceleration,
    compute_temporal_acceleration_smoothed,
    compute_temporal_jerk,
)
from src.features.temporal_calendar import compute_all_temporal_features
from src.features.trend import (
    compute_cross_ma,
    compute_moving_averages,
    compute_price_zscore,
    compute_return_streak,
)
from src.features.vpin import compute_vpin
from src.path import (
    DOLLAR_BARS_PARQUET,
    DATASET_FEATURES_LINEAR_TEST_PARQUET,
    DATASET_FEATURES_LINEAR_TRAIN_PARQUET,
    DATASET_FEATURES_LINEAR_PARQUET,
    DATASET_FEATURES_LSTM_TEST_PARQUET,
    DATASET_FEATURES_LSTM_TRAIN_PARQUET,
    DATASET_FEATURES_LSTM_PARQUET,
    DATASET_FEATURES_TEST_PARQUET,
    DATASET_FEATURES_TRAIN_PARQUET,
    DATASET_FEATURES_PARQUET,
    FEATURES_DIR,
    MINMAX_SCALER_FILE,
    SCALERS_DIR,
    ZSCORE_SCALER_FILE,
)

logger = get_logger(__name__)

# Train/test split ratio (80% train, 20% test)
TRAIN_RATIO = 0.8


def load_input_data() -> pd.DataFrame:
    """Load input data from log_returns_split.parquet.

    Returns:
        DataFrame with dollar bars and log returns.
    """
    logger.info("Loading input data from %s", DOLLAR_BARS_PARQUET)

    if not DOLLAR_BARS_PARQUET.exists():
        raise FileNotFoundError(
            f"Input file not found: {DOLLAR_BARS_PARQUET}. "
            "Please run data_preparation first."
        )

    df = pd.read_parquet(DOLLAR_BARS_PARQUET)

    logger.info(
        "Loaded %d rows, %d columns. Columns: %s",
        len(df),
        len(df.columns),
        list(df.columns),
    )

    return df


def compute_all_features(df_bars: pd.DataFrame) -> pd.DataFrame:
    """Compute all features from dollar bars.

    Args:
        df_bars: DataFrame with dollar bars (OHLCV + log_return).

    Returns:
        DataFrame with all computed features.
    """
    logger.info("Computing all features...")

    feature_dfs = [df_bars.copy()]

    # =========================================================================
    # 1. MOMENTUM FEATURES
    # =========================================================================
    logger.info("Computing momentum features...")

    df_cum = compute_cumulative_returns(df_bars, return_col="log_return")
    feature_dfs.append(df_cum)

    df_extremes = compute_recent_extremes(df_bars, return_col="log_return")
    feature_dfs.append(df_extremes)

    # =========================================================================
    # 2. REALIZED VOLATILITY & HIGHER MOMENTS
    # =========================================================================
    logger.info("Computing realized volatility features...")

    df_vol = compute_realized_volatility(df_bars, return_col="log_return")
    feature_dfs.append(df_vol)

    df_rvr = compute_return_volatility_ratio(df_bars, return_col="log_return")
    feature_dfs.append(df_rvr)

    # Realized skewness (third moment - captures asymmetry)
    df_skew = compute_realized_skewness(df_bars, return_col="log_return")
    feature_dfs.append(df_skew)

    # Realized kurtosis (fourth moment - captures tail risk)
    df_kurt = compute_realized_kurtosis(df_bars, return_col="log_return")
    feature_dfs.append(df_kurt)

    # Jump detection features (bipower variation, jump component)
    df_jumps = compute_all_jump_features(df_bars, return_col="log_return")
    feature_dfs.append(df_jumps)

    # =========================================================================
    # 3. TREND FEATURES
    # =========================================================================
    logger.info("Computing trend features...")

    df_ma = compute_moving_averages(df_bars, price_col="close")
    feature_dfs.append(df_ma)

    df_zscore_price = compute_price_zscore(df_bars, price_col="close")
    feature_dfs.append(df_zscore_price)

    df_cross = compute_cross_ma(df_bars, price_col="close")
    feature_dfs.append(df_cross)

    df_streak = compute_return_streak(df_bars, return_col="log_return")
    feature_dfs.append(df_streak.to_frame())

    # =========================================================================
    # 4. RANGE-BASED VOLATILITY
    # =========================================================================
    logger.info("Computing range-based volatility features...")

    df_park = compute_parkinson_volatility(df_bars)
    feature_dfs.append(df_park)

    df_gk = compute_garman_klass_volatility(df_bars)
    feature_dfs.append(df_gk)

    df_rs = compute_rogers_satchell_volatility(df_bars)
    feature_dfs.append(df_rs)

    df_yz = compute_yang_zhang_volatility(df_bars)
    feature_dfs.append(df_yz)

    df_ratios = compute_range_ratios(df_bars)
    feature_dfs.append(df_ratios)

    # =========================================================================
    # 5. TEMPORAL ACCELERATION
    # =========================================================================
    if "duration_sec" in df_bars.columns:
        logger.info("Computing temporal acceleration features...")

        df_accel = compute_temporal_acceleration(df_bars, duration_col="duration_sec")
        feature_dfs.append(df_accel.to_frame())

        df_accel_smooth = compute_temporal_acceleration_smoothed(
            df_bars, duration_col="duration_sec"
        )
        feature_dfs.append(df_accel_smooth.to_frame())

        df_jerk = compute_temporal_jerk(df_bars, duration_col="duration_sec")
        feature_dfs.append(df_jerk.to_frame())

    # =========================================================================
    # 6. ORDER FLOW / MICROSTRUCTURE
    # =========================================================================
    logger.info("Computing order flow features...")

    if "buy_volume" in df_bars.columns and "sell_volume" in df_bars.columns:
        vi = (df_bars["buy_volume"] - df_bars["sell_volume"]) / (
            df_bars["buy_volume"] + df_bars["sell_volume"] + 1e-10
        )
        feature_dfs.append(vi.rename("volume_imbalance").to_frame())

        df_vpin = compute_vpin(
            df_bars,
            v_buy_col="buy_volume",
            v_sell_col="sell_volume",
        )
        feature_dfs.append(df_vpin.to_frame())

        df_kyle = compute_kyle_lambda(
            df_bars,
            price_col="close",
            v_buy_col="buy_volume",
            v_sell_col="sell_volume",
        )
        feature_dfs.append(df_kyle.to_frame())

    # =========================================================================
    # 7. ENTROPY FEATURES
    # =========================================================================
    logger.info("Computing entropy features...")

    df_shannon = compute_shannon_entropy(df_bars, return_col="log_return")
    feature_dfs.append(df_shannon.to_frame())

    df_apen = compute_approximate_entropy(df_bars, return_col="log_return")
    feature_dfs.append(df_apen.to_frame())

    df_sampen = compute_sample_entropy(df_bars, return_col="log_return", window=30)
    feature_dfs.append(df_sampen.to_frame())

    # =========================================================================
    # 8. TEMPORAL / CALENDAR / REGIME
    # =========================================================================
    logger.info("Computing temporal/calendar/regime features...")

    timestamp_col = None
    for col in ["datetime_close", "datetime_open", "timestamp_close", "timestamp", "datetime", "date"]:
        if col in df_bars.columns:
            timestamp_col = col
            break

    if timestamp_col:
        df_temporal = compute_all_temporal_features(
            df_bars,
            timestamp_col=timestamp_col,
            return_col="log_return",
            price_col="close",
        )
        feature_dfs.append(df_temporal)

    # =========================================================================
    # 9. FRACTIONAL DIFFERENTIATION
    # =========================================================================
    logger.info("Computing fractional differentiation features...")

    df_frac = compute_frac_diff_features(
        df_bars,
        price_col="close",
        d_values=[0.3, 0.5],
    )
    feature_dfs.append(df_frac)

    # =========================================================================
    # 10. TECHNICAL ANALYSIS INDICATORS (ta library)
    # =========================================================================
    logger.info("Computing technical analysis indicators...")

    try:
        df_ta = compute_all_technical_indicators(
            df_bars,
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            volume_col="volume",
            fillna=False,  # Preserve NaN for causality
        )
        feature_dfs.append(df_ta)
    except ImportError as e:
        logger.warning("Skipping TA indicators: %s", e)
    except ValueError as e:
        logger.warning("Skipping TA indicators (missing columns): %s", e)

    # =========================================================================
    # COMBINE ALL FEATURES
    # =========================================================================
    logger.info("Combining all features...")

    df_all = pd.concat(feature_dfs, axis=1)
    df_all = df_all.loc[:, ~df_all.columns.duplicated()]

    logger.info("Total features computed: %d columns", len(df_all.columns))

    return df_all


def apply_lags(df_features: pd.DataFrame) -> pd.DataFrame:
    """Apply intelligent lag structure to features.

    Args:
        df_features: DataFrame with all features.

    Returns:
        DataFrame with lagged features.
    """
    logger.info("Applying intelligent lag structure...")

    exclude_cols = [
        "timestamp",
        "timestamp_open",
        "timestamp_close",
        "datetime",
        "date"
    ]

    df_lagged = generate_all_lags(
        df_features,
        exclude_columns=exclude_cols,
        include_original=True,
    )

    return df_lagged


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

    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Count NaN before
    nan_before = df_result[numeric_cols].isna().sum().sum()

    if nan_before == 0:
        logger.info("No NaN values to interpolate")
        return df_result

    # Forward-fill NaN values (respects causality)
    df_result[numeric_cols] = df_result[numeric_cols].ffill()

    # If any NaN remain at the start (before first valid), backfill
    nan_after_ffill = df_result[numeric_cols].isna().sum().sum()
    if nan_after_ffill > 0:
        df_result[numeric_cols] = df_result[numeric_cols].bfill()

    nan_after = df_result[numeric_cols].isna().sum().sum()

    logger.info(
        "Interpolated %d sporadic NaN values (forward-fill). Remaining: %d",
        nan_before - nan_after,
        nan_after,
    )

    # Log which columns had NaN
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

    # Calculate NaN ratio for each column
    nan_ratio = df.isna().sum() / n_rows

    # Identify columns to drop (entirely empty or above threshold)
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

    The lag generation creates NaN in the first rows (up to max lag).
    Rolling windows also create NaN at the start.
    We drop these rows to have a clean dataset.

    Args:
        df: DataFrame with potential NaN rows at the start.

    Returns:
        DataFrame with initial NaN rows removed.
    """
    # First, drop empty or mostly-empty columns
    df = drop_empty_columns(df)

    initial_rows = len(df)

    # Find the first row where all columns have valid values
    # Use numeric columns only for NaN detection
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        logger.warning("No numeric columns found after cleanup")
        return df

    # Find first valid index (where no NaN in numeric columns)
    valid_rows = df[numeric_cols].dropna()

    if valid_rows.empty:
        # No rows without NaN - use threshold approach instead
        # Keep rows where at least 95% of numeric columns are valid
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
            # Last resort: find the row with minimum NaN count
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
        "Dropped %d initial NaN rows (lags/rolling windows). "
        "Remaining: %d rows",
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
    # Patterns to exclude from scaling
    # Note: Use suffixes for sin/cos to avoid matching "bars_since"
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

        # Check exact suffixes (e.g., _sin, _cos)
        for suffix in exclude_exact_suffixes:
            if col_lower.endswith(suffix):
                exclude = True
                break

        # Check substring patterns
        if not exclude:
            for pattern in exclude_patterns:
                if pattern in col_lower:
                    exclude = True
                    break

        # Handle log_return specially
        if col == "log_return" and exclude_target:
            exclude = True
        elif "log_return_lag" in col_lower:
            # Include lags of log_return if requested
            if not include_log_return_lags:
                exclude = True
            else:
                exclude = False  # Override other exclusions

        if not exclude:
            cols_to_scale.append(col)

    return cols_to_scale


def fit_and_save_scalers(
    df_train: pd.DataFrame,
) -> None:
    """Fit scalers on training data and save them for later use.

    NOTE: Scalers are fit here but NOT applied. The actual normalization
    happens in clear_features after PCA reduction and log transformation.

    CRITICAL: Scalers are fit on TRAIN only to prevent data leakage.

    Args:
        df_train: Training DataFrame.
    """
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Z-SCORE FOR LINEAR MODELS
    # Exclude log_return (target) and its lags
    # =========================================================================
    logger.info("Fitting z-score scaler for linear models (NOT applying)...")

    cols_zscore = get_columns_to_scale(
        df_train,
        exclude_target=True,
        include_log_return_lags=False,  # Don't z-score log_return lags
    )

    zscore_scaler = StandardScalerCustom()
    zscore_scaler.fit(df_train, cols_zscore)
    zscore_scaler.save(ZSCORE_SCALER_FILE)

    logger.info(
        "Z-score scaler fit on %d columns, saved to %s",
        len(cols_zscore),
        ZSCORE_SCALER_FILE,
    )

    # =========================================================================
    # MIN-MAX FOR LSTM
    # Include log_return lags (they are features), exclude log_return (target)
    # =========================================================================
    logger.info("Fitting min-max scaler for LSTM (NOT applying)...")

    cols_minmax = get_columns_to_scale(
        df_train,
        exclude_target=True,  # Don't scale target
        include_log_return_lags=True,  # DO scale log_return lags
    )

    minmax_scaler = MinMaxScalerCustom()
    minmax_scaler.fit(df_train, cols_minmax)
    minmax_scaler.save(MINMAX_SCALER_FILE)

    logger.info(
        "Min-max scaler fit on %d columns, saved to %s",
        len(cols_minmax),
        MINMAX_SCALER_FILE,
    )

    logger.info("Scalers saved. Normalization will be applied in clear_features module.")


def compute_timestamp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features from timestamps before dropping them.

    Features computed:
    - bar_duration_sec: Duration of bar (timestamp_close - timestamp_open)
    - time_since_prev_bar: Time gap from previous bar close to current bar open
    - bar_overlap: Whether bar overlaps with previous (negative time_since_prev)

    Args:
        df: DataFrame with timestamp columns.

    Returns:
        DataFrame with additional timestamp-derived features.
    """
    result = df.copy()

    # Bar duration from timestamps (more precise than duration_sec if available)
    if "timestamp_open" in df.columns and "timestamp_close" in df.columns:
        result["bar_duration_ts"] = (
            df["timestamp_close"] - df["timestamp_open"]
        ) / 1000.0  # Convert ms to seconds

        # Time gap between bars
        result["time_gap_bars"] = (
            df["timestamp_open"] - df["timestamp_close"].shift(1)
        ) / 1000.0

        # Fill first NaN
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

    These columns are not useful for ML models and can cause issues:
    - timestamp_open, timestamp_close: Raw Unix timestamps
    - datetime_open: Datetime object

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


def save_outputs(
    df_features: pd.DataFrame,
) -> None:
    """Save feature dataset to files.

    NOTE: Only saves the raw (unscaled) dataset. Linear and LSTM datasets
    with normalization will be created by clear_features module after
    PCA reduction and log transformation.

    Args:
        df_features: Raw features for ML models.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Drop timestamp columns before saving
    df_features = drop_timestamp_columns(df_features)

    # Save raw ML features (tree-based models don't need scaling)
    logger.info("Saving dataset_features to %s", DATASET_FEATURES_PARQUET)
    df_features.to_parquet(DATASET_FEATURES_PARQUET, index=False)
    logger.info(
        "Saved dataset_features: %d rows, %d columns",
        len(df_features),
        len(df_features.columns),
    )

    # Also save copies for linear and lstm (will be normalized in clear_features)
    logger.info("Saving dataset_features_linear to %s (unscaled, will be normalized in clear_features)", DATASET_FEATURES_LINEAR_PARQUET)
    df_features.to_parquet(DATASET_FEATURES_LINEAR_PARQUET, index=False)

    logger.info("Saving dataset_features_lstm to %s (unscaled, will be normalized in clear_features)", DATASET_FEATURES_LSTM_PARQUET)
    df_features.to_parquet(DATASET_FEATURES_LSTM_PARQUET, index=False)


def _split_dataframe_for_output(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into chronological train/test slices.

    Uses an existing 'split' column if present; otherwise applies an 80/20
    chronological split and tags the rows with a new 'split' column.
    """
    if "split" in df.columns:
        mask_train = df["split"] == "train"
        mask_test = df["split"] == "test"
        df_train = df.loc[mask_train].copy()
        df_test = df.loc[mask_test].copy()

        if df_train.empty or df_test.empty:
            logger.warning("Existing split column is empty; falling back to ratio split")
        else:
            return df_train, df_test

    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_train["split"] = "train"
    df_test = df.iloc[split_idx:].copy()
    df_test["split"] = "test"
    return df_train, df_test


def save_train_test_splits() -> None:
    """Create and save 80/20 train/test splits for all feature variants."""
    datasets = [
        (
            DATASET_FEATURES_PARQUET,
            DATASET_FEATURES_TRAIN_PARQUET,
            DATASET_FEATURES_TEST_PARQUET,
        ),
        (
            DATASET_FEATURES_LINEAR_PARQUET,
            DATASET_FEATURES_LINEAR_TRAIN_PARQUET,
            DATASET_FEATURES_LINEAR_TEST_PARQUET,
        ),
        (
            DATASET_FEATURES_LSTM_PARQUET,
            DATASET_FEATURES_LSTM_TRAIN_PARQUET,
            DATASET_FEATURES_LSTM_TEST_PARQUET,
        ),
    ]

    for input_path, train_parquet, test_parquet in datasets:
        if not input_path.exists():
            logger.warning("Cannot create split: missing %s", input_path)
            continue

        df = pd.read_parquet(input_path)
        df_train, df_test = _split_dataframe_for_output(df)

        df_train.to_parquet(train_parquet, index=False)
        df_test.to_parquet(test_parquet, index=False)

        logger.info(
            "Saved train/test splits for %s -> train: %s, test: %s",
            input_path.name,
            train_parquet.name,
            test_parquet.name,
        )


# =============================================================================
# BATCH PROCESSING - Memory Efficient Pipeline
# =============================================================================

# Default batch size (number of bars per batch)
DEFAULT_BATCH_SIZE = 200_000

# Overlap to handle rolling windows and lags at batch boundaries
# Should be >= max(rolling window size, max lag) used in features
# Max lag is 50, max rolling window is ~100, so 200 is safe
BATCH_OVERLAP = 200


def run_batch_pipeline(
    batch_size: int = DEFAULT_BATCH_SIZE,
    overlap: int = BATCH_OVERLAP,
    sample_fraction: float = 1.0,
) -> None:
    """Run feature engineering with batch processing for large datasets.

    Processes dollar bars in batches using PyArrow iter_batches to minimize
    memory usage. Maintains an overlap buffer between batches to handle
    rolling windows and lags correctly.

    Train/test split is applied DURING batch processing to avoid reloading
    the entire dataset at the end.

    Args:
        batch_size: Number of bars per batch (default 200K).
        overlap: Number of rows overlap between batches for rolling windows/lags.
        sample_fraction: Fraction of data to use (0.0-1.0). Default 1.0 (all data).
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE - BATCH MODE")
    logger.info("=" * 60)
    logger.info(f"  Batch size: {batch_size:,}")
    logger.info(f"  Overlap: {overlap:,}")
    if sample_fraction < 1.0:
        logger.info(f"  Sample fraction: {sample_fraction:.1%} (TEST MODE)")

    if not DOLLAR_BARS_PARQUET.exists():
        raise FileNotFoundError(
            f"Input file not found: {DOLLAR_BARS_PARQUET}. "
            "Please run data_preparation first."
        )

    # Get total row count without loading
    parquet_file = pq.ParquetFile(DOLLAR_BARS_PARQUET)
    total_rows_full = parquet_file.metadata.num_rows

    # Apply sampling if requested
    if sample_fraction < 1.0:
        total_rows = int(total_rows_full * sample_fraction)
        logger.info(f"  Full dataset: {total_rows_full:,} bars")
        logger.info(f"  Sampling {sample_fraction:.1%}: {total_rows:,} bars")
    else:
        total_rows = total_rows_full

    # Calculate train/test split index (80/20 chronological split)
    # This is based on output rows, not input rows - we'll estimate
    # and apply split during consolidation based on cumulative position
    logger.info(f"  Total input bars: {total_rows:,}")
    logger.info(f"  Train/test split: {TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}%")

    # Create output directories
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    # Temporary directory for batch outputs
    batch_output_dir = FEATURES_DIR / "batches"

    # Clean up any leftover batch files from previous runs
    if batch_output_dir.exists():
        for old_batch in batch_output_dir.glob("*.parquet"):
            old_batch.unlink()
        logger.info("  Cleaned up old batch files")

    batch_output_dir.mkdir(parents=True, exist_ok=True)

    # Track batch files for later consolidation
    batch_files: list[Path] = []
    total_features_rows = 0

    # Overlap buffer from previous batch (raw data for rolling windows/lags)
    overlap_buffer: pd.DataFrame | None = None
    # Track the last timestamp processed to avoid duplicates
    last_processed_timestamp: int | None = None
    batch_num = 0
    # Track input rows processed for sampling
    input_rows_processed = 0

    # Process using PyArrow iter_batches (memory efficient - doesn't load all data)
    for arrow_batch in parquet_file.iter_batches(batch_size=batch_size):
        # Check if we've processed enough rows (for sampling)
        if input_rows_processed >= total_rows:
            logger.info(f"  Reached sample limit ({total_rows:,} rows), stopping")
            break

        df_batch = arrow_batch.to_pandas()

        # Truncate batch if it would exceed sample limit
        remaining_rows = total_rows - input_rows_processed
        if len(df_batch) > remaining_rows:
            df_batch = df_batch.iloc[:remaining_rows].copy()
            logger.info(f"  Truncated batch to {len(df_batch):,} rows (sample limit)")

        input_rows_processed += len(df_batch)
        batch_num += 1

        logger.info(f"\n{'=' * 50}")
        logger.info(f"BATCH {batch_num}")
        logger.info(f"  Raw batch rows: {len(df_batch):,}")

        # Prepend overlap buffer from previous batch (needed for rolling windows/lags)
        if overlap_buffer is not None:
            df_batch = pd.concat([overlap_buffer, df_batch], ignore_index=True)
            logger.info(f"  With overlap: {len(df_batch):,} rows")

        # Save overlap for next batch BEFORE processing (raw data)
        if len(df_batch) > overlap:
            overlap_buffer = df_batch.tail(overlap).copy()
        else:
            overlap_buffer = df_batch.copy()

        # Compute timestamp features
        df_batch = compute_timestamp_features(df_batch)

        # Compute all features
        df_features = compute_all_features(df_batch)

        # Free memory
        del df_batch
        gc.collect()

        # Apply lags
        df_features_lagged = apply_lags(df_features)

        # Free memory
        del df_features
        gc.collect()

        # Clean NaN and interpolate
        df_features_clean = drop_initial_nan_rows(df_features_lagged)

        # Free memory
        del df_features_lagged
        gc.collect()

        df_features_clean = interpolate_sporadic_nan(df_features_clean)

        # Shift target
        df_features_clean = shift_target_to_future_return(df_features_clean, target_col="log_return")

        # Remove overlap rows using timestamp-based filtering (more robust than index-based)
        # This avoids issues when drop_initial_nan_rows removes variable numbers of rows
        if last_processed_timestamp is not None and len(df_features_clean) > 0:
            # Find timestamp column (prefer timestamp_close for bar data)
            ts_col = None
            for col in ["timestamp_close", "timestamp_open", "timestamp"]:
                if col in df_features_clean.columns:
                    ts_col = col
                    break

            if ts_col is not None:
                # Keep only rows with timestamp > last processed timestamp
                before_filter = len(df_features_clean)
                df_features_clean = df_features_clean[
                    df_features_clean[ts_col] > last_processed_timestamp
                ].copy()
                filtered_count = before_filter - len(df_features_clean)
                if filtered_count > 0:
                    logger.info(f"  Filtered {filtered_count} overlap rows (timestamp <= {last_processed_timestamp})")
            else:
                # Fallback to index-based removal if no timestamp column
                rows_to_skip = min(overlap, len(df_features_clean) - 1)
                if rows_to_skip > 0:
                    df_features_clean = df_features_clean.iloc[rows_to_skip:].copy()
                    logger.info(f"  After removing overlap prefix (index-based): {len(df_features_clean):,} rows")

        if len(df_features_clean) == 0:
            logger.warning(f"  Batch {batch_num} produced no valid rows, skipping")
            continue

        # Update last_processed_timestamp before saving
        ts_col = None
        for col in ["timestamp_close", "timestamp_open", "timestamp"]:
            if col in df_features_clean.columns:
                ts_col = col
                break
        if ts_col is not None:
            last_processed_timestamp = int(df_features_clean[ts_col].max())

        # Save batch
        batch_file = batch_output_dir / f"batch_{batch_num:04d}.parquet"
        df_features_clean.to_parquet(batch_file, index=False)
        batch_files.append(batch_file)
        total_features_rows += len(df_features_clean)

        logger.info(f"  Saved {len(df_features_clean):,} rows to {batch_file.name}")

        # Clear memory
        del df_features_clean
        gc.collect()

    # =========================================================================
    # CONSOLIDATE BATCHES WITH TRAIN/TEST SPLIT
    # NOTE: Scaler fitting moved to clear_features module (after PCA)
    # =========================================================================
    logger.info(f"\n{'=' * 50}")
    logger.info("CONSOLIDATING BATCHES WITH TRAIN/TEST SPLIT")
    logger.info(f"  Total batch files: {len(batch_files)}")
    logger.info(f"  Total rows: {total_features_rows:,}")

    if not batch_files:
        raise ValueError("No batches were generated")

    # Calculate train split index based on total output rows
    train_split_idx = int(total_features_rows * TRAIN_RATIO)
    logger.info(f"  Train split at row: {train_split_idx:,}")

    # =========================================================================
    # FIRST PASS: Collect all unique columns to create unified schema
    # This is necessary because some features (e.g., bars_since_shock_*)
    # have different threshold values depending on the data in each batch
    # =========================================================================
    logger.info("  Scanning batch schemas...")
    all_columns: set[str] = set()
    for batch_file in batch_files:
        pf = pq.ParquetFile(batch_file)
        schema = pf.schema_arrow
        batch_cols = set(schema.names)
        all_columns.update(batch_cols)

    unified_columns = sorted(all_columns)
    logger.info(f"  Unified schema: {len(unified_columns)} columns")

    # Track cumulative rows for split
    cumulative_rows = 0

    # Writers for the 3 output files
    writers: dict[str, pq.ParquetWriter | None] = {
        "tree_based": None,
        "linear": None,
        "lstm": None,
    }
    output_paths = {
        "tree_based": DATASET_FEATURES_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_PARQUET,
    }

    for batch_idx, batch_file in enumerate(batch_files):
        logger.info(f"  Processing batch {batch_idx + 1}/{len(batch_files)}: {batch_file.name}")

        df_batch = pd.read_parquet(batch_file)
        batch_len = len(df_batch)

        # Reindex to unified schema (add missing columns as NaN)
        # Use pd.concat to avoid DataFrame fragmentation warning
        missing_cols = set(unified_columns) - set(df_batch.columns)
        if missing_cols:
            # Create all missing columns at once using concat (avoids fragmentation)
            missing_df = pd.DataFrame(
                np.nan,
                index=df_batch.index,
                columns=pd.Index(list(missing_cols)),
            )
            df_batch = pd.concat([df_batch, missing_df], axis=1)
            # Reorder to match unified schema
            df_batch = df_batch[unified_columns].copy()  # copy() defragments

        # Determine split for each row in this batch
        batch_start = cumulative_rows
        batch_end = cumulative_rows + batch_len

        # Determine train portion for this batch (COPY to avoid issues)
        train_portion: pd.DataFrame | None = None

        if batch_end <= train_split_idx:
            # Entire batch is train
            df_batch = df_batch.copy()  # Defragment before adding column
            df_batch["split"] = "train"
            train_portion = cast(pd.DataFrame, df_batch.copy())
        elif batch_start >= train_split_idx:
            # Entire batch is test
            if not missing_cols:  # Only copy if not already copied above
                df_batch = df_batch.copy()
            df_batch["split"] = "test"
        else:
            # Split within this batch
            local_split = train_split_idx - batch_start
            if not missing_cols:  # Only copy if not already copied above
                df_batch = df_batch.copy()
            df_batch["split"] = "test"
            df_batch.iloc[:local_split, df_batch.columns.get_loc("split")] = "train"
            train_portion = df_batch.iloc[:local_split].copy()

        cumulative_rows = batch_end

        # NOTE: Scaler fitting removed - now done in clear_features after PCA

        # Drop timestamp columns before saving
        df_batch = drop_timestamp_columns(cast(pd.DataFrame, df_batch))

        # Write to all 3 output files
        table = pa.Table.from_pandas(df_batch, preserve_index=False)

        for key in writers:
            if writers[key] is None:
                try:
                    writers[key] = pq.ParquetWriter(
                        output_paths[key],
                        table.schema,
                        compression="snappy",
                    )
                except Exception as e:
                    logger.error(f"Failed to create ParquetWriter for {key}: {e}")
                    continue
            if writers[key] is not None:
                cast(pq.ParquetWriter, writers[key]).write_table(table)

        del df_batch, table
        if train_portion is not None:
            del train_portion
        gc.collect()

    logger.info("  All batches consolidated")

    # Close all writers
    for key in writers:
        if writers[key] is not None:
            cast(pq.ParquetWriter, writers[key]).close()

    # Clean up batch files and directory
    import shutil
    if batch_output_dir.exists():
        shutil.rmtree(batch_output_dir)
        logger.info(f"  Cleaned up batch directory: {batch_output_dir}")

    logger.info(f"  Saved to: {DATASET_FEATURES_PARQUET}")
    logger.info(f"  Saved to: {DATASET_FEATURES_LINEAR_PARQUET}")
    logger.info(f"  Saved to: {DATASET_FEATURES_LSTM_PARQUET}")

    # NOTE: Scaler fitting moved to clear_features module (after PCA transformation)

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE (BATCH MODE)")
    logger.info("=" * 60)
    logger.info(f"  Total features rows: {total_features_rows:,}")
    logger.info(f"  Train rows: {train_split_idx:,}")
    logger.info(f"  Test rows: {total_features_rows - train_split_idx:,}")
    logger.info("")
    logger.info("NEXT STEP: Run clear_features to apply:")
    logger.info("  1. PCA reduction on correlated features")
    logger.info("  2. Scaler fitting on PCA-transformed features (train only)")
    logger.info("  3. Normalization (z-score for linear, minmax for LSTM)")


def main() -> None:
    """Run the complete feature engineering pipeline.

    Pipeline flow:
    1. Load dollar bars
    2. Compute all features + lags
    3. Clean NaN rows
    4. Split train/test
    5. Save raw datasets (unscaled)

    Uses batch mode by default to handle large datasets efficiently.

    Next step: Run clear_features to apply PCA, fit scalers, and normalize.
    """
    parser = argparse.ArgumentParser(
        description="Feature Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for processing (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=BATCH_OVERLAP,
        help=f"Overlap between batches for rolling windows (default: {BATCH_OVERLAP})",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch mode (load all in memory - may crash on large datasets)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Sample fraction of input data (0.0-1.0). Use 0.05 for 5%% sample for testing.",
    )
    args = parser.parse_args()

    setup_logging()

    try:
        if not args.no_batch:
            # Batch mode (default) - memory efficient
            run_batch_pipeline(
                batch_size=args.batch_size,
                overlap=args.overlap,
                sample_fraction=args.sample,
            )
        else:
            # Standard mode (load all in memory) - only for small datasets
            logger.info("=" * 60)
            logger.info("FEATURE ENGINEERING PIPELINE (NON-BATCH MODE)")
            logger.info("=" * 60)
            logger.warning("Non-batch mode may crash on large datasets!")

            # 1. Load input data
            df_bars = load_input_data()

            # 2. Compute timestamp-derived features (before they get dropped)
            df_bars = compute_timestamp_features(df_bars)

            # 3. Compute all features
            df_features = compute_all_features(df_bars)

            # 4. Apply intelligent lag structure
            df_features_lagged = apply_lags(df_features)

            # 5. Drop initial NaN rows from lags/rolling windows
            df_features_clean = drop_initial_nan_rows(df_features_lagged)

            # 6. Interpolate sporadic NaN (body_ratio, sampen edge cases)
            df_features_clean = interpolate_sporadic_nan(df_features_clean)

            # 7. Shift target to next-bar return to avoid leakage
            df_features_clean = shift_target_to_future_return(df_features_clean, target_col="log_return")

            # 8. Split into train/test
            df_train, df_test, _ = split_train_test(df_features_clean)
            df_train["split"] = "train"
            df_test["split"] = "test"
            df_features_with_split = pd.concat([df_train, df_test], axis=0)

            # 9. NOTE: Scaler fitting moved to clear_features module
            # Scalers must be fit AFTER PCA transformation to match final columns
            # fit_and_save_scalers(df_train)  # REMOVED - now done in clear_features

            # 10. Save raw datasets (all 3 copies unscaled)
            save_outputs(df_features_with_split)

            logger.info("=" * 60)
            logger.info("FEATURE ENGINEERING COMPLETE")
            logger.info("=" * 60)
            logger.info("Output files (all unscaled):")
            logger.info("  - %s (Raw features for tree-based ML)", DATASET_FEATURES_PARQUET)
            logger.info("  - %s (Copy for linear models)", DATASET_FEATURES_LINEAR_PARQUET)
            logger.info("  - %s (Copy for LSTM)", DATASET_FEATURES_LSTM_PARQUET)
            logger.info("")
            logger.info("NEXT STEP: Run clear_features to apply:")
            logger.info("  1. PCA reduction on correlated features")
            logger.info("  2. Scaler fitting on PCA-transformed features (train only)")
            logger.info("  3. Normalization (z-score for linear, minmax for LSTM)")

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
