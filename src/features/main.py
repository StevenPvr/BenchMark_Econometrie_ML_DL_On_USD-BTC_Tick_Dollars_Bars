"""Main pipeline for feature generation.

This module orchestrates the complete feature engineering pipeline:

1. Load input: log_returns_split.parquet
2. Compute all features (microstructure, volatility, momentum, etc.)
3. Apply intelligent lag structure
4. Split into train/test
5. Fit scalers on TRAIN only, transform both (no data leakage!)
6. Save outputs:
   - dataset_features.parquet/csv: Raw features for tree-based ML (XGBoost, etc.)
   - dataset_features_linear.parquet/csv: Z-scored features for linear models (Ridge, Lasso)
   - dataset_features_lstm.parquet/csv: Min-max [-1,1] scaled for LSTM

Usage:
    python -m src.features.main
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger, setup_logging
from src.features.entropy import (
    compute_approximate_entropy,
    compute_sample_entropy,
    compute_shannon_entropy,
)
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
    compute_local_sharpe,
    compute_realized_volatility,
)
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
    DATASET_FEATURES_CSV,
    DATASET_FEATURES_LINEAR_CSV,
    DATASET_FEATURES_LINEAR_PARQUET,
    DATASET_FEATURES_LSTM_CSV,
    DATASET_FEATURES_LSTM_PARQUET,
    DATASET_FEATURES_PARQUET,
    FEATURES_DIR,
    LOG_RETURNS_SPLIT_PARQUET,
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
    logger.info("Loading input data from %s", LOG_RETURNS_SPLIT_PARQUET)

    if not LOG_RETURNS_SPLIT_PARQUET.exists():
        raise FileNotFoundError(
            f"Input file not found: {LOG_RETURNS_SPLIT_PARQUET}. "
            "Please run data_preparation first."
        )

    df = pd.read_parquet(LOG_RETURNS_SPLIT_PARQUET)

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
    # 2. REALIZED VOLATILITY
    # =========================================================================
    logger.info("Computing realized volatility features...")

    df_vol = compute_realized_volatility(df_bars, return_col="log_return")
    feature_dfs.append(df_vol)

    df_sharpe = compute_local_sharpe(df_bars, return_col="log_return")
    feature_dfs.append(df_sharpe)

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
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_volume",
        "sell_volume",
        "n_ticks",
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
    initial_rows = len(df)

    # Find the first row where all columns have valid values
    # Use numeric columns only for NaN detection
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Find first valid index (where no NaN in numeric columns)
    first_valid_idx = df[numeric_cols].dropna().index[0]
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


def create_scaled_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create scaled datasets for linear models and LSTM.

    CRITICAL: Scalers are fit on TRAIN only, then applied to both train and test.
    This prevents data leakage from test to train.

    Args:
        df_train: Training DataFrame.
        df_test: Test DataFrame.

    Returns:
        Tuple of (train_linear, test_linear, train_lstm, test_lstm).
    """
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Z-SCORE FOR LINEAR MODELS
    # Exclude log_return (target) and its lags
    # =========================================================================
    logger.info("Creating z-scored dataset for linear models...")

    cols_zscore = get_columns_to_scale(
        df_train,
        exclude_target=True,
        include_log_return_lags=False,  # Don't z-score log_return lags
    )

    zscore_scaler = StandardScalerCustom()
    zscore_scaler.fit(df_train, cols_zscore)
    zscore_scaler.save(ZSCORE_SCALER_FILE)

    train_linear = zscore_scaler.transform(df_train)
    test_linear = zscore_scaler.transform(df_test)

    logger.info(
        "Z-score scaler fit on %d columns, saved to %s",
        len(cols_zscore),
        ZSCORE_SCALER_FILE,
    )

    # =========================================================================
    # MIN-MAX FOR LSTM
    # Include log_return lags (they are features), exclude log_return (target)
    # =========================================================================
    logger.info("Creating min-max scaled dataset for LSTM...")

    cols_minmax = get_columns_to_scale(
        df_train,
        exclude_target=True,  # Don't scale target
        include_log_return_lags=True,  # DO scale log_return lags
    )

    minmax_scaler = MinMaxScalerCustom()
    minmax_scaler.fit(df_train, cols_minmax)
    minmax_scaler.save(MINMAX_SCALER_FILE)

    train_lstm = minmax_scaler.transform(df_train)
    test_lstm = minmax_scaler.transform(df_test)

    logger.info(
        "Min-max scaler fit on %d columns, saved to %s",
        len(cols_minmax),
        MINMAX_SCALER_FILE,
    )

    return train_linear, test_linear, train_lstm, test_lstm


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
    df_features_linear: pd.DataFrame,
    df_features_lstm: pd.DataFrame,
) -> None:
    """Save feature datasets to files.

    Args:
        df_features: Raw features for ML models.
        df_features_linear: Z-scored features for linear models.
        df_features_lstm: Min-max scaled features for LSTM.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Drop timestamp columns from all datasets before saving
    df_features = drop_timestamp_columns(df_features)
    df_features_linear = drop_timestamp_columns(df_features_linear)
    df_features_lstm = drop_timestamp_columns(df_features_lstm)

    # Save raw ML features
    logger.info("Saving dataset_features to %s", DATASET_FEATURES_PARQUET)
    df_features.to_parquet(DATASET_FEATURES_PARQUET, index=False)
    df_features.to_csv(DATASET_FEATURES_CSV, index=False)
    logger.info(
        "Saved dataset_features: %d rows, %d columns",
        len(df_features),
        len(df_features.columns),
    )

    # Save z-scored linear features
    logger.info("Saving dataset_features_linear to %s", DATASET_FEATURES_LINEAR_PARQUET)
    df_features_linear.to_parquet(DATASET_FEATURES_LINEAR_PARQUET, index=False)
    df_features_linear.to_csv(DATASET_FEATURES_LINEAR_CSV, index=False)
    logger.info(
        "Saved dataset_features_linear: %d rows, %d columns",
        len(df_features_linear),
        len(df_features_linear.columns),
    )

    # Save min-max LSTM features
    logger.info("Saving dataset_features_lstm to %s", DATASET_FEATURES_LSTM_PARQUET)
    df_features_lstm.to_parquet(DATASET_FEATURES_LSTM_PARQUET, index=False)
    df_features_lstm.to_csv(DATASET_FEATURES_LSTM_CSV, index=False)
    logger.info(
        "Saved dataset_features_lstm: %d rows, %d columns",
        len(df_features_lstm),
        len(df_features_lstm.columns),
    )


def main() -> None:
    """Run the complete feature engineering pipeline."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    try:
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

        # 9. Create scaled datasets (fit on train only!)
        train_linear, test_linear, train_lstm, test_lstm = create_scaled_datasets(
            df_train, df_test
        )

        # 10. Concatenate train + test for final datasets
        df_features_linear = pd.concat([train_linear, test_linear], axis=0)
        df_features_lstm = pd.concat([train_lstm, test_lstm], axis=0)

        # 11. Save outputs (timestamp columns dropped, cleaned data)
        save_outputs(df_features_clean, df_features_linear, df_features_lstm)

        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 60)
        logger.info("Output files:")
        logger.info("  - %s (Raw features for tree-based ML)", DATASET_FEATURES_PARQUET)
        logger.info("  - %s (Z-scored for Ridge/Lasso)", DATASET_FEATURES_LINEAR_PARQUET)
        logger.info("  - %s (Min-max [-1,1] for LSTM)", DATASET_FEATURES_LSTM_PARQUET)
        logger.info("Scalers saved to: %s", SCALERS_DIR)
        logger.info("  - zscore_scaler.joblib")
        logger.info("  - minmax_scaler.joblib")

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
