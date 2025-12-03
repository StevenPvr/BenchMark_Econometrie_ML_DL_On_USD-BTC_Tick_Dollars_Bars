"""Feature computation functions.

This module orchestrates computing all features from dollar bars:
- Momentum features
- Realized volatility and higher moments
- Trend features
- Range-based volatility
- Temporal acceleration
- Order flow / microstructure
- Entropy features
- Temporal / calendar / regime
- Fractional differentiation
- Technical analysis indicators
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from src.constants import EPS
from src.utils import get_logger
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

logger = get_logger(__name__)

__all__ = [
    "compute_all_features",
    "apply_lags",
]


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

    df_skew = compute_realized_skewness(df_bars, return_col="log_return")
    feature_dfs.append(df_skew)

    df_kurt = compute_realized_kurtosis(df_bars, return_col="log_return")
    feature_dfs.append(df_kurt)

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
            df_bars["buy_volume"] + df_bars["sell_volume"] + EPS
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
            fillna=False,
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
