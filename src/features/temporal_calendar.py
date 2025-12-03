"""Temporal, calendar, and regime features.

This module computes time-based and regime detection features:

1. Cyclical time encoding (hour, day):
   sin(2π · hour/24), cos(2π · hour/24)
   sin(2π · day/7), cos(2π · day/7)

2. Time since last shock:
   Number of bars since |r_t| > threshold

3. Volatility regime dummies:
   1 if σ_t > quantile(σ, q) else 0

4. Drawdown regime:
   Current drawdown from rolling max

Interpretation:
    - Cyclical encoding: Captures intraday/weekly patterns without discontinuities
    - Time since shock: Mean reversion timing signal
    - Vol regime: Identifies turbulent vs calm periods
    - Drawdown: Crash detection and recovery tracking

Reference:
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 5: Fractionally Differentiated Features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.utils import get_logger
from src.features.temporal_calendar_core import (
    _compute_drawdown,
    _compute_drawup,
    _expanding_mean,
    _expanding_std,
    _time_since_negative_shock,
    _time_since_positive_shock,
    _time_since_shock,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_cyclical_time_features",
    "compute_time_since_shock",
    "compute_volatility_regime",
    "compute_drawdown_features",
    "compute_all_temporal_features",
]


# =============================================================================
# CYCLICAL TIME ENCODING
# =============================================================================


def compute_cyclical_time_features(
    df_bars: pd.DataFrame,
    timestamp_col: str = "timestamp_close",
) -> pd.DataFrame:
    """Compute cyclical time encoding features AND numeric time features.

    Two types of encoding:
    1. Cyclical (sin/cos): Captures cyclical patterns without discontinuities
    2. Numeric: Raw values that can be normalized like other features

    Cyclical encoding:
        hour_sin = sin(2π · hour/24)
        hour_cos = cos(2π · hour/24)
        day_sin = sin(2π · dayofweek/7)
        day_cos = cos(2π · dayofweek/7)

    Numeric features (scalable):
        hour_of_day: 0-23 (fractional with minutes)
        day_of_week: 0-6 (Monday=0, Sunday=6)
        day_of_month: 1-31
        week_of_year: 1-52
        month_of_year: 1-12
        quarter: 1-4

    Args:
        df_bars: DataFrame with timestamp column.
        timestamp_col: Name of timestamp column.

    Returns:
        DataFrame with cyclical and numeric time features.

    Example:
        >>> df_time = compute_cyclical_time_features(df_bars)
        >>> df_bars = pd.concat([df_bars, df_time], axis=1)
    """
    result = pd.DataFrame(index=df_bars.index)

    # Convert to datetime if needed
    timestamps = pd.to_datetime(df_bars[timestamp_col])

    # =========================================================================
    # NUMERIC TIME FEATURES (can be scaled/normalized)
    # =========================================================================

    # Hour of day (0-23, fractional)
    hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
    result["hour_of_day"] = hour

    # Day of week (0=Monday, 6=Sunday)
    dayofweek = timestamps.dt.dayofweek.astype(float)
    result["day_of_week"] = dayofweek

    # Day of month (1-31)
    result["day_of_month"] = timestamps.dt.day.astype(float)

    # Week of year (1-52)
    result["week_of_year"] = timestamps.dt.isocalendar().week.astype(float)

    # Month of year (1-12)
    month = timestamps.dt.month.astype(float)
    result["month_of_year"] = month

    # Quarter (1-4)
    result["quarter"] = timestamps.dt.quarter.astype(float)

    # =========================================================================
    # CYCLICAL TIME FEATURES (sin/cos encoding)
    # =========================================================================

    # Hour cyclical
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Day of week cyclical
    result["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    result["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)

    # Month cyclical
    result["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Log statistics
    logger.info(
        "Time features computed: numeric (hour, day, week, month, quarter) + "
        "cyclical (sin/cos encoding)"
    )

    return result


# =============================================================================
# TIME SINCE LAST SHOCK
# =============================================================================


def compute_time_since_shock(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute time (bars) since last shock event.

    A shock is defined as |r_t| > threshold. This feature captures
    mean reversion timing and volatility clustering.

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        thresholds: List of shock thresholds (default: computed from quantiles).

    Returns:
        DataFrame with time-since-shock features.

    Example:
        >>> df_shock = compute_time_since_shock(df_bars)
        >>> df_bars = pd.concat([df_bars, df_shock], axis=1)
    """
    returns = df_bars[return_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    # Default thresholds based on return distribution
    if thresholds is None:
        valid_returns = returns[~np.isnan(returns)]
        if len(valid_returns) > 0:
            # Use 90th, 95th, 99th percentile of absolute returns
            abs_returns = np.abs(valid_returns)
            thresholds = [
                float(np.percentile(abs_returns, 90)),
                float(np.percentile(abs_returns, 95)),
                float(np.percentile(abs_returns, 99)),
            ]
        else:
            thresholds = [0.01, 0.02, 0.03]

    for thresh in thresholds:
        # Time since any shock (positive or negative)
        time_since = _time_since_shock(returns, thresh)
        thresh_str = f"{thresh:.4f}".rstrip("0").rstrip(".")
        result[f"bars_since_shock_{thresh_str}"] = time_since

        # Time since positive shock
        time_since_pos = _time_since_positive_shock(returns, thresh)
        result[f"bars_since_pos_shock_{thresh_str}"] = time_since_pos

        # Time since negative shock
        time_since_neg = _time_since_negative_shock(returns, thresh)
        result[f"bars_since_neg_shock_{thresh_str}"] = time_since_neg

        # Log statistics
        valid = time_since[~np.isnan(time_since)]
        if len(valid) > 0:
            logger.info(
                "Time since shock (thresh=%.4f): mean=%.1f bars, max=%.0f bars",
                thresh,
                np.mean(valid),
                np.max(valid),
            )

    return result


# =============================================================================
# VOLATILITY REGIME
# =============================================================================


def _rolling_std(
    values: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling standard deviation using pandas (optimized in C).

    Args:
        values: Input array.
        window: Rolling window size.

    Returns:
        Array of rolling standard deviations.
    """
    # Use pandas rolling std - highly optimized in C with O(n) complexity
    series = pd.Series(values)
    result = series.rolling(window=window, min_periods=2).std().to_numpy(dtype=np.float64)
    return result


def _expanding_quantile(
    values: NDArray[np.float64],
    quantile: float,
    min_periods: int,
) -> NDArray[np.float64]:
    """Compute expanding quantile using pandas (O(n) optimized).

    Args:
        values: Input array.
        quantile: Quantile to compute (0-1).
        min_periods: Minimum observations required.

    Returns:
        Array of expanding quantiles.
    """
    # Use pandas expanding quantile - highly optimized in C
    series = pd.Series(values)
    result = series.expanding(min_periods=min_periods).quantile(quantile).to_numpy(dtype=np.float64)
    return result


def compute_volatility_regime(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    vol_window: int = 20,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Compute volatility regime indicators.

    Creates dummy variables indicating when volatility exceeds
    historical quantiles:

        vol_regime_high = 1 if σ_t > quantile(σ, q) else 0

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        vol_window: Window for volatility calculation.
        quantiles: List of quantiles for regime detection (default: [0.75, 0.90, 0.95]).

    Returns:
        DataFrame with volatility regime features.

    Example:
        >>> df_regime = compute_volatility_regime(df_bars)
        >>> df_bars = pd.concat([df_bars, df_regime], axis=1)
    """
    if quantiles is None:
        quantiles = [0.75, 0.90, 0.95]

    returns = df_bars[return_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    # Compute rolling volatility
    vol = _rolling_std(returns, vol_window)
    result[f"volatility_{vol_window}"] = vol

    # Compute regime dummies using expanding quantiles
    for q in quantiles:
        # Expanding quantile of volatility (no lookahead)
        vol_quantile = _expanding_quantile(vol, q, min_periods=vol_window * 2)

        # Regime dummy
        regime = np.where(vol > vol_quantile, 1.0, 0.0)
        regime = np.where(np.isnan(vol) | np.isnan(vol_quantile), np.nan, regime)

        q_pct = int(q * 100)
        result[f"vol_regime_q{q_pct}"] = regime

        # Log statistics
        valid = regime[~np.isnan(regime)]
        if len(valid) > 0:
            pct_high = 100 * np.mean(valid)
            logger.info(
                "Volatility regime (q=%d): %.1f%% of bars in high-vol regime",
                q_pct,
                pct_high,
            )

    # Add volatility z-score (normalized)
    vol_mean = _expanding_mean(vol, min_periods=vol_window * 2)
    vol_std = _expanding_std(vol, min_periods=vol_window * 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        vol_zscore = np.where(vol_std > 1e-10, (vol - vol_mean) / vol_std, np.nan)
    result["vol_zscore"] = vol_zscore

    return result


# =============================================================================
# DRAWDOWN FEATURES
# =============================================================================


def compute_drawdown_features(
    df_bars: pd.DataFrame,
    price_col: str = "close",
    crash_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute drawdown and crash detection features.

    1. Drawdown: (P_t - max(P)) / max(P)
    2. Drawup: (P_t - min(P)) / min(P)
    3. Bars since all-time high/low
    4. Crash dummies: 1 if drawdown < threshold

    Args:
        df_bars: DataFrame with price data.
        price_col: Name of price column.
        crash_thresholds: Drawdown thresholds for crash dummies (default: [-0.05, -0.10, -0.20]).

    Returns:
        DataFrame with drawdown features.

    Example:
        >>> df_dd = compute_drawdown_features(df_bars)
        >>> df_bars = pd.concat([df_bars, df_dd], axis=1)
    """
    if crash_thresholds is None:
        crash_thresholds = [-0.05, -0.10, -0.20]

    prices = df_bars[price_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    # Compute drawdown from all-time high
    drawdown, rolling_max, bars_since_high = _compute_drawdown(prices)
    result["drawdown"] = drawdown
    result["rolling_max"] = rolling_max
    result["bars_since_high"] = bars_since_high

    # Compute drawup from all-time low
    drawup, rolling_min, bars_since_low = _compute_drawup(prices)
    result["drawup"] = drawup
    result["rolling_min"] = rolling_min
    result["bars_since_low"] = bars_since_low

    # Crash dummies
    for thresh in crash_thresholds:
        crash_dummy = np.where(drawdown < thresh, 1.0, 0.0)
        crash_dummy = np.where(np.isnan(drawdown), np.nan, crash_dummy)

        thresh_pct = int(abs(thresh) * 100)
        result[f"crash_{thresh_pct}pct"] = crash_dummy

        # Log statistics
        valid = crash_dummy[~np.isnan(crash_dummy)]
        if len(valid) > 0:
            pct_crash = 100 * np.mean(valid)
            logger.info(
                "Crash regime (<%d%%): %.2f%% of bars",
                thresh_pct,
                pct_crash,
            )

    # Log drawdown statistics
    valid_dd = drawdown[~np.isnan(drawdown)]
    if len(valid_dd) > 0:
        logger.info(
            "Drawdown stats: mean=%.2f%%, min=%.2f%%, current=%.2f%%",
            100 * np.mean(valid_dd),
            100 * np.min(valid_dd),
            100 * valid_dd[-1] if len(valid_dd) > 0 else 0,
        )

    return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def compute_all_temporal_features(
    df_bars: pd.DataFrame,
    timestamp_col: str = "timestamp_close",
    return_col: str = "log_return",
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute all temporal, calendar, and regime features.

    Convenience function that computes:
    - Cyclical time features (hour, day, month)
    - Time since shock
    - Volatility regime
    - Drawdown features

    Args:
        df_bars: DataFrame with bar data.
        timestamp_col: Name of timestamp column.
        return_col: Name of return column.
        price_col: Name of price column.

    Returns:
        DataFrame with all temporal features.

    Example:
        >>> df_temporal = compute_all_temporal_features(df_bars)
        >>> df_bars = pd.concat([df_bars, df_temporal], axis=1)
    """
    dfs = []

    # Cyclical time
    if timestamp_col in df_bars.columns:
        dfs.append(compute_cyclical_time_features(df_bars, timestamp_col))

    # Time since shock
    if return_col in df_bars.columns:
        dfs.append(compute_time_since_shock(df_bars, return_col))

    # Volatility regime
    if return_col in df_bars.columns:
        dfs.append(compute_volatility_regime(df_bars, return_col))

    # Drawdown
    if price_col in df_bars.columns:
        dfs.append(compute_drawdown_features(df_bars, price_col))

    if dfs:
        result = pd.concat(dfs, axis=1)
        logger.info("Computed %d temporal/regime features", len(result.columns))
        return result

    return pd.DataFrame(index=df_bars.index)
