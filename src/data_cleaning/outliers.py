"""Robust outlier detection methods for financial tick data.

All methods are CAUSAL (use only past data via expanding/rolling windows)
to prevent temporal data leakage.

Methods
-------
- MAD (Median Absolute Deviation) - robust to fat-tailed distributions
- Rolling Z-score - adapts to local volatility regimes
- Volume anomaly detection - filters dust trades and manipulation
- Dollar value filtering - combined price*volume anomalies

References
----------
Huber, P.J. (1981). Robust Statistics.
Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import (
    OUTLIER_DOLLAR_VALUE_MAD_THRESHOLD,
    OUTLIER_LEGACY_MAX_PCT_CHANGE,
    OUTLIER_MAD_SCALING_FACTOR,
    OUTLIER_MAD_THRESHOLD,
    OUTLIER_MIN_PERIODS,
    OUTLIER_MIN_VOLUME,
    OUTLIER_ROBUST_MAD_MULTIPLIER,
    OUTLIER_ROLLING_WINDOW,
    OUTLIER_ROLLING_ZSCORE_THRESHOLD,
    OUTLIER_VOLUME_MAD_THRESHOLD,
    SYMBOL,
)
from src.utils import get_logger

logger = get_logger(__name__)

# Stablecoins considered as USD-like for notional calculation
USD_LIKE_CODES: frozenset[str] = frozenset({"USD", "USDT", "USDC", "BUSD", "DAI"})

__all__ = [
    "OutlierReport",
    "filter_outliers_robust",
    "filter_price_outliers",
    "merge_outlier_reports",
]


@dataclass
class OutlierReport:
    """Summary of outliers detected and removed by each filtering method."""

    total_ticks: int
    removed_mad_price: int = 0
    removed_rolling_zscore: int = 0
    removed_volume_outliers: int = 0
    removed_dollar_value: int = 0
    removed_dust_trades: int = 0
    final_ticks: int = 0

    def log_summary(self) -> None:
        """Log a human-readable summary of the outlier detection results."""
        total_removed = (
            self.removed_mad_price
            + self.removed_rolling_zscore
            + self.removed_volume_outliers
            + self.removed_dollar_value
            + self.removed_dust_trades
        )
        pct_removed = (total_removed / self.total_ticks * 100) if self.total_ticks > 0 else 0

        logger.info("=" * 60)
        logger.info("OUTLIER DETECTION SUMMARY (Causal Filters)")
        logger.info("=" * 60)
        logger.info(f"  Initial ticks:          {self.total_ticks:>12,}")
        logger.info(f"  MAD price outliers:     {self.removed_mad_price:>12,}")
        logger.info(f"  Rolling Z-score:        {self.removed_rolling_zscore:>12,}")
        logger.info(f"  Volume outliers:        {self.removed_volume_outliers:>12,}")
        logger.info(f"  Dollar value outliers:  {self.removed_dollar_value:>12,}")
        logger.info(f"  Dust trades (<min vol): {self.removed_dust_trades:>12,}")
        logger.info("-" * 60)
        logger.info(f"  Total removed:          {total_removed:>12,} ({pct_removed:.4f}%)")
        logger.info(f"  Final ticks:            {self.final_ticks:>12,}")
        logger.info("=" * 60)


def merge_outlier_reports(
    aggregate: OutlierReport | None,
    current: OutlierReport,
) -> OutlierReport:
    """Accumulate OutlierReport metrics across partitions."""
    if aggregate is None:
        return current

    return OutlierReport(
        total_ticks=aggregate.total_ticks + current.total_ticks,
        removed_mad_price=aggregate.removed_mad_price + current.removed_mad_price,
        removed_rolling_zscore=aggregate.removed_rolling_zscore + current.removed_rolling_zscore,
        removed_volume_outliers=aggregate.removed_volume_outliers + current.removed_volume_outliers,
        removed_dollar_value=aggregate.removed_dollar_value + current.removed_dollar_value,
        removed_dust_trades=aggregate.removed_dust_trades + current.removed_dust_trades,
        final_ticks=aggregate.final_ticks + current.final_ticks,
    )


def _compute_expanding_mad_mask(
    values: pd.Series,
    threshold: float,
    min_periods: int,
    filter_high_only: bool = False,
) -> pd.Series:
    """Compute validity mask using expanding MAD statistics (causal)."""
    expanding_median = values.expanding(min_periods=min_periods).median()
    expanding_mad = (
        (values - expanding_median).abs().expanding(min_periods=min_periods).median()
    )
    scaled_mad = OUTLIER_MAD_SCALING_FACTOR * expanding_mad
    scaled_mad = scaled_mad.replace(0, np.nan)

    if filter_high_only:
        deviation = values - expanding_median
        return (deviation <= threshold * scaled_mad) | scaled_mad.isna()

    deviation = (values - expanding_median).abs()
    return (deviation <= threshold * scaled_mad) | scaled_mad.isna()


def _compute_dollar_notional(
    df: pd.DataFrame,
    price_col: str,
    volume_col: str,
    symbol: str,
) -> pd.Series:
    """Compute USD-like notional regardless of symbol orientation."""
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
    else:
        base, quote = symbol, ""

    base_upper = base.upper()
    quote_upper = quote.upper()

    if base_upper in USD_LIKE_CODES:
        return pd.Series(df[volume_col])
    if quote_upper in USD_LIKE_CODES:
        return pd.Series(df[price_col] * df[volume_col])

    return pd.Series(df[price_col] * df[volume_col])


def filter_price_outliers(
    df: pd.DataFrame,
    max_pct_change: float = OUTLIER_LEGACY_MAX_PCT_CHANGE,
) -> pd.DataFrame:
    """Filter ticks with aberrant price changes. Deprecated: use filter_outliers_robust."""
    if "price" not in df.columns or df.empty:
        return df

    pct_change = df["price"].pct_change().abs()
    mask_valid = pct_change.isna() | (pct_change <= max_pct_change)
    filtered = df.loc[mask_valid].copy()

    removed = len(df) - len(filtered)
    if removed > 0:
        logger.info("Filtered %d ticks with price change > %.1f%%", removed, max_pct_change * 100)

    return filtered


def _filter_mad_price_outliers(
    df: pd.DataFrame,
    price_col: str = "price",
    threshold: float = OUTLIER_MAD_THRESHOLD,
    min_periods: int = OUTLIER_MIN_PERIODS,
) -> tuple[pd.DataFrame, int]:
    """Filter price outliers using expanding MAD (causal, no look-ahead)."""
    if price_col not in df.columns or df.empty:
        return df, 0

    mask_valid = _compute_expanding_mad_mask(pd.Series(df[price_col]), threshold, min_periods)
    filtered = df.loc[mask_valid].reset_index(drop=True)
    removed = len(df) - len(filtered)

    if removed > 0:
        logger.info(
            "MAD filter (causal): removed %d price outliers (threshold=%.1f MADs)",
            removed,
            threshold,
        )

    return filtered, removed


def _filter_rolling_zscore_outliers(
    df: pd.DataFrame,
    price_col: str = "price",
    window: int = OUTLIER_ROLLING_WINDOW,
    threshold: float = OUTLIER_ROLLING_ZSCORE_THRESHOLD,
    min_periods: int = OUTLIER_MIN_PERIODS,
) -> tuple[pd.DataFrame, int]:
    """Filter price outliers using rolling z-score (causal, adapts to volatility)."""
    if price_col not in df.columns or df.empty:
        return df, 0

    prices = pd.Series(df[price_col])

    # Compute rolling mean and std (causal)
    rolling_mean = prices.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = prices.rolling(window=window, min_periods=min_periods).std()

    # Z-score relative to rolling window
    z_scores = (prices - rolling_mean) / rolling_std

    # Keep prices where |z-score| <= threshold or where rolling stats are NaN
    mask_valid = z_scores.abs().le(threshold) | rolling_std.isna()
    filtered = df.loc[mask_valid].reset_index(drop=True)
    removed = len(df) - len(filtered)

    if removed > 0:
        logger.info(
            "Rolling Z-score filter (causal): removed %d outliers (threshold=%.1f, window=%d)",
            removed,
            threshold,
            window,
        )

    return filtered, removed


def _filter_dollar_value_outliers(
    df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "amount",
    symbol: str = SYMBOL,
    threshold: float = OUTLIER_DOLLAR_VALUE_MAD_THRESHOLD,
    min_periods: int = OUTLIER_MIN_PERIODS,
) -> tuple[pd.DataFrame, int]:
    """Filter combined price*volume outliers using expanding MAD."""
    if price_col not in df.columns or volume_col not in df.columns or df.empty:
        return df, 0

    # Compute dollar notional values
    dollar_values = _compute_dollar_notional(df, price_col, volume_col, symbol)

    # Use expanding MAD to detect outliers in dollar values
    mask_valid = _compute_expanding_mad_mask(dollar_values, threshold, min_periods)
    filtered = df.loc[mask_valid].reset_index(drop=True)
    removed = len(df) - len(filtered)

    if removed > 0:
        logger.info(
            "Dollar value filter (causal): removed %d outliers (threshold=%.1f MADs)",
            removed,
            threshold,
        )

    return filtered, removed


def _filter_volume_outliers(
    df: pd.DataFrame,
    volume_col: str = "amount",
    threshold: float = OUTLIER_VOLUME_MAD_THRESHOLD,
    min_volume: float = OUTLIER_MIN_VOLUME,
    min_periods: int = OUTLIER_MIN_PERIODS,
    apply_mad: bool = True,
) -> tuple[pd.DataFrame, int, int]:
    """Filter volume outliers using expanding MAD and minimum threshold."""
    if volume_col not in df.columns or df.empty:
        return df, 0, 0

    volumes = pd.Series(df[volume_col])
    mask_min_volume = volumes >= min_volume
    removed_dust = (~mask_min_volume).sum()

    df_no_dust = df.loc[mask_min_volume].copy()
    if df_no_dust.empty:
        return df_no_dust, 0, removed_dust

    if not apply_mad:
        filtered = df_no_dust.reset_index(drop=True)
        if removed_dust > 0:
            logger.info(
                "Volume filter (dust only): removed %d dust trades, skipped MAD filtering",
                removed_dust,
            )
        return filtered, 0, removed_dust

    mask_valid = _compute_expanding_mad_mask(
        df_no_dust[volume_col], threshold, min_periods, filter_high_only=True
    )
    filtered = df_no_dust.loc[mask_valid].reset_index(drop=True)
    removed_outliers = len(df_no_dust) - len(filtered)

    if removed_dust > 0 or removed_outliers > 0:
        logger.info(
            "Volume filter (causal): removed %d dust trades, %d outliers (threshold=%.1f MADs)",
            removed_dust,
            removed_outliers,
            threshold,
        )

    return filtered, removed_outliers, removed_dust


def filter_outliers_robust(
    df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "amount",
) -> tuple[pd.DataFrame, OutlierReport]:
    """Apply robust outlier detection methods in sequence (all causal)."""
    report = OutlierReport(total_ticks=len(df))

    if df.empty:
        report.final_ticks = 0
        return df, report

    price_threshold = OUTLIER_MAD_THRESHOLD * OUTLIER_ROBUST_MAD_MULTIPLIER

    # 1. Volume outliers first (dust-only removal; MAD skipped)
    df, removed_vol, removed_dust = _filter_volume_outliers(
        df,
        volume_col=volume_col,
        threshold=OUTLIER_VOLUME_MAD_THRESHOLD,
        apply_mad=False,
    )
    report.removed_volume_outliers = removed_vol
    report.removed_dust_trades = removed_dust

    # 2. MAD-based price outliers (expanding, causal)
    df, removed = _filter_mad_price_outliers(
        df,
        price_col=price_col,
        threshold=price_threshold,
    )
    report.removed_mad_price = removed

    report.final_ticks = len(df)
    return df, report
