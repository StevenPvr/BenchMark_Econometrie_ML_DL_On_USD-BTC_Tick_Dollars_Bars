"""Computation functions for data preparation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.constants import (
    LIGHTGBM_REALIZED_VOL_WINDOW,
    REQUIRED_COLS_CLOSE_PRICE,
    REQUIRED_COLS_LOG_RETURN,
)
from src.utils import get_logger, validate_required_columns

logger = get_logger(__name__)


def compute_log_returns_for_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns per ticker from close prices with comprehensive validation.

    TEMPORAL CORRECTNESS:
    Log return at time t uses ONLY prices from t and t-1, ensuring no look-ahead bias.
    Formula: log_return[t] = log(close[t] / close[t-1])

    Args:
        df: DataFrame with ticker, date, and close columns.

    Returns:
        DataFrame with log_return column added, rows with NaN removed.

    Raises:
        ValueError: If required columns are missing or prices are invalid.
    """
    validate_required_columns(df, REQUIRED_COLS_CLOSE_PRICE)

    # Validate price positivity across all tickers
    n_total = len(df)
    n_non_positive = (df["close"] <= 0).sum()
    if n_non_positive > 0:
        pct_non_positive = 100 * n_non_positive / n_total
        logger.warning(
            f"Found {n_non_positive} non-positive close prices ({pct_non_positive:.2f}%), "
            "will result in NaN log returns"
        )

    # Compute log returns per ticker
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Return at date t uses close price at t and t-1 (no future info)
    def _safe_log_return(x: pd.Series) -> pd.Series:
        x = x.astype(float)

        # Validate price positivity within group
        if (x <= 0).any():
            n_invalid = (x <= 0).sum()
            logger.debug(f"Group has {n_invalid} non-positive prices")

        prev = x.shift(1)

        # Check for zero previous prices before division
        if (prev == 0).any():
            n_zero = (prev == 0).sum()
            logger.warning(f"Group has {n_zero} zero prices in previous period")

        ratio = x / prev
        # Clean problematic ratios before log
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        ratio = ratio.mask(ratio <= 0)  # Zero or negative ratios → NaN

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = cast(pd.Series, np.log(ratio))

        return log_ret

    df["log_return"] = df.groupby("ticker")["close"].transform(_safe_log_return)

    # Log NaN statistics before removal
    n_nan = df["log_return"].isna().sum()
    if n_nan > 0:
        pct_nan = 100 * n_nan / n_total
        logger.info(
            f"Log return calculation produced {n_nan} NaN values ({pct_nan:.2f}% of data). "
            "This includes first observation per ticker (expected)."
        )

    # Remove rows with NaN log_return (first observation per ticker + invalid prices)
    n_before = len(df)
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    n_after = len(df)
    n_removed = n_before - n_after

    logger.info(
        f"Computed log_return per ticker. Removed {n_removed} rows with NaN log returns "
        f"({n_before} → {n_after} observations)"
    )
    return df


def _compute_volatility_per_group(log_ret: pd.Series) -> pd.Series:
    """Compute volatility for a group of log returns.

    Args:
        log_ret: Series of log returns for a single ticker.

    Returns:
        Series of log volatility values.
    """
    ret_sq = cast(pd.Series, log_ret**2)

    # Trailing rolling window (no look-ahead)
    realized_vol = cast(
        pd.Series,
        np.sqrt(
            ret_sq.rolling(
                LIGHTGBM_REALIZED_VOL_WINDOW,
                min_periods=LIGHTGBM_REALIZED_VOL_WINDOW,
            ).sum(),
        ),
    )
    return cast(pd.Series, np.log1p(realized_vol.astype(float)))


def compute_volatility_for_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log volatility per ticker from log returns.

    Calculates log_volatility as log1p of sqrt of LIGHTGBM_REALIZED_VOL_WINDOW-day
    sum of squared returns. This column is the RF target.

    Args:
        df: DataFrame with ticker, date, and log_return columns.
            Must be sorted by (ticker, date).

    Returns:
        DataFrame with log_volatility column added.

    Raises:
        ValueError: If required columns are missing.
    """
    validate_required_columns(df, REQUIRED_COLS_LOG_RETURN)

    # Ensure data is sorted by ticker and date
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Apply per ticker without accessing grouping columns
    df = df.assign(
        log_volatility=df.groupby("ticker")["log_return"].transform(_compute_volatility_per_group)
    )

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info("Computed log_volatility per ticker")
    return df


def compute_log_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log_volume from volume column if present, with validation.

    Uses log1p (log(1 + x)) to handle zero volumes gracefully.
    Validates that volume is non-negative before transformation.

    Args:
        df: DataFrame with volume column.

    Returns:
        DataFrame with log_volume column added if volume exists.

    Raises:
        ValueError: If volume contains negative values (data quality issue).
    """
    if "volume" in df.columns and "log_volume" not in df.columns:
        # Validate volume range
        n_negative = (df["volume"] < 0).sum()
        if n_negative > 0:
            pct_negative = 100 * n_negative / len(df)
            logger.error(
                f"Found {n_negative} negative volume values ({pct_negative:.2f}%) - "
                "this is a data quality issue!"
            )
            # Replace negative volumes with zero (conservative approach)
            df.loc[df["volume"] < 0, "volume"] = 0
            logger.warning("Replaced negative volumes with zero")

        # Log if many zero-volume observations (may indicate halted trading)
        n_zero = (df["volume"] == 0).sum()
        if n_zero > 0:
            pct_zero = 100 * n_zero / len(df)
            logger.info(
                f"Found {n_zero} zero-volume observations ({pct_zero:.2f}%). "
                "This may indicate halted trading or data quality issues."
            )

        logger.info("Computing log_volume from volume using log1p")
        df["log_volume"] = np.log1p(df["volume"])

    return df
