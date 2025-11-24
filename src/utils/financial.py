"""Financial utilities for the project.

Provides functions for computing financial metrics and weights.
Used in data preparation and conversion pipelines for weighted aggregations.
"""

from __future__ import annotations

import pandas as pd

from src.config_logging import get_logger

__all__ = [
    "compute_rolling_liquidity_weights",
]


def compute_rolling_liquidity_weights(
    df: pd.DataFrame,
    *,
    volume_col: str = "volume",
    price_col: str = "close",
    date_col: str = "date",
    ticker_col: str = "ticker",
    window: int | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute time-varying liquidity weights without look-ahead bias.

    Uses a trailing rolling mean of volume * price per ticker, shifted by one day
    to avoid using same-day information. Returns unnormalized liquidity scores;
    normalization should be applied at aggregation stage.

    This function prevents look-ahead bias by ensuring weights at date t only use
    information up to t-1.

    Args:
        df: DataFrame with ticker time series data (sorted by ticker, date).
        volume_col: Name of volume column. Default is 'volume'.
        price_col: Name of price column. Default is 'close'.
        date_col: Name of date column. Default is 'date'.
        ticker_col: Name of ticker column. Default is 'ticker'.
        window: Trailing window size in days for rolling mean. If None, uses
            LIQUIDITY_WEIGHTS_WINDOW_DEFAULT (20).
        min_periods: Minimum observations in window. Defaults to window if None.

    Returns:
        DataFrame with columns [date, ticker, weight] where weight is a trailing
        liquidity score using only information up to t-1.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.

    Examples:
        Standard liquidity weights:
        >>> weights = compute_rolling_liquidity_weights(
        ...     df,
        ...     volume_col="volume",
        ...     price_col="close",
        ...     window=20
        ... )

        Custom window:
        >>> weights = compute_rolling_liquidity_weights(
        ...     df,
        ...     window=60,
        ...     min_periods=30
        ... )

    Usage in project:
        - Replaces src/data_conversion/data_conversion.py:compute_liquidity_weights_timevarying
        - Used in weighted log returns computation for GARCH models
        - Ensures no look-ahead bias in weight computation
    """
    from src.utils.validation import validate_dataframe_not_empty, validate_required_columns

    required_cols = {date_col, ticker_col, volume_col, price_col}
    validate_required_columns(df, required_cols, "Rolling liquidity weights")
    validate_dataframe_not_empty(df, "Input")

    logger = get_logger(__name__)
    logger.info("Computing time-varying liquidity weights (no look-ahead)")

    df = df.sort_values([ticker_col, date_col]).copy()

    # Filter valid liquidity data (positive volume and price)
    df_valid = df.dropna(subset=[volume_col, price_col]).copy()
    mask = (df_valid[volume_col] > 0) & (df_valid[price_col] > 0)
    df_valid = df_valid.loc[mask].copy()

    # Compute liquidity score
    df_valid["liquidity_score"] = df_valid[volume_col] * df_valid[price_col]

    if window is None:
        from src.constants import LIQUIDITY_WEIGHTS_WINDOW_DEFAULT

        window = LIQUIDITY_WEIGHTS_WINDOW_DEFAULT

    mp = window if min_periods is None else min_periods

    # Compute trailing rolling mean
    df_valid["liquidity_score"] = df_valid.groupby(ticker_col)["liquidity_score"].transform(
        lambda s: s.rolling(window, min_periods=mp).mean()
    )

    # CRITICAL: shift by 1 to enforce causality
    # Weights at date t only use information up to t-1
    df_valid["weight"] = df_valid.groupby(ticker_col)["liquidity_score"].shift(1)

    # Drop rows with missing weights
    df_with_weights = df_valid.dropna(subset=["weight"])

    # Return weights DataFrame
    return pd.DataFrame(df_with_weights[[date_col, ticker_col, "weight"]].reset_index(drop=True))
