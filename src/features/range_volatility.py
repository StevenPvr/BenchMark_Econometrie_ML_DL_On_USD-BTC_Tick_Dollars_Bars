"""Range-based volatility estimators.

This module computes volatility estimators using intrabar price information
(OHLC data), which are more efficient than close-to-close estimators.

1. Parkinson (1980):
   σ²_Park = (1 / 4·ln(2)) · (ln(H/L))²
   - Uses high-low range only
   - 5x more efficient than close-to-close

2. Garman-Klass (1980):
   σ²_GK = 0.5·(ln(H/L))² - (2·ln(2) - 1)·(ln(C/O))²
   - Uses OHLC data
   - More efficient than Parkinson

3. Rogers-Satchell (1991):
   σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)
   - Handles drift (trending markets)
   - Unbiased for non-zero mean returns

4. Yang-Zhang (2000):
   σ²_YZ = σ²_overnight + k·σ²_open + (1-k)·σ²_RS
   - Combines overnight, open-to-close, and RS
   - Handles both drift and opening jumps

5. Simple ratios:
   - Range ratio: (H - L) / C
   - Body ratio: |C - O| / (H - L)

Reference:
    Parkinson, M. (1980). The Extreme Value Method for Estimating
    the Variance of the Rate of Return. Journal of Business.

    Yang, D., & Zhang, Q. (2000). Drift Independent Volatility Estimation
    Based on High, Low, Open, and Close Prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.constants import EPS
from src.utils import get_logger
from src.features.range_volatility_core import (
    _compute_range_ratios,
    _rolling_garman_klass,
    _rolling_parkinson,
    _rolling_rogers_satchell,
    _rolling_yang_zhang,
)

logger = get_logger(__name__)

__all__ = [
    "compute_parkinson_volatility",
    "compute_garman_klass_volatility",
    "compute_rogers_satchell_volatility",
    "compute_yang_zhang_volatility",
    "compute_range_ratios",
    # Aliases for backward compatibility
    "compute_parkinson",
    "compute_garman_klass",
    "compute_rogers_satchell",
    "compute_yang_zhang",
    "compute_all_range_volatility",
    "compute_body_ratio",
    "compute_range_ratio",
]


def compute_parkinson_volatility(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Parkinson volatility estimator.

    σ²_Park = (1 / 4·ln(2)) · (ln(H/L))²

    The Parkinson estimator uses the high-low range and is approximately
    5x more efficient than the close-to-close estimator.

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        windows: List of window sizes (default: [5, 10, 20]).

    Returns:
        DataFrame with Parkinson volatility columns.
    """
    if windows is None:
        windows = [5, 10, 20]

    high = df_bars[high_col].values.astype(np.float64)
    low = df_bars[low_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in windows:
        vol = _rolling_parkinson(high, low, k)
        col_name = f"parkinson_vol_{k}"
        result[col_name] = vol

        valid = vol[~np.isnan(vol)]
        if len(valid) > 0:
            logger.info(
                "Parkinson volatility (k=%d) stats: mean=%.6f, std=%.6f",
                k, np.mean(valid), np.std(valid),
            )

    return result


def compute_garman_klass_volatility(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Garman-Klass volatility estimator.

    σ²_GK = 0.5·(ln(H/L))² - (2·ln(2) - 1)·(ln(C/O))²

    More efficient than Parkinson as it uses all OHLC information.

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        open_col: Name of open price column.
        close_col: Name of close price column.
        windows: List of window sizes (default: [5, 10, 20]).

    Returns:
        DataFrame with Garman-Klass volatility columns.
    """
    if windows is None:
        windows = [5, 10, 20]

    high = df_bars[high_col].values.astype(np.float64)
    low = df_bars[low_col].values.astype(np.float64)
    open_ = df_bars[open_col].values.astype(np.float64)
    close = df_bars[close_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in windows:
        vol = _rolling_garman_klass(high, low, open_, close, k)
        col_name = f"garman_klass_vol_{k}"
        result[col_name] = vol

        valid = vol[~np.isnan(vol)]
        if len(valid) > 0:
            logger.info(
                "Garman-Klass volatility (k=%d) stats: mean=%.6f, std=%.6f",
                k, np.mean(valid), np.std(valid),
            )

    return result


def compute_rogers_satchell_volatility(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Rogers-Satchell volatility estimator.

    σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)

    This estimator is unbiased for non-zero mean returns (handles drift).

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        open_col: Name of open price column.
        close_col: Name of close price column.
        windows: List of window sizes (default: [5, 10, 20]).

    Returns:
        DataFrame with Rogers-Satchell volatility columns.
    """
    if windows is None:
        windows = [5, 10, 20]

    high = df_bars[high_col].values.astype(np.float64)
    low = df_bars[low_col].values.astype(np.float64)
    open_ = df_bars[open_col].values.astype(np.float64)
    close = df_bars[close_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in windows:
        vol = _rolling_rogers_satchell(high, low, open_, close, k)
        col_name = f"rogers_satchell_vol_{k}"
        result[col_name] = vol

        valid = vol[~np.isnan(vol)]
        if len(valid) > 0:
            logger.info(
                "Rogers-Satchell volatility (k=%d) stats: mean=%.6f, std=%.6f",
                k, np.mean(valid), np.std(valid),
            )

    return result


def compute_yang_zhang_volatility(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Yang-Zhang volatility estimator.

    σ²_YZ = σ²_overnight + k·σ²_open + (1-k)·σ²_RS

    This is the most efficient estimator that handles both drift and
    opening price jumps. Requires previous close price.

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        open_col: Name of open price column.
        close_col: Name of close price column.
        windows: List of window sizes (default: [5, 10, 20]).

    Returns:
        DataFrame with Yang-Zhang volatility columns.
    """
    if windows is None:
        windows = [5, 10, 20]

    high = df_bars[high_col].values.astype(np.float64)
    low = df_bars[low_col].values.astype(np.float64)
    open_ = df_bars[open_col].values.astype(np.float64)
    close = df_bars[close_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in windows:
        vol = _rolling_yang_zhang(high, low, open_, close, k)
        col_name = f"yang_zhang_vol_{k}"
        result[col_name] = vol

        valid = vol[~np.isnan(vol)]
        if len(valid) > 0:
            logger.info(
                "Yang-Zhang volatility (k=%d) stats: mean=%.6f, std=%.6f",
                k, np.mean(valid), np.std(valid),
            )

    return result


def compute_range_ratios(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute simple range-based ratios.

    1. Range ratio: (H - L) / C
       - Normalized range (relative to price level)
       - Higher values indicate higher intrabar volatility

    2. Body ratio: |C - O| / (H - L)
       - Proportion of range captured by open-close move
       - High value: Directional bar (trending)
       - Low value: Indecisive bar (doji-like)

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        open_col: Name of open price column.
        close_col: Name of close price column.

    Returns:
        DataFrame with range_ratio and body_ratio columns.
    """
    high = df_bars[high_col].values.astype(np.float64)
    low = df_bars[low_col].values.astype(np.float64)
    open_ = df_bars[open_col].values.astype(np.float64)
    close = df_bars[close_col].values.astype(np.float64)

    range_ratio, body_ratio = _compute_range_ratios(high, low, open_, close)

    result = pd.DataFrame(index=df_bars.index)
    result["range_ratio"] = range_ratio
    result["body_ratio"] = body_ratio

    valid_range = range_ratio[~np.isnan(range_ratio)]
    valid_body = body_ratio[~np.isnan(body_ratio)]

    if len(valid_range) > 0:
        logger.info(
            "Range ratio stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
            np.mean(valid_range), np.std(valid_range),
            np.min(valid_range), np.max(valid_range),
        )

    if len(valid_body) > 0:
        logger.info(
            "Body ratio stats: mean=%.4f, std=%.4f (1=directional, 0=doji)",
            np.mean(valid_body), np.std(valid_body),
        )

    return result


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================


def compute_parkinson(
    df_bars: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Alias for compute_parkinson_volatility."""
    return compute_parkinson_volatility(df_bars, windows=windows)


def compute_garman_klass(
    df_bars: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Alias for compute_garman_klass_volatility."""
    return compute_garman_klass_volatility(df_bars, windows=windows)


def compute_rogers_satchell(
    df_bars: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Alias for compute_rogers_satchell_volatility."""
    return compute_rogers_satchell_volatility(df_bars, windows=windows)


def compute_yang_zhang(
    df_bars: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Alias for compute_yang_zhang_volatility."""
    return compute_yang_zhang_volatility(df_bars, windows=windows)


def compute_all_range_volatility(
    df_bars: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute all range-based volatility estimators."""
    results = []
    results.append(compute_parkinson_volatility(df_bars, windows=windows))
    results.append(compute_garman_klass_volatility(df_bars, windows=windows))
    results.append(compute_rogers_satchell_volatility(df_bars, windows=windows))
    results.append(compute_yang_zhang_volatility(df_bars, windows=windows))
    results.append(compute_range_ratios(df_bars))
    return pd.concat(results, axis=1)


def compute_body_ratio(df_bars: pd.DataFrame) -> pd.Series:
    """Compute body ratio only."""
    return pd.Series(compute_range_ratios(df_bars)["body_ratio"])


def compute_range_ratio(df_bars: pd.DataFrame) -> pd.Series:
    """Compute range ratio only."""
    return pd.Series(compute_range_ratios(df_bars)["range_ratio"])
