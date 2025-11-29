"""Realized volatility and risk-adjusted return features.

This module computes volatility-based features from past returns:

1. Realized volatility over rolling window:
   σ_t^{(k)} = √(Σ_{j=1}^{k} r_{t-j}²)

2. Local Sharpe ratio (return/vol):
   R_t^{(k)} / σ_t^{(k)}

Interpretation:
    - High realized vol: Turbulent market regime
    - Low realized vol: Calm market regime
    - High Sharpe: Strong momentum with low risk
    - Low Sharpe: Weak momentum or high risk

Reference:
    Andersen, T. G., & Bollerslev, T. (1998). Answering the Skeptics:
    Yes, Standard Volatility Models do Provide Accurate Forecasts.
    International Economic Review, 39(4), 885-905.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.config_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_realized_volatility",
    "compute_local_sharpe",
]


def _rolling_sum_squares(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling sum of squared returns using pandas (numerically stable).

    Uses pandas rolling which is implemented in C with proper numerical stability.

    Args:
        returns: Array of returns.
        window: Rolling window size.

    Returns:
        Array of rolling sum of squares.
    """
    # Use pandas rolling sum - numerically stable implementation
    series = pd.Series(returns ** 2)
    result = series.rolling(window=window, min_periods=window).sum().values
    return result


def _rolling_sum(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling sum using pandas (numerically stable).

    Args:
        returns: Array of returns.
        window: Rolling window size.

    Returns:
        Array of rolling sums.
    """
    series = pd.Series(returns)
    result = series.rolling(window=window, min_periods=window).sum().values
    return result


def compute_realized_volatility(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute realized volatility over multiple horizons.

    σ_t^{(k)} = √(Σ_{j=1}^{k} r_{t-j}²)

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [5, 10, 20, 50]).

    Returns:
        DataFrame with realized volatility columns for each horizon.

    Example:
        >>> df_vol = compute_realized_volatility(df_bars)
        >>> df_bars = pd.concat([df_bars, df_vol], axis=1)
    """
    if horizons is None:
        horizons = [5, 10, 20, 50]

    returns = df_bars[return_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in horizons:
        sum_sq = _rolling_sum_squares(returns, k)
        realized_vol = np.sqrt(sum_sq)

        col_name = f"realized_vol_{k}"
        result[col_name] = realized_vol

        # Log statistics
        valid = realized_vol[~np.isnan(realized_vol)]
        if len(valid) > 0:
            logger.info(
                "Realized volatility (k=%d) stats: mean=%.4f, std=%.4f, "
                "min=%.4f, max=%.4f",
                k,
                np.mean(valid),
                np.std(valid),
                np.min(valid),
                np.max(valid),
            )

    return result


def compute_local_sharpe(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute local Sharpe ratio (return / volatility).

    Sharpe_t^{(k)} = R_t^{(k)} / σ_t^{(k)}

    Where:
        R_t^{(k)} = Σ_{j=0}^{k-1} r_{t-j} (cumulative return)
        σ_t^{(k)} = √(Σ_{j=1}^{k} r_{t-j}²) (realized volatility)

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [5, 10, 20]).

    Returns:
        DataFrame with local Sharpe ratio columns for each horizon.

    Example:
        >>> df_sharpe = compute_local_sharpe(df_bars)
        >>> df_bars = pd.concat([df_bars, df_sharpe], axis=1)
    """
    if horizons is None:
        horizons = [5, 10, 20]

    returns = df_bars[return_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in horizons:
        # Cumulative return
        cum_ret = _rolling_sum(returns, k)

        # Realized volatility
        sum_sq = _rolling_sum_squares(returns, k)
        realized_vol = np.sqrt(sum_sq)

        # Local Sharpe (0 when vol is ~0, as there's no risk-adjusted return)
        sharpe = np.where(
            realized_vol > 1e-10,
            cum_ret / realized_vol,
            0.0,  # No vol = no risk-adjusted return
        )

        col_name = f"local_sharpe_{k}"
        result[col_name] = sharpe

        # Log statistics
        valid = sharpe[~np.isnan(sharpe)]
        if len(valid) > 0:
            logger.info(
                "Local Sharpe (k=%d) stats: mean=%.4f, std=%.4f, "
                "min=%.4f, max=%.4f",
                k,
                np.mean(valid),
                np.std(valid),
                np.min(valid),
                np.max(valid),
            )

    return result
