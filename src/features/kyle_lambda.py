"""Kyle's Lambda (Price Impact Coefficient).

Kyle's Lambda measures the permanent price impact per unit of signed volume,
estimated via rolling OLS regression:

    ΔP_t = λ · SignedVolume_t + ε_t

Where:
    - ΔP_t = P_t - P_{t-1} (price change)
    - SignedVolume_t = V_buy - V_sell (order flow imbalance)
    - λ = Kyle's Lambda (price impact coefficient)

A higher λ indicates:
    - Lower liquidity
    - Higher information asymmetry
    - Greater price impact per unit of volume

Reference:
    Kyle, A. S. (1985). Continuous Auctions and Insider Trading.
    Econometrica, 53(6), 1315-1335.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.constants import EPS
from src.utils import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = ["compute_kyle_lambda"]

# NOTE: Numba functions use literal values equivalent to EPS (1e-10)
# because Numba cannot import Python constants at compile time.


@njit(cache=True)
def _ols_slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Compute OLS slope coefficient (numba optimized).

    β = Cov(x, y) / Var(x) = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²

    Args:
        x: Independent variable array.
        y: Dependent variable array.

    Returns:
        OLS slope coefficient (NaN if insufficient variance).
    """
    n = len(x)
    if n < 2:
        return np.nan

    # Compute means
    x_mean = 0.0
    y_mean = 0.0
    for i in range(n):
        x_mean += x[i]
        y_mean += y[i]
    x_mean /= n
    y_mean /= n

    # Compute covariance and variance
    cov_xy = 0.0
    var_x = 0.0
    for i in range(n):
        dx = x[i] - x_mean
        cov_xy += dx * (y[i] - y_mean)
        var_x += dx * dx

    if var_x < 1e-10:
        return np.nan

    return cov_xy / var_x


@njit(cache=True)
def _compute_kyle_lambda_rolling(
    delta_price: NDArray[np.float64],
    signed_volume: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute Kyle's Lambda using rolling OLS regression (numba optimized).

    ΔP_t = λ · SignedVolume_t + ε_t

    Args:
        delta_price: Array of price changes (ΔP).
        signed_volume: Array of signed volumes (V_buy - V_sell).
        window: Rolling window size.

    Returns:
        Array of Kyle's Lambda values (NaN for first window-1 elements).
    """
    n = len(delta_price)
    kyle_lambda = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return kyle_lambda

    # Rolling OLS
    for i in range(window - 1, n):
        start_idx = i - window + 1
        x = signed_volume[start_idx : i + 1]
        y = delta_price[start_idx : i + 1]
        kyle_lambda[i] = _ols_slope(x, y)

    return kyle_lambda


def compute_kyle_lambda(
    df_bars: pd.DataFrame,
    window: int = 50,
    price_col: str = "close",
    v_buy_col: str = "v_buy",
    v_sell_col: str = "v_sell",
) -> pd.Series:
    """Compute Kyle's Lambda (price impact coefficient).

    Kyle's Lambda measures the permanent price impact per unit of signed volume,
    estimated via rolling OLS regression:

        ΔP_t = λ · SignedVolume_t + ε_t

    Args:
        df_bars: DataFrame with price and volume imbalance data.
        window: Rolling window size for OLS estimation (default: 50).
        price_col: Name of price column.
        v_buy_col: Name of buy volume column.
        v_sell_col: Name of sell volume column.

    Returns:
        Series with Kyle's Lambda values (NaN for first window rows).

    Example:
        >>> df_bars = compute_volume_imbalance_bars(df_ticks, df_bars)
        >>> df_bars["kyle_lambda"] = compute_kyle_lambda(df_bars, window=50)
    """
    # Compute price changes
    prices = df_bars[price_col].values.astype(np.float64)
    delta_price = np.diff(prices, prepend=np.nan)

    # Compute signed volume (order flow)
    v_buy = df_bars[v_buy_col].values.astype(np.float64)
    v_sell = df_bars[v_sell_col].values.astype(np.float64)
    signed_volume = v_buy - v_sell

    # Compute Kyle's Lambda via rolling OLS
    kyle_lambda = _compute_kyle_lambda_rolling(delta_price, signed_volume, window)

    # Log statistics (excluding NaN)
    valid_lambda = kyle_lambda[~np.isnan(kyle_lambda)]
    if len(valid_lambda) > 0:
        logger.info(
            "Kyle's Lambda (window=%d) stats: mean=%.2e, std=%.2e, min=%.2e, max=%.2e",
            window,
            np.mean(valid_lambda),
            np.std(valid_lambda),
            np.min(valid_lambda),
            np.max(valid_lambda),
        )

    return pd.Series(kyle_lambda, index=df_bars.index, name=f"kyle_lambda_{window}")
