"""VPIN (Volume-Synchronized Probability of Informed Trading).

VPIN is an adaptation of the PIN model (Easley & O'Hara) for high-frequency data.
It measures order flow toxicity by computing the average absolute imbalance
over a rolling window of volume buckets:

    VPIN = Σ|V_buy_i - V_sell_i| / Σ(V_buy_i + V_sell_i)

A high VPIN indicates elevated probability of informed trading,
which has been associated with market stress events (e.g., Flash Crash 2010).

Reference:
    Easley, D., Lopez de Prado, M., & O'Hara, M. (2012).
    The Volume Clock: Insights into the High-Frequency Paradigm.
    Journal of Portfolio Management, 39(1), 19-29.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.utils import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = ["compute_vpin"]


@njit(cache=True)
def _compute_vpin_rolling(
    v_buy: NDArray[np.float64],
    v_sell: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute VPIN using rolling window (numba optimized).

    VPIN = sum(|V_buy_i - V_sell_i|) / sum(V_buy_i + V_sell_i)

    over a rolling window of n buckets.

    Args:
        v_buy: Array of buy volumes per bucket.
        v_sell: Array of sell volumes per bucket.
        window: Rolling window size (number of buckets).

    Returns:
        Array of VPIN values (NaN for first window-1 elements).
    """
    n = len(v_buy)
    vpin = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return vpin

    # Initialize first window
    sum_abs_imbalance = 0.0
    sum_total_volume = 0.0

    for i in range(window):
        sum_abs_imbalance += abs(v_buy[i] - v_sell[i])
        sum_total_volume += v_buy[i] + v_sell[i]

    if sum_total_volume > 0:
        vpin[window - 1] = sum_abs_imbalance / sum_total_volume

    # Rolling computation
    for i in range(window, n):
        # Remove oldest bucket
        old_idx = i - window
        sum_abs_imbalance -= abs(v_buy[old_idx] - v_sell[old_idx])
        sum_total_volume -= v_buy[old_idx] + v_sell[old_idx]

        # Add newest bucket
        sum_abs_imbalance += abs(v_buy[i] - v_sell[i])
        sum_total_volume += v_buy[i] + v_sell[i]

        if sum_total_volume > 0:
            vpin[i] = sum_abs_imbalance / sum_total_volume

    return vpin


def compute_vpin(
    df_bars: pd.DataFrame,
    window: int = 50,
    v_buy_col: str = "v_buy",
    v_sell_col: str = "v_sell",
) -> pd.Series:
    """Compute VPIN (Volume-Synchronized Probability of Informed Trading).

    VPIN measures order flow toxicity by computing the average absolute
    imbalance over a rolling window of volume buckets:

        VPIN = sum(|V_buy_i - V_sell_i|) / sum(V_buy_i + V_sell_i)

    Args:
        df_bars: DataFrame with volume imbalance data (must have v_buy, v_sell).
        window: Rolling window size in number of buckets (default: 50).
        v_buy_col: Name of buy volume column.
        v_sell_col: Name of sell volume column.

    Returns:
        Series with VPIN values (NaN for first window-1 rows).

    Example:
        >>> df_bars = compute_volume_imbalance_bars(df_ticks, df_bars)
        >>> df_bars["vpin"] = compute_vpin(df_bars, window=50)
    """
    v_buy = df_bars[v_buy_col].values.astype(np.float64)
    v_sell = df_bars[v_sell_col].values.astype(np.float64)

    vpin = _compute_vpin_rolling(v_buy, v_sell, window)

    logger.info(
        "VPIN (window=%d) stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        window,
        np.nanmean(vpin),
        np.nanstd(vpin),
        np.nanmin(vpin),
        np.nanmax(vpin),
    )

    return pd.Series(vpin, index=df_bars.index, name=f"vpin_{window}")
