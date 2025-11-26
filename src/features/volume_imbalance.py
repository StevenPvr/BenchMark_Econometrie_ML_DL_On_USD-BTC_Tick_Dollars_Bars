"""Volume Imbalance (VI) feature computation.

Volume Imbalance measures the order flow toxicity by comparing
buy-initiated vs sell-initiated volume:

    VI = (V_buy - V_sell) / (V_buy + V_sell) ∈ [-1, 1]

A VI close to +1 indicates dominant buying pressure.
A VI close to -1 indicates dominant selling pressure.

This is a proxy for order flow toxicity — the information that market makers
don't see coming.

Reference:
    Easley, D., Lopez de Prado, M., & O'Hara, M. (2012).
    Flow Toxicity and Liquidity in a High-Frequency World.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.features.trade_classification import (
    classify_trades_direct,
    classify_trades_tick_rule,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_volume_imbalance",
    "compute_volume_imbalance_bars",
]


@njit(cache=True)
def _compute_vi(
    volumes: NDArray[np.float64],
    signs: NDArray[np.int8],
) -> tuple[float, float, float]:
    """Compute volume imbalance (numba optimized).

    Args:
        volumes: Array of trade volumes.
        signs: Array of trade signs (+1 buy, -1 sell).

    Returns:
        Tuple of (volume_imbalance, v_buy, v_sell).
    """
    v_buy = 0.0
    v_sell = 0.0

    for i in range(len(volumes)):
        if signs[i] > 0:
            v_buy += volumes[i]
        elif signs[i] < 0:
            v_sell += volumes[i]

    total = v_buy + v_sell
    if total > 0:
        vi = (v_buy - v_sell) / total
    else:
        vi = 0.0

    return vi, v_buy, v_sell


def compute_volume_imbalance(
    df: pd.DataFrame,
    volume_col: str = "amount",
    side_col: str | None = "side",
    price_col: str = "price",
    use_tick_rule: bool = False,
) -> dict[str, float]:
    """Compute volume imbalance for a set of trades.

    Args:
        df: DataFrame with trade data.
        volume_col: Name of volume column.
        side_col: Name of side column (if available).
        price_col: Name of price column (for tick rule).
        use_tick_rule: If True, use tick rule instead of direct side.

    Returns:
        Dictionary with vi, v_buy, v_sell.
    """
    if use_tick_rule or side_col is None or side_col not in df.columns:
        signs = classify_trades_tick_rule(df, price_col).values
    else:
        signs = classify_trades_direct(df, side_col).values

    volumes = df[volume_col].values.astype(np.float64)
    vi, v_buy, v_sell = _compute_vi(volumes, signs.astype(np.int8))

    return {
        "volume_imbalance": vi,
        "v_buy": v_buy,
        "v_sell": v_sell,
    }


@njit(cache=True)
def _aggregate_vi_by_bar(
    bar_ids: NDArray[np.int64],
    volumes: NDArray[np.float64],
    signs: NDArray[np.int8],
    n_bars: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate volume imbalance by bar (numba optimized).

    Args:
        bar_ids: Array of bar IDs for each tick.
        volumes: Array of trade volumes.
        signs: Array of trade signs (+1 buy, -1 sell).
        n_bars: Number of bars.

    Returns:
        Tuple of (vi_array, v_buy_array, v_sell_array).
    """
    v_buy = np.zeros(n_bars, dtype=np.float64)
    v_sell = np.zeros(n_bars, dtype=np.float64)

    for i in range(len(bar_ids)):
        bar_id = bar_ids[i]
        if 0 <= bar_id < n_bars:
            if signs[i] > 0:
                v_buy[bar_id] += volumes[i]
            elif signs[i] < 0:
                v_sell[bar_id] += volumes[i]

    vi = np.zeros(n_bars, dtype=np.float64)
    for j in range(n_bars):
        total = v_buy[j] + v_sell[j]
        if total > 0:
            vi[j] = (v_buy[j] - v_sell[j]) / total

    return vi, v_buy, v_sell


def compute_volume_imbalance_bars(
    df_ticks: pd.DataFrame,
    df_bars: pd.DataFrame,
    volume_col: str = "amount",
    side_col: str | None = "side",
    price_col: str = "price",
    timestamp_col: str = "timestamp",
    bar_timestamp_open: str = "timestamp_open",
    bar_timestamp_close: str = "timestamp_close",
    use_tick_rule: bool = False,
) -> pd.DataFrame:
    """Compute volume imbalance for each dollar bar.

    This function maps each tick to its corresponding bar and computes
    the volume imbalance within each bar.

    Args:
        df_ticks: DataFrame with tick-level trade data.
        df_bars: DataFrame with dollar bars.
        volume_col: Name of volume column in ticks.
        side_col: Name of side column in ticks (if available).
        price_col: Name of price column in ticks.
        timestamp_col: Name of timestamp column in ticks.
        bar_timestamp_open: Name of bar open timestamp column.
        bar_timestamp_close: Name of bar close timestamp column.
        use_tick_rule: If True, use tick rule instead of direct side.

    Returns:
        DataFrame with bars and volume imbalance columns added.
    """
    logger.info(
        "Computing volume imbalance for %d bars from %d ticks",
        len(df_bars),
        len(df_ticks),
    )

    # Classify trades
    if use_tick_rule or side_col is None or side_col not in df_ticks.columns:
        logger.info("Using tick rule for trade classification")
        signs = classify_trades_tick_rule(df_ticks, price_col).values
    else:
        logger.info("Using direct side classification")
        signs = classify_trades_direct(df_ticks, side_col).values

    # Get tick timestamps as int64
    if pd.api.types.is_datetime64_any_dtype(df_ticks[timestamp_col]):
        tick_ts = (df_ticks[timestamp_col].astype("int64") // 10**6).values
    else:
        tick_ts = df_ticks[timestamp_col].values.astype(np.int64)

    # Get bar boundaries
    bar_open = df_bars[bar_timestamp_open].values.astype(np.int64)
    bar_close = df_bars[bar_timestamp_close].values.astype(np.int64)

    # Assign each tick to a bar using searchsorted
    bar_ids = np.searchsorted(bar_close, tick_ts, side="left")
    bar_ids = np.clip(bar_ids, 0, len(df_bars) - 1)

    # Verify assignment
    valid_mask = (tick_ts >= bar_open[bar_ids]) & (tick_ts <= bar_close[bar_ids])
    bar_ids_valid = np.where(valid_mask, bar_ids, -1)

    n_valid = np.sum(valid_mask)
    logger.info(
        "Mapped %d/%d ticks to bars (%.1f%%)",
        n_valid,
        len(tick_ts),
        100 * n_valid / len(tick_ts),
    )

    # Get volumes
    volumes = df_ticks[volume_col].values.astype(np.float64)

    # Aggregate by bar
    vi, v_buy, v_sell = _aggregate_vi_by_bar(
        bar_ids_valid.astype(np.int64),
        volumes,
        signs.astype(np.int8),
        len(df_bars),
    )

    # Add to bars DataFrame
    df_result = df_bars.copy()
    df_result["volume_imbalance"] = vi
    df_result["v_buy"] = v_buy
    df_result["v_sell"] = v_sell

    # Log statistics
    logger.info(
        "Volume imbalance stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        vi.mean(),
        vi.std(),
        vi.min(),
        vi.max(),
    )

    return df_result
