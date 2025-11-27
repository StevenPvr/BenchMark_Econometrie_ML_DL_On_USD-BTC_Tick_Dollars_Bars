"""Triple-barrier method for event labeling.

This module implements the triple-barrier method from De Prado (2018)
for generating labels based on price movements hitting barriers.

The three barriers are:
    1. Upper (Profit-Taking): Price rises above entry + volatility * pt_mult
    2. Lower (Stop-Loss): Price falls below entry - volatility * sl_mult
    3. Vertical (Time): Maximum holding period elapsed

Reference:
    De Prado, M. L. (2018). Advances in Financial Machine Learning.
    John Wiley & Sons. Chapter 3.
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
    "get_triple_barrier_events",
    "get_vertical_barriers",
    "apply_triple_barrier_labels",
]


# =============================================================================
# VERTICAL BARRIER (TIME LIMIT)
# =============================================================================


def get_vertical_barriers(
    t_events: pd.DatetimeIndex | pd.Index,
    max_holding_period: int,
    n_bars: int,
) -> pd.Series:
    """Get vertical barrier timestamps (time limit).

    Args:
        t_events: Index of event start times.
        max_holding_period: Maximum number of bars to hold.
        n_bars: Total number of bars in the dataset.

    Returns:
        Series with vertical barrier indices for each event.
    """
    # For integer index, add max_holding_period
    if isinstance(t_events, pd.RangeIndex) or t_events.dtype in (
        np.int64,
        np.int32,
        int,
    ):
        # Convert to numpy for clipping
        vertical_barriers = np.array(t_events) + max_holding_period
        # Clip to max index
        vertical_barriers = np.clip(vertical_barriers, 0, n_bars - 1)
        return pd.Series(vertical_barriers, index=t_events)

    # For datetime index, shift by max_holding_period bars
    # We'll return the index position instead
    positions = np.arange(len(t_events))
    vertical_barriers = positions + max_holding_period
    vertical_barriers = np.clip(vertical_barriers, 0, n_bars - 1)
    return pd.Series(vertical_barriers, index=t_events)


# =============================================================================
# NUMBA-OPTIMIZED BARRIER DETECTION
# =============================================================================


@njit(cache=True)
def _find_first_barrier_touch(
    prices: NDArray[np.float64],
    entry_idx: int,
    vertical_barrier_idx: int,
    upper_barrier: float,
    lower_barrier: float,
) -> tuple[int, int, float]:
    """Find the first barrier touched (Numba optimized).

    Args:
        prices: Array of prices (close).
        entry_idx: Index of trade entry.
        vertical_barrier_idx: Index of vertical barrier (time limit).
        upper_barrier: Upper barrier price (profit-taking).
        lower_barrier: Lower barrier price (stop-loss).

    Returns:
        Tuple of (exit_idx, barrier_type, return):
            - exit_idx: Index when barrier was touched
            - barrier_type: 1 (upper/PT), -1 (lower/SL), 0 (vertical/time)
            - return: Log return from entry to exit
    """
    entry_price = prices[entry_idx]

    # Search from entry+1 to vertical barrier (inclusive)
    for i in range(entry_idx + 1, vertical_barrier_idx + 1):
        price = prices[i]

        # Check upper barrier (profit-taking)
        if price >= upper_barrier:
            ret = np.log(price / entry_price)
            return i, 1, ret

        # Check lower barrier (stop-loss)
        if price <= lower_barrier:
            ret = np.log(price / entry_price)
            return i, -1, ret

    # Vertical barrier reached (time limit)
    exit_price = prices[vertical_barrier_idx]
    ret = np.log(exit_price / entry_price)
    return vertical_barrier_idx, 0, ret


@njit(cache=True)
def _compute_all_barriers(
    prices: NDArray[np.float64],
    event_indices: NDArray[np.int64],
    vertical_barriers: NDArray[np.int64],
    volatilities: NDArray[np.float64],
    pt_mult: float,
    sl_mult: float,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
]:
    """Compute barrier touches for all events (Numba optimized).

    Args:
        prices: Array of prices (close).
        event_indices: Array of event start indices.
        vertical_barriers: Array of vertical barrier indices.
        volatilities: Array of volatilities at each event.
        pt_mult: Profit-taking multiplier.
        sl_mult: Stop-loss multiplier.

    Returns:
        Tuple of arrays (exit_indices, barrier_types, returns).
    """
    n_events = len(event_indices)
    exit_indices = np.empty(n_events, dtype=np.int64)
    barrier_types = np.empty(n_events, dtype=np.int64)
    returns = np.empty(n_events, dtype=np.float64)

    for i in range(n_events):
        entry_idx = event_indices[i]
        vertical_idx = vertical_barriers[i]
        vol = volatilities[i]
        entry_price = prices[entry_idx]

        # Skip if volatility is NaN or zero
        if np.isnan(vol) or vol <= 0:
            exit_indices[i] = vertical_idx
            barrier_types[i] = 0
            returns[i] = np.nan
            continue

        # Calculate barriers
        upper_barrier = entry_price * (1.0 + pt_mult * vol)
        lower_barrier = entry_price * (1.0 - sl_mult * vol)

        # Find first barrier touch
        exit_idx, barrier_type, ret = _find_first_barrier_touch(
            prices, entry_idx, vertical_idx, upper_barrier, lower_barrier
        )

        exit_indices[i] = exit_idx
        barrier_types[i] = barrier_type
        returns[i] = ret

    return exit_indices, barrier_types, returns


# =============================================================================
# MAIN TRIPLE-BARRIER FUNCTION
# =============================================================================


def get_triple_barrier_events(
    prices: pd.Series,
    t_events: pd.DatetimeIndex | pd.Index | None = None,
    pt_sl: list[float] | tuple[float, float] | None = None,
    target_volatility: pd.Series | None = None,
    max_holding_period: int = 20,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Calculate triple barrier events (De Prado methodology).

    This function identifies when prices hit one of three barriers:
    1. Upper (Profit-Taking): price >= entry * (1 + pt_mult * volatility)
    2. Lower (Stop-Loss): price <= entry * (1 - sl_mult * volatility)
    3. Vertical (Time): max_holding_period bars elapsed

    Args:
        prices: Series of prices (typically close prices).
        t_events: Index of event start times. If None, uses all indices.
        pt_sl: [profit_taking_mult, stop_loss_mult] in volatility units.
               Default is [1.0, 1.0] (symmetric).
        target_volatility: Rolling volatility estimate at each bar.
                          If None, uses rolling std of log returns.
        max_holding_period: Maximum bars to hold position.
        min_ret: Minimum return threshold (not used for labeling).

    Returns:
        DataFrame with columns:
            - t_start: Event start index
            - t_end: Time first barrier was touched
            - ret: Log return at t_end
            - barrier: Which barrier (1=PT, -1=SL, 0=Vertical)
            - label: Direction label based on sign of return

    Example:
        >>> prices = df_bars["close"]
        >>> volatility = compute_realized_volatility(df_bars)["realized_vol_20"]
        >>> events = get_triple_barrier_events(
        ...     prices=prices,
        ...     pt_sl=[1.0, 1.0],  # Symmetric barriers
        ...     target_volatility=volatility,
        ...     max_holding_period=20,
        ... )
    """
    # Default pt_sl
    if pt_sl is None:
        pt_sl = [1.0, 1.0]
    pt_mult, sl_mult = pt_sl[0], pt_sl[1]

    # Default events: all indices
    if t_events is None:
        t_events = prices.index

    # Ensure prices is aligned with events
    prices = prices.loc[prices.index.isin(t_events) | True]  # Keep all prices

    # Convert to numpy for Numba
    prices_arr = prices.values.astype(np.float64)
    n_bars = len(prices_arr)

    # Get event indices (position in prices array)
    if isinstance(prices.index, pd.DatetimeIndex):
        # For datetime index, get positions
        event_positions = prices.index.get_indexer(t_events)
        event_positions = event_positions[event_positions >= 0]
    else:
        # For integer index, use directly
        event_positions = t_events.values.astype(np.int64)
        event_positions = event_positions[event_positions < n_bars]

    n_events = len(event_positions)
    logger.info(
        "Computing triple-barrier events for %d events (pt=%.2f, sl=%.2f, max_hold=%d)",
        n_events,
        pt_mult,
        sl_mult,
        max_holding_period,
    )

    # Compute volatility if not provided
    if target_volatility is None:
        logger.info("Computing default volatility (rolling std of log returns)")
        log_returns = np.diff(np.log(prices_arr))
        log_returns = np.concatenate([[np.nan], log_returns])

        # Rolling std with window=20
        vol_window = min(20, n_bars // 4)
        volatility = pd.Series(log_returns).rolling(vol_window).std().values
    else:
        # Align volatility with prices
        volatility = target_volatility.reindex(prices.index).values.astype(np.float64)

    # Get volatility at event times
    event_volatilities = volatility[event_positions].astype(np.float64)

    # Compute vertical barriers
    vertical_barriers = np.minimum(
        event_positions + max_holding_period, n_bars - 1
    ).astype(np.int64)

    # Compute all barriers using Numba
    exit_indices, barrier_types, returns = _compute_all_barriers(
        prices_arr,
        event_positions.astype(np.int64),
        vertical_barriers,
        event_volatilities,
        pt_mult,
        sl_mult,
    )

    # Create result DataFrame
    result = pd.DataFrame(
        {
            "t_start": event_positions,
            "t_end": exit_indices,
            "ret": returns,
            "barrier": barrier_types,
        }
    )

    # Generate labels based on return sign
    # Label: 1 if ret > min_ret, -1 if ret < -min_ret, 0 otherwise
    result["label"] = np.where(
        result["ret"] > min_ret,
        1,
        np.where(result["ret"] < -min_ret, -1, 0),
    )

    # Filter out NaN returns
    valid_mask = ~np.isnan(result["ret"])
    result = pd.DataFrame(result[valid_mask].copy().reset_index(drop=True))

    # Log statistics
    n_pt = (result["barrier"] == 1).sum()
    n_sl = (result["barrier"] == -1).sum()
    n_vert = (result["barrier"] == 0).sum()
    n_up = (result["label"] == 1).sum()
    n_down = (result["label"] == -1).sum()
    n_neutral = (result["label"] == 0).sum()

    logger.info(
        "Barrier touches: PT=%d (%.1f%%), SL=%d (%.1f%%), Vertical=%d (%.1f%%)",
        n_pt,
        100 * n_pt / len(result) if len(result) > 0 else 0,
        n_sl,
        100 * n_sl / len(result) if len(result) > 0 else 0,
        n_vert,
        100 * n_vert / len(result) if len(result) > 0 else 0,
    )
    logger.info(
        "Labels: Up=%d (%.1f%%), Down=%d (%.1f%%), Neutral=%d (%.1f%%)",
        n_up,
        100 * n_up / len(result) if len(result) > 0 else 0,
        n_down,
        100 * n_down / len(result) if len(result) > 0 else 0,
        n_neutral,
        100 * n_neutral / len(result) if len(result) > 0 else 0,
    )

    return result


def apply_triple_barrier_labels(
    df: pd.DataFrame,
    price_col: str = "close",
    volatility_col: str | None = None,
    pt_sl: list[float] | tuple[float, float] | None = None,
    max_holding_period: int = 20,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Apply triple-barrier labels to a DataFrame.

    This is a convenience function that adds a 'label' column to the DataFrame.

    Args:
        df: DataFrame with price data.
        price_col: Name of price column (default: "close").
        volatility_col: Name of volatility column. If None, computes from returns.
        pt_sl: [profit_taking_mult, stop_loss_mult].
        max_holding_period: Maximum bars to hold.
        min_ret: Minimum return threshold.

    Returns:
        DataFrame with added 'label' column (-1, 0, 1).
    """
    prices = pd.Series(df[price_col].copy())

    # Get volatility
    if volatility_col is not None and volatility_col in df.columns:
        volatility = pd.Series(df[volatility_col].copy())
    else:
        volatility = None

    # Compute events
    events = get_triple_barrier_events(
        prices=prices,
        t_events=None,  # Use all indices
        pt_sl=pt_sl,
        target_volatility=volatility,
        max_holding_period=max_holding_period,
        min_ret=min_ret,
    )

    # Map labels back to original DataFrame
    result = df.copy()
    result["label"] = np.nan

    # Set labels at event start times
    for _, row in events.iterrows():
        t_start = int(row["t_start"])
        if t_start < len(result):
            result.iloc[t_start, result.columns.get_loc("label")] = row["label"]

    # Forward-fill NaN labels (optional: can be removed if sparse labels are desired)
    # result["label"] = result["label"].ffill()

    # Convert to int where not NaN
    result["label"] = result["label"].astype("Int64")  # Nullable int

    return result
