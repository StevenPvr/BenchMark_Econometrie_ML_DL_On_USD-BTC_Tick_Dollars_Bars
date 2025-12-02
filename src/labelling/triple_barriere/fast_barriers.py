"""
Optimized triple-barrier functions using Numba JIT compilation and vectorization.

These functions replace the slow Python loops in the original implementation
with highly optimized NumPy/Numba code for 10-100x speedup.

Key optimizations:
- JIT compilation with Numba for C-level performance
- Parallel execution across events with prange
- Cache compiled functions to avoid recompilation
- Vectorized operations where possible
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange


# =============================================================================
# LOW-LEVEL NUMBA FUNCTIONS
# =============================================================================


@njit(cache=True)
def _find_barrier_touch_numba(
    path_returns: np.ndarray,
    pt_barrier: float,
    sl_barrier: float,
) -> int:
    """
    Find index of first barrier touch in price path.

    Args:
        path_returns: Array of cumulative returns from entry.
        pt_barrier: Profit-taking barrier (positive).
        sl_barrier: Stop-loss barrier (negative).

    Returns:
        Index of first touch, or -1 if no barrier is touched.
    """
    n = len(path_returns)
    for i in range(n):
        ret = path_returns[i]
        # Check profit-taking (positive barrier)
        if pt_barrier > 0 and ret >= pt_barrier:
            return i
        # Check stop-loss (negative barrier)
        if sl_barrier < 0 and ret <= sl_barrier:
            return i
    return -1


@njit(cache=True, parallel=True)
def compute_vertical_barriers_numba(
    event_positions: np.ndarray,
    n_close: int,
    max_holding: int,
) -> np.ndarray:
    """
    Compute vertical barrier positions for all events in parallel.

    Args:
        event_positions: Array of position indices for each event in close.
        n_close: Total length of close price array.
        max_holding: Maximum holding period in bars.

    Returns:
        Array of t1 positions for each event.
    """
    n_events = len(event_positions)
    t1_positions = np.empty(n_events, dtype=np.int64)

    for i in prange(n_events):
        t0_pos = event_positions[i]
        if t0_pos >= 0:
            t1_positions[i] = min(t0_pos + max_holding, n_close - 1)
        else:
            t1_positions[i] = -1  # Invalid

    return t1_positions


@njit(cache=True)
def compute_return_numba(
    price_t0: float,
    price_t1: float,
) -> float:
    """Compute return between two prices."""
    if price_t0 == 0:
        return 0.0
    return (price_t1 - price_t0) / price_t0


@njit(cache=True)
def compute_label_numba(ret: float, min_return: float) -> int:
    """Compute label from return."""
    if abs(ret) < min_return:
        return 0
    elif ret > 0:
        return 1
    else:
        return -1


@njit(cache=True, parallel=True)
def compute_all_returns_and_labels_numba(
    close_values: np.ndarray,
    t0_positions: np.ndarray,
    t1_positions: np.ndarray,
    min_return: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute returns and labels for all events in parallel.

    Args:
        close_values: Array of close prices.
        t0_positions: Array of entry position indices.
        t1_positions: Array of exit position indices.
        min_return: Minimum return threshold for labeling.

    Returns:
        Tuple of (returns array, labels array).
    """
    n_events = len(t0_positions)
    returns = np.empty(n_events, dtype=np.float64)
    labels = np.empty(n_events, dtype=np.int64)

    for i in prange(n_events):
        t0_pos = t0_positions[i]
        t1_pos = t1_positions[i]

        if t0_pos < 0 or t1_pos < 0 or t0_pos >= len(close_values) or t1_pos >= len(close_values):
            returns[i] = np.nan
            labels[i] = 0
            continue

        price_t0 = float(close_values[t0_pos])
        price_t1 = float(close_values[t1_pos])

        ret = compute_return_numba(price_t0, price_t1)
        returns[i] = ret
        labels[i] = compute_label_numba(ret, min_return)

    return returns, labels


@njit(cache=True)
def update_t1_with_barrier_touch_single(
    close_values: np.ndarray,
    t0_pos: int,
    t1_pos: int,
    pt_barrier: float,
    sl_barrier: float,
) -> int:
    """
    Update t1 position based on barrier touch for a single event.

    Args:
        close_values: Array of close prices.
        t0_pos: Entry position index.
        t1_pos: Current exit position index (vertical barrier).
        pt_barrier: Profit-taking barrier level.
        sl_barrier: Stop-loss barrier level.

    Returns:
        Updated t1 position (may be earlier if barrier was touched).
    """
    if t0_pos < 0 or t1_pos < 0:
        return t1_pos
    if t0_pos >= len(close_values) or t1_pos >= len(close_values):
        return t1_pos
    if t1_pos <= t0_pos:
        return t1_pos

    entry_price = close_values[t0_pos]
    if entry_price == 0:
        return t1_pos

    # Scan price path for barrier touch
    for pos in range(t0_pos + 1, t1_pos + 1):
        ret = (close_values[pos] - entry_price) / entry_price

        # Check barriers
        if pt_barrier > 0 and ret >= pt_barrier:
            return pos
        if sl_barrier < 0 and ret <= sl_barrier:
            return pos

    return t1_pos


@njit(cache=True, parallel=True)
def update_all_t1_with_barriers_numba(
    close_values: np.ndarray,
    t0_positions: np.ndarray,
    t1_positions: np.ndarray,
    pt_barriers: np.ndarray,
    sl_barriers: np.ndarray,
) -> np.ndarray:
    """
    Update t1 positions based on barrier touches for all events in parallel.

    Args:
        close_values: Array of close prices.
        t0_positions: Array of entry position indices.
        t1_positions: Array of exit position indices (vertical barriers).
        pt_barriers: Array of profit-taking barriers for each event.
        sl_barriers: Array of stop-loss barriers for each event.

    Returns:
        Updated t1 positions array.
    """
    n_events = len(t0_positions)
    updated_t1 = np.empty(n_events, dtype=np.int64)

    for i in prange(n_events):
        updated_t1[i] = update_t1_with_barrier_touch_single(
            close_values,
            t0_positions[i],
            t1_positions[i],
            pt_barriers[i],
            sl_barriers[i],
        )

    return updated_t1


# =============================================================================
# FULL PIPELINE - SINGLE PARALLEL PASS
# =============================================================================


@njit(cache=True, parallel=True)
def compute_triple_barrier_labels_full(
    close_values: np.ndarray,
    t0_positions: np.ndarray,
    volatility_values: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    min_return: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full triple-barrier labeling pipeline in a single parallel pass.

    This is the most optimized version that combines all steps:
    1. Compute vertical barriers (t1)
    2. Compute horizontal barriers (pt, sl)
    3. Find barrier touches and update t1
    4. Compute returns and labels

    Args:
        close_values: Array of close prices.
        t0_positions: Array of entry position indices in close.
        volatility_values: Array of volatility values for each event.
        pt_mult: Profit-taking multiplier.
        sl_mult: Stop-loss multiplier.
        max_holding: Maximum holding period.
        min_return: Minimum return threshold.

    Returns:
        Tuple of (t1_positions, returns, labels).
    """
    n_events = len(t0_positions)
    n_close = len(close_values)

    t1_positions = np.empty(n_events, dtype=np.int64)
    returns = np.empty(n_events, dtype=np.float64)
    labels = np.empty(n_events, dtype=np.int64)

    for i in prange(n_events):
        t0_pos = t0_positions[i]
        vol = volatility_values[i]

        # Skip invalid events
        if t0_pos < 0 or t0_pos >= n_close or np.isnan(vol):
            t1_positions[i] = -1
            returns[i] = np.nan
            labels[i] = 0
            continue

        # 1. Compute vertical barrier (t1)
        t1_pos = min(t0_pos + max_holding, n_close - 1)

        # 2. Compute horizontal barriers
        pt_barrier = pt_mult * vol if pt_mult > 0 else 0.0
        sl_barrier = -sl_mult * vol if sl_mult > 0 else 0.0

        # 3. Find barrier touch and update t1
        entry_price = close_values[t0_pos]
        if entry_price > 0:
            for pos in range(t0_pos + 1, t1_pos + 1):
                ret = (close_values[pos] - entry_price) / entry_price

                if pt_barrier > 0 and ret >= pt_barrier:
                    t1_pos = pos
                    break
                if sl_barrier < 0 and ret <= sl_barrier:
                    t1_pos = pos
                    break

        t1_positions[i] = t1_pos

        # 4. Compute return and label
        if entry_price > 0 and t1_pos >= 0 and t1_pos < n_close:
            exit_price = close_values[t1_pos]
            ret = (exit_price - entry_price) / entry_price
            returns[i] = ret

            if abs(ret) < min_return:
                labels[i] = 0
            elif ret > 0:
                labels[i] = 1
            else:
                labels[i] = -1
        else:
            returns[i] = np.nan
            labels[i] = 0

    return t1_positions, returns, labels


# =============================================================================
# PANDAS WRAPPER
# =============================================================================


def get_events_primary_fast(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_mult: float,
    sl_mult: float,
    trgt: pd.Series,
    max_holding: int,
    min_return: float = 0.0,
) -> pd.DataFrame:
    """
    Generate triple-barrier events using optimized Numba functions.

    This is a drop-in replacement for get_events_primary with 10-100x speedup.

    Parameters
    ----------
    close : pd.Series
        Close prices with datetime index.
    t_events : pd.DatetimeIndex
        Event timestamps to label.
    pt_mult : float
        Profit-taking multiplier.
    sl_mult : float
        Stop-loss multiplier.
    trgt : pd.Series
        Volatility target series.
    max_holding : int
        Maximum holding period in bars.
    min_return : float, default=0.0
        Minimum return threshold for labeling.

    Returns
    -------
    pd.DataFrame
        Events DataFrame with columns: t1, trgt, ret, label.
    """
    # Filter to valid events (those with valid volatility)
    valid_mask = t_events.isin(trgt.index)
    t_events_valid = t_events[valid_mask]

    if len(t_events_valid) == 0:
        return pd.DataFrame(
            {
                "t1": pd.Series([], dtype="datetime64[ns]"),
                "trgt": pd.Series([], dtype="float64"),
                "ret": pd.Series([], dtype="float64"),
                "label": pd.Series([], dtype="int64"),
            }
        )

    # Get volatility values for valid events
    vol_values = trgt.loc[t_events_valid].values
    valid_vol_mask = ~np.isnan(vol_values)
    t_events_final = t_events_valid[valid_vol_mask]
    vol_final = vol_values[valid_vol_mask]

    if len(t_events_final) == 0:
        return pd.DataFrame(
            {
                "t1": pd.Series([], dtype="datetime64[ns]"),
                "trgt": pd.Series([], dtype="float64"),
                "ret": pd.Series([], dtype="float64"),
                "label": pd.Series([], dtype="int64"),
            }
        )

    # Convert to numpy arrays for Numba
    close_values = close.values.astype(np.float64)
    close_index = close.index

    # Get position indices for events in close
    # Use searchsorted for vectorized lookup (much faster than get_loc in loop)
    close_index_values = np.asarray(close_index.values)
    t_events_values = np.asarray(t_events_final.values)

    # For datetime indices, we need to find positions
    t0_positions = np.searchsorted(close_index_values, t_events_values)

    # Validate positions (searchsorted may return len(close) for out-of-range)
    valid_pos_mask = (t0_positions < len(close_values)) & (t0_positions >= 0)

    # Check that the found positions actually match the event times
    for i in range(len(t0_positions)):
        if valid_pos_mask[i]:
            if t0_positions[i] < len(close_index_values):
                if close_index_values[t0_positions[i]] != t_events_values[i]:
                    valid_pos_mask[i] = False

    # Filter to valid positions
    t_events_final = t_events_final[valid_pos_mask]
    t0_positions = t0_positions[valid_pos_mask]
    vol_final = vol_final[valid_pos_mask]

    if len(t_events_final) == 0:
        return pd.DataFrame(
            {
                "t1": pd.Series([], dtype="datetime64[ns]"),
                "trgt": pd.Series([], dtype="float64"),
                "ret": pd.Series([], dtype="float64"),
                "label": pd.Series([], dtype="int64"),
            }
        )

    # Run optimized Numba computation
    t1_positions, returns, labels = compute_triple_barrier_labels_full(
        close_values=close_values,
        t0_positions=t0_positions.astype(np.int64),
        volatility_values=vol_final.astype(np.float64),
        pt_mult=float(pt_mult),
        sl_mult=float(sl_mult),
        max_holding=int(max_holding),
        min_return=float(min_return),
    )

    # Convert t1 positions back to timestamps
    valid_t1_mask = t1_positions >= 0
    t1_timestamps = np.empty(len(t1_positions), dtype=object)
    t1_timestamps[:] = pd.NaT

    for i in range(len(t1_positions)):
        if valid_t1_mask[i] and t1_positions[i] < len(close_index):
            t1_timestamps[i] = close_index[t1_positions[i]]

    # Build result DataFrame
    result = pd.DataFrame(
        {
            "t1": t1_timestamps,
            "trgt": vol_final,
            "ret": returns,
            "label": labels.astype(int),
        },
        index=t_events_final,
    )

    return result


# =============================================================================
# MODULAR FUNCTIONS FOR STEP-BY-STEP PIPELINE
# =============================================================================


def compute_vertical_barriers_fast(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    max_holding: int,
) -> pd.Series:
    """
    Compute vertical barriers using Numba optimization.

    Parameters
    ----------
    t_events : pd.DatetimeIndex
        Event timestamps.
    close : pd.Series
        Close prices with datetime index.
    max_holding : int
        Maximum holding period in bars.

    Returns
    -------
    pd.Series
        Series with t1 timestamps for each event.
    """
    close_index = close.index
    close_index_values = np.asarray(close_index.values)
    t_events_values = np.asarray(t_events.values)

    # Get positions using searchsorted
    t0_positions = np.searchsorted(close_index_values, t_events_values)

    # Compute vertical barriers
    t1_positions = compute_vertical_barriers_numba(
        event_positions=t0_positions.astype(np.int64),
        n_close=len(close),
        max_holding=int(max_holding),
    )

    # Convert back to timestamps
    t1_timestamps = []
    for i, t1_pos in enumerate(t1_positions):
        if t1_pos >= 0 and t1_pos < len(close_index):
            t1_timestamps.append(close_index[t1_pos])
        else:
            t1_timestamps.append(pd.NaT)

    return pd.Series(t1_timestamps, index=t_events, dtype="datetime64[ns]")


def update_barriers_with_touches_fast(
    close: pd.Series,
    events: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
) -> pd.DataFrame:
    """
    Update t1 positions based on horizontal barrier touches using Numba.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    events : pd.DataFrame
        Events DataFrame with 't1' and 'trgt' columns.
    pt_mult : float
        Profit-taking multiplier.
    sl_mult : float
        Stop-loss multiplier.

    Returns
    -------
    pd.DataFrame
        Updated events DataFrame with barrier-adjusted t1.
    """
    close_values = close.values.astype(np.float64)
    close_index = close.index
    close_index_values = np.asarray(close_index.values)

    # Get t0 positions
    t0_positions = np.searchsorted(close_index_values, np.asarray(events.index.values))

    # Get t1 positions
    t1_values = events["t1"].values
    t1_positions = np.array([
        np.searchsorted(close_index_values, np.asarray([t1]))[0] if pd.notna(t1) else -1
        for t1 in t1_values
    ], dtype=np.int64)

    # Compute barriers from volatility
    vol_values = events["trgt"].values.astype(np.float64)
    pt_barriers = pt_mult * vol_values if pt_mult > 0 else np.zeros(len(vol_values))
    sl_barriers = -sl_mult * vol_values if sl_mult > 0 else np.zeros(len(vol_values))

    # Update t1 with barrier touches
    updated_t1_positions = update_all_t1_with_barriers_numba(
        close_values=close_values,
        t0_positions=t0_positions.astype(np.int64),
        t1_positions=t1_positions,
        pt_barriers=pt_barriers.astype(np.float64),
        sl_barriers=sl_barriers.astype(np.float64),
    )

    # Convert back to timestamps
    updated_t1_timestamps = []
    for t1_pos in updated_t1_positions:
        if t1_pos >= 0 and t1_pos < len(close_index):
            updated_t1_timestamps.append(close_index[t1_pos])
        else:
            updated_t1_timestamps.append(pd.NaT)

    events = events.copy()
    events["t1"] = updated_t1_timestamps
    return events


def compute_returns_and_labels_fast(
    close: pd.Series,
    events: pd.DataFrame,
    min_return: float,
) -> pd.DataFrame:
    """
    Compute returns and labels using Numba optimization.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    events : pd.DataFrame
        Events DataFrame with 't1' column.
    min_return : float
        Minimum return threshold for labeling.

    Returns
    -------
    pd.DataFrame
        Events DataFrame with 'ret' and 'label' columns added.
    """
    close_values = close.values.astype(np.float64)
    close_index = close.index
    close_index_values = np.asarray(close_index.values)

    # Get t0 positions
    t0_positions = np.searchsorted(close_index_values, np.asarray(events.index.values))

    # Get t1 positions
    t1_values = events["t1"].values
    t1_positions = np.array([
        np.searchsorted(close_index_values, np.asarray([t1]))[0] if pd.notna(t1) else -1
        for t1 in t1_values
    ], dtype=np.int64)

    # Compute returns and labels
    returns, labels = compute_all_returns_and_labels_numba(
        close_values=close_values,
        t0_positions=t0_positions.astype(np.int64),
        t1_positions=t1_positions,
        min_return=float(min_return),
    )

    events = events.copy()
    events["ret"] = returns
    events["label"] = labels.astype(int)
    return events
