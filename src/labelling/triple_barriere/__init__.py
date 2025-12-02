"""
Triple Barriere - Optimized Triple-Barrier Labeling with Numba.

This module provides Numba-optimized triple-barrier labeling functions
for 10-100x speedup over pure Python implementations.

Key Functions:
    - get_events_primary_fast: Drop-in replacement for get_events_primary
    - compute_triple_barrier_labels_full: Low-level Numba function for full pipeline
    - compute_vertical_barriers_fast: Vectorized vertical barrier computation
    - update_barriers_with_touches_fast: Parallel barrier touch detection
    - compute_returns_and_labels_fast: Vectorized return/label computation

Usage:
    from src.labelling.triple_barriere import get_events_primary_fast

    events = get_events_primary_fast(
        close=close_prices,
        t_events=event_timestamps,
        pt_mult=1.5,
        sl_mult=1.0,
        trgt=volatility,
        max_holding=15,
        min_return=0.0003,
    )
"""

from src.labelling.triple_barriere.fast_barriers import (
    # Main high-level function (drop-in replacement)
    get_events_primary_fast,
    # Full pipeline Numba function
    compute_triple_barrier_labels_full,
    # Modular functions for step-by-step pipeline
    compute_vertical_barriers_fast,
    update_barriers_with_touches_fast,
    compute_returns_and_labels_fast,
    # Low-level Numba functions (for advanced use)
    compute_vertical_barriers_numba,
    update_all_t1_with_barriers_numba,
    compute_all_returns_and_labels_numba,
)

__all__ = [
    # High-level API
    "get_events_primary_fast",
    # Full pipeline
    "compute_triple_barrier_labels_full",
    # Modular functions
    "compute_vertical_barriers_fast",
    "update_barriers_with_touches_fast",
    "compute_returns_and_labels_fast",
    # Low-level Numba
    "compute_vertical_barriers_numba",
    "update_all_t1_with_barriers_numba",
    "compute_all_returns_and_labels_numba",
]
