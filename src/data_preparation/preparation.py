"""Dollar Bars implementation following De Prado's methodology.

This module implements Dollar Bars (monetary volume bars) as described in
Marcos Lopez de Prado's "Advances in Financial Machine Learning" (Chapter 2).

Dollar Bars sample data each time a predefined monetary value (threshold T)
is exchanged, rather than at fixed time intervals. This approach:
- Produces more bars during high activity periods (volatile markets)
- Produces fewer bars during low activity periods (quiet markets)
- Improves statistical properties (closer to IID Gaussian) of returns
- Synchronizes sampling with market activity information flow

Mathematical Formulation (De Prado):
    Let each tick t have price p_t and volume v_t.
    Dollar value: dv_t = p_t * v_t
    Bar k closes at tick t when: sum(dv_i for i in [t_start, t]) >= T_k

Adaptive Threshold (De Prado EWMA method):
    T_0 = initial calibration
    After bar k with dollar_value D_k:
        E_k = alpha * D_k + (1 - alpha) * E_{k-1}  (EWMA of bar dollar values)
        T_{k+1} = E_k  (threshold adapts to market regime)

Reference:
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 2: Financial Data Structures, pp. 23-30.

This module re-exports all public functions from submodules for convenience.
"""

from __future__ import annotations

# Re-export numba core functions (for tests and advanced usage)
from src.data_preparation.numba_core import (
    _accumulate_dollar_bars_adaptive,
    _accumulate_dollar_bars_fixed,
    _compute_robust_percentile_threshold,
    _compute_threshold_from_target_bars,
)

# Re-export dollar bars computation API
from src.data_preparation.dollar_bars import (
    _create_empty_bars_df,
    _validate_dollar_bars_params,
    compute_dollar_bars,
    create_empty_bars_df,
    generate_dollar_bars,
)

# Re-export pipeline functions
from src.data_preparation.pipeline import (
    _BarAccumulatorState,
    _process_batch_with_state,
    add_log_returns_to_bars_file,
    prepare_dollar_bars,
    run_dollar_bars_pipeline,
    run_dollar_bars_pipeline_batch,
)

__all__ = [
    # Public API
    "compute_dollar_bars",
    "generate_dollar_bars",
    "prepare_dollar_bars",
    "run_dollar_bars_pipeline",
    "run_dollar_bars_pipeline_batch",
    "add_log_returns_to_bars_file",
    "create_empty_bars_df",
    # Internal (exported for tests)
    "_compute_threshold_from_target_bars",
    "_compute_robust_percentile_threshold",
    "_accumulate_dollar_bars_adaptive",
    "_accumulate_dollar_bars_fixed",
    "_create_empty_bars_df",
    "_validate_dollar_bars_params",
    "_BarAccumulatorState",
    "_process_batch_with_state",
]
