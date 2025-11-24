"""EGARCH and GARCH MLE estimation module.

This module provides Maximum Likelihood Estimation for EGARCH and GARCH models:
- MLE estimation via SLSQP optimizer
- Parameter initialization and bounds
- Convergence tracking and diagnostics

All estimation follows Bollerslev (1986) conditional MLE approach.
"""

from __future__ import annotations

from src.garch.garch_params.estimation.common import (
    ConvergenceResult,
    ConvergenceTracker,
    extract_convergence_info,
)
from src.garch.garch_params.estimation.initialization import (
    build_bounds,
    build_initial_params,
    build_result_dict,
    compute_initial_omega,
    count_params,
    extract_params_from_array,
    get_default_params,
)
from src.garch.garch_params.estimation.mle import (
    create_negloglik_function,
    estimate_egarch_mle,
    run_slsqp_optimizer,
)

__all__ = [
    # MLE
    "estimate_egarch_mle",
    "create_negloglik_function",
    "run_slsqp_optimizer",
    # Initialization
    "build_bounds",
    "build_initial_params",
    "build_result_dict",
    "compute_initial_omega",
    "count_params",
    "extract_params_from_array",
    "get_default_params",
    # Convergence
    "ConvergenceResult",
    "ConvergenceTracker",
    "extract_convergence_info",
]
