"""Hyperparameter optimization for EGARCH models.

This module provides:
- Optuna-based hyperparameter search
- Walk-forward cross-validation
- Anti-leakage validation

Implements hyperparameter search using walk-forward cross-validation
on TRAIN data only, minimizing QLIKE out-of-sample.
"""

from __future__ import annotations

from src.garch.garch_params.optimization.cross_validation import walk_forward_cv
from src.garch.garch_params.optimization.optuna import (
    optimize_egarch_hyperparameters,
    optuna_objective,
    save_optimization_results,
)
from src.garch.garch_params.optimization.validation import (
    assert_no_future_information,
    validate_cv_fold,
    validate_no_test_data_used,
    validate_temporal_ordering,
    validate_window_bounds,
)

__all__ = [
    # Cross-validation
    "walk_forward_cv",
    # Optuna optimization
    "optimize_egarch_hyperparameters",
    "optuna_objective",
    "save_optimization_results",
    # Validation
    "validate_temporal_ordering",
    "validate_window_bounds",
    "validate_no_test_data_used",
    "validate_cv_fold",
    "assert_no_future_information",
]
