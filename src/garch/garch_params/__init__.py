"""EGARCH parameter estimation and optimization.

This module provides a complete EGARCH estimation pipeline:

**Core Functions** (variance computation, distributions, validation):
- Variance computation (EGARCH(1,1) and EGARCH(o,p))
- Distribution functions (kappa, log-likelihood)
- Parameter validation

**MLE Estimation**:
- Maximum Likelihood Estimation via SLSQP
- Parameter initialization and bounds
- Convergence tracking
- Batch estimation for all distributions (preparation step)

**Hyperparameter Optimization**:
- Walk-forward cross-validation
- Optuna-based search
- Anti-leakage validation

**Refit Management**:
- Periodic refit with expanding/rolling windows
- Unified refit logic
- History tracking

## Architecture

The module is organized into submodules:

- `core/`: Variance computation, distributions, validation
- `estimation/`: MLE estimation and convergence tracking
- `optimization/`: Hyperparameter optimization with Optuna
- `refit/`: Refit management (windows, schedule, manager)

## Example Usage

```python
from src.garch.garch_params import (
    estimate_egarch_mle,
    optimize_hyperparameters,
    RefitManager,
)

# MLE estimation
params, convergence = estimate_egarch_mle(residuals, o=1, p=1, dist="student")

# Hyperparameter optimization
results = optimize_hyperparameters(residuals_train, n_trials=50)

# Refit management
refit_mgr = RefitManager(
    frequency=20,
    window_type="expanding",
    o=1,
    p=1,
    dist="student",
)
params, conv = refit_mgr.perform_refit(residuals, position=100)
```

## References

- Nelson (1991): EGARCH specification
- Bollerslev (1986): GARCH estimation via MLE
- Patton (2011): QLIKE loss for variance forecasting
"""

from __future__ import annotations

# Core functions
from src.garch.garch_params.core import (
    clip_and_exp_log_variance,
    compute_kappa,
    compute_kappa_skewt,
    compute_kappa_student,
    compute_loglik_skewt,
    compute_loglik_student,
    compute_variance_path,
    compute_variance_path_egarch11,
    initialize_variance,
    safe_variance,
    validate_beta,
    validate_param_types,
    validate_residuals,
    validate_skewt_params,
    validate_student_params,
)

# Data loading
from src.garch.garch_params.data import (
    estimate_egarch_models,
    estimate_single_model,
    load_and_prepare_data,
)

# Estimation
from src.garch.garch_params.estimation import (
    ConvergenceResult,
    ConvergenceTracker,
    build_bounds,
    build_initial_params,
    build_result_dict,
    compute_initial_omega,
    count_params,
    create_negloglik_function,
    estimate_egarch_mle,
    extract_convergence_info,
    extract_params_from_array,
    get_default_params,
    run_slsqp_optimizer,
)

# Models (imported last to avoid circular imports)
from src.garch.garch_params.models import (
    EGARCHParams,
    create_egarch_params_from_array,
    create_egarch_params_from_dict,
    create_egarch_params_from_optimization_result,
)

# Optimization
from src.garch.garch_params.optimization import (
    assert_no_future_information,
    optimize_egarch_hyperparameters,
    optuna_objective,
    save_optimization_results,
    validate_cv_fold,
    validate_no_test_data_used,
    validate_temporal_ordering,
    validate_window_bounds,
    walk_forward_cv,
)

# Refit
from src.garch.garch_params.refit import (
    ExpandingWindow,
    RefitEvent,
    RefitManager,
    RefitSchedule,
    RollingWindow,
    Window,
    create_periodic_schedule,
    create_window_manager,
)

# Backward compatibility: keep old function names
egarch11_variance = compute_variance_path_egarch11  # Alias for backward compatibility

__all__ = [
    # Core - Variance
    "compute_variance_path",
    "compute_variance_path_egarch11",
    "initialize_variance",
    "safe_variance",
    "clip_and_exp_log_variance",
    "validate_param_types",
    "egarch11_variance",  # Backward compatibility
    # Core - Distributions
    "compute_kappa",
    "compute_kappa_student",
    "compute_kappa_skewt",
    "compute_loglik_student",
    "compute_loglik_skewt",
    # Core - Validation
    "validate_beta",
    "validate_residuals",
    "validate_student_params",
    "validate_skewt_params",
    # Estimation - MLE
    "estimate_egarch_mle",
    "create_negloglik_function",
    "run_slsqp_optimizer",
    # Estimation - Initialization
    "build_bounds",
    "build_initial_params",
    "build_result_dict",
    "compute_initial_omega",
    "count_params",
    "extract_params_from_array",
    "get_default_params",
    # Estimation - Convergence
    "ConvergenceResult",
    "ConvergenceTracker",
    "extract_convergence_info",
    # Models
    "EGARCHParams",
    "create_egarch_params_from_array",
    "create_egarch_params_from_dict",
    "create_egarch_params_from_optimization_result",
    # Optimization - Cross-validation
    "walk_forward_cv",
    # Optimization - Optuna
    "optimize_egarch_hyperparameters",
    "optuna_objective",
    "save_optimization_results",
    # Optimization - Validation
    "validate_temporal_ordering",
    "validate_window_bounds",
    "validate_no_test_data_used",
    "validate_cv_fold",
    "assert_no_future_information",
    # Refit - Manager
    "RefitManager",
    "RefitEvent",
    # Refit - Windows
    "Window",
    "ExpandingWindow",
    "RollingWindow",
    "create_window_manager",
    # Refit - Schedule
    "RefitSchedule",
    "create_periodic_schedule",
    # Data
    "load_and_prepare_data",
    "estimate_egarch_models",
    "estimate_single_model",
]
