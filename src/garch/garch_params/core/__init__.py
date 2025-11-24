"""Core EGARCH computation and validation functions.

This module provides the foundational functions for EGARCH models:
- Variance computation (EGARCH(1,1) and EGARCH(o,p))
- Distribution functions (kappa and log-likelihood)
- Parameter validation

All variance computations are causal (real-time filtering).
"""

from __future__ import annotations

import numpy as np

# Distribution functions
from src.garch.garch_params.core.distributions import (
    compute_kappa,
    compute_kappa_skewt,
    compute_kappa_student,
    compute_loglik_skewt,
    compute_loglik_student,
)

# Validation
from src.garch.garch_params.core.validation import (
    validate_beta,
    validate_residuals,
    validate_skewt_params,
    validate_student_params,
)

# Variance computation
from src.garch.garch_params.core.variance import (
    clip_and_exp_log_variance,
    compute_variance_path,
    compute_variance_path_egarch11,
    compute_variance_step_egarch11,
    initialize_variance,
    safe_variance,
    validate_param_types,
)


# Backward compatibility wrappers with original computation.py API
def egarch11_variance(
    e: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    *,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
    init: float | None = None,
) -> np.ndarray:
    """EGARCH(1,1) variance - backward compatibility wrapper.

    Automatically computes kappa from dist/nu/lambda_skew parameters.
    """
    kappa = compute_kappa(dist, nu, lambda_skew)
    return compute_variance_path_egarch11(e, omega, alpha, gamma, beta, kappa, init=init)


def egarch_variance(
    e: np.ndarray,
    omega: float,
    alpha: float | tuple[float, float],
    gamma: float | tuple[float, float],
    beta: float | tuple[float, float] | tuple[float, float, float],
    *,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
    init: float | None = None,
    o: int = 1,
    p: int = 1,
) -> np.ndarray:
    """EGARCH(o,p) variance - backward compatibility wrapper.

    Automatically computes kappa from dist/nu/lambda_skew parameters.
    """
    kappa = compute_kappa(dist, nu, lambda_skew)
    return compute_variance_path(e, omega, alpha, gamma, beta, kappa, init=init, o=o, p=p)


# Other backward compatibility aliases
egarch_kappa = compute_kappa

# Additional backward compatibility aliases for log-likelihood functions
compute_student_loglikelihood = compute_loglik_student
compute_skewt_loglikelihood = compute_loglik_skewt

# Alias for validation functions
validate_series = validate_residuals

# Import estimate_egarch_mle here to avoid circular imports
# (it's now in estimation module but old code imports from core)
from src.garch.garch_params.estimation import estimate_egarch_mle  # noqa: E402

__all__ = [
    # Variance
    "clip_and_exp_log_variance",
    "compute_variance_path",
    "compute_variance_path_egarch11",
    "compute_variance_step_egarch11",
    "initialize_variance",
    "safe_variance",
    "validate_param_types",
    # Distributions
    "compute_kappa",
    "compute_kappa_skewt",
    "compute_kappa_student",
    "compute_loglik_skewt",
    "compute_loglik_student",
    # Validation
    "validate_beta",
    "validate_residuals",
    "validate_series",  # Alias for validate_residuals
    "validate_skewt_params",
    "validate_student_params",
    # Backward compatibility aliases
    "egarch11_variance",
    "egarch_variance",
    "egarch_kappa",
    "compute_student_loglikelihood",
    "compute_skewt_loglikelihood",
    "estimate_egarch_mle",  # Now in estimation module
]
