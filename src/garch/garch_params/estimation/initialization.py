"""EGARCH parameter initialization and bounds.

This module provides functions for setting up MLE optimization:
- Initial parameter values
- Parameter bounds
- Parameter extraction from arrays/dicts
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GARCH_ESTIMATION_ALPHA_BOUND_MAX,
    GARCH_ESTIMATION_ALPHA_BOUND_MIN,
    GARCH_ESTIMATION_BETA_MAX,
    GARCH_ESTIMATION_BETA_MIN,
    GARCH_ESTIMATION_GAMMA_BOUND_MAX,
    GARCH_ESTIMATION_GAMMA_BOUND_MIN,
    GARCH_ESTIMATION_INIT_ALPHA,
    GARCH_ESTIMATION_INIT_BETA,
    GARCH_ESTIMATION_INIT_BETA2,
    GARCH_ESTIMATION_INIT_GAMMA,
    GARCH_ESTIMATION_OMEGA_BOUND_MAX,
    GARCH_ESTIMATION_OMEGA_BOUND_MIN,
    GARCH_SKEWT_LAMBDA_INIT,
    GARCH_SKEWT_LAMBDA_MAX,
    GARCH_SKEWT_LAMBDA_MIN,
    GARCH_STUDENT_NU_INIT,
    GARCH_STUDENT_NU_MAX,
    GARCH_STUDENT_NU_MIN,
)


def compute_initial_omega(variance_estimate: float) -> float:
    """Compute initial omega parameter from variance estimate.

    Uses formula: ω₀ = (1 - β₀) · log(σ²)
    where β₀ is the default initial beta value.

    Args:
        variance_estimate: Initial variance estimate.

    Returns:
        Initial omega value.
    """
    beta_init = GARCH_ESTIMATION_INIT_BETA
    omega_init = (1.0 - beta_init) * np.log(variance_estimate)
    return float(omega_init)


def get_default_params() -> dict[str, float]:
    """Get default initial parameter values.

    Returns:
        Dictionary with default parameter values.
    """
    return {
        "beta": GARCH_ESTIMATION_INIT_BETA,
        "alpha": GARCH_ESTIMATION_INIT_ALPHA,
        "gamma": GARCH_ESTIMATION_INIT_GAMMA,
        "nu": GARCH_STUDENT_NU_INIT,  # Used for student, skewt
        "lambda": GARCH_SKEWT_LAMBDA_INIT,  # Used for skewt
    }


def build_bounds(o: int, p: int, dist: str) -> list[tuple[float, float]]:
    """Build parameter bounds for EGARCH(o,p) with given distribution.

    Args:
        o: ARCH order (1 or 2).
        p: GARCH order (1 or 2).
        dist: Distribution name ('student', 'skewt').

    Returns:
        List of (min, max) tuples for all parameters.
    """
    bounds: list[tuple[float, float]] = []

    # Omega
    bounds.append((GARCH_ESTIMATION_OMEGA_BOUND_MIN, GARCH_ESTIMATION_OMEGA_BOUND_MAX))

    # Alpha parameters (o times)
    bounds.extend([(GARCH_ESTIMATION_ALPHA_BOUND_MIN, GARCH_ESTIMATION_ALPHA_BOUND_MAX)] * o)

    # Gamma parameters (o times)
    bounds.extend([(GARCH_ESTIMATION_GAMMA_BOUND_MIN, GARCH_ESTIMATION_GAMMA_BOUND_MAX)] * o)

    # Beta parameters (p times)
    bounds.extend([(GARCH_ESTIMATION_BETA_MIN, GARCH_ESTIMATION_BETA_MAX)] * p)

    # Distribution-specific parameters
    dist_lower = dist.lower()
    if dist_lower == "student":
        bounds.append((GARCH_STUDENT_NU_MIN, GARCH_STUDENT_NU_MAX))
    elif dist_lower == "skewt":
        bounds.extend(
            [
                (GARCH_STUDENT_NU_MIN, GARCH_STUDENT_NU_MAX),
                (GARCH_SKEWT_LAMBDA_MIN, GARCH_SKEWT_LAMBDA_MAX),
            ]
        )

    return bounds


def count_params(o: int, p: int, dist: str) -> int:
    """Count number of parameters for EGARCH(o,p) with given distribution.

    Args:
        o: ARCH order (1 or 2).
        p: GARCH order (1 or 2).
        dist: Distribution name.

    Returns:
        Number of parameters.
    """
    n_params = 1 + 2 * o + p  # omega + o*alpha + o*gamma + p*beta
    dist_lower = dist.lower()
    if dist_lower == "student":
        n_params += 1  # nu
    elif dist_lower == "skewt":
        n_params += 2  # nu, lambda
    return n_params


def build_initial_params(omega_init: float, o: int, p: int, dist: str) -> np.ndarray:
    """Build initial parameter vector for EGARCH(o,p).

    Args:
        omega_init: Initial omega value.
        o: ARCH order (1 or 2).
        p: GARCH order (1 or 2).
        dist: Distribution name.

    Returns:
        Initial parameter vector.
    """
    defaults = get_default_params()
    params_list = [omega_init]

    # Alpha parameters
    params_list.extend([defaults["alpha"]] * o)

    # Gamma parameters
    params_list.extend([defaults["gamma"]] * o)

    # Beta parameters
    if p == 1:
        params_list.append(defaults["beta"])
    else:  # p == 2
        params_list.extend([defaults["beta"], GARCH_ESTIMATION_INIT_BETA2])

    # Distribution parameters
    dist_lower = dist.lower()
    if dist_lower == "student":
        params_list.append(defaults["nu"])
    elif dist_lower == "skewt":
        params_list.extend([defaults["nu"], defaults["lambda"]])

    return np.array(params_list, dtype=float)


def extract_params_from_array(params: np.ndarray, o: int, p: int, dist: str) -> tuple[
    float,
    float | tuple[float, float],
    float | tuple[float, float],
    float | tuple[float, float] | tuple[float, float, float],
    float | None,
    float | None,
]:
    """Extract EGARCH parameters from parameter array.

    Args:
        params: Parameter vector from optimizer.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        Tuple of (omega, alpha, gamma, beta, nu, lambda_skew).

    Note:
        This function is kept for backward compatibility but delegates to
        EGARCHParams for actual extraction logic.
    """
    from src.garch.garch_params.models import create_egarch_params_from_array

    egarch_params = create_egarch_params_from_array(params, o, p, dist)
    return (
        egarch_params.omega,
        egarch_params.alpha,
        egarch_params.gamma,
        egarch_params.beta,
        egarch_params.nu,
        egarch_params.lambda_skew,
    )


def build_result_dict(
    params: np.ndarray, o: int, p: int, dist: str, loglik: float, converged: bool
) -> dict[str, float]:
    """Build result dictionary from parameter array.

    Args:
        params: Estimated parameter vector.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.
        loglik: Log-likelihood value.
        converged: Convergence status.

    Returns:
        Dictionary with all parameters and metadata.

    Note:
        This function is kept for backward compatibility but delegates to
        EGARCHParams for actual conversion logic.
    """
    from src.garch.garch_params.models import create_egarch_params_from_array

    egarch_params = create_egarch_params_from_array(
        params, o, p, dist, loglik=loglik, converged=converged
    )
    return egarch_params.to_dict()
