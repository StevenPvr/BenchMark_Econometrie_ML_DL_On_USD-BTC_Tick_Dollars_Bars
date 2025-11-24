"""EGARCH parameter validation functions.

This module provides validation for EGARCH parameters:
- Beta stationarity constraints
- Distribution parameter constraints
- Residual series validation
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GARCH_ESTIMATION_BETA_MAX,
    GARCH_ESTIMATION_BETA_MIN,
    GARCH_ESTIMATION_MIN_OBSERVATIONS,
    GARCH_ESTIMATION_NU_MIN_THRESHOLD,
    GARCH_SKEWT_LAMBDA_MAX,
    GARCH_SKEWT_LAMBDA_MIN,
)
from src.utils import get_logger

logger = get_logger(__name__)


def validate_residuals(residuals: np.ndarray) -> np.ndarray:
    """Validate and convert residual series to 1D float array.

    Args:
        residuals: Residual series to validate.

    Returns:
        Validated 1D float array.

    Raises:
        ValueError: If insufficient observations.
    """
    residuals_arr = np.asarray(residuals, dtype=float).ravel()
    if residuals_arr.size < GARCH_ESTIMATION_MIN_OBSERVATIONS:
        msg = (
            f"Need at least {GARCH_ESTIMATION_MIN_OBSERVATIONS} observations "
            f"to estimate EGARCH, got {residuals_arr.size}."
        )
        raise ValueError(msg)
    return residuals_arr


def validate_beta(beta: float) -> bool:
    """Validate beta parameter for stationarity.

    For EGARCH stationarity: |β| < 1

    Args:
        beta: EGARCH beta parameter.

    Returns:
        True if parameter satisfies stationarity constraint.
    """
    return GARCH_ESTIMATION_BETA_MIN < beta < GARCH_ESTIMATION_BETA_MAX


def validate_student_params(beta: float, nu: float) -> bool:
    """Validate Student-t distribution parameters.

    Args:
        beta: EGARCH beta parameter (stationarity).
        nu: Degrees of freedom (must be > 2).

    Returns:
        True if parameters are valid.
    """
    return validate_beta(beta) and nu > GARCH_ESTIMATION_NU_MIN_THRESHOLD


def validate_skewt_params(beta: float, nu: float, lambda_skew: float) -> bool:
    """Validate Skew-t distribution parameters.

    Args:
        beta: EGARCH beta parameter (stationarity).
        nu: Degrees of freedom (must be > 2).
        lambda_skew: Skewness parameter (must be in (-1, 1)).

    Returns:
        True if parameters are valid.
    """
    return (
        validate_beta(beta)
        and nu > GARCH_ESTIMATION_NU_MIN_THRESHOLD
        and GARCH_SKEWT_LAMBDA_MIN < lambda_skew < GARCH_SKEWT_LAMBDA_MAX
    )
