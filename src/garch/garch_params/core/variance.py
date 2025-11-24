"""EGARCH variance computation functions.

This module provides a minimal, explicit implementation for EGARCH variance:
- Single-point variance initialization (no complex heuristics)
- Variance step computation
- Full variance path computation for EGARCH(1,1) and EGARCH(o,p)

All functions implement real-time filtering (causal computation).
The initialization deliberately avoids implicit behavior: when ``init`` is not
provided, the sample variance of the provided residuals is used, clamped to the
minimum threshold. This keeps behavior simple and reproducible.
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GARCH_LOG_VAR_EXPLOSION_THRESHOLD,
    GARCH_LOG_VAR_MAX,
    GARCH_LOG_VAR_MIN,
    GARCH_MIN_INIT_VAR,
)
from src.utils import get_logger

logger = get_logger(__name__)


def initialize_variance(residuals: np.ndarray, init: float | None) -> float:
    """Initialize variance for EGARCH recursion.

    Simple rule by design:
    - If ``init`` is provided and positive, use it as-is.
    - Otherwise, use the sample variance of the provided residuals and clamp to
      ``GARCH_MIN_INIT_VAR``.

    This removes unnecessary heuristics while keeping the behavior explicit and
    reproducible.

    Args:
        residuals: Residual series.
        init: Optional initial variance value.

    Returns:
        Initial variance value (>= ``GARCH_MIN_INIT_VAR``).
    """
    if init is not None and init > 0:
        return float(init)
    if residuals.size == 0:
        return GARCH_MIN_INIT_VAR
    # Use ddof=0 for consistency with MLE log-likelihood computation
    v = float(np.var(residuals, ddof=0))
    return max(v, GARCH_MIN_INIT_VAR)


def safe_variance(variance: float) -> float:
    """Ensure variance is at least minimum threshold.

    Args:
        variance: Variance value.

    Returns:
        Variance clamped to minimum threshold.
    """
    return max(variance, GARCH_MIN_INIT_VAR)


def clip_and_exp_log_variance(log_variance: float) -> float:
    """Clip log-variance and compute exp, returning NaN if invalid.

    This prevents numerical overflow/underflow in variance computation.

    Args:
        log_variance: Log-variance value.

    Returns:
        Exp of clipped log-variance, or NaN if invalid.
    """
    if not np.isfinite(log_variance):
        return float("nan")
    log_var_clipped = np.clip(log_variance, GARCH_LOG_VAR_MIN, GARCH_LOG_VAR_MAX)
    variance = float(np.exp(log_var_clipped))
    if not np.isfinite(variance) or variance <= 0:
        return float("nan")
    return variance


def compute_variance_step_egarch11(
    residual: float,
    variance_prev: float,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    kappa: float,
) -> float:
    """Compute next variance step for EGARCH(1,1).

    Implements: log(σ²ₜ) = ω + β·log(σ²ₜ₋₁) + α·(|zₜ₋₁| - κ) + γ·zₜ₋₁
    where zₜ₋₁ = εₜ₋₁ / σₜ₋₁.

    Explosion handling: if |log(σ²ₜ)| exceeds the explicit
    ``GARCH_LOG_VAR_EXPLOSION_THRESHOLD``, we clamp the log-variance to
    ``±GARCH_LOG_VAR_EXPLOSION_THRESHOLD`` instead of aborting the entire
    variance path. This keeps forecasts finite and traceable while making the
    stabilization explicit (no silent fallbacks).

    Args:
        residual: Previous residual value εₜ₋₁.
        variance_prev: Previous variance σ²ₜ₋₁.
        omega: EGARCH omega parameter.
        alpha: EGARCH alpha parameter.
        gamma: EGARCH gamma (leverage) parameter.
        beta: EGARCH beta parameter.
        kappa: E[|Z|] constant for distribution.

    Returns:
        Next variance σ²ₜ (finite and strictly positive if clamped).
    """
    variance_safe = safe_variance(variance_prev)
    z_prev = float(residual / np.sqrt(variance_safe))
    log_variance_next = (
        omega + beta * np.log(variance_safe) + alpha * (abs(z_prev) - kappa) + gamma * z_prev
    )

    # Explicit explosion control: clamp instead of returning NaN
    if abs(log_variance_next) > GARCH_LOG_VAR_EXPLOSION_THRESHOLD:
        logger.debug(
            "Clamping EGARCH(1,1) log-variance: %.2f -> ±%.1f",
            log_variance_next,
            GARCH_LOG_VAR_EXPLOSION_THRESHOLD,
        )
        sign = 1.0 if log_variance_next >= 0.0 else -1.0
        log_variance_next = sign * GARCH_LOG_VAR_EXPLOSION_THRESHOLD

    return clip_and_exp_log_variance(log_variance_next)


def _compute_recursion_step_general(
    residuals: np.ndarray,
    variances: np.ndarray,
    t: int,
    omega: float,
    alpha1: float,
    alpha2: float,
    gamma1: float,
    gamma2: float,
    beta1: float,
    beta2: float,
    beta3: float,
    kappa: float,
    o: int,
    p: int,
) -> float | None:
    """Compute one step of EGARCH(o,p) variance recursion.

    Args:
        residuals: Full residual series.
        variances: Variance series (being filled).
        t: Current time index.
        omega: Omega parameter.
        alpha1: Alpha1 parameter.
        alpha2: Alpha2 parameter.
        gamma1: Gamma1 parameter.
        gamma2: Gamma2 parameter.
        beta1: Beta1 parameter.
        beta2: Beta2 parameter.
        beta3: Beta3 parameter.
        kappa: Kappa constant.
        o: ARCH order (1 or 2).
        p: GARCH order (1, 2, or 3).

    Returns:
        Next variance value, or None if invalid or variance explosion detected.
    """
    log_variance = omega

    # GARCH terms (lagged log-variance)
    if p >= 1:
        variance_lag1 = safe_variance(variances[t - 1])
        log_variance += beta1 * np.log(variance_lag1)
    if p >= 2:
        variance_lag2 = safe_variance(variances[t - 2])
        log_variance += beta2 * np.log(variance_lag2)
    if p >= 3:
        variance_lag3 = safe_variance(variances[t - 3])
        log_variance += beta3 * np.log(variance_lag3)

    # ARCH terms (lagged standardized residuals)
    if o >= 1:
        variance_lag1 = safe_variance(variances[t - 1])
        z_lag1 = float(residuals[t - 1] / np.sqrt(variance_lag1))
        log_variance += alpha1 * (abs(z_lag1) - kappa) + gamma1 * z_lag1
    if o >= 2:
        variance_lag2 = safe_variance(variances[t - 2])
        z_lag2 = float(residuals[t - 2] / np.sqrt(variance_lag2))
        log_variance += alpha2 * (abs(z_lag2) - kappa) + gamma2 * z_lag2

    # Explicit explosion control BEFORE clipping: clamp instead of aborting
    # This indicates non-stationarity in the EGARCH recursion, so we cap
    # the log-variance to keep the path finite and auditable.
    if abs(log_variance) > GARCH_LOG_VAR_EXPLOSION_THRESHOLD:
        logger.debug(
            "Clamping EGARCH(o,p) log-variance at t=%d: %.2f -> ±%.1f",
            t,
            log_variance,
            GARCH_LOG_VAR_EXPLOSION_THRESHOLD,
        )
        sign = 1.0 if log_variance >= 0.0 else -1.0
        log_variance = sign * GARCH_LOG_VAR_EXPLOSION_THRESHOLD

    variance_next = clip_and_exp_log_variance(log_variance)
    if not np.isfinite(variance_next) or variance_next <= 0:
        return None
    return variance_next


def _initialize_variance_array(
    residuals_arr: np.ndarray, init: float | None, max_lag: int
) -> np.ndarray:
    """Initialize variance array with proper lag values.

    Args:
        residuals_arr: Residual array.
        init: Optional initial variance.
        max_lag: Maximum lag required.

    Returns:
        Initialized variance array.
    """
    n = residuals_arr.size
    variances = np.empty(n, dtype=float)
    init_val = initialize_variance(residuals_arr, init)
    # Fill the first max_lag elements (indices 0 to max_lag-1) with the same initialized variance
    # The recursion starts at index max_lag, so we only need to initialize up to max_lag-1
    end = min(max_lag, n)
    variances[:end] = init_val
    return variances


def _compute_variance_path_core(
    residuals: np.ndarray,
    omega: float,
    alpha1: float,
    alpha2: float,
    gamma1: float,
    gamma2: float,
    beta1: float,
    beta2: float,
    beta3: float,
    kappa: float,
    init: float | None,
    o: int,
    p: int,
) -> np.ndarray:
    """Compute EGARCH(o,p) variance path using recursion.

    Args:
        residuals: Residual series.
        omega: Omega parameter.
        alpha1: Alpha1 parameter.
        alpha2: Alpha2 parameter.
        gamma1: Gamma1 parameter.
        gamma2: Gamma2 parameter.
        beta1: Beta1 parameter.
        beta2: Beta2 parameter.
        beta3: Beta3 parameter.
        kappa: Kappa constant.
        init: Optional initial variance.
        o: ARCH order (1 or 2).
        p: GARCH order (1, 2, or 3).

    Returns:
        Variance path array (NaN if computation fails).
    """
    residuals_arr = np.asarray(residuals, dtype=float).ravel()
    n = residuals_arr.size
    max_lag = max(o, p)

    variances = _initialize_variance_array(residuals_arr, init, max_lag)
    if n < max_lag + 1:
        return variances

    for t in range(max_lag, n):
        variance_next = _compute_recursion_step_general(
            residuals_arr,
            variances,
            t,
            omega,
            alpha1,
            alpha2,
            gamma1,
            gamma2,
            beta1,
            beta2,
            beta3,
            kappa,
            o,
            p,
        )
        if variance_next is None:
            return np.full(n, np.nan)
        variances[t] = variance_next

    return variances


def compute_variance_path_egarch11(
    residuals: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    kappa: float,
    init: float | None = None,
) -> np.ndarray:
    """Compute EGARCH(1,1) variance path.

    log(σ²ₜ) = ω + β·log(σ²ₜ₋₁) + α·(|zₜ₋₁| - κ) + γ·zₜ₋₁
    where zₜ₋₁ = εₜ₋₁/σₜ₋₁

    Args:
        residuals: Residual series.
        omega: Omega parameter.
        alpha: Alpha parameter.
        gamma: Gamma (leverage) parameter.
        beta: Beta parameter.
        kappa: E[|Z|] constant.
        init: Optional initial variance.

    Returns:
        Variance path array (NaN if computation fails).
    """
    residuals_arr = np.asarray(residuals, dtype=float).ravel()
    n = residuals_arr.size
    variances = np.empty(n, dtype=float)
    variances[0] = initialize_variance(residuals_arr, init)

    for t in range(1, n):
        variances[t] = compute_variance_step_egarch11(
            residuals_arr[t - 1], variances[t - 1], omega, alpha, gamma, beta, kappa
        )
        if not np.isfinite(variances[t]) or variances[t] <= 0:
            return np.full(n, np.nan)

    return variances


def _validate_and_extract_alpha(alpha: float | tuple[float, float], o: int) -> tuple[float, float]:
    """Extract alpha parameters based on ARCH order."""
    if o == 1:
        if isinstance(alpha, tuple):
            msg = "For o=1, alpha must be float, not tuple"
            raise ValueError(msg)
        return float(alpha), 0.0
    # o == 2
    if not isinstance(alpha, tuple):
        msg = "For o=2, alpha must be tuple (alpha1, alpha2)"
        raise ValueError(msg)
    return float(alpha[0]), float(alpha[1])


def _validate_and_extract_gamma(gamma: float | tuple[float, float], o: int) -> tuple[float, float]:
    """Extract gamma parameters based on ARCH order."""
    if o == 1:
        if isinstance(gamma, tuple):
            msg = "For o=1, gamma must be float, not tuple"
            raise ValueError(msg)
        return float(gamma), 0.0
    # o == 2
    if not isinstance(gamma, tuple):
        msg = "For o=2, gamma must be tuple (gamma1, gamma2)"
        raise ValueError(msg)
    return float(gamma[0]), float(gamma[1])


def _validate_and_extract_beta(
    beta: float | tuple[float, float] | tuple[float, float, float], p: int
) -> tuple[float, float, float]:
    """Extract beta parameters based on GARCH order."""
    if p == 1:
        if isinstance(beta, tuple):
            msg = "For p=1, beta must be float, not tuple"
            raise ValueError(msg)
        return float(beta), 0.0, 0.0
    if p == 2:
        if not isinstance(beta, tuple) or len(beta) != 2:
            msg = "For p=2, beta must be tuple (beta1, beta2)"
            raise ValueError(msg)
        return float(beta[0]), float(beta[1]), 0.0
    # p == 3
    if not isinstance(beta, tuple) or len(beta) != 3:
        msg = "For p=3, beta must be tuple (beta1, beta2, beta3)"
        raise ValueError(msg)
    return float(beta[0]), float(beta[1]), float(beta[2])


def validate_param_types(
    alpha: float | tuple[float, float],
    gamma: float | tuple[float, float],
    beta: float | tuple[float, float] | tuple[float, float, float],
    o: int,
    p: int,
) -> tuple[float, float, float, float, float, float, float]:
    """Validate and extract EGARCH parameters according to orders.

    Args:
        alpha: Alpha parameter(s).
        gamma: Gamma parameter(s).
        beta: Beta parameter(s).
        o: ARCH order (1 or 2).
        p: GARCH order (1, 2, or 3).

    Returns:
        Tuple of (alpha1, alpha2, gamma1, gamma2, beta1, beta2, beta3).

    Raises:
        ValueError: If parameter types don't match orders.
    """
    alpha1, alpha2 = _validate_and_extract_alpha(alpha, o)
    gamma1, gamma2 = _validate_and_extract_gamma(gamma, o)
    beta1, beta2, beta3 = _validate_and_extract_beta(beta, p)
    return alpha1, alpha2, gamma1, gamma2, beta1, beta2, beta3


def compute_variance_path(
    residuals: np.ndarray,
    omega: float,
    alpha: float | tuple[float, float],
    gamma: float | tuple[float, float],
    beta: float | tuple[float, float] | tuple[float, float, float],
    kappa: float,
    *,
    init: float | None = None,
    o: int = 1,
    p: int = 1,
) -> np.ndarray:
    """Compute EGARCH(o,p) variance path for any valid order.

    Supports:
    - EGARCH(1,1): Standard specification
    - EGARCH(1,2): Two GARCH lags
    - EGARCH(1,3): Three GARCH lags
    - EGARCH(2,1): Two ARCH lags
    - EGARCH(2,2): Two ARCH and two GARCH lags
    - EGARCH(2,3): Two ARCH and three GARCH lags

    Args:
        residuals: Residual series.
        omega: Omega parameter.
        alpha: Alpha parameter(s). Float for o=1, tuple for o=2.
        gamma: Gamma parameter(s). Float for o=1, tuple for o=2.
        beta: Beta parameter(s). Float for p=1, tuple for p=2 or p=3.
        kappa: E[|Z|] constant.
        init: Optional initial variance.
        o: ARCH order (1 or 2).
        p: GARCH order (1, 2, or 3).

    Returns:
        Variance path array (NaN if computation fails).

    Raises:
        ValueError: If orders invalid or parameter types don't match.
    """
    if o not in (1, 2):
        msg = f"ARCH order o={o} not supported (only o=1 or 2)"
        raise ValueError(msg)
    if p not in (1, 2, 3):
        msg = f"GARCH order p={p} not supported (only p=1, 2, or 3)"
        raise ValueError(msg)

    alpha1, alpha2, gamma1, gamma2, beta1, beta2, beta3 = validate_param_types(
        alpha, gamma, beta, o, p
    )

    return _compute_variance_path_core(
        residuals, omega, alpha1, alpha2, gamma1, gamma2, beta1, beta2, beta3, kappa, init, o, p
    )
