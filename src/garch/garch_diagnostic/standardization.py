"""Residual standardization utilities for GARCH diagnostics.

Contains functions for standardizing residuals using GARCH variance estimates.
"""

from __future__ import annotations

import numpy as np

from src.garch.garch_params.core import egarch_variance
from src.utils import get_logger

logger = get_logger(__name__)


def _detect_egarch_orders(params: dict[str, float]) -> tuple[int, int]:
    """Detect EGARCH orders o and p from parameters dictionary.

    Args:
    ----
        params: GARCH parameters dictionary.

    Returns:
    -------
        Tuple of (o, p) orders.

    """
    # Check if orders are explicitly provided
    if "o" in params and "p" in params:
        o = int(params["o"])
        p = int(params["p"])
        if o in (1, 2) and p in (1, 2):
            return o, p

    # Detect from parameter names
    has_alpha2 = "alpha2" in params
    has_gamma2 = "gamma2" in params
    has_beta2 = "beta2" in params

    o = 2 if (has_alpha2 or has_gamma2) else 1
    p = 2 if has_beta2 else 1

    return o, p


def _extract_egarch_params_for_variance(
    params: dict[str, float],
    o: int,
    p: int,
) -> tuple[
    float, float | tuple[float, float], float | tuple[float, float], float | tuple[float, float]
]:
    """Extract EGARCH parameters for variance computation.

    Args:
    ----
        params: GARCH parameters dictionary.
        o: ARCH order.
        p: GARCH order.

    Returns:
    -------
        Tuple of (omega, alpha, gamma, beta).

    """
    omega = float(params["omega"])

    if o == 1:
        alpha: float | tuple[float, float] = float(params["alpha"])
        gamma: float | tuple[float, float] = float(params.get("gamma", 0.0))
    else:  # o == 2
        alpha = (float(params["alpha1"]), float(params["alpha2"]))
        gamma = (float(params.get("gamma1", 0.0)), float(params.get("gamma2", 0.0)))

    if p == 1:
        beta: float | tuple[float, float] = float(params["beta"])
    else:  # p == 2
        beta = (float(params["beta1"]), float(params["beta2"]))

    return omega, alpha, gamma, beta


def _validate_variance_path(sigma2: np.ndarray) -> None:
    """Validate that variance path is finite and positive.

    Args:
    ----
        sigma2: Variance array.

    Raises:
    ------
        ValueError: If variance path is invalid.

    """
    if not (np.all(np.isfinite(sigma2)) and np.all(sigma2 > 0)):
        msg = "Invalid variance path for standardization."
        raise ValueError(msg)


def standardize_residuals(
    residuals: np.ndarray,
    params: dict[str, float],
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
    clean: bool = False,
) -> np.ndarray:
    """Return standardized residuals z_t = e_t / sigma_t using EGARCH(o,p) params.

    Args:
    ----
        residuals: Residual array εt.
        params: GARCH parameters dictionary.
        dist: Distribution name ('student', 'skewt').
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t). If None, extracted from params.
        clean: If True, remove non-finite values from output.

    Returns:
    -------
        Standardized residuals array z_t = e_t / sigma_t.
        If clean=True, only finite values are returned.

    Raises:
    ------
        ValueError: If variance path is invalid or required parameters missing.

    """
    e = np.asarray(residuals, dtype=float)
    e = e[np.isfinite(e)]

    o, p = _detect_egarch_orders(params)
    omega, alpha, gamma, beta = _extract_egarch_params_for_variance(params, o, p)

    # Extract lambda_skew if not provided (supports both new/legacy param formats)
    if lambda_skew is None:
        params_dict: dict[str, float] = params
        if "params" in params and isinstance(params["params"], dict):
            params_dict = params["params"]  # type: ignore[assignment]
        lambda_val = params_dict.get("lambda_skew") or params_dict.get("lambda")
        if lambda_val is None and dist.lower() == "skewt":
            available_keys = list(params_dict.keys())
            raise ValueError(
                "Skew-t distribution requires 'lambda_skew' (or 'lambda') parameter. "
                f"Available keys: {available_keys}"
            )
        lambda_skew = float(lambda_val) if lambda_val is not None else None

    sigma2 = egarch_variance(
        e, omega, alpha, gamma, beta, dist=dist, nu=nu, lambda_skew=lambda_skew, o=o, p=p
    )
    _validate_variance_path(sigma2)
    z = e / np.sqrt(sigma2)

    if clean:
        z = z[np.isfinite(z)]

    return z


def compute_standardized_residuals_for_plot(
    all_res: np.ndarray,
    garch_params: dict[str, float] | None,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> np.ndarray | None:
    """Compute standardized residuals if GARCH params are provided.

    Args:
    ----
        all_res: Residual array.
        garch_params: GARCH parameters dictionary.
        dist: Distribution name.
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t). If None, extracted from params.

    Returns:
    -------
        Standardized residuals array or None if params not provided.

    Raises:
    ------
        ValueError: If variance path is invalid or required parameters missing.

    """
    if garch_params is None:
        return None
    return standardize_residuals(all_res, garch_params, dist=dist, nu=nu, lambda_skew=lambda_skew)
