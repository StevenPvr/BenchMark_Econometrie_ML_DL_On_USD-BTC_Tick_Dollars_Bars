"""Model selection and distribution utilities for GARCH evaluation.

Includes:
- Model selection using AIC
- Distribution functions for Hansen skew-t, Student-t, and Normal
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GARCH_EVAL_AIC_MULTIPLIER,
    GARCH_MODEL_NAMES,
    GARCH_MODEL_PARAMS_COUNT,
    GARCH_STUDENT_NU_MIN,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Model selection
# ============================================================================


def aic(ll: float, k: int) -> float:
    """Calculate AIC: 2k - 2*loglik.

    Args:
    ----
        ll: Log-likelihood value.
        k: Number of parameters.

    Returns:
    -------
        AIC score.

    """
    return GARCH_EVAL_AIC_MULTIPLIER * k - GARCH_EVAL_AIC_MULTIPLIER * float(ll)


def collect_converged_candidates(
    payload: dict,
    keys: list[str],
    k_params: dict[str, int],
) -> list[tuple[str, dict, float]]:
    """Collect converged model candidates with their AIC scores.

    Args:
    ----
        payload: Estimation payload dictionary.
        keys: List of model keys to check.
        k_params: Dictionary mapping model names to parameter counts.

    Returns:
    -------
        List of tuples (name, params_dict, aic_score).

    """
    cand: list[tuple[str, dict, float]] = []
    for name in keys:
        d = payload.get(name)
        if isinstance(d, dict) and d.get("converged"):
            k = k_params[name]
            # Extract loglik from new format (log_likelihood) or legacy format (loglik in params)
            loglik_val = d.get("log_likelihood")
            if loglik_val is None:
                params_dict = d.get("params", {})
                loglik_val = params_dict.get("loglik")
            if loglik_val is None:
                # Skip if no loglik found
                continue
            cand.append((name, d, aic(float(loglik_val), k)))
    return cand


def choose_best_from_estimation(
    payload: dict,
) -> tuple[dict[str, float], str, float | None, float | None]:
    """Pick best model from estimation JSON using AIC and preference order.

    Preference order on ties: skew-t → student.

    Args:
    ----
        payload: Estimation payload dictionary.

    Returns:
    -------
        Tuple of (params_dict, model_name, nu, lambda_skew).

    """
    keys = list(GARCH_MODEL_NAMES)
    k_params = GARCH_MODEL_PARAMS_COUNT

    cand = collect_converged_candidates(payload, keys, k_params)
    if not cand:
        msg = "No converged volatility model found in estimation file"
        raise RuntimeError(msg)

    # Sort by AIC then by preference order
    order = {k: i for i, k in enumerate(keys)}
    cand.sort(key=lambda t: (t[2], order.get(t[0], 999)))
    name, model_dict, _ = cand[0]

    # Extract params dictionary (new format has params nested, legacy format is direct)
    params_dict = (
        model_dict.get("params", model_dict)
        if isinstance(model_dict.get("params"), dict)
        else model_dict
    )

    nu_val = params_dict.get("nu")
    nu = float(nu_val) if nu_val is not None else None
    lambda_val = params_dict.get("lambda")
    lambda_skew = float(lambda_val) if lambda_val is not None else None
    return params_dict, name, nu, lambda_skew


# ============================================================================
# Distribution utilities (Hansen skew-t)
# ============================================================================


def _skewt_abc(nu: float, lam: float) -> tuple[float, float, float]:
    """Return Hansen skew-t constants (a, b, c) for nu>2 and |lam|<1.

    Parameters
    ----------
    nu : float
        Degrees of freedom (must be > 2 for finite variance).
    lam : float
        Skewness parameter in (-1, 1).

    Returns
    -------
    (a, b, c) : tuple of floats
        Hansen skew-t constants.

    """
    if not (nu is not None and float(nu) > GARCH_STUDENT_NU_MIN):
        raise ValueError("Skew-t requires nu > 2.0")
    lam = float(lam)
    try:
        from scipy.special import gammaln  # type: ignore
    except Exception as exc:  # pragma: no cover - SciPy is required in this project
        raise RuntimeError("SciPy required for skew-t constants") from exc

    c_log = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * (np.log(np.pi) + np.log(nu - 2.0))
    c = float(np.exp(c_log))
    a = float(4.0 * lam * c * (nu - 2.0) / (nu - 1.0))
    b_sq = float(1.0 + 3.0 * lam * lam - a * a)
    if b_sq <= 0.0:
        raise ValueError("Invalid (nu, lambda) for skew-t: b^2 <= 0")
    b = float(np.sqrt(b_sq))
    return a, b, c


def _compute_skewt_transforms(
    z_arr: np.ndarray, a: float, b: float, s: float, lam: float, thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute transformed values for skew-t CDF."""
    x_left = s * (b * z_arr + a) / (1.0 - lam)
    x_right = s * (b * z_arr + a) / (1.0 + lam)
    mask_left = z_arr < thr
    return x_left, x_right, mask_left


def skewt_cdf(z: np.ndarray | float, nu: float, lam: float) -> np.ndarray | float:
    """CDF of the standardized Hansen skew-t at z.

    Parameters
    ----------
    z : array-like or float
        Evaluation point(s).
    nu : float
        Degrees of freedom (>2).
    lam : float
        Skewness parameter (-1 < lam < 1).

    Returns
    -------
    F : array-like or float
        CDF value(s) at z.

    """
    try:
        from scipy.stats import t as student_t  # type: ignore
    except Exception as exc:  # pragma: no cover - SciPy is required
        raise RuntimeError("SciPy required for skew-t CDF") from exc

    a, b, _ = _skewt_abc(float(nu), float(lam))
    z_arr = np.asarray(z, dtype=float)
    s = float(np.sqrt(nu / (nu - 2.0)))
    thr = float(-a / b)

    x_left, x_right, mask_left = _compute_skewt_transforms(z_arr, a, b, s, lam, thr)

    F = np.empty_like(z_arr, dtype=float)
    if np.any(mask_left):
        F[mask_left] = (1.0 - lam) * student_t.cdf(x_left[mask_left], df=nu)
    if np.any(~mask_left):
        F[~mask_left] = (1.0 + lam) * student_t.cdf(x_right[~mask_left], df=nu) - lam

    return float(F.item()) if np.isscalar(z) else F


def _compute_ppf_left_branch(
    q_arr: np.ndarray,
    mask_left: np.ndarray,
    lam: float,
    a: float,
    b: float,
    s_inv: float,
    nu: float,
) -> np.ndarray:
    """Compute left branch of skew-t PPF."""
    from scipy.stats import t as student_t  # type: ignore

    z_out = np.empty_like(q_arr, dtype=float)
    if np.any(mask_left):
        qq = q_arr[mask_left] / (1.0 - lam)
        x = student_t.ppf(qq, df=nu)
        z_out[mask_left] = ((1.0 - lam) * s_inv * x - a) / b
    return z_out


def _compute_ppf_right_branch(
    q_arr: np.ndarray,
    mask_right: np.ndarray,
    lam: float,
    a: float,
    b: float,
    s_inv: float,
    nu: float,
) -> np.ndarray:
    """Compute right branch of skew-t PPF."""
    from scipy.stats import t as student_t  # type: ignore

    z_out = np.empty_like(q_arr, dtype=float)
    if np.any(mask_right):
        qq = (q_arr[mask_right] + lam) / (1.0 + lam)
        x = student_t.ppf(qq, df=nu)
        z_out[mask_right] = ((1.0 + lam) * s_inv * x - a) / b
    return z_out


def skewt_ppf(q: np.ndarray | float, nu: float, lam: float) -> np.ndarray | float:
    """PPF (inverse CDF) of the standardized Hansen skew-t at probability q.

    Piecewise inversion using the closed-form CDF mapping to Student-t.

    Parameters
    ----------
    q : array-like or float
        Probability level(s) in (0,1).
    nu : float
        Degrees of freedom (>2).
    lam : float
        Skewness parameter (-1 < lam < 1).

    Returns
    -------
    z_q : array-like or float
        Quantile(s) such that P(Z <= z_q) = q.

    """
    a, b, _ = _skewt_abc(float(nu), float(lam))
    q_arr = np.asarray(q, dtype=float)
    s_inv = float(np.sqrt((nu - 2.0) / nu))
    p0 = 0.5 * (1.0 - lam)

    mask_left = q_arr < p0
    z_out = _compute_ppf_left_branch(q_arr, mask_left, lam, a, b, s_inv, nu)
    mask_right = ~mask_left
    z_out_right = _compute_ppf_right_branch(q_arr, mask_right, lam, a, b, s_inv, nu)
    z_out[mask_right] = z_out_right[mask_right]

    return float(z_out.item()) if np.isscalar(q) else z_out


__all__ = [
    # Model selection
    "aic",
    "collect_converged_candidates",
    "choose_best_from_estimation",
    # Distributions
    "skewt_cdf",
    "skewt_ppf",
]
