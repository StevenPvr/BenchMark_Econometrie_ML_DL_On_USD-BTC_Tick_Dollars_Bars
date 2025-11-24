"""Helper functions for heteroskedasticity tests (Breusch-Pagan and White).

This module contains shared logic to avoid duplication between
breusch_pagan_test and white_test implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from scipy import stats

from src.utils import get_logger

logger = get_logger(__name__)


def validate_heteroskedasticity_inputs(
    residuals: np.ndarray, min_obs: int
) -> tuple[np.ndarray, int]:
    """Validate inputs for heteroskedasticity tests.

    Args:
        residuals: Residual series.
        min_obs: Minimum required observations.

    Returns:
        Tuple of (filtered residuals, sample size).
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    n = len(res)
    return res, n


def create_nan_result(n: int, df: float | None = None) -> dict[str, Any]:
    """Create a standardized NaN result for invalid test cases.

    Args:
        n: Sample size.
        df: Degrees of freedom (if known).

    Returns:
        Dict with NaN values for all statistics.
    """
    return {
        "lm_stat": float("nan"),
        "p_value": float("nan"),
        "df": float("nan") if df is None else float(df),
        "reject": False,
        "n": n,
    }


def compute_heteroskedasticity_lm_statistic(
    e2: np.ndarray, exog: np.ndarray, n: int
) -> tuple[float, int]:
    """Compute LM statistic for heteroskedasticity tests.

    Args:
        e2: Squared residuals.
        exog: Exogenous variables matrix (n x k).
        n: Sample size.

    Returns:
        Tuple of (lm_stat, df).

    Raises:
        LinAlgError: If OLS regression fails.
    """
    k = exog.shape[1]

    # Check if we have enough observations
    if n <= k:
        msg = f"Insufficient observations: n={n} <= k={k} regressors"
        raise ValueError(msg)

    # OLS regression: e² = X*β + u
    beta = np.linalg.lstsq(exog, e2, rcond=None)[0]
    fitted = exog @ beta

    # R² from regression
    ss_tot = np.sum((e2 - np.mean(e2)) ** 2)
    ss_res = np.sum((e2 - fitted) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # LM statistic: n*R²
    lm_stat = n * max(0.0, r2)

    # Degrees of freedom = k-1 (exclude intercept)
    df = k - 1

    return float(lm_stat), df


def format_heteroskedasticity_result(
    lm_stat: float, df: int, alpha: float, n: int
) -> dict[str, Any]:
    """Format heteroskedasticity test result.

    Args:
        lm_stat: LM test statistic.
        df: Degrees of freedom.
        alpha: Significance level.
        n: Sample size.

    Returns:
        Dict with test results.
    """
    # Chi-square test
    p_val = float(1 - stats.chi2.cdf(lm_stat, df))
    reject = bool(p_val < alpha)

    return {
        "lm_stat": float(lm_stat),
        "p_value": p_val,
        "df": float(df),
        "reject": reject,
        "n": n,
    }


def safe_heteroskedasticity_test(
    residuals: np.ndarray,
    exog: np.ndarray,
    alpha: float,
    min_obs: int,
    test_name: str,
) -> dict[str, Any]:
    """Safely execute heteroskedasticity test with error handling.

    Args:
        residuals: Residual series.
        exog: Exogenous variables matrix.
        alpha: Significance level.
        min_obs: Minimum required observations.
        test_name: Name of test for logging.

    Returns:
        Dict with test results.
    """
    res, n = validate_heteroskedasticity_inputs(residuals, min_obs)

    if n < min_obs:
        return create_nan_result(n)

    e2 = res**2
    k = exog.shape[1]

    if n <= k:
        return create_nan_result(n, df=k - 1)

    try:
        lm_stat, df = compute_heteroskedasticity_lm_statistic(e2, exog, n)
        return format_heteroskedasticity_result(lm_stat, df, alpha, n)
    except (LinAlgError, ValueError) as exc:
        logger.warning("%s test failed: %s", test_name, exc)
        return create_nan_result(n, df=k - 1)


__all__ = [
    "validate_heteroskedasticity_inputs",
    "create_nan_result",
    "compute_heteroskedasticity_lm_statistic",
    "format_heteroskedasticity_result",
    "safe_heteroskedasticity_test",
]
