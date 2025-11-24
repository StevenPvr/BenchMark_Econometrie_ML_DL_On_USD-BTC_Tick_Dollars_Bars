"""Numerical tests for GARCH structure detection (pre-EGARCH, post-ARIMA).

Provides implementations of:
- Ljung-Box test on residuals
- Ljung-Box test on squared residuals
- Engle ARCH-LM test
- McLeod-Li test
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.constants import (
    GARCH_LJUNG_BOX_LAGS_DEFAULT,
    GARCH_LM_LAGS_DEFAULT,
    GARCH_NUMERICAL_BREUSCH_PAGAN_POLYNOMIAL_ORDER,
    GARCH_NUMERICAL_DEFAULT_ALPHA,
    GARCH_NUMERICAL_MIN_OBS_BREUSCH_PAGAN,
    GARCH_NUMERICAL_MIN_OBS_WHITE,
    GARCH_NUMERICAL_WHITE_NORMALIZATION_EPSILON,
    GARCH_NUMERICAL_WHITE_POLYNOMIAL_ORDER,
)
from src.garch.garch_numerical_test.helpers import safe_heteroskedasticity_test
from src.garch.garch_numerical_test.utils import (
    compute_arch_lm_statistic,
    compute_ljung_box_stats,
    prepare_lags_list,
)
from src.utils import chi2_sf, get_logger

logger = get_logger(__name__)


def _validate_residuals_and_lags(
    residuals: np.ndarray, lags: int | list[int]
) -> tuple[np.ndarray, list[int]]:
    """Validate residuals and prepare lags list.

    Args:
        residuals: Residual series.
        lags: Number of lags or list of lags.

    Returns:
        Tuple of (filtered residuals, lags list).

    Raises:
        ValueError: If residuals are empty or lags are invalid.
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    lags_list = prepare_lags_list(lags)

    if res.size == 0:
        msg = "Residuals array is empty after filtering finite values"
        raise ValueError(msg)
    if not lags_list:
        msg = f"Invalid lags parameter: {lags}. Must be positive integer or non-empty list"
        raise ValueError(msg)

    return res, lags_list


def ljung_box_test(
    residuals: np.ndarray,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    alpha: float = GARCH_NUMERICAL_DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Ljung-Box test on residuals to assess whiteness.

    Tests the null hypothesis that the residuals are independently
    distributed (no autocorrelation).

    Args:
        residuals: Residual series.
        lags: Number of lags to test.
        alpha: Significance level.

    Returns:
        Dict with lags, lb_stat, lb_pvalue, reject (bool), and n.

    Raises:
        ValueError: If residuals are empty or lags are invalid.
    """
    res, lags_list = _validate_residuals_and_lags(residuals, lags)
    lb_stats, lb_pvalues = compute_ljung_box_stats(res, lags_list)
    reject = bool(lb_pvalues[-1] < alpha) if lb_pvalues else False
    return {
        "lags": lags_list[: len(lb_stats)],
        "lb_stat": lb_stats,
        "lb_pvalue": lb_pvalues,
        "reject": reject,
        "n": int(res.size),
    }


def ljung_box_squared_test(
    residuals: np.ndarray,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    alpha: float = GARCH_NUMERICAL_DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Ljung-Box test on squared residuals to detect ARCH effects.

    Tests the null hypothesis that squared residuals are independently
    distributed (no autocorrelation in squared residuals).

    Args:
        residuals: Residual series.
        lags: Number of lags to test.
        alpha: Significance level.

    Returns:
        Dict with lags, lb_stat, lb_pvalue, reject (bool), and n.
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    res_squared = res**2
    return ljung_box_test(res_squared, lags=lags, alpha=alpha)


def _validate_arch_lm_inputs(e2: np.ndarray, lags: int) -> int:
    """Validate inputs for ARCH-LM test.

    Args:
        e2: Squared residuals.
        lags: Number of lags.

    Returns:
        Sample size.

    Raises:
        ValueError: If residuals are empty or insufficient for lags.
    """
    n = int(e2.size)
    if n == 0:
        msg = "Residuals array is empty after filtering NaN values"
        raise ValueError(msg)
    if n <= lags:
        msg = f"Insufficient data: {n} observations <= {lags} lags required"
        raise ValueError(msg)
    return n


def engle_arch_lm_test(
    residuals: np.ndarray,
    lags: int = GARCH_LM_LAGS_DEFAULT,
    alpha: float = GARCH_NUMERICAL_DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Engle's ARCH-LM test for heteroskedasticity.

    Tests the null hypothesis of no ARCH effects using Lagrange multiplier
    test on squared residuals regressed on lagged squared residuals.

    Args:
        residuals: Residual series.
        lags: Number of lags in regression.
        alpha: Significance level.

    Returns:
        Dict with lm_stat, p_value, df, reject (bool), and n.

    Raises:
        ValueError: If residuals are empty or insufficient for lags.
    """
    e2 = np.asarray(residuals, dtype=float) ** 2
    e2 = e2[~np.isnan(e2)]
    n = _validate_arch_lm_inputs(e2, lags)

    lm = compute_arch_lm_statistic(e2, lags)
    p_val = chi2_sf(lm, lags)
    reject = bool(np.isfinite(p_val) and p_val < alpha)

    return {
        "lm_stat": float(lm),
        "p_value": float(p_val),
        "df": float(lags),
        "reject": reject,
        "n": n,
    }


def mcleod_li_test(
    residuals: np.ndarray,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    alpha: float = GARCH_NUMERICAL_DEFAULT_ALPHA,
) -> dict[str, Any]:
    """McLeod-Li test for ARCH effects.

    Tests for ARCH effects by applying Ljung-Box test to squared residuals.
    This is similar to Ljung-Box on squared residuals but with a different
    interpretation focused on detecting ARCH/GARCH structure.

    Args:
        residuals: Residual series.
        lags: Number of lags to test.
        alpha: Significance level.

    Returns:
        Dict with lags, lb_stat, lb_pvalue, reject (bool), and n.

    Raises:
        ValueError: If residuals are empty or lags are invalid.
    """
    res, lags_list = _validate_residuals_and_lags(residuals, lags)
    res_squared = res**2
    lb_stats, lb_pvalues = compute_ljung_box_stats(res_squared, lags_list)
    reject = bool(lb_pvalues[-1] < alpha) if lb_pvalues else False
    return {
        "lags": lags_list[: len(lb_stats)],
        "lb_stat": lb_stats,
        "lb_pvalue": lb_pvalues,
        "reject": reject,
        "n": int(res_squared.size),
    }


def _create_breusch_pagan_exog(residuals: np.ndarray) -> np.ndarray:
    """Create exogenous variables for Breusch-Pagan test.

    Uses time trend polynomial: [1, t, t²].

    Args:
        residuals: Residual series.

    Returns:
        Exogenous variables matrix.
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    n = len(res)
    t = np.arange(n, dtype=float)
    poly_order = GARCH_NUMERICAL_BREUSCH_PAGAN_POLYNOMIAL_ORDER
    return np.column_stack([np.ones(n)] + [t**i for i in range(1, poly_order + 1)])


def breusch_pagan_test(
    residuals: np.ndarray,
    exog: np.ndarray | None = None,
    alpha: float = GARCH_NUMERICAL_DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Breusch-Pagan test for heteroskedasticity.

    Tests H0: Homoskedasticity (constant variance)
    against H1: Heteroskedasticity (variance depends on exogenous variables).

    If no exogenous variables provided, tests against time trend and squared time.

    Args:
        residuals: Residual series from mean model (ARIMA).
        exog: Optional exogenous variables (n x k matrix). If None, uses time trend.
        alpha: Significance level for test.

    Returns:
        Dict with lm_stat, p_value, df, reject, n.

    Note:
        - p-value < alpha suggests heteroskedasticity (motivation for GARCH)
        - Rejection provides statistical justification for using GARCH models

    Reference:
        Breusch, T. S., & Pagan, A. R. (1979). "A simple test for heteroscedasticity
        and random coefficient variation." Econometrica, 47(5), 1287-1294.
    """
    # Prepare exogenous variables
    if exog is None:
        exog = _create_breusch_pagan_exog(residuals)
    else:
        exog = np.asarray(exog, dtype=float)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        # Add intercept if not present
        if not np.allclose(exog[:, 0], 1.0):
            res = np.asarray(residuals, dtype=float)
            res = res[np.isfinite(res)]
            exog = np.column_stack([np.ones(len(res)), exog])

    return safe_heteroskedasticity_test(
        residuals=residuals,
        exog=exog,
        alpha=alpha,
        min_obs=GARCH_NUMERICAL_MIN_OBS_BREUSCH_PAGAN,
        test_name="Breusch-Pagan",
    )


def _create_white_test_exog(residuals: np.ndarray) -> np.ndarray:
    """Create exogenous variables for White test.

    Uses normalized time trend polynomial up to 4th order: [1, t, t², t³, t⁴].

    Args:
        residuals: Residual series.

    Returns:
        Exogenous variables matrix.
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    n = len(res)

    t = np.arange(n, dtype=float)
    t_normalized = (t - np.mean(t)) / (np.std(t) + GARCH_NUMERICAL_WHITE_NORMALIZATION_EPSILON)

    poly_order = GARCH_NUMERICAL_WHITE_POLYNOMIAL_ORDER
    return np.column_stack([np.ones(n)] + [t_normalized**i for i in range(1, poly_order + 1)])


def white_test(
    residuals: np.ndarray,
    alpha: float = GARCH_NUMERICAL_DEFAULT_ALPHA,
) -> dict[str, Any]:
    """White test for heteroskedasticity.

    Tests H0: Homoskedasticity
    against H1: Heteroskedasticity (unspecified form).

    More general than Breusch-Pagan as it doesn't assume specific form
    of heteroskedasticity. Uses squared residuals regressed on original
    regressors, their squares, and cross-products.

    For time series without exogenous variables, uses time trend up to 4th order.

    Args:
        residuals: Residual series from mean model (ARIMA).
        alpha: Significance level for test.

    Returns:
        Dict with lm_stat, p_value, df, reject, n.

    Note:
        - More powerful than Breusch-Pagan when form of heteroskedasticity unknown
        - Commonly used in time series to justify GARCH models

    Reference:
        White, H. (1980). "A heteroskedasticity-consistent covariance matrix estimator
        and a direct test for heteroskedasticity." Econometrica, 48(4), 817-838.
    """
    exog = _create_white_test_exog(residuals)

    return safe_heteroskedasticity_test(
        residuals=residuals,
        exog=exog,
        alpha=alpha,
        min_obs=GARCH_NUMERICAL_MIN_OBS_WHITE,
        test_name="White",
    )


def run_all_tests(
    residuals: np.ndarray,
    *,
    ljung_box_lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    arch_lm_lags: int = GARCH_LM_LAGS_DEFAULT,
    alpha: float = GARCH_NUMERICAL_DEFAULT_ALPHA,
    include_heteroskedasticity: bool = True,
) -> dict[str, Any]:
    """Run all numerical tests for GARCH structure detection.

    Performs complete battery of tests:
    1. Ljung-Box on residuals (autocorrelation)
    2. Ljung-Box on squared residuals (ARCH effects)
    3. Engle ARCH-LM test (heteroskedasticity)
    4. McLeod-Li test (ARCH effects via squared residuals)
    5. Breusch-Pagan test (heteroskedasticity with time trend)
    6. White test (general heteroskedasticity)

    Args:
        residuals: Residual series from ARIMA model.
        ljung_box_lags: Lags for Ljung-Box tests.
        arch_lm_lags: Lags for ARCH-LM test.
        alpha: Significance level.
        include_heteroskedasticity: Include Breusch-Pagan and White tests.

    Returns:
        Dict containing results from all tests.

    Note:
        Tests 5-6 provide additional evidence for heteroskedasticity,
        complementing ARCH-LM and McLeod-Li tests. Consistent rejection
        across multiple tests strengthens justification for GARCH modeling.
    """
    results = {
        "ljung_box_residuals": ljung_box_test(residuals, lags=ljung_box_lags, alpha=alpha),
        "ljung_box_squared": ljung_box_squared_test(residuals, lags=ljung_box_lags, alpha=alpha),
        "engle_arch_lm": engle_arch_lm_test(residuals, lags=arch_lm_lags, alpha=alpha),
        "mcleod_li": mcleod_li_test(residuals, lags=ljung_box_lags, alpha=alpha),
    }

    if include_heteroskedasticity:
        results["breusch_pagan"] = breusch_pagan_test(residuals, alpha=alpha)
        results["white"] = white_test(residuals, alpha=alpha)

    return results


__all__ = [
    "ljung_box_test",
    "ljung_box_squared_test",
    "engle_arch_lm_test",
    "mcleod_li_test",
    "breusch_pagan_test",
    "white_test",
    "run_all_tests",
]
