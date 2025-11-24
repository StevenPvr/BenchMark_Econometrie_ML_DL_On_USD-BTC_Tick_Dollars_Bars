"""Classic evaluation metrics for GARCH volatility forecasts.

Includes:
- Variance forecast losses: QLIKE, MSE, MAE
- Mincer-Zarnowitz regression on e_t^2 vs sigma_t^2
- VaR backtests: Kupiec POF and Christoffersen independence + combined
- Quantile and VaR utilities
- Metrics computation utilities


All functions are small, typed, and dependency-light (SciPy optional for p-values).
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GARCH_CALIBRATION_EPS,
    GARCH_EVAL_EPSILON,
    GARCH_EVAL_HALF,
    GARCH_EVAL_METRICS_FILE,
    GARCH_EVAL_MIN_OBS,
    GARCH_EVAL_VAR_SUMMARY_FILE,
    GARCH_STUDENT_NU_INIT,
    GARCH_STUDENT_NU_MIN,
)
from src.garch.garch_eval.data_loading import load_dataset_for_metrics
from src.garch.garch_eval.helpers import _clip_probability, _validate_and_filter_arrays, chi2_sf
from src.garch.garch_eval.models import skewt_ppf
from src.garch.garch_eval.mz_calibration import (
    compute_mz_pvalues,
    filter_test_data,
    load_test_resid_sigma2,
)
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


def qlike_loss(e: np.ndarray, sigma2: np.ndarray) -> float:
    """Return average QLIKE loss: log(sigma2) + e^2 / sigma2.

    Args:
    ----
        e: Residuals aligned to sigma2 (length n)
        sigma2: Conditional variance sequence (positive; length n)

    Returns:
    -------
        Mean QLIKE over finite pairs.

    """
    e_filt, s2_filt, _ = _validate_and_filter_arrays(e, sigma2)
    if e_filt.size == 0:
        return float("nan")
    return float(np.mean(np.log(s2_filt) + (e_filt**2) / s2_filt))


def mse_mae_variance(e: np.ndarray, sigma2: np.ndarray) -> dict[str, float]:
    """Compute MSE and MAE between realized e^2 and forecast sigma^2."""
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2)
    if not np.any(m):
        return {"mse": float("nan"), "mae": float("nan")}
    y = e[m] ** 2
    f = s2[m]
    return {
        "mse": float(np.mean((y - f) ** 2)),
        "mae": float(np.mean(np.abs(y - f))),
    }


def r2_variance(e: np.ndarray, sigma2: np.ndarray) -> float:
    """Compute R² for variance predictions.

    R² = correlation(realized_volatility, predicted_volatility)²

    Args:
        e: Residuals (realized volatility = e²)
        sigma2: Predicted variance (sigma²)

    Returns:
        R² coefficient in [0, 1], or NaN if computation fails
    """
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2)
    if not np.any(m):
        return float("nan")

    y = e[m] ** 2  # Realized volatility
    f = s2[m]  # Predicted volatility

    if y.size < 2:
        return float("nan")

    # R² = correlation coefficient squared
    corr_matrix = np.corrcoef(y, f)
    corr = corr_matrix[0, 1]

    if not np.isfinite(corr):
        return float("nan")

    return float(max(0.0, corr**2))  # Ensure non-negative


def mincer_zarnowitz(e: np.ndarray, sigma2: np.ndarray) -> dict[str, float]:
    """Run Mincer-Zarnowitz regression: e^2 = c + b * sigma^2 + u.

    Returns intercept c, slope b, R^2 and (optionally) p-values when SciPy is available.
    """
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2)
    y = (e[m] ** 2).astype(float)
    x = s2[m].astype(float)
    if y.size < GARCH_EVAL_MIN_OBS:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}

    x_mat = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
    y_hat = x_mat @ beta
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    out: dict[str, float] = {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(r2),
    }
    out.update(compute_mz_pvalues(beta, x_mat, ss_res))
    return out


def mz_calibration_params(e: np.ndarray, sigma2: np.ndarray) -> dict[str, float]:
    """Return intercept and slope (a,b) from MZ regression to calibrate variance.

    Why: If b>1 (like in your diagnostics), the raw sigma² underestimates
    realized variance. Typically use multiplicative calibration: h_adj = b * h
    (intercept usually not significant and can cause instability).
    """
    mz = mincer_zarnowitz(e, sigma2)
    return {
        "intercept": float(mz.get("intercept", np.nan)),
        "slope": float(mz.get("slope", np.nan)),
        "p_intercept": float(mz.get("p_intercept", np.nan)),
        "p_slope": float(mz.get("p_slope", np.nan)),
    }


def apply_mz_calibration(
    sigma2: np.ndarray,
    intercept: float,
    slope: float,
    *,
    eps: float = GARCH_CALIBRATION_EPS,
    use_intercept: bool = False,
) -> np.ndarray:
    """Apply MZ calibration: h_adj = max(eps, a + b * h) or h_adj = b * h.

    Args:
    ----
        sigma2: Variance array to calibrate.
        intercept: MZ regression intercept.
        slope: MZ regression slope.
        eps: Minimum variance threshold.
        use_intercept: If False, use multiplicative calibration only (slope * h).
                      If True, use full additive calibration (intercept + slope * h).

    Returns:
    -------
        Calibrated variance array.

    """
    s2 = np.asarray(sigma2, dtype=float)
    # Multiplicative calibration only (more stable) if use_intercept is False
    h_adj = intercept + slope * s2 if use_intercept else slope * s2
    return np.asarray(np.maximum(eps, h_adj), dtype=float)


# ============================================================================
# Quantile and VaR utilities
# ============================================================================


def _quantile_skewt(p: float, nu: float, lambda_skew: float) -> float:
    """Return quantile for skew-t distribution.

    Args:
    ----
        p: Probability level.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
    -------
        Quantile value.

    """
    if nu <= GARCH_STUDENT_NU_MIN or lambda_skew is None:
        msg = "Skew-t requires nu>2 and lambda for quantiles"
        raise ValueError(msg)
    return float(skewt_ppf(float(p), float(nu), float(lambda_skew)))


def _quantile_student(p: float, nu: float) -> float:
    """Return quantile for Student-t distribution.

    Args:
    ----
        p: Probability level.
        nu: Degrees of freedom.

    Returns:
    -------
        Quantile value.

    """
    from scipy.stats import t  # type: ignore

    if nu <= GARCH_STUDENT_NU_MIN:
        msg = "Student-t requires nu>2 for quantiles"
        raise ValueError(msg)
    return float(t.ppf(p, df=float(nu)))


def quantile(dist: str, p: float, nu: float | None, lambda_skew: float | None = None) -> float:
    """Return left-tail quantile for Student-t or Hansen skew-t.

    Args:
    ----
        dist: Distribution type ('student', 'skewt').
        p: Probability level.
        nu: Degrees of freedom (for Student-t/Skew-t).
            Uses GARCH_STUDENT_NU_INIT as default for student distribution.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Quantile value.

    """
    dist_l = dist.lower()

    # Use default nu for student distribution if not provided
    effective_nu = (
        nu if nu is not None else (GARCH_STUDENT_NU_INIT if dist_l == "student" else None)
    )

    if dist_l == "skewt":
        if effective_nu is None or lambda_skew is None:
            msg = "Skew-t requires nu and lambda for quantiles"
            raise ValueError(msg)
        return _quantile_skewt(p, effective_nu, lambda_skew)
    if dist_l == "student":
        if effective_nu is None:
            msg = "Student-t requires nu for quantiles"
            raise ValueError(msg)
        return _quantile_student(p, effective_nu)
    msg = f"Unsupported distribution: {dist}. Must be 'student' or 'skewt'."
    raise ValueError(msg)


def var_quantile(
    alpha: float, dist: str, nu: float | None, lambda_skew: float | None = None
) -> float:
    """Quantile for VaR under Normal/Student/Skew-t innovations (left tail).

    Args:
    ----
        alpha: Tail probability level.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        VaR quantile value.

    """
    return quantile(dist, alpha, nu, lambda_skew)


def build_var_series(
    sigma2: np.ndarray,
    alpha: float,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> np.ndarray:
    """Return VaR_t series at level alpha with zero mean: VaR = q_alpha * sigma_t.

    Args:
    ----
        sigma2: Variance series.
        alpha: Tail probability level.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        VaR series.

    """
    s2 = np.asarray(sigma2, dtype=float).ravel()
    q = var_quantile(float(alpha), dist, nu, lambda_skew)
    return q * np.sqrt(s2)


def kupiec_pof_test(hits: np.ndarray, alpha: float) -> dict[str, float]:
    """Kupiec Proportion-of-Failures (POF) test.

    Args:
    ----
        hits: 1 if return < VaR, else 0.
        alpha: Target tail probability (e.g. 0.01).

    Returns:
    -------
        Dict with n, x, hit_rate, lr_uc, p_value.

    """
    h = np.asarray(hits, dtype=float).ravel()
    m = np.isfinite(h)
    h = h[m]
    n = int(h.size)
    x = int(np.sum(h > GARCH_EVAL_HALF))
    if n == 0:
        return {
            "n": 0.0,
            "x": 0.0,
            "hit_rate": float("nan"),
            "lr_uc": float("nan"),
            "p_value": float("nan"),
        }
    phat = x / max(1, n)

    # Likelihood ratio for unconditional coverage
    def _lnp(p: float) -> float:
        p_clipped = _clip_probability(p, GARCH_EVAL_EPSILON)
        return (n - x) * np.log(1 - p_clipped) + x * np.log(p_clipped)

    lr_uc = -2.0 * (_lnp(alpha) - _lnp(phat))
    p_val = chi2_sf(float(lr_uc), df=1)
    return {
        "n": float(n),
        "x": float(x),
        "hit_rate": float(phat),
        "lr_uc": float(lr_uc),
        "p_value": float(p_val),
    }


def _compute_transition_counts(h: np.ndarray) -> tuple[int, int, int, int]:
    """Compute transition counts for Markov chain."""
    n00 = int(np.sum((h[1:] == 0) & (h[:-1] == 0)))
    n01 = int(np.sum((h[1:] == 1) & (h[:-1] == 0)))
    n10 = int(np.sum((h[1:] == 0) & (h[:-1] == 1)))
    n11 = int(np.sum((h[1:] == 1) & (h[:-1] == 1)))
    return n00, n01, n10, n11


def _compute_lr_independence(n00: int, n01: int, n10: int, n11: int) -> tuple[float, float]:
    """Compute likelihood ratio statistic for independence test."""
    n0 = max(1, n00 + n01)
    n1 = max(1, n10 + n11)
    p01 = n01 / n0
    p11 = n11 / n1
    p = (n01 + n11) / max(1, n00 + n01 + n10 + n11)

    def _ll(p0: float, p1: float) -> float:
        p0_clipped = _clip_probability(p0, GARCH_EVAL_EPSILON)
        p1_clipped = _clip_probability(p1, GARCH_EVAL_EPSILON)
        return (
            n00 * np.log(1 - p0_clipped)
            + n01 * np.log(p0_clipped)
            + n10 * np.log(1 - p1_clipped)
            + n11 * np.log(p1_clipped)
        )

    l1 = _ll(p01, p11)
    l0 = _ll(p, p)
    lr_ind = -2.0 * (l0 - l1)
    p_val = chi2_sf(float(lr_ind), df=1)
    return float(lr_ind), float(p_val)


def christoffersen_ind_test(hits: np.ndarray) -> dict[str, float]:
    """Christoffersen independence test (first-order Markov).

    Returns LR_ind and p-value (df=1) and the transition counts.
    """
    h = (np.asarray(hits, dtype=float).ravel() > GARCH_EVAL_HALF).astype(int)
    if h.size <= 1:
        return {
            "lr_ind": float("nan"),
            "p_value": float("nan"),
            "n00": 0.0,
            "n01": 0.0,
            "n10": 0.0,
            "n11": 0.0,
        }

    n00, n01, n10, n11 = _compute_transition_counts(h)
    lr_ind, p_val = _compute_lr_independence(n00, n01, n10, n11)

    return {
        "lr_ind": lr_ind,
        "p_value": p_val,
        "n00": float(n00),
        "n01": float(n01),
        "n10": float(n10),
        "n11": float(n11),
    }


# ==============================================================================


def var_backtest_metrics(
    e: np.ndarray,
    sigma2: np.ndarray,
    *,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
    alphas: list[float],
) -> dict[str, dict[str, float]]:
    """Compute VaR backtests (Kupiec/Christoffersen) for multiple alphas."""
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2) & (s2 > 0)
    out: dict[str, dict[str, float]] = {}
    for a in alphas:
        alpha_f = float(a)
        var_t = build_var_series(s2[m], alpha_f, dist, nu, lambda_skew)
        hits = (e[m] < var_t).astype(int)
        kup = kupiec_pof_test(hits, alpha_f)
        ind = christoffersen_ind_test(hits)
        expected_violations = float(alpha_f * kup["n"])
        violation_pct = (
            float(kup["hit_rate"] * 100.0) if np.isfinite(kup["hit_rate"]) else float("nan")
        )
        expected_pct = float(alpha_f * 100.0)
        excess_rate_pct = (
            violation_pct - expected_pct if np.isfinite(violation_pct) else float("nan")
        )
        excess_violations = float(kup["x"] - expected_violations)
        out[str(a)] = {
            "alpha": float(alpha_f),
            "n": kup["n"],
            "violations": kup["x"],
            "expected_violations": expected_violations,
            "hit_rate": kup["hit_rate"],
            "expected_rate": alpha_f,
            "violation_rate_pct": violation_pct,
            "expected_rate_pct": expected_pct,
            "excess_violations": excess_violations,
            "excess_rate_pct": excess_rate_pct,
            "lr_uc": kup["lr_uc"],
            "p_uc": kup["p_value"],
            "lr_ind": ind["lr_ind"],
            "p_ind": ind["p_value"],
            "lr_cc": float(kup["lr_uc"] + ind["lr_ind"]),
            "p_cc": chi2_sf(float(kup["lr_uc"] + ind["lr_ind"]), df=2),
        }
    return out


def summarize_var_backtests(
    e: np.ndarray,
    sigma2: np.ndarray,
    *,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
    alphas: list[float],
) -> dict[str, object]:
    """Build structured VaR summary for reporting and persistence (test split only)."""
    alpha_list = [float(a) for a in alphas]
    metrics = var_backtest_metrics(
        e,
        sigma2,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
        alphas=alpha_list,
    )
    first_level = next(iter(metrics.values()), None)
    n_obs = int(first_level["n"]) if first_level is not None else 0
    qlike_val = qlike_loss(e, sigma2)
    totals = {
        "violations": float(sum(level["violations"] for level in metrics.values())),
        "expected_violations": float(
            sum(level["expected_violations"] for level in metrics.values())
        ),
    }
    return {
        "n_obs": n_obs,
        "qlike": float(qlike_val),
        "alphas": alpha_list,
        "distribution": {
            "name": dist,
            "nu": float(nu) if nu is not None else None,
            "lambda_skew": float(lambda_skew) if lambda_skew is not None else None,
        },
        "levels": metrics,
        "totals": totals,
    }


def save_var_summary_json(summary: dict[str, object]) -> None:
    """Persist VaR summary to the dedicated evaluation JSON file."""
    save_json_pretty(summary, GARCH_EVAL_VAR_SUMMARY_FILE)
    logger.info("Saved GARCH VaR summary to: %s", GARCH_EVAL_VAR_SUMMARY_FILE)


def compute_test_evaluation_metrics(
    resid: np.ndarray,
    sigma2: np.ndarray,
    *,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
    alphas: list[float],
) -> dict[str, object]:
    """Compute comprehensive test evaluation metrics (QLIKE, MSE, VaR).

    Uses sigma2_garch (variance forecasts) as the primary input for evaluation.

    Args:
        resid: Test residuals (realized values).
        sigma2: Variance forecasts (sigma2_garch).
        dist: Distribution type ('student' or 'skewt').
        nu: Degrees of freedom for Student-t/Skew-t.
        lambda_skew: Skewness parameter for Skew-t.
        alphas: VaR alpha levels (e.g., [0.01, 0.05]).

    Returns:
        Dictionary containing QLIKE, MSE, and VaR metrics.
    """
    e = np.asarray(resid, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2) & (s2 > 0)
    e_valid = e[m]
    s2_valid = s2[m]

    if e_valid.size == 0:
        return {
            "n_obs": 0,
            "qlike": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "var_metrics": {},
        }

    # Compute core metrics
    qlike_val = qlike_loss(e_valid, s2_valid)
    variance_metrics = mse_mae_variance(e_valid, s2_valid)

    # Compute VaR metrics for each alpha
    var_metrics_dict = var_backtest_metrics(
        e_valid,
        s2_valid,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
        alphas=alphas,
    )

    return {
        "n_obs": int(e_valid.size),
        "qlike": float(qlike_val),
        "mse": variance_metrics["mse"],
        "mae": variance_metrics["mae"],
        "var_metrics": var_metrics_dict,
    }


def save_test_evaluation_metrics(metrics: dict[str, object]) -> None:
    """Persist test evaluation metrics to JSON file in results/garch/evaluation.

    Args:
        metrics: Dictionary with test evaluation metrics.
    """
    from src.path import GARCH_EVAL_TEST_METRICS_FILE

    save_json_pretty(metrics, GARCH_EVAL_TEST_METRICS_FILE)
    logger.info("Saved GARCH test evaluation metrics to: %s", GARCH_EVAL_TEST_METRICS_FILE)


def _load_and_filter_test_data(
    params: dict[str, float],
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset and extract filtered test residuals and variance."""
    dataset_df = load_dataset_for_metrics()
    e_test, s2_test = load_test_resid_sigma2(
        params,
        dataset_df,
        model_name=model_name,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )
    return filter_test_data(e_test, s2_test)


def compute_classic_metrics_from_artifacts(
    *,
    params: dict[str, float],
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
    alphas: list[float],
    apply_mz_calibration: bool = False,
) -> dict[str, object]:
    """Compute classic GARCH metrics on the test split and return a summary dict.

    DATA LEAKAGE PREVENTION:
    - Uses ex-ante variance forecasts for test period (computed via walk-forward)
    - Does NOT use test residuals to compute test variance a posteriori
    - Forecasts are computed using only data up to each forecast point

    Args:
    ----
        params: Model parameters dictionary.
        model_name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).
        alphas: VaR alpha levels (default: [0.01, 0.05]).
        apply_mz_calibration: Whether to apply MZ calibration to variances.
            WARNING: Setting this to True introduces data leakage (uses test data
            to calibrate forecasts). Use only for diagnostic purposes.

    Returns:
    -------
        Dictionary with all computed metrics.

    """
    # Load and filter test data
    e_test, s2_test = _load_and_filter_test_data(params, model_name, dist, nu, lambda_skew)

    # Compute all metrics
    return compute_all_metrics(
        e_test,
        s2_test,
        dist,
        nu,
        alphas,
        lambda_skew=lambda_skew,
        use_mz_calibration=apply_mz_calibration,
    )


def save_metrics_json(payload: dict[str, object]) -> None:
    """Persist metrics to JSON path in constants.

    Delegates to src.utils.save_json_pretty() for consistency.
    """
    save_json_pretty(payload, GARCH_EVAL_METRICS_FILE)
    logger.info("Saved GARCH evaluation metrics to: %s", GARCH_EVAL_METRICS_FILE)


# ============================================================================
# Metrics computation utilities
# ============================================================================


def compute_variance_metrics(
    e_test: np.ndarray,
    s2_test: np.ndarray,
) -> dict[str, float]:
    """Compute variance forecast metrics (QLIKE, MSE, MAE, R²).

    Args:
    ----
        e_test: Test residuals.
        s2_test: Test variance.

    Returns:
    -------
        Dictionary with variance metrics.

    """
    out_losses = mse_mae_variance(e_test, s2_test)
    return {
        "n_test": int(e_test.size),
        "qlike": qlike_loss(e_test, s2_test),
        "mse_var": out_losses["mse"],
        "mae_var": out_losses["mae"],
        "r2_var": r2_variance(e_test, s2_test),
    }


def apply_mz_calibration_if_requested(
    e_test: np.ndarray,
    s2_test: np.ndarray,
    use_mz_calibration: bool,
) -> tuple[np.ndarray, dict[str, float], float, float]:
    """Apply MZ calibration to test variances if requested.

    Args:
    ----
        e_test: Test residuals.
        s2_test: Test variance.
        use_mz_calibration: Whether to apply calibration.

    Returns:
    -------
        Tuple of (calibrated_variance, mz_results, intercept, slope).

    """
    mz_results = mincer_zarnowitz(e_test, s2_test)
    mz_intercept = mz_results.get("intercept", 0.0)
    mz_slope = mz_results.get("slope")
    if mz_slope is None:
        raise ValueError("Mincer-Zarnowitz results must include 'slope'")

    if use_mz_calibration:
        s2_calibrated = apply_mz_calibration(s2_test, mz_intercept, mz_slope, use_intercept=False)
    else:
        s2_calibrated = s2_test

    return s2_calibrated, mz_results, mz_intercept, mz_slope


def add_comparison_metrics(
    out: dict[str, object],
    e_test: np.ndarray,
    s2_test: np.ndarray,
    s2_calibrated: np.ndarray,
    use_mz_calibration: bool,
) -> None:
    """Add comparison metrics between original and calibrated variances.

    Args:
    ----
        out: Output dictionary to update.
        e_test: Test residuals.
        s2_test: Original test variance.
        s2_calibrated: Calibrated test variance.
        use_mz_calibration: Whether calibration was applied.

    """
    if use_mz_calibration:
        variance_metrics_original = compute_variance_metrics(e_test, s2_test)
        out["variance_metrics_original"] = variance_metrics_original
        mz_calibrated = mincer_zarnowitz(e_test, s2_calibrated)
        out["mz_calibrated"] = {f"mz_{k}": v for k, v in mz_calibrated.items()}


def _add_mz_metrics(
    out: dict[str, object],
    mz_results: dict[str, float],
    mz_intercept: float,
    mz_slope: float,
    use_mz_calibration: bool,
) -> None:
    """Add Mincer-Zarnowitz metrics to output dictionary."""
    out.update({f"mz_{k}": v for k, v in mz_results.items()})
    out["mz_calibration"] = {
        "intercept": float(mz_intercept),
        "slope": float(mz_slope),
        "applied": use_mz_calibration,
    }


def _prepare_mz_calibration(
    e_test: np.ndarray,
    s2_test: np.ndarray,
    use_mz_calibration: bool,
) -> tuple[np.ndarray, dict[str, float], float, float]:
    """Prepare MZ calibration data.

    Args:
    ----
        e_test: Test residuals.
        s2_test: Test variance.
        use_mz_calibration: Whether to apply calibration.

    Returns:
    -------
        Tuple of (calibrated_variance, mz_results, intercept, slope).

    """
    return apply_mz_calibration_if_requested(e_test, s2_test, use_mz_calibration)


def _compute_core_metrics(
    e_test: np.ndarray,
    s2_for_metrics: np.ndarray,
    mz_results: dict[str, float],
    mz_intercept: float,
    mz_slope: float,
    use_mz_calibration: bool,
) -> dict[str, object]:
    """Compute core variance and MZ metrics.

    Args:
    ----
        e_test: Test residuals.
        s2_for_metrics: Variance to use for metrics computation.
        mz_results: MZ regression results.
        mz_intercept: MZ intercept.
        mz_slope: MZ slope.
        use_mz_calibration: Whether calibration was applied.

    Returns:
    -------
        Dictionary with variance and MZ metrics.

    """
    out: dict[str, object] = {}

    # Add variance metrics
    variance_metrics = compute_variance_metrics(e_test, s2_for_metrics)
    out.update(variance_metrics)

    # Add MZ metrics
    _add_mz_metrics(out, mz_results, mz_intercept, mz_slope, use_mz_calibration)

    return out


def _add_var_backtests_to_metrics(
    out: dict[str, object],
    e_test: np.ndarray,
    s2_test: np.ndarray,
    dist: str,
    nu: float | None,
    alphas: list[float],
    lambda_skew: float | None = None,
) -> None:
    """Add VaR backtests to metrics dictionary.

    Args:
    ----
        out: Output metrics dictionary to update.
        e_test: Test residuals.
        s2_test: Test variance.
        dist: Distribution type.
        nu: Degrees of freedom.
        alphas: VaR alpha levels.
        lambda_skew: Skewness parameter (for Skew-t).

    """
    out["var_backtests"] = var_backtest_metrics(
        e_test, s2_test, dist=dist, nu=nu, lambda_skew=lambda_skew, alphas=alphas
    )


def compute_all_metrics(
    e_test: np.ndarray,
    s2_test: np.ndarray,
    dist: str,
    nu: float | None,
    alphas: list[float],
    *,
    lambda_skew: float | None = None,
    use_mz_calibration: bool = False,
) -> dict[str, object]:
    """Compute all GARCH evaluation metrics.

    Args:
    ----
        e_test: Test residuals.
        s2_test: Test variance.
        dist: Distribution type.
        nu: Degrees of freedom.
        alphas: VaR alpha levels.
        lambda_skew: Skewness parameter (for Skew-t).
        use_mz_calibration: Whether to apply MZ calibration.
            Defaults to False to avoid data leakage. Set to True only for diagnostic purposes.

    Returns:
    -------
        Dictionary with all metrics.

    """
    # Prepare MZ calibration
    s2_calibrated, mz_results, mz_intercept, mz_slope = _prepare_mz_calibration(
        e_test, s2_test, use_mz_calibration
    )

    # Compute core metrics
    s2_for_metrics = s2_calibrated if use_mz_calibration else s2_test
    out = _compute_core_metrics(
        e_test, s2_for_metrics, mz_results, mz_intercept, mz_slope, use_mz_calibration
    )

    # Add VaR backtests
    _add_var_backtests_to_metrics(out, e_test, s2_test, dist, nu, alphas, lambda_skew)

    # Add comparison metrics
    add_comparison_metrics(out, e_test, s2_test, s2_calibrated, use_mz_calibration)

    return out


__all__ = [
    # Basic metrics
    "qlike_loss",
    "mse_mae_variance",
    "mincer_zarnowitz",
    "kupiec_pof_test",
    "christoffersen_ind_test",
    "var_backtest_metrics",
    "summarize_var_backtests",
    "compute_classic_metrics_from_artifacts",
    "save_metrics_json",
    "save_var_summary_json",
    "compute_test_evaluation_metrics",
    "save_test_evaluation_metrics",
    # Quantiles
    "quantile",
    "var_quantile",
    "build_var_series",
    # Metrics computation
    "compute_variance_metrics",
    "r2_variance",
    "apply_mz_calibration_if_requested",
    "add_comparison_metrics",
    "compute_all_metrics",
]
