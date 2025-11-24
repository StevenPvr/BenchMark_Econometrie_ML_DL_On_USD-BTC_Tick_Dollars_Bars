"""Statistical utilities for GARCH diagnostics (ACF/PACF + Ljung-Box).

This module consolidates autocorrelation and hypothesis testing helpers
to reduce fragmentation and follow DRY/KISS principles.
"""

from __future__ import annotations

import numpy as np

# =============================
# Autocorrelation and PACF
# =============================


def compute_autocorr_denominator(x: np.ndarray) -> float:
    """Compute denominator for autocorrelation calculation."""
    return float(np.sum(x * x))


def compute_autocorr_lag(x: np.ndarray, k: int, denom: float) -> float:
    """Compute autocorrelation for a single lag."""
    if denom == 0.0:
        return 0.0
    num = float(np.sum(x[k:] * x[:-k]))
    return num / denom


def autocorr(x: np.ndarray, nlags: int) -> np.ndarray:
    """Return sample autocorrelation r_k for k=0..nlags.

    Uses a mean-centered series with a biased denominator (sum of squares)
    to avoid dependency on external packages.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.zeros(nlags + 1, dtype=float)
    x = x - float(np.nanmean(x))
    denom = compute_autocorr_denominator(x)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros(nlags + 1, dtype=float)
    r = np.empty(nlags + 1, dtype=float)
    r[0] = 1.0
    for k in range(1, nlags + 1):
        r[k] = compute_autocorr_lag(x, k, denom)
    return r


def pacf_init_first_lag(r: np.ndarray, phi_prev: np.ndarray) -> float:
    """Initialize PACF for first lag (k=1)."""
    phi_kk = r[1]
    phi_prev[0] = phi_kk
    return phi_kk


def pacf_compute_lag(
    r: np.ndarray, k: int, phi_prev: np.ndarray, den_prev: float
) -> tuple[float, float]:
    """Compute PACF for lag k > 1 via Durbin-Levinson recursion."""
    num = r[k] - float(np.dot(phi_prev[: k - 1], r[1:k][::-1]))
    den = den_prev
    phi_kk = 0.0 if den <= 0.0 or not np.isfinite(den) else num / den
    phi_new = phi_prev[: k - 1] - phi_kk * phi_prev[: k - 1][::-1]
    phi_prev[: k - 1] = phi_new
    phi_prev[k - 1] = phi_kk
    den_prev = 1.0 - float(np.dot(phi_prev[:k], r[1 : k + 1]))
    return phi_kk, den_prev


def pacf_from_autocorr(r: np.ndarray, nlags: int) -> np.ndarray:
    """Compute PACF(1..nlags) from autocorrelation r[0..nlags]."""
    nlags = int(nlags)
    if nlags <= 0:
        return np.asarray([], dtype=float)
    if r.size < (nlags + 1):
        r = np.pad(r, (0, nlags + 1 - r.size), constant_values=0.0)
    pacf = np.empty(nlags, dtype=float)
    phi_prev = np.zeros(nlags, dtype=float)
    den_prev = 1.0
    for k in range(1, nlags + 1):
        if k == 1:
            pacf_init_first_lag(r, phi_prev)
            den_prev = 1.0 - phi_prev[0] * r[1]
        else:
            _, den_prev = pacf_compute_lag(r, k, phi_prev, den_prev)
        pacf[k - 1] = float(np.clip(phi_prev[k - 1], -1.0, 1.0))
    return pacf


# =============================
# Ljung-Box test
# =============================


def compute_ljung_box_statistics(
    series: np.ndarray,
    lags: int,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test statistics for a given series."""
    lags_list = list(range(1, int(lags) + 1))
    r = autocorr(series, max(lags_list))
    n = float(np.sum(np.isfinite(series)))
    q_stats: list[float] = []
    p_values: list[float] = []
    try:
        from scipy.stats import chi2  # type: ignore

        has_scipy = True
    except ImportError:  # pragma: no cover - optional
        has_scipy = False
    s = 0.0
    for h in lags_list:
        rk = r[h]
        s += (rk * rk) / max(1.0, (n - h))
        q = n * (n + 2.0) * s
        q_stats.append(float(q))
        if has_scipy:
            p = float(chi2.sf(q, df=h))  # type: ignore[attr-defined]
        else:
            p = float("nan")
        p_values.append(p)
    return {"lags": lags_list, "lb_stat": q_stats, "lb_pvalue": p_values}


__all__ = [
    # ACF/PACF
    "autocorr",
    "compute_autocorr_denominator",
    "compute_autocorr_lag",
    "pacf_init_first_lag",
    "pacf_compute_lag",
    "pacf_from_autocorr",
    # Ljung-Box
    "compute_ljung_box_statistics",
]
