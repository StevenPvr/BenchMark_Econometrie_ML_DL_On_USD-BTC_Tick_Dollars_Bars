"""Advanced diagnostics for GARCH models.

Implements comprehensive diagnostic checks:
- ARCH-LM test on standardized residuals
- Ljung-Box tests at specific lags (10, 20)
- Mean and variance validation for standardized residuals
- Distribution moment checks (skewness/kurtosis)
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, cast

import numpy as np

from src.constants import GARCH_LM_LAGS_DEFAULT, GARCH_STD_EPSILON
from src.path import GARCH_DIAGNOSTIC_DIR
from src.garch.garch_diagnostic.io import save_diagnostics_json
from src.garch.garch_diagnostic.standardization import standardize_residuals
from src.garch.garch_diagnostic.statistics import compute_ljung_box_statistics
from src.garch.structure_garch.utils import compute_arch_lm_test, run_all_engle_ng_tests
from src.utils import get_logger

logger = get_logger(__name__)

DiagnosticResultType: TypeAlias = dict[
    str,
    dict[str, float]
    | dict[str, list[int] | list[float]]
    | dict[str, dict[str, float]]
    | dict[str, float | dict[str, float] | int],
]


def compute_arch_lm_on_standardized(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_LM_LAGS_DEFAULT,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> dict[str, float]:
    """Compute ARCH-LM test on standardized residuals z_t.

    Tests for residual ARCH effects in standardized residuals.
    If GARCH model is adequate, z_t should show no ARCH effects.

    Args:
    ----
        residuals: Raw residuals εt from mean model.
        garch_params: GARCH parameters dictionary.
        lags: Number of lags in ARCH-LM regression.
        dist: Distribution name.
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Dictionary with lm_stat, p_value, df.

    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu, lambda_skew=lambda_skew)
    return compute_arch_lm_test(z, lags=lags)


def compute_ljung_box_at_specific_lags(
    series: np.ndarray,
    lags_list: list[int],
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test at specific lags.

    Args:
    ----
        series: Time series to test.
        lags_list: List of specific lags to test (e.g., [10, 20]).

    Returns:
    -------
        Dictionary with lags, lb_stat, lb_pvalue for each lag.

    """
    results = compute_ljung_box_statistics(series, max(lags_list))

    # Filter to only requested lags
    filtered_lags = []
    filtered_stats = []
    filtered_pvalues = []

    for lag in lags_list:
        if lag in results["lags"]:
            idx = results["lags"].index(lag)
            filtered_lags.append(lag)
            filtered_stats.append(results["lb_stat"][idx])
            filtered_pvalues.append(results["lb_pvalue"][idx])

    return {
        "lags": filtered_lags,
        "lb_stat": filtered_stats,
        "lb_pvalue": filtered_pvalues,
    }


def compute_standardized_residual_moments(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> dict[str, float]:
    """Compute and validate moments of standardized residuals z_t.

    For adequate GARCH model:
    - Mean should be ≈ 0
    - Variance should be ≈ 1
    - Skewness/kurtosis should be consistent with chosen distribution

    Args:
    ----
        residuals: Raw residuals εt from mean model.
        garch_params: GARCH parameters dictionary.
        dist: Distribution name.
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Dictionary with mean, variance, std, skewness, kurtosis, n_obs.

    """
    z_clean = standardize_residuals(
        residuals, garch_params, dist=dist, nu=nu, lambda_skew=lambda_skew, clean=True
    )

    if z_clean.size == 0:
        msg = "No valid standardized residuals for moment computation"
        raise ValueError(msg)

    mean_z = float(np.mean(z_clean))
    var_z = float(np.var(z_clean, ddof=0))  # Population variance
    std_z = float(np.std(z_clean, ddof=0))

    # Compute skewness and kurtosis
    zc = z_clean - mean_z
    skew = float(np.mean(zc**3) / (std_z**3 + GARCH_STD_EPSILON))
    kurt = float(np.mean(zc**4) / (std_z**4 + GARCH_STD_EPSILON))

    return {
        "mean": mean_z,
        "variance": var_z,
        "std": std_z,
        "skewness": skew,
        "kurtosis": kurt,
        "n_obs": int(z_clean.size),
    }


def _get_critical_values_by_k(k: int) -> dict[str, float]:
    """Get Nyblom test critical values based on number of parameters.

    Args:
    ----
        k: Number of parameters.

    Returns:
    -------
        Dictionary with critical values at 10%, 5%, 1%.

    """
    if k == 1:
        return {"10%": 0.35, "5%": 0.47, "1%": 0.75}
    if k == 2:
        return {"10%": 0.47, "5%": 0.58, "1%": 0.94}
    # k >= 3: Use approximate values for k=3 (most common case)
    return {"10%": 0.56, "5%": 0.68, "1%": 1.07}


def _compute_nyblom_statistic(z_clean: np.ndarray) -> float:
    """Compute Nyblom test statistic.

    Args:
    ----
        z_clean: Cleaned standardized residuals.

    Returns:
    -------
        Nyblom test statistic.

    """
    z2 = z_clean**2
    mean_z2 = np.mean(z2)
    cumsum = np.cumsum(z2 - mean_z2)
    n = len(z_clean)
    return float(np.sum(cumsum**2) / (n**2 * np.var(z2, ddof=1)))


def nyblom_stability_test(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> dict[str, float | dict[str, float] | int]:
    """Nyblom (1989) test for parameter stability in GARCH models.

    Tests H0: GARCH parameters are constant over time
    against H1: Parameters follow a random walk (time-varying).

    Args:
    ----
        residuals: Raw residuals εt from mean model.
        garch_params: GARCH parameters dictionary.
        dist: Distribution name.
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Dictionary with statistic, critical_values, n, k.

    Raises:
    ------
        ValueError: If sample size is too small (n < 10).

    Reference:
    ---------
        Nyblom, J. (1989). "Testing for the constancy of parameters over time."
        Journal of the American Statistical Association, 84(405), 223-230.

    """
    z_clean = standardize_residuals(
        residuals, garch_params, dist=dist, nu=nu, lambda_skew=lambda_skew, clean=True
    )
    n = len(z_clean)

    if n < 10:
        msg = f"Nyblom test requires at least 10 observations, got {n}"
        raise ValueError(msg)

    nyblom_stat = _compute_nyblom_statistic(z_clean)
    k = len([v for v in garch_params.values() if v is not None])
    crit_values = _get_critical_values_by_k(k)

    return {
        "statistic": nyblom_stat,
        "critical_values": crit_values,
        "n": n,
        "k": k,
    }


def compute_engle_ng_diagnostics(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> dict[str, dict[str, float]]:
    """Compute Engle-Ng (1993) sign bias tests for asymmetric volatility.

    Tests for presence of asymmetric effects in volatility:
    - Sign bias: different impact of positive vs negative shocks
    - Negative size bias: magnitude of negative shocks matters
    - Positive size bias: magnitude of positive shocks matters
    - Joint test: all three effects simultaneously

    Critical for validating EGARCH specification vs standard GARCH.

    Args:
    ----
        residuals: Raw residuals εt from mean model.
        garch_params: GARCH parameters dictionary.
        dist: Distribution name.
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Dictionary with sign_bias, negative_size_bias, positive_size_bias, joint tests.
        Each contains: coef, t_stat, p_value (or f_stat for joint test).

    Note:
    ----
        - For EGARCH: expect joint test to NOT reject H0 (model captures asymmetry)
        - For standard GARCH: expect joint test to REJECT H0 (missing asymmetry)

    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu, lambda_skew=lambda_skew)
    return run_all_engle_ng_tests(z)


def _compute_core_diagnostics(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    z: np.ndarray,
    *,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> dict[str, dict[str, float] | dict[str, list[int] | list[float]]]:
    """Compute core diagnostic tests (ARCH-LM, Ljung-Box, moments).

    Args:
    ----
        residuals: Raw residuals εt from mean model.
        garch_params: GARCH parameters dictionary.
        z: Standardized residuals.
        dist: Distribution name.
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Dictionary with core diagnostic results.

    """
    arch_lm = compute_arch_lm_on_standardized(
        residuals, garch_params, dist=dist, nu=nu, lambda_skew=lambda_skew
    )
    lb_z = compute_ljung_box_at_specific_lags(z, [10, 20])
    z_squared = z**2 - np.mean(z**2)
    lb_z2 = compute_ljung_box_at_specific_lags(z_squared, [10, 20])
    moments = compute_standardized_residual_moments(
        residuals, garch_params, dist=dist, nu=nu, lambda_skew=lambda_skew
    )
    return {
        "arch_lm": arch_lm,
        "ljung_box_z": lb_z,
        "ljung_box_z2": lb_z2,
        "moments": moments,
    }


def compute_comprehensive_diagnostics(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
    include_engle_ng: bool = True,
    include_nyblom: bool = True,
) -> DiagnosticResultType:
    """Compute comprehensive diagnostic tests for standardized residuals.

    Performs: ARCH-LM, Ljung-Box (z and z²), moments, Engle-Ng, Nyblom tests.

    Args:
    ----
        residuals: Raw residuals εt from mean model.
        garch_params: GARCH parameters dictionary.
        dist: Distribution name.
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).
        include_engle_ng: Whether to include Engle-Ng asymmetry tests.
        include_nyblom: Whether to include Nyblom stability test.

    Returns:
    -------
        Dictionary with all diagnostic results.

    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu, lambda_skew=lambda_skew)
    results: DiagnosticResultType = cast(
        DiagnosticResultType,
        _compute_core_diagnostics(
            residuals, garch_params, z, dist=dist, nu=nu, lambda_skew=lambda_skew
        ),
    )

    if include_engle_ng:
        results["engle_ng"] = compute_engle_ng_diagnostics(
            residuals, garch_params, dist=dist, nu=nu, lambda_skew=lambda_skew
        )

    if include_nyblom:
        results["nyblom"] = nyblom_stability_test(
            residuals, garch_params, dist=dist, nu=nu, lambda_skew=lambda_skew
        )

    return results


def save_comprehensive_diagnostics(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    output_file: Path | None = None,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
    include_engle_ng: bool = True,
    include_nyblom: bool = True,
) -> Path:
    """Compute and save comprehensive diagnostics to JSON file.

    Args:
    ----
        residuals: Raw residuals εt from mean model.
        garch_params: GARCH parameters dictionary.
        output_file: Output file path. Defaults to
            GARCH_DIAGNOSTIC_DIR / "comprehensive_diagnostics.json".
        dist: Distribution name.
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).
        include_engle_ng: Whether to include Engle-Ng asymmetry tests.
        include_nyblom: Whether to include Nyblom stability test.

    Returns:
    -------
        Path to saved file.

    """
    if output_file is None:
        output_file = GARCH_DIAGNOSTIC_DIR / "comprehensive_diagnostics.json"

    output_file = cast(Path, output_file)

    diagnostics = compute_comprehensive_diagnostics(
        residuals,
        garch_params,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
        include_engle_ng=include_engle_ng,
        include_nyblom=include_nyblom,
    )

    save_diagnostics_json(diagnostics, output_file, "comprehensive diagnostics")
    return output_file
