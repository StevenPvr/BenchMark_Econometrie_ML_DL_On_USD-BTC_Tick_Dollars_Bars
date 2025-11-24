"""CLI for post-estimation diagnostics of GARCH models.

Implements methodology for verifying GARCH model adequacy:
1. Verify standardized residuals εt/σt behave as centered white noise
2. Verify squared standardized residuals show no significant autocorrelation
   (ACF/PACF plots + Ljung-Box tests)
3. Verify distribution adequacy for zt (Normal or Student-t)
   (graphical diagnostics + normality tests)

Results saved to results/garch/diagnostic/
Plots saved to plots/garch/diagnostics/
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root before any src imports
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


import numpy as np

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DIAGNOSTIC_DIR,
    GARCH_DIAGNOSTICS_PLOTS_DIR,
    GARCH_DISTRIBUTION_DIAGNOSTICS_FILE,
    GARCH_LJUNG_BOX_LAGS_DEFAULT,
    GARCH_LJUNG_BOX_SPECIFIC_LAGS,
    GARCH_LJUNGBOX_FILE,
    GARCH_STD_ACF_PACF_PLOT,
    GARCH_STD_HISTOGRAM_PLOT,
    GARCH_STD_QQ_PLOT,
    GARCH_STD_SQUARED_ACF_PACF_PLOT,
)
from src.garch.garch_diagnostic.advanced_diagnostics import (
    compute_comprehensive_diagnostics,
    compute_ljung_box_at_specific_lags,
    save_comprehensive_diagnostics,
)
from src.garch.garch_diagnostic.diagnostics import (
    compute_distribution_diagnostics,
    compute_ljung_box_on_std_squared,
    save_acf_pacf_std_plots,
    save_acf_pacf_std_squared_plots,
    save_histogram_std_residuals,
    save_qq_plot_std_residuals,
)
from src.garch.garch_diagnostic.io import (
    load_data_and_params,
    save_diagnostics_json,
    validate_dict_field,
)
from src.garch.garch_diagnostic.standardization import standardize_residuals
from src.utils import get_logger

logger = get_logger(__name__)


def _generate_acf_pacf_plots(
    resid_test: np.ndarray,
    best_params: dict[str, float],
    dist: str | None,
    nu: float | None,
) -> None:
    """Generate and save ACF/PACF plots for standardized residuals.

    Verifies:
    1. Standardized residuals (z) behave as centered white noise
    2. Squared standardized residuals (z²) show no significant autocorrelation

    Args:
        resid_test: Test residuals array εt.
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name ('student' or 'skewt').
        nu: Degrees of freedom for Student-t distribution (if applicable).
    """
    save_acf_pacf_std_squared_plots(
        resid_test,
        best_params,
        lags=GARCH_ACF_LAGS_DEFAULT,
        outdir=GARCH_DIAGNOSTICS_PLOTS_DIR,
        filename=GARCH_STD_SQUARED_ACF_PACF_PLOT.name,
        dist=(dist or "student"),
        nu=nu,
    )
    save_acf_pacf_std_plots(
        resid_test,
        best_params,
        lags=GARCH_ACF_LAGS_DEFAULT,
        outdir=GARCH_DIAGNOSTICS_PLOTS_DIR,
        filename=GARCH_STD_ACF_PACF_PLOT.name,
        dist=(dist or "student"),
        nu=nu,
    )
    logger.info("Saved ACF/PACF plots to %s", GARCH_DIAGNOSTICS_PLOTS_DIR)


def _save_ljung_box_squared_results(
    resid_test: np.ndarray,
    best_params: dict[str, float],
    dist: str,
    nu: float | None,
) -> None:
    """Save Ljung-Box test results for squared standardized residuals.

    Includes results at all lags up to default, plus specific results at lags 10 and 20.
    """

    # Full Ljung-Box up to default lag
    lb2_full = compute_ljung_box_on_std_squared(
        resid_test,
        best_params,
        lags=GARCH_LJUNG_BOX_LAGS_DEFAULT,
        dist=dist,
        nu=nu,
    )

    # Specific lags from constants
    z = standardize_residuals(resid_test, best_params, dist=dist, nu=nu)
    z_squared = z**2 - np.mean(z**2)
    lb2_specific = compute_ljung_box_at_specific_lags(z_squared, GARCH_LJUNG_BOX_SPECIFIC_LAGS)

    # Combine results
    results = {
        "full": lb2_full,
        "lags_10_20": lb2_specific,
    }

    save_diagnostics_json(results, GARCH_LJUNGBOX_FILE, "Ljung-Box(z^2) results")


def _run_ljung_box_test_squared(
    resid_test: np.ndarray,
    best_params: dict[str, float],
    dist: str | None,
    nu: float | None,
) -> None:
    """Run Ljung-Box test on squared standardized residuals (z²).

    Tests that volatility is correctly captured (z² should be uncorrelated).

    Raises:
        ValueError: If Ljung-Box test computation fails.
    """
    _save_ljung_box_squared_results(resid_test, best_params, (dist or "student"), nu)


def _run_ljung_box_tests(
    resid_test: np.ndarray,
    best_params: dict[str, float],
    dist: str | None,
    nu: float | None,
) -> None:
    """Run Ljung-Box tests on squared standardized residuals.

    Tests that volatility is correctly captured (z² should be uncorrelated).
    Results saved to results/garch/diagnostic/

    Args:
        resid_test: Test residuals array εt.
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name ('student' or 'skewt').
        nu: Degrees of freedom for Student-t distribution (if applicable).
    """
    _run_ljung_box_test_squared(resid_test, best_params, dist, nu)


def _save_distribution_diagnostics(
    diag: dict[str, float | str | None],
) -> None:
    """Save distribution diagnostics to file."""
    save_diagnostics_json(diag, GARCH_DISTRIBUTION_DIAGNOSTICS_FILE, "distribution diagnostics")
    logger.info("Distribution diagnostics: %s", diag)


def _run_distribution_diagnostics(
    resid_test: np.ndarray,
    best_params: dict[str, float],
    dist: str | None,
) -> None:
    """Run distribution diagnostics and generate QQ plot.

    Verifies adequacy of chosen distribution (Normal or Student-t) for zt:
    - Graphical diagnostics: QQ plot
    - Numerical tests: Jarque-Bera, Kolmogorov-Smirnov

    Results saved to results/garch/diagnostic/
    Plots saved to plots/garch/diagnostics/

    Args:
        resid_test: Test residuals array εt.
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name ('student' or 'skewt').
    """
    # Note: best_params is already the params dict, but extract_nu_from_params expects full result
    # For now, we use nu directly as it's already extracted in load_data_and_params
    # This will be cleaned up when we refactor the parameter passing
    nu_from_params = best_params.get("nu")
    nu = float(nu_from_params) if nu_from_params is not None else None
    dist_name = dist or "student"
    diag = compute_distribution_diagnostics(resid_test, best_params, dist=dist_name, nu=nu)
    save_qq_plot_std_residuals(
        resid_test,
        best_params,
        dist=dist_name,
        nu=nu,
        outdir=GARCH_DIAGNOSTICS_PLOTS_DIR,
        filename=GARCH_STD_QQ_PLOT.name,
    )
    save_histogram_std_residuals(
        resid_test,
        best_params,
        dist=dist_name,
        nu=nu,
        outdir=GARCH_DIAGNOSTICS_PLOTS_DIR,
        filename=GARCH_STD_HISTOGRAM_PLOT.name,
    )
    _save_distribution_diagnostics(diag)


def _log_ljung_box_results(test_name: str, lb_results: dict) -> None:
    """Log Ljung-Box test results at multiple lags.

    Args:
        test_name: Description of the test (e.g., "z_t" or "z_t²").
        lb_results: Dictionary with keys "lags", "lb_stat", "lb_pvalue".
    """
    logger.info("Ljung-Box on %s (lags 10, 20):", test_name)
    lags = lb_results["lags"]
    stats = lb_results["lb_stat"]
    pvals = lb_results["lb_pvalue"]
    if isinstance(lags, list) and isinstance(stats, list) and isinstance(pvals, list):
        for lag, stat, pval in zip(lags, stats, pvals, strict=True):
            logger.info("  Lag %d: Q=%.4f, p_value=%.4f", lag, stat, pval)


def _log_comprehensive_diagnostics(diagnostics: dict) -> None:
    """Log comprehensive diagnostic results.

    Args:
        diagnostics: Dictionary with keys arch_lm, moments, ljung_box_z, ljung_box_z2.
    """
    arch_lm = diagnostics["arch_lm"]
    moments = diagnostics["moments"]
    lb_z = diagnostics["ljung_box_z"]
    lb_z2 = diagnostics["ljung_box_z2"]

    logger.info("=" * 60)
    logger.info("COMPREHENSIVE DIAGNOSTICS")
    logger.info("=" * 60)
    logger.info(
        "ARCH-LM test on z_t: LM_stat=%.4f, p_value=%.4f, df=%.0f",
        arch_lm["lm_stat"],
        arch_lm["p_value"],
        arch_lm["df"],
    )
    logger.info(
        "Moments of z_t: mean=%.4f, variance=%.4f, std=%.4f, skewness=%.4f, kurtosis=%.4f",
        moments["mean"],
        moments["variance"],
        moments["std"],
        moments["skewness"],
        moments["kurtosis"],
    )
    _log_ljung_box_results("z_t", lb_z)
    _log_ljung_box_results("z_t²", lb_z2)


def _extract_comprehensive_diagnostics_params(
    best_params: dict[str, float],
    dist: str | None,
) -> tuple[str, float | None]:
    """Extract parameters for comprehensive diagnostics computation.

    Args:
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name.

    Returns:
        Tuple of (distribution_name, lambda_skew).
    """
    lambda_skew_val = best_params.get("lambda_skew") or best_params.get("lambda")
    lambda_skew = float(lambda_skew_val) if lambda_skew_val is not None else None
    dist_name = dist or "student"
    return dist_name, lambda_skew


def _validate_comprehensive_diagnostics_structure(diagnostics: dict) -> None:
    """Validate that comprehensive diagnostics has expected structure.

    Args:
        diagnostics: Computed diagnostics dictionary.

    Raises:
        KeyError: If required fields are missing.
        TypeError: If required fields are not dictionaries.
    """
    validate_dict_field(diagnostics, "arch_lm")
    validate_dict_field(diagnostics, "moments")
    validate_dict_field(diagnostics, "ljung_box_z")
    validate_dict_field(diagnostics, "ljung_box_z2")


def _save_comprehensive_diagnostics_results(
    resid_test: np.ndarray,
    best_params: dict[str, float],
    diagnostics: dict,
    dist_name: str,
    nu: float | None,
    lambda_skew: float | None,
) -> None:
    """Save comprehensive diagnostics results to file.

    Args:
        resid_test: Test residuals array.
        best_params: Best GARCH parameters dictionary.
        diagnostics: Computed diagnostics dictionary.
        dist_name: Distribution name.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.
    """
    output_file = GARCH_DIAGNOSTIC_DIR / "comprehensive_diagnostics.json"
    save_comprehensive_diagnostics(
        resid_test,
        best_params,
        output_file=output_file,
        dist=dist_name,
        nu=nu,
        lambda_skew=lambda_skew,
    )


def _run_comprehensive_diagnostics(
    resid_test: np.ndarray,
    best_params: dict[str, float],
    dist: str | None,
    nu: float | None,
) -> None:
    """Run comprehensive diagnostic tests.

    Performs:
    1. ARCH-LM test on z_t (should not reject H0: no ARCH effects)
    2. Ljung-Box on z_t at lags 10, 20 (should not reject H0: no autocorrelation)
    3. Ljung-Box on z_t² at lags 10, 20 (should not reject H0: no autocorrelation)
    4. Moment validation (mean ≈ 0, variance ≈ 1)

    Results saved to results/garch/diagnostic/comprehensive_diagnostics.json

    Args:
        resid_test: Test residuals array εt (1-step ahead forecast residuals).
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name.
        nu: Degrees of freedom (for Student-t/Skew-t).
    """
    # Extract parameters
    dist_name, lambda_skew = _extract_comprehensive_diagnostics_params(best_params, dist)

    # Compute diagnostics
    diagnostics = compute_comprehensive_diagnostics(
        resid_test,
        best_params,
        dist=dist_name,
        nu=nu,
        lambda_skew=lambda_skew,
    )

    # Validate structure
    _validate_comprehensive_diagnostics_structure(diagnostics)

    # Log results
    _log_comprehensive_diagnostics(diagnostics)

    # Save to file
    _save_comprehensive_diagnostics_results(
        resid_test, best_params, diagnostics, dist_name, nu, lambda_skew
    )


def main() -> None:
    """Run post-estimation GARCH diagnostics.

    Methodology:
    1. Verify standardized residuals εt/σt behave as centered white noise
    2. Verify squared standardized residuals show no significant autocorrelation
       (ACF/PACF plots + Ljung-Box tests)
    3. Verify distribution adequacy for zt (Normal or Student-t)
       (graphical diagnostics + normality tests)
    4. Comprehensive diagnostics:
       - ARCH-LM test on z_t (should not reject H0: no ARCH effects)
       - Ljung-Box on z_t and z_t² at lags 10, 20
       - Moment validation (mean ≈ 0, variance ≈ 1)
       - Engle-Ng asymmetry tests (if EGARCH)
       - Nyblom parameter stability test
    5. Generate comprehensive diagnostic report (JSON + plots)

    CRITICAL: All diagnostics are performed on 1-step ahead forecast residuals
    (test set), NOT in-sample residuals, to ensure out-of-sample validity.

    Results saved to results/garch/diagnostic/
    Plots saved to plots/garch/diagnostics/
    """
    logger.info("=" * 60)
    logger.info("GARCH DIAGNOSTICS (post-estimation)")
    logger.info("=" * 60)

    try:
        resid_test, dist, best, nu = load_data_and_params()
    except ValueError as e:
        logger.error("Cannot run diagnostics: %s", e)
        return

    _generate_acf_pacf_plots(resid_test, best, dist, nu)
    _run_ljung_box_tests(resid_test, best, dist, nu)
    _run_distribution_diagnostics(resid_test, best, dist)
    _run_comprehensive_diagnostics(resid_test, best, dist, nu)


if __name__ == "__main__":
    main()
