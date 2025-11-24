"""Post-estimation diagnostics for GARCH models.

Implements methodology for verifying GARCH model adequacy:
1. Verify standardized residuals εt/σt behave as centered white noise
2. Verify squared standardized residuals show no significant autocorrelation
   (ACF/PACF plots + Ljung-Box tests)
3. Verify distribution adequacy for zt (Normal or Student-t)
   (graphical diagnostics + normality tests)
"""

from __future__ import annotations

from typing import cast

from pathlib import Path

import numpy as np

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_LJUNG_BOX_LAGS_DEFAULT,
    GARCH_STD_EPSILON,
)
from src.path import GARCH_DIAGNOSTICS_PLOTS_DIR
from src.garch.garch_diagnostic.plotting import (
    _create_diagnostic_figure_canvas,
    compute_acf_pacf_data,
    compute_qq_data,
    create_residual_plots_figure,
    plot_acf_subplot,
    plot_pacf_subplot,
    plot_qq_scatter,
    prepare_output_path,
    prepare_residual_data,
    save_figure_or_placeholder,
    write_placeholder_png,
)
from src.garch.garch_diagnostic.standardization import (
    compute_standardized_residuals_for_plot,
    standardize_residuals,
)
from src.garch.garch_diagnostic.statistics import compute_ljung_box_statistics
from src.utils import get_logger
from src.data_visualisation import create_figure_canvas, plot_histogram_with_normal_overlay

logger = get_logger(__name__)
_DIAGNOSTIC_FIGURE_SIZE: tuple[int, int] = (10, 6)
_QQ_FIGURE_SIZE: tuple[int, int] = (6, 6)
_HISTOGRAM_COLOR = "#1f77b4"
_CONFIDENCE_COLOR = "red"
_DISTRIBUTION_HISTOGRAM_BINS = 100


def _compute_ljung_box_on_standardized(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    dist: str = "student",
    nu: float | None = None,
    squared: bool = False,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test on standardized residuals (optionally squared).

    Args:
        residuals: Raw residuals from GARCH model.
        garch_params: GARCH model parameters.
        lags: Number of lags for Ljung-Box test.
        dist: Distribution type.
        nu: Degrees of freedom for Student-t distribution.
        squared: If True, test squared residuals; if False, test raw standardized residuals.

    Returns:
        Ljung-Box test results.
    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    if squared:
        y = (z**2) - np.mean(z**2)
        return compute_ljung_box_statistics(y, lags)
    return compute_ljung_box_statistics(z, lags)


def compute_ljung_box_on_std_squared(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    dist: str = "student",
    nu: float | None = None,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test on standardized squared residuals (z²).

    Verifies that squared standardized residuals show no significant autocorrelation.
    If the model captures volatility correctly, z² should be uncorrelated.
    """
    return _compute_ljung_box_on_standardized(
        residuals, garch_params, lags=lags, dist=dist, nu=nu, squared=True
    )


def compute_ljung_box_on_std(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    dist: str = "student",
    nu: float | None = None,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test on standardized residuals (z).

    Tests for white noise behavior: standardized residuals should show
    no significant autocorrelation if the model captures volatility correctly.
    """
    return _compute_ljung_box_on_standardized(
        residuals, garch_params, lags=lags, dist=dist, nu=nu, squared=False
    )


def _get_plot_config(squared: bool) -> tuple[str, str, str, str, str]:
    """Get plot configuration based on whether residuals are squared.

    Args:
        squared: Whether to plot squared residuals.

    Returns:
        Tuple of (plot_name, acf_title, pacf_title, suptitle, log_message).
    """
    if squared:
        return (
            "garch_std_squared_acf_pacf",
            "ACF of standardized squared residuals (z^2)",
            "PACF of standardized squared residuals (z^2)",
            "ACF/PACF of standardized squared residuals",
            "Saved ACF/PACF(z^2) plots",
        )
    return (
        "garch_std_acf_pacf",
        "ACF of standardized residuals (z)",
        "PACF of standardized residuals (z)",
        "ACF/PACF of standardized residuals",
        "Saved ACF/PACF(z) plots",
    )


def _save_acf_pacf_plots(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    squared: bool = False,
    lags: int = GARCH_ACF_LAGS_DEFAULT,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
    dist: str = "student",
    nu: float | None = None,
) -> Path:
    """Save ACF and PACF plots of standardized residuals (z or z²).

    Args:
        residuals: Model residuals.
        garch_params: GARCH model parameters.
        squared: If True, plot ACF/PACF of z², otherwise of z.
        lags: Number of lags for ACF/PACF computation.
        outdir: Output directory for plots.
        filename: Custom filename (optional).
        dist: Distribution assumption.
        nu: Degrees of freedom for t-distribution.

    Returns:
        Path to saved plot file.
    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    y = (z**2 - np.mean(z**2)) if squared else (z - np.mean(z))
    plot_name, acf_title, pacf_title, suptitle, log_message = _get_plot_config(squared)

    acf, pacf_vals, se = compute_acf_pacf_data(y, lags)
    out_path = prepare_output_path(outdir, filename, plot_name)

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig, canvas = _create_diagnostic_figure_canvas(figsize=_DIAGNOSTIC_FIGURE_SIZE)
        ax1, ax2 = fig.subplots(2, 1)
        plot_acf_subplot(ax1, acf, se, acf_title, lags)
        plot_pacf_subplot(ax2, pacf_vals, se, pacf_title, lags)
        fig.suptitle(suptitle)
        # Convert Figure to FigureCanvasAgg for compatibility
        canvas_agg = FigureCanvasAgg(fig)
        save_figure_or_placeholder(canvas_agg, out_path, log_message)
    except ImportError:  # pragma: no cover - matplotlib optional
        write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


def save_acf_pacf_std_squared_plots(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_ACF_LAGS_DEFAULT,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
    dist: str = "student",
    nu: float | None = None,
) -> Path:
    """Save ACF and PACF plots of standardized squared residuals (z²).

    Verifies that squared standardized residuals show no significant autocorrelation.
    ACF/PACF should be near zero for all lags if volatility is correctly captured.
    """
    return _save_acf_pacf_plots(
        residuals,
        garch_params,
        squared=True,
        lags=lags,
        outdir=outdir,
        filename=filename,
        dist=dist,
        nu=nu,
    )


def save_acf_pacf_std_plots(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_ACF_LAGS_DEFAULT,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
    dist: str = "student",
    nu: float | None = None,
) -> Path:
    """Save ACF and PACF plots of standardized residuals (z).

    Verifies that standardized residuals εt/σt behave as centered white noise.
    For white noise, ACF/PACF should be near zero for all lags.
    """
    return _save_acf_pacf_plots(
        residuals,
        garch_params,
        squared=False,
        lags=lags,
        outdir=outdir,
        filename=filename,
        dist=dist,
        nu=nu,
    )


def _compute_moments(zc: np.ndarray) -> tuple[float, float]:
    """Compute skewness and kurtosis from centered standardized residuals."""
    skew = float(np.nanmean(zc**3))
    kurt = float(np.nanmean(zc**4))
    return skew, kurt


def _compute_jarque_bera_test(z: np.ndarray) -> tuple[float, float]:
    """Compute Jarque-Bera test statistics."""
    from scipy.stats import jarque_bera  # type: ignore

    jb = jarque_bera(z)
    jb_stat = float(jb.statistic)  # type: ignore[attr-defined]
    jb_p = float(jb.pvalue)  # type: ignore[attr-defined]
    return jb_stat, jb_p


def _compute_kolmogorov_smirnov_test(
    z: np.ndarray, dist: str, nu: float | None
) -> tuple[float, float, str]:
    """Compute Kolmogorov-Smirnov test statistics."""
    from scipy.stats import kstest, norm, t  # type: ignore

    if dist.lower() == "student" and nu is not None and nu > 2:
        ks = kstest(z, lambda x: t.cdf(x, df=nu))
        used = "student"
    else:
        ks = kstest(z, lambda x: norm.cdf(x))
        used = "student"
    ks_stat = float(ks.statistic)  # type: ignore[attr-defined]
    ks_p = float(ks.pvalue)  # type: ignore[attr-defined]
    return ks_stat, ks_p, used


def _compute_basic_statistics(z_clean: np.ndarray) -> tuple[float, float, float]:
    """Compute mean, variance, and std of standardized residuals.

    Args:
    ----
        z_clean: Cleaned standardized residuals.

    Returns:
    -------
        Tuple of (mean, variance, std).

    """
    mean_z = float(np.mean(z_clean))
    var_z = float(np.var(z_clean, ddof=0))
    std_z = float(np.std(z_clean, ddof=0))
    return mean_z, var_z, std_z


def _compute_distribution_test_results(
    z_clean: np.ndarray,
    dist: str,
    nu: float | None,
) -> tuple[float, float, float, float, str]:
    """Compute statistical test results for distribution diagnostics.

    Args:
    ----
        z_clean: Cleaned standardized residuals.
        dist: Distribution name.
        nu: Degrees of freedom.

    Returns:
    -------
        Tuple of (jb_stat, jb_p, ks_stat, ks_p, used_dist).
    """
    jb_stat, jb_p = _compute_jarque_bera_test(z_clean)
    ks_stat, ks_p, used = _compute_kolmogorov_smirnov_test(z_clean, dist, nu)
    return jb_stat, jb_p, ks_stat, ks_p, used


def _build_distribution_diagnostics_dict(
    used_dist: str,
    nu: float | None,
    mean_z: float,
    var_z: float,
    std_z: float,
    skew: float,
    kurt: float,
    jb_stat: float,
    jb_p: float,
    ks_stat: float,
    ks_p: float,
    n_obs: int,
) -> dict[str, float | str | None]:
    """Build the distribution diagnostics result dictionary.

    Args:
    ----
        used_dist: Distribution that was actually used.
        nu: Degrees of freedom.
        mean_z: Mean of standardized residuals.
        var_z: Variance of standardized residuals.
        std_z: Standard deviation of standardized residuals.
        skew: Skewness.
        kurt: Kurtosis.
        jb_stat: Jarque-Bera test statistic.
        jb_p: Jarque-Bera p-value.
        ks_stat: Kolmogorov-Smirnov test statistic.
        ks_p: Kolmogorov-Smirnov p-value.
        n_obs: Number of observations.

    Returns:
    -------
        Dictionary with diagnostic results.
    """
    return {
        "dist": used_dist,
        "nu": float(nu) if nu is not None else None,
        "mean": mean_z,
        "variance": var_z,
        "std": std_z,
        "skewness": skew,
        "kurtosis": kurt,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_p,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
        "n_obs": n_obs,
    }


def _compute_distribution_statistics(
    z_clean: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute all statistical measures for distribution diagnostics.

    Args:
    ----
        z_clean: Cleaned standardized residuals.

    Returns:
    -------
        Tuple of (mean_z, var_z, std_z, skew, kurt).
    """
    mean_z, var_z, std_z = _compute_basic_statistics(z_clean)
    zc = (z_clean - mean_z) / (std_z + GARCH_STD_EPSILON)
    skew, kurt = _compute_moments(zc)
    return mean_z, var_z, std_z, skew, kurt


def compute_distribution_diagnostics(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str,
    nu: float | None = None,
) -> dict[str, float | str | None]:
    """Compute distribution diagnostics for standardized residuals.

    Verifies adequacy of chosen distribution (Normal or Student-t) for zt.

    Args:
    ----
        residuals: Raw residuals εt from mean model.
        garch_params: GARCH parameter dictionary.
        dist: Distribution name ('student', 'skewt').
        nu: Degrees of freedom for Student-t/Skew-t (if applicable).

    Returns:
    -------
        Dictionary with diagnostic statistics and test results.

    Raises:
    ------
        ValueError: If no valid standardized residuals found.

    """
    z_clean = standardize_residuals(residuals, garch_params, dist=dist, nu=nu, clean=True)

    if z_clean.size == 0:
        msg = "No valid standardized residuals for diagnostics"
        raise ValueError(msg)

    # Compute all statistics and tests
    mean_z, var_z, std_z, skew, kurt = _compute_distribution_statistics(z_clean)
    jb_stat, jb_p, ks_stat, ks_p, used = _compute_distribution_test_results(z_clean, dist, nu)

    return _build_distribution_diagnostics_dict(
        used, nu, mean_z, var_z, std_z, skew, kurt, jb_stat, jb_p, ks_stat, ks_p, int(z_clean.size)
    )


def save_qq_plot_std_residuals(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str,
    nu: float | None = None,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
) -> Path:
    """Save QQ-plot of standardized residuals vs chosen distribution.

    Graphical diagnostic to verify distribution adequacy:
    - Points should lie along the diagonal line if distribution is appropriate
    - Deviations indicate distribution misspecification
    """
    # Extract lambda parameter for Skew-t distribution
    lam = garch_params.get("lambda") if dist.lower() == "skewt" else None
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu, lambda_skew=lam)
    theo_q, z_sorted, title = compute_qq_data(z, dist, nu, lam)
    out_path = prepare_output_path(outdir, filename, "garch_std_residuals_qq")

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig, canvas, ax = create_figure_canvas(figsize=_QQ_FIGURE_SIZE, n_rows=1, n_cols=1)
        plot_qq_scatter(ax, theo_q, z_sorted, title)
        # Convert Figure to FigureCanvasAgg for compatibility
        canvas_agg = FigureCanvasAgg(fig)
        save_figure_or_placeholder(canvas_agg, out_path, "Saved QQ plot")
    except ImportError:  # pragma: no cover - matplotlib optional
        write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


def save_histogram_std_residuals(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str,
    nu: float | None = None,
    bins: int = _DISTRIBUTION_HISTOGRAM_BINS,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
) -> Path:
    """Save histogram of standardized residuals with theoretical distribution overlay.

    Graphical diagnostic to verify distribution adequacy:
    - Histogram shows empirical distribution of standardized residuals
    - Overlay shows theoretical Skew-t distribution fit

    Args:
        residuals: Raw residuals array εt from mean model.
        garch_params: GARCH parameter dictionary.
        dist: Distribution name ('student', 'skewt').
        nu: Degrees of freedom for Student-t/Skew-t (if applicable).
        bins: Number of histogram bins.
        outdir: Output directory for plots.
        filename: Custom filename (optional).

    Returns:
        Path to saved plot file.
    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    out_path = prepare_output_path(outdir, filename, "garch_std_residuals_histogram")

    try:
        from matplotlib.axes import Axes
        fig, canvas, ax = create_figure_canvas(figsize=_DIAGNOSTIC_FIGURE_SIZE, n_rows=1, n_cols=1)
        # Type cast: for single subplot, ax is a single Axes object, not ndarray
        ax = cast(Axes, ax)

        # Plot histogram with theoretical distribution overlay
        # Extract Skew-t parameters for proper overlay
        lam_param = garch_params.get("lambda")
        mean_z, std_z = plot_histogram_with_normal_overlay(
            z,  # residuals
            ax,  # axes
            bins=bins,
        )

        # Set labels and title
        ax.set_title(
            "Distribution of standardized residuals (with theoretical distribution overlay)"
        )
        ax.set_xlabel("Standardized residuals (z_t)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")

        # Convert Figure to FigureCanvasAgg for compatibility
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas_agg = FigureCanvasAgg(fig)
        save_figure_or_placeholder(canvas_agg, out_path, "Saved histogram plot")
        logger.info(
            "Histogram saved: mean=%.4f, std=%.4f, n=%d",
            mean_z,
            std_z,
            len(z),
        )
    except ImportError:  # pragma: no cover - matplotlib optional
        write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


def save_residual_plots(
    resid_train: np.ndarray,
    resid_test: np.ndarray,
    *,
    garch_params: dict[str, float] | None = None,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
    dist: str = "student",
    nu: float | None = None,
) -> Path:
    """Save plots of residuals and standardized residuals (if params provided)."""
    train, test, n_train = prepare_residual_data(resid_train, resid_test)
    all_res = np.concatenate([train, test]) if test.size else train
    z_all = compute_standardized_residuals_for_plot(all_res, garch_params, dist, nu)
    out_path = prepare_output_path(outdir, filename, "garch_residuals")

    try:
        fig, canvas = create_residual_plots_figure(train, test, z_all, n_train)
        # Convert Figure to FigureCanvasAgg for compatibility if needed
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        if not isinstance(canvas, FigureCanvasAgg):
            canvas = FigureCanvasAgg(fig)
        save_figure_or_placeholder(canvas, out_path, "Saved residual plots")
    except ImportError:  # pragma: no cover - matplotlib optional
        write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


__all__ = [
    "save_acf_pacf_std_squared_plots",
    "compute_ljung_box_on_std_squared",
    "compute_ljung_box_on_std",
    "save_acf_pacf_std_plots",
    "compute_distribution_diagnostics",
    "save_histogram_std_residuals",
    "save_qq_plot_std_residuals",
    "save_residual_plots",
]
