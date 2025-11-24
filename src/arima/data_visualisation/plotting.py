"""Plotting utilities for visualization."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd # type: ignore[import-untyped]
from statsmodels.tsa.seasonal import seasonal_decompose # type: ignore[import-untyped]

from src.constants import (
    ARIMA_SEASONALITY_DAILY_PERIOD,
    ARIMA_SEASONALITY_MONTHLY_PERIOD,
    FONTSIZE_AXIS,
    FONTSIZE_SUBTITLE,
    FONTSIZE_TEXT,
    LINEWIDTH_BOLD,
    LINEWIDTH_THIN,
    PLOT_ALPHA_DEFAULT,
    PLOT_ALPHA_FILL,
    PLOT_ALPHA_LIGHT,
    PLOT_ALPHA_MEDIUM,
    SEASONALITY_SEPARATOR_LENGTH,
    STATIONARITY_TEXT_BOX_X,
    STATIONARITY_TEXT_BOX_Y,
)
from src.utils import get_logger
from src.visualization import (
    add_grid,
    add_legend,
    add_zero_line,
    format_date_axis,
    save_plot_wrapper,
)

logger = get_logger(__name__)


def _format_date_axis(ax: Axes) -> None:
    """Format x-axis dates for better readability.

    Wrapper for backward compatibility.

    Args:
        ax: Matplotlib axes to format.
    """
    format_date_axis(ax)


def _save_plot(output_file: str) -> None:
    """Save the current plot to file.

    Wrapper for backward compatibility.

    Args:
        output_file: Path to save the plot.
    """
    save_plot_wrapper(output_file)


# Legacy seasonal decomposition plotting helpers removed.


def plot_stationarity_series_with_bands(
    ax: Axes,
    weighted_series: pd.Series,
    rolling_mean: pd.Series,
    rolling_std: pd.Series,
    rolling_window: int,
) -> None:
    """Plot original series with rolling mean and std bands.

    Args:
        ax: Matplotlib axes to plot on.
        weighted_series: Original time series.
        rolling_mean: Rolling mean series.
        rolling_std: Rolling standard deviation series.
        rolling_window: Window size for rolling statistics.
    """
    ax.plot(
        weighted_series.index,
        weighted_series,
        label="Série originale",
        linewidth=LINEWIDTH_THIN,
        alpha=PLOT_ALPHA_MEDIUM,
        color="blue",
    )
    ax.plot(
        rolling_mean.index,
        rolling_mean,
        label=f"Moyenne mobile (window={rolling_window})",
        linewidth=LINEWIDTH_BOLD,
        color="red",
    )
    ax.fill_between(
        rolling_mean.index,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=PLOT_ALPHA_FILL,
        color="red",
        label=f"±1 écart-type (window={rolling_window})",
    )
    add_zero_line(ax, linewidth=LINEWIDTH_THIN, alpha=0.5)
    ax.set_title(
        "Série temporelle avec statistiques mobiles",
        fontsize=FONTSIZE_SUBTITLE,
        fontweight="bold",
    )
    ax.set_ylabel("Rendement logarithmique pondéré", fontsize=FONTSIZE_AXIS)
    add_legend(ax)
    add_grid(ax, alpha=PLOT_ALPHA_LIGHT, linestyle="--")
    _format_date_axis(ax)


def _determine_test_verdict(p_value: float, alpha: float, test_type: str) -> str:
    """Determine verdict for a statistical test."""
    if test_type == "adf":
        return "Stationnaire" if p_value < alpha else "Non stationnaire"
    elif test_type == "kpss":
        return "Stationnaire" if p_value > alpha or (p_value != p_value) else "Non stationnaire"
    else:
        msg = f"Unknown test type: {test_type}"
        raise ValueError(msg)


def _format_comparison_operator(p_value: float, alpha: float, test_type: str) -> str:
    """Format the comparison operator for p-value display."""
    if test_type == "adf":
        return "<" if p_value < alpha else "≥"
    elif test_type == "kpss":
        return ">" if p_value > alpha or (p_value != p_value) else "≤"
    else:
        msg = f"Unknown test type: {test_type}"
        raise ValueError(msg)


def _format_adf_section(alpha: float, adf_stat: float, adf_pval: float, adf_lags: int) -> str:
    """Format the ADF test section."""
    adf_verdict = _determine_test_verdict(adf_pval, alpha, "adf")
    comparison = _format_comparison_operator(adf_pval, alpha, "adf")

    return (
        f"Test ADF (Augmented Dickey-Fuller):\n"
        f"  Statistique: {adf_stat:.6f}\n"
        f"  p-value: {adf_pval:.6f}\n"
        f"  Lags: {adf_lags}\n"
        f"  Verdict: {adf_verdict} (p-value {comparison} {alpha})\n"
    )


def _format_kpss_section(alpha: float, kpss_stat: float, kpss_pval: float, kpss_lags: int) -> str:
    """Format the KPSS test section."""
    kpss_verdict = _determine_test_verdict(kpss_pval, alpha, "kpss")
    comparison = _format_comparison_operator(kpss_pval, alpha, "kpss")

    return (
        f"Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin):\n"
        f"  Statistique: {kpss_stat:.6f}\n"
        f"  p-value: {kpss_pval:.6f}\n"
        f"  Lags: {kpss_lags}\n"
        f"  Verdict: {kpss_verdict} "
        f"(p-value {comparison} {alpha})"
    )


def format_stationarity_test_text(
    alpha: float,
    adf_stat: float,
    adf_pval: float,
    adf_lags: int,
    kpss_stat: float,
    kpss_pval: float,
    kpss_lags: int,
    overall_verdict: str,
) -> str:
    """Format stationarity test results as text.

    Args:
        alpha: Significance level.
        adf_stat: ADF test statistic.
        adf_pval: ADF test p-value.
        adf_lags: ADF test lags.
        kpss_stat: KPSS test statistic.
        kpss_pval: KPSS test p-value.
        kpss_lags: KPSS test lags.
        overall_verdict: Overall stationarity verdict.

    Returns:
        Formatted text string with test results.
    """
    adf_section = _format_adf_section(alpha, adf_stat, adf_pval, adf_lags)
    kpss_section = _format_kpss_section(alpha, kpss_stat, kpss_pval, kpss_lags)

    return (
        f"Résultats des tests de stationnarité (α = {alpha})\n"
        f"{'=' * SEASONALITY_SEPARATOR_LENGTH}\n"
        f"Verdict global: {overall_verdict}\n"
        f"\n"
        f"{adf_section}\n"
        f"{kpss_section}"
    )


def add_stationarity_text_box(ax: Axes, test_text: str) -> None:
    """Add stationarity test results as text box to axes.

    Args:
        ax: Matplotlib axes to add text box to.
        test_text: Formatted test results text.
    """
    ax.text(
        STATIONARITY_TEXT_BOX_X,
        STATIONARITY_TEXT_BOX_Y,
        test_text,
        transform=ax.transAxes,
        fontsize=FONTSIZE_TEXT,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=PLOT_ALPHA_DEFAULT, edgecolor="black"),
        family="monospace",
    )


def _plot_seasonal_decomposition(
    series: pd.Series,
    *,
    period: int,
    title: str,
    model: str = "additive",
) -> tuple[Figure, tuple[Axes, ...]]:
    """Plot seasonal decomposition with given parameters.

    Args:
        series: Input time series indexed by date.
        period: Seasonal period for decomposition.
        title: Title for the plot.
        model: seasonal_decompose model type ("additive" or "multiplicative").

    Returns:
        Tuple with the Matplotlib figure and axes created by statsmodels.

    Raises:
        ValueError: If the provided series is empty.
    """
    if series.empty:
        msg = f"Series is empty; cannot compute {title.lower()} decomposition"
        raise ValueError(msg)

    decomposition = seasonal_decompose(
        series,
        model=model,
        period=period,
        extrapolate_trend="freq",  # type: ignore
    )
    fig = decomposition.plot()
    fig.suptitle(
        title,
        fontsize=FONTSIZE_SUBTITLE,
        fontweight="bold",
    )
    axes = tuple(fig.axes)
    for axis in axes:
        add_grid(axis, alpha=PLOT_ALPHA_LIGHT, linestyle="--")
        _format_date_axis(axis)
    return fig, axes


def plot_seasonal_decomposition_daily(
    series: pd.Series,
    *,
    model: str = "additive",
) -> tuple[Figure, tuple[Axes, ...]]:
    """Plot seasonal decomposition using the daily window.

    Args:
        series: Input time series indexed by date.
        model: seasonal_decompose model type ("additive" or "multiplicative").

    Returns:
        Tuple with the Matplotlib figure and axes created by statsmodels.

    Raises:
        ValueError: If the provided series is empty.
    """
    return _plot_seasonal_decomposition(
        series,
        period=ARIMA_SEASONALITY_DAILY_PERIOD,
        title="Décomposition saisonnière – fenêtre quotidienne",
        model=model,
    )


def plot_seasonal_decomposition_monthly(
    series: pd.Series,
    *,
    model: str = "additive",
) -> tuple[Figure, tuple[Axes, ...]]:
    """Plot seasonal decomposition using the monthly window.

    Args:
        series: Input time series indexed by date.
        model: seasonal_decompose model type ("additive" or "multiplicative").

    Returns:
        Tuple with the Matplotlib figure and axes created by statsmodels.

    Raises:
        ValueError: If the provided series is empty.
    """
    return _plot_seasonal_decomposition(
        series,
        period=ARIMA_SEASONALITY_MONTHLY_PERIOD,
        title="Décomposition saisonnière – fenêtre mensuelle",
        model=model,
    )
