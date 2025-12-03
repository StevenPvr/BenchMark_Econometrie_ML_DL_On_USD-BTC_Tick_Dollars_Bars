"""Trend analysis and plotting for time series."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats
from scipy.stats import linregress

from src.constants import FONTSIZE_TITLE, PLOT_ALPHA_LIGHT, PLOT_ALPHA_MEDIUM
from src.utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "mann_kendall_test",
    "compute_trend_statistics",
    "plot_trend_extraction",
    "plot_trend_analysis",
]

# Trend analysis constants
_FIGSIZE_TREND_EXTRACTION: tuple[int, int] = (14, 12)
_FIGSIZE_TREND_ANALYSIS: tuple[int, int] = (14, 10)
_DEFAULT_MA_WINDOWS: list[int] = [20, 50, 100, 200]
_MANN_KENDALL_MAX_SAMPLES: int = 5000
_SIGNIFICANCE_LEVEL: float = 0.05


def mann_kendall_test(
    series: np.ndarray,
    max_samples: int = _MANN_KENDALL_MAX_SAMPLES,
) -> dict[str, object]:
    """
    Run Mann-Kendall test to detect monotonic trend.

    H0: No monotonic trend.
    H1: Monotonic trend is present.

    Args:
        series: Time series as numpy array.
        max_samples: Maximum samples for computation. Subsamples if n > max_samples
            to avoid O(n²) complexity on large datasets.

    Returns:
        Dictionary with test statistic, p-value, and trend interpretation.
    """
    n = len(series)

    # Subsample if data is too large (O(n²) becomes prohibitive)
    if n > max_samples:
        indices = np.linspace(0, n - 1, max_samples, dtype=int)
        series = series[indices]
        n = max_samples

    # Vectorized computation of statistic S
    # For each pair (i, j) with i < j, compute sign(series[j] - series[i])
    diff_matrix = series[np.newaxis, :] - series[:, np.newaxis]
    s = int(np.sum(np.sign(diff_matrix[np.triu_indices(n, k=1)])))

    # Variance of S
    var_s = n * (n - 1) * (2 * n + 5) / 18

    # Normalized Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Trend interpretation
    if p_value < _SIGNIFICANCE_LEVEL:
        if z > 0:
            trend = "Tendance croissante significative"
        else:
            trend = "Tendance decroissante significative"
    else:
        trend = "Pas de tendance significative"

    return {
        "statistic_S": s,
        "statistic_Z": float(z),
        "p_value": float(p_value),
        "trend": trend,
        "hypothesis": "H0: Pas de tendance monotone (p < 0.05 => rejeter H0)",
    }


def compute_trend_statistics(
    series: pd.Series,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """
    Compute trend statistics for a time series.

    Includes:
    - Mann-Kendall test (monotonic trend)
    - Linear regression (slope, R²)
    - Change statistics between first and second halves

    Args:
        series: Time series data (e.g., prices or cumulative log-returns).
        output_dir: Directory to save results as JSON.

    Returns:
        Dictionary with trend analysis results.
    """
    import json

    data = np.asarray(series.dropna().values)
    n = len(data)
    x = np.arange(n)

    results: dict[str, object] = {}

    # 1. Mann-Kendall test
    mk_result = mann_kendall_test(data)
    results["Mann_Kendall"] = mk_result

    # 2. Linear regression
    lr_result = linregress(x, data)
    slope = float(lr_result.slope)  # type: ignore[attr-defined]
    intercept = float(lr_result.intercept)  # type: ignore[attr-defined]
    r_value = float(lr_result.rvalue)  # type: ignore[attr-defined]
    p_value = float(lr_result.pvalue)  # type: ignore[attr-defined]
    std_err = float(lr_result.stderr)  # type: ignore[attr-defined]
    r_squared = r_value**2

    trend_direction = "Haussiere" if slope > 0 else "Baissiere" if slope < 0 else "Neutre"

    results["Linear_Regression"] = {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "std_error": std_err,
        "trend_direction": trend_direction,
        "slope_significance": "Significatif" if p_value < _SIGNIFICANCE_LEVEL else "Non significatif",
    }

    # 3. Change statistics between halves
    first_half = np.asarray(data[: n // 2])
    second_half = np.asarray(data[n // 2 :])

    results["Change_Statistics"] = {
        "mean_first_half": float(first_half.mean()),
        "mean_second_half": float(second_half.mean()),
        "mean_change": float(second_half.mean() - first_half.mean()),
        "std_first_half": float(first_half.std()),
        "std_second_half": float(second_half.std()),
        "std_change": float(second_half.std() - first_half.std()),
    }

    # Log results
    logger.info("=" * 80)
    logger.info("ANALYSE DE TENDANCE")
    logger.info("=" * 80)

    logger.info("[Mann-Kendall Test]")
    logger.info("  Statistique S: %d", mk_result["statistic_S"])
    logger.info("  Statistique Z: %.4f", mk_result["statistic_Z"])
    logger.info("  P-value: %.6f", mk_result["p_value"])
    logger.info("  Conclusion: %s", mk_result["trend"])

    logger.info("[Regression Lineaire]")
    logger.info("  Pente: %.8f", slope)
    logger.info("  R²: %.6f", r_squared)
    logger.info("  P-value: %.6f", p_value)
    logger.info("  Direction: %s", trend_direction)

    change_stats = results["Change_Statistics"]
    if isinstance(change_stats, dict):
        logger.info("[Changement entre les deux moities]")
        logger.info("  Moyenne 1ere moitie: %.6f", change_stats["mean_first_half"])
        logger.info("  Moyenne 2eme moitie: %.6f", change_stats["mean_second_half"])
        logger.info("  Changement de moyenne: %.6f", change_stats["mean_change"])

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Convert for JSON serialization
        json_results = {
            "Mann_Kendall": mk_result,
            "Linear_Regression": results["Linear_Regression"],
            "Change_Statistics": results["Change_Statistics"],
        }
        filepath = output_dir / "trend_analysis.json"
        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)
        logger.info("Trend analysis saved: %s", filepath)

    return results


def plot_trend_extraction(
    series: pd.Series,
    windows: list[int] | None = None,
    output_dir: Path | None = None,
    figsize: tuple[int, int] = _FIGSIZE_TREND_EXTRACTION,
) -> Figure:
    """
    Extract and visualize trend via moving averages.

    For dollar bars (volume-sampled), classical seasonal decomposition is not
    appropriate. Moving averages are used instead to extract the trend.

    Args:
        series: Price series from dollar bars.
        windows: Windows for moving averages. Defaults to [20, 50, 100, 200].
        output_dir: Directory to save the plot.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    if windows is None:
        windows = _DEFAULT_MA_WINDOWS.copy()

    data = series.dropna()
    data_arr = np.asarray(data.values)
    n = len(data_arr)

    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # Plot 1: Original series with long-term trend
    ax1 = axes[0]
    ax1.plot(data_arr, linewidth=0.5, color="steelblue", alpha=PLOT_ALPHA_MEDIUM, label="Prix")
    long_window = max(w for w in windows if w < n) if any(w < n for w in windows) else 50
    ma_long = cast(pd.Series, pd.Series(data_arr).rolling(window=long_window).mean())
    ax1.plot(ma_long.values, linewidth=2, color="red", label=f"Tendance (MA{long_window})")
    ax1.set_title("Dollar Bars - Prix et Tendance", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax1.set_ylabel("Prix")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=PLOT_ALPHA_LIGHT)

    # Plot 2: Extracted trend component
    ax2 = axes[1]
    short_window = min(windows)
    trend_component = ma_long - ma_long.iloc[long_window - 1]
    ax2.plot(trend_component.values, linewidth=1.5, color="red")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title(f"Composante de Tendance (MA{long_window} - depart)", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax2.set_ylabel("Tendance relative")
    ax2.grid(True, alpha=PLOT_ALPHA_LIGHT)

    # Plot 3: Residuals (price - trend)
    ax3 = axes[2]
    residuals = data_arr - ma_long.values
    ax3.plot(residuals, linewidth=0.8, color="orange")
    ax3.axhline(0, color="black", linestyle="--", linewidth=1)
    ax3.fill_between(
        range(n),
        0,
        residuals,
        alpha=PLOT_ALPHA_LIGHT,
        color="orange",
        where=~np.isnan(residuals),
    )
    ax3.set_title("Residus (Prix - Tendance)", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax3.set_ylabel("Residu")
    ax3.grid(True, alpha=PLOT_ALPHA_LIGHT)

    # Residual statistics
    valid_res = residuals[~np.isnan(residuals)]
    res_std = np.std(valid_res)
    res_mean = np.mean(valid_res)
    ax3.text(
        0.02,
        0.95,
        f"Mean={res_mean:.4f}, Std={res_std:.4f}",
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # Plot 4: Momentum (difference between short and long MA)
    ax4 = axes[3]
    ma_short = cast(pd.Series, pd.Series(data_arr).rolling(window=short_window).mean())
    momentum = cast(pd.Series, ma_short - ma_long)
    ax4.plot(momentum.values, linewidth=1.2, color="purple")
    ax4.axhline(0, color="black", linestyle="--", linewidth=1)
    momentum_values = cast(np.ndarray, momentum.values)
    ax4.fill_between(
        range(n),
        0,
        momentum_values,
        alpha=PLOT_ALPHA_LIGHT,
        where=(momentum_values > 0) & ~np.isnan(momentum_values),
        color="green",
    )
    ax4.fill_between(
        range(n),
        0,
        momentum_values,
        alpha=PLOT_ALPHA_LIGHT,
        where=(momentum_values < 0) & ~np.isnan(momentum_values),
        color="red",
    )
    ax4.set_title(f"Momentum (MA{short_window} - MA{long_window})", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax4.set_xlabel("Index (Dollar Bars)")
    ax4.set_ylabel("Momentum")
    ax4.grid(True, alpha=PLOT_ALPHA_LIGHT)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "trend_extraction.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info("Trend extraction plot saved: %s", filepath)

    return fig


def plot_trend_analysis(
    series: pd.Series,
    trend_results: dict[str, object],
    series_name: str = "Dollar Bars",
    output_dir: Path | None = None,
    figsize: tuple[int, int] = _FIGSIZE_TREND_ANALYSIS,
) -> Figure:
    """
    Plot comprehensive trend analysis with linear regression and moving averages.

    Adapted for dollar bars (volume-sampled data).

    Args:
        series: Price series from dollar bars.
        trend_results: Results from compute_trend_statistics.
        series_name: Name for the series in plot titles.
        output_dir: Directory to save the plot.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    data = series.dropna()
    n = len(data)
    x = np.arange(n)

    # Extract regression parameters
    lr = trend_results.get("Linear_Regression", {})
    slope = lr.get("slope", 0) if isinstance(lr, dict) else 0
    intercept = lr.get("intercept", 0) if isinstance(lr, dict) else 0
    r_squared = lr.get("r_squared", 0) if isinstance(lr, dict) else 0

    mk = trend_results.get("Mann_Kendall", {})

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Series with linear regression
    ax1 = axes[0, 0]
    ax1.plot(data.values, linewidth=0.8, color="steelblue", alpha=PLOT_ALPHA_MEDIUM, label=series_name)
    ax1.plot(
        x,
        slope * x + intercept,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Tendance (pente={slope:.2e})",
    )
    ax1.set_title(f"{series_name} avec Regression Lineaire", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax1.set_xlabel("Index (Dollar Bars)")
    ax1.set_ylabel("Prix")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=PLOT_ALPHA_LIGHT)

    # R² annotation
    ax1.text(
        0.95,
        0.95,
        f"R² = {r_squared:.4f}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # Plot 2: Multiple moving averages
    ax2 = axes[0, 1]
    data_arr = np.asarray(data.values)
    ax2.plot(data_arr, linewidth=0.5, color="steelblue", alpha=0.5, label="Original")

    windows = [20, 50, 100]
    colors = ["orange", "red", "purple"]
    for window, color in zip(windows, colors):
        if window < n:
            ma = cast(pd.Series, pd.Series(data_arr).rolling(window=window).mean())
            ax2.plot(ma.values, linewidth=1.5, color=color, label=f"MA({window})")

    ax2.set_title("Moyennes Mobiles", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Valeur")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=PLOT_ALPHA_LIGHT)

    # Plot 3: Distribution first vs second half
    ax3 = axes[1, 0]
    first_half = data_arr[: n // 2]
    second_half = data_arr[n // 2 :]

    ax3.hist(first_half, bins=40, alpha=0.6, color="blue", label="1ere moitie", density=True)
    ax3.hist(second_half, bins=40, alpha=0.6, color="red", label="2eme moitie", density=True)
    ax3.axvline(first_half.mean(), color="blue", linestyle="--", linewidth=2)
    ax3.axvline(second_half.mean(), color="red", linestyle="--", linewidth=2)
    ax3.set_title("Distribution: 1ere vs 2eme moitie", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax3.set_xlabel("Valeur")
    ax3.set_ylabel("Densite")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=PLOT_ALPHA_LIGHT)

    # Plot 4: Trend residuals
    ax4 = axes[1, 1]
    trend_line = slope * x + intercept
    residuals = np.asarray(data.values) - np.asarray(trend_line)
    ax4.plot(residuals, linewidth=0.8, color="orange")
    ax4.axhline(0, color="black", linestyle="--", linewidth=1)
    ax4.fill_between(x, 0, residuals, alpha=PLOT_ALPHA_LIGHT, color="orange")
    ax4.set_title("Residus apres suppression de la tendance", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax4.set_xlabel("Index")
    ax4.set_ylabel("Residu")
    ax4.grid(True, alpha=PLOT_ALPHA_LIGHT)

    # Residual statistics
    res_std = np.std(residuals)
    ax4.text(
        0.95,
        0.95,
        f"Std residus = {res_std:.4f}",
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # Main title with Mann-Kendall results
    mk_trend = mk.get("trend", "N/A") if isinstance(mk, dict) else "N/A"
    mk_pvalue = mk.get("p_value", 0) if isinstance(mk, dict) else 0
    fig.suptitle(
        f"Analyse de Tendance - Mann-Kendall: {mk_trend} (p={mk_pvalue:.4f})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "trend_analysis.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info("Trend analysis plot saved: %s", filepath)

    return fig
