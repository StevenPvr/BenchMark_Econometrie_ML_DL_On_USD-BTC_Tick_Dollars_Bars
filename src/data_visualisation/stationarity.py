"""Stationarity analysis and plotting for time series."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from statsmodels.tsa.stattools import adfuller, kpss

from src.constants import FONTSIZE_TITLE, LINEWIDTH_BOLD, LINEWIDTH_THIN, PLOT_ALPHA_LIGHT
from src.utils import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

__all__ = [
    "run_stationarity_tests",
    "plot_stationarity",
]

# Stationarity plot constants
_FIGSIZE_STATIONARITY: tuple[int, int] = (14, 8)
_DEFAULT_ROLLING_WINDOW: int = 50
_SIGNIFICANCE_LEVEL: float = 0.05


def run_stationarity_tests(
    log_returns: pd.Series,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """
    Run ADF and KPSS stationarity tests on log-returns.

    ADF (Augmented Dickey-Fuller): H0 = Series is non-stationary.
    KPSS (Kwiatkowski-Phillips-Schmidt-Shin): H0 = Series is stationary.

    Args:
        log_returns: Series of log-returns.
        output_dir: Directory to save results as JSON.

    Returns:
        Dictionary with test results for ADF and KPSS.
    """
    import json

    series = log_returns.dropna().values

    results: dict[str, object] = {}

    # ADF Test (H0: series is non-stationary)
    try:
        adf_result = adfuller(series, autolag="AIC")
        adf_stat = float(adf_result[0])
        adf_pvalue = float(adf_result[1])
        adf_critical = adf_result[4] if len(adf_result) > 4 else {}
        adf_conclusion = "Stationnaire" if adf_pvalue < _SIGNIFICANCE_LEVEL else "Non-stationnaire"
        results["ADF"] = {
            "statistic": adf_stat,
            "p_value": adf_pvalue,
            "conclusion": adf_conclusion,
            "critical_values": {
                "1%": float(adf_critical.get("1%", 0)) if isinstance(adf_critical, dict) else None,
                "5%": float(adf_critical.get("5%", 0)) if isinstance(adf_critical, dict) else None,
                "10%": float(adf_critical.get("10%", 0)) if isinstance(adf_critical, dict) else None,
            },
            "hypothesis": "H0: serie non stationnaire (p < 0.05 => rejeter H0 => stationnaire)",
        }
    except Exception as e:
        logger.warning("ADF test failed: %s", e)
        results["ADF"] = {"error": str(e)}

    # KPSS Test (H0: series is stationary)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*p-value.*")
            kpss_result = kpss(series, regression="c", nlags="auto")
        kpss_stat = float(kpss_result[0])
        kpss_pvalue = float(kpss_result[1])
        kpss_critical = kpss_result[3]
        kpss_conclusion = "Stationnaire" if kpss_pvalue > _SIGNIFICANCE_LEVEL else "Non-stationnaire"
        results["KPSS"] = {
            "statistic": kpss_stat,
            "p_value": kpss_pvalue,
            "conclusion": kpss_conclusion,
            "critical_values": {
                "1%": float(kpss_critical.get("1%", 0)) if isinstance(kpss_critical, dict) else None,
                "5%": float(kpss_critical.get("5%", 0)) if isinstance(kpss_critical, dict) else None,
                "10%": float(kpss_critical.get("10%", 0)) if isinstance(kpss_critical, dict) else None,
            },
            "hypothesis": "H0: serie stationnaire (p > 0.05 => ne pas rejeter H0 => stationnaire)",
        }
    except Exception as e:
        logger.warning("KPSS test failed: %s", e)
        results["KPSS"] = {"error": str(e)}

    # Log results
    logger.info("=" * 80)
    logger.info("TESTS DE STATIONNARITE - LOG-RETURNS")
    logger.info("=" * 80)
    for test_name, test_results in results.items():
        if isinstance(test_results, dict) and "error" not in test_results:
            logger.info("%s:", test_name)
            logger.info("  Statistic: %.6f", test_results["statistic"])
            logger.info("  P-value: %.6f", test_results["p_value"])
            logger.info("  Conclusion: %s", test_results["conclusion"])

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "stationarity_tests.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Stationarity tests saved: %s", filepath)

    return results


def plot_stationarity(
    log_returns: pd.Series,
    stationarity_results: dict[str, object],
    output_dir: Path | None = None,
    window: int = _DEFAULT_ROLLING_WINDOW,
    figsize: tuple[int, int] = _FIGSIZE_STATIONARITY,
) -> Figure:
    """
    Plot stationarity analysis with rolling mean and rolling std of log-returns.

    Args:
        log_returns: Series of log-returns.
        stationarity_results: Results from run_stationarity_tests.
        output_dir: Directory to save the plot.
        window: Window size for rolling statistics.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    data = log_returns.dropna()

    # Compute rolling statistics
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Plot 1: Series + Rolling Mean
    ax1 = axes[0]
    ax1.plot(data, linewidth=0.5, color="steelblue", alpha=0.6, label="Log-Returns")
    ax1.plot(rolling_mean, linewidth=2, color="red", label=f"Rolling Mean ({window})")
    ax1.axhline(
        data.mean(),
        color="green",
        linestyle="--",
        linewidth=LINEWIDTH_BOLD,
        label=f"Mean globale: {data.mean():.6f}",
    )
    ax1.set_ylabel("Log-Return", fontsize=10)
    ax1.set_title("Log-Returns et Rolling Mean", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=PLOT_ALPHA_LIGHT)

    # Plot 2: Rolling Std
    ax2 = axes[1]
    ax2.plot(rolling_std.values, linewidth=LINEWIDTH_BOLD, color="orange", label=f"Rolling Std ({window})")
    ax2.axhline(
        data.std(),
        color="green",
        linestyle="--",
        linewidth=LINEWIDTH_BOLD,
        label=f"Std globale: {data.std():.6f}",
    )
    ax2.fill_between(range(len(rolling_std)), 0, rolling_std.values, alpha=PLOT_ALPHA_LIGHT, color="orange")
    ax2.set_xlabel("Index", fontsize=10)
    ax2.set_ylabel("Ecart-type", fontsize=10)
    ax2.set_title("Rolling Standard Deviation (Volatilite)", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=PLOT_ALPHA_LIGHT)

    # Add test results annotation
    adf_result = stationarity_results.get("ADF", {})
    kpss_result = stationarity_results.get("KPSS", {})

    test_text = "Tests de Stationnarite:\n"
    if isinstance(adf_result, dict) and "conclusion" in adf_result:
        test_text += f"ADF: {adf_result['conclusion']} (p={adf_result['p_value']:.4f})\n"
    if isinstance(kpss_result, dict) and "conclusion" in kpss_result:
        test_text += f"KPSS: {kpss_result['conclusion']} (p={kpss_result['p_value']:.4f})"

    fig.text(
        0.02,
        0.98,
        test_text,
        transform=fig.transFigure,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "stationarity_plot.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info("Stationarity plot saved: %s", filepath)

    return fig
