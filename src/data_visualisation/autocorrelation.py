"""Autocorrelation analysis and plotting for time series."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.constants import FONTSIZE_TITLE, TEMPORAL_ACF_LAGS
from src.utils import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

__all__ = [
    "compute_autocorrelation",
    "compute_autocorrelation_squared",
    "run_ljung_box_test",
]

# Autocorrelation constants
_FIGSIZE_ACF: tuple[int, int] = (14, 5)
_ACF_ALPHA: float = 0.05
_DEFAULT_LJUNG_BOX_LAGS: list[int] = [10, 20, 40]
_SIGNIFICANCE_LEVEL: float = 0.05


def compute_autocorrelation(
    log_returns: pd.Series,
    output_dir: Path | None = None,
    max_lags: int = TEMPORAL_ACF_LAGS,
    figsize: tuple[int, int] = _FIGSIZE_ACF,
) -> Figure:
    """
    Plot ACF and PACF of log-returns.

    Args:
        log_returns: Series of log-returns.
        output_dir: Directory to save the plot.
        max_lags: Maximum number of lags to plot.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    data = log_returns.dropna()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ACF of log-returns (excluding lag 0)
    plot_acf(data, lags=max_lags, ax=axes[0], alpha=_ACF_ALPHA, zero=False)
    axes[0].set_title("ACF - Log-Returns", fontsize=FONTSIZE_TITLE, fontweight="bold")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")

    # PACF of log-returns (excluding lag 0)
    plot_pacf(data, lags=max_lags, ax=axes[1], alpha=_ACF_ALPHA, zero=False)
    axes[1].set_title("PACF - Log-Returns", fontsize=FONTSIZE_TITLE, fontweight="bold")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Partial Autocorrelation")

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "log_returns_acf.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info("ACF plot saved: %s", filepath)

    return fig


def compute_autocorrelation_squared(
    log_returns: pd.Series,
    output_dir: Path | None = None,
    max_lags: int = TEMPORAL_ACF_LAGS,
    figsize: tuple[int, int] = _FIGSIZE_ACF,
) -> Figure:
    """
    Plot ACF and PACF of squared log-returns for volatility clustering detection.

    Autocorrelation of squared log-returns helps detect volatility clustering,
    a characteristic feature of financial time series.

    Args:
        log_returns: Series of log-returns.
        output_dir: Directory to save the plot.
        max_lags: Maximum number of lags to plot.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    data = log_returns.dropna()
    squared = data**2

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ACF of squared log-returns (excluding lag 0)
    plot_acf(squared, lags=max_lags, ax=axes[0], alpha=_ACF_ALPHA, zero=False)
    axes[0].set_title("ACF - Log-Returns² (Volatilite)", fontsize=FONTSIZE_TITLE, fontweight="bold")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")

    # PACF of squared log-returns (excluding lag 0)
    plot_pacf(squared, lags=max_lags, ax=axes[1], alpha=_ACF_ALPHA, zero=False)
    axes[1].set_title("PACF - Log-Returns² (Volatilite)", fontsize=FONTSIZE_TITLE, fontweight="bold")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Partial Autocorrelation")

    fig.suptitle(
        "Autocorrelation des Log-Returns² (detection du clustering de volatilite)",
        fontsize=11,
        y=1.02,
    )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "log_returns_squared_acf.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info("ACF squared plot saved: %s", filepath)

    return fig


def run_ljung_box_test(
    series: pd.Series,
    lags: list[int] | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """
    Run Ljung-Box test for autocorrelation.

    H0: No autocorrelation up to lag k.
    H1: Autocorrelation is present.

    Args:
        series: Time series data (log-returns or residuals).
        lags: List of lags to test. Defaults to [10, 20, 40].
        output_dir: Directory to save results as JSON.

    Returns:
        Dictionary with test results for each lag.
    """
    import json

    from statsmodels.stats.diagnostic import acorr_ljungbox

    if lags is None:
        lags = _DEFAULT_LJUNG_BOX_LAGS.copy()

    data = series.dropna()

    results: dict[str, object] = {
        "test_name": "Ljung-Box",
        "hypothesis": "H0: Pas d'autocorrelation jusqu'au lag k (p < 0.05 => rejeter H0)",
        "lags_tested": {},
    }

    try:
        lb_results = acorr_ljungbox(data, lags=lags, return_df=True)

        lags_tested: dict[str, object] = {}
        for lag in lags:
            if lag in lb_results.index:
                lb_stat = float(lb_results.loc[lag, "lb_stat"])
                lb_pvalue = float(lb_results.loc[lag, "lb_pvalue"])
                conclusion = (
                    "Autocorrelation significative"
                    if lb_pvalue < _SIGNIFICANCE_LEVEL
                    else "Pas d'autocorrelation"
                )
                lags_tested[str(lag)] = {
                    "statistic": lb_stat,
                    "p_value": lb_pvalue,
                    "conclusion": conclusion,
                }
        results["lags_tested"] = lags_tested
    except Exception as e:
        logger.warning("Ljung-Box test failed: %s", e)
        results["error"] = str(e)

    # Log results
    logger.info("=" * 60)
    logger.info("TEST DE LJUNG-BOX (Autocorrelation)")
    logger.info("=" * 60)
    if "error" not in results:
        lags_tested_raw = results.get("lags_tested", {})
        if isinstance(lags_tested_raw, dict):
            lags_tested_dict: dict[str, object] = lags_tested_raw
            for lag_str, lag_results in lags_tested_dict.items():
                if isinstance(lag_results, dict):
                    logger.info(
                        "  Lag %s: stat=%.4f, p=%.4f -> %s",
                        lag_str,
                        lag_results["statistic"],
                        lag_results["p_value"],
                        lag_results["conclusion"],
                    )
    else:
        logger.info("  Erreur: %s", results["error"])

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "ljung_box_test.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Ljung-Box test saved: %s", filepath)

    return results
