"""Distribution analysis and plotting for log-returns."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import stats

from src.constants import (
    DISTRIBUTION_HISTOGRAM_BINS,
    FONTSIZE_SUBTITLE,
    FONTSIZE_TITLE,
    PLOT_ALPHA_MEDIUM,
)
from src.utils import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

__all__ = [
    "plot_log_returns_distribution",
    "run_normality_tests",
]

# Distribution plot constants
_KDE_POINTS: int = 200
_LINEWIDTH_KDE: float = 2.0
_LINEWIDTH_AXVLINE: float = 1.5
_AXVLINE_ALPHA: float = 0.7
_FIGSIZE_DISTRIBUTION: tuple[int, int] = (14, 5)
_SUPTITLE_Y_OFFSET: float = 1.02


def plot_log_returns_distribution(
    log_returns: pd.Series,
    output_dir: Path | None = None,
    figsize: tuple[int, int] = _FIGSIZE_DISTRIBUTION,
) -> Figure:
    """
    Plot the distribution of log-returns with histogram, KDE and normal overlay.

    Creates a two-panel figure with:
    - Left: Histogram with KDE and normal distribution overlay
    - Right: Q-Q plot against normal distribution

    Args:
        log_returns: Series of log-returns.
        output_dir: Directory to save the plot. If None, plot is not saved.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    data = log_returns.dropna()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram with KDE
    ax1 = axes[0]
    ax1.hist(
        data,
        bins=DISTRIBUTION_HISTOGRAM_BINS,
        alpha=PLOT_ALPHA_MEDIUM,
        color="steelblue",
        edgecolor="white",
        density=True,
    )

    # KDE overlay
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), _KDE_POINTS)
    ax1.plot(x_range, kde(x_range), color="red", linewidth=_LINEWIDTH_KDE, label="KDE")

    # Normal distribution overlay
    mu, std = data.mean(), data.std()
    normal_pdf = stats.norm.pdf(x_range, mu, std)
    ax1.plot(
        x_range,
        normal_pdf,
        color="green",
        linewidth=_LINEWIDTH_KDE,
        linestyle="--",
        label="Normal",
    )

    ax1.axvline(
        mu,
        color="red",
        linestyle=":",
        linewidth=_LINEWIDTH_AXVLINE,
        alpha=_AXVLINE_ALPHA,
    )
    ax1.set_title("Distribution des Log-Returns", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax1.set_xlabel("Log-Return")
    ax1.set_ylabel("Densite")
    ax1.legend()

    # Q-Q plot
    ax2 = axes[1]
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Normal)", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Summary statistics
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    fig.suptitle(
        f"Log-Returns: Mean={mu:.6f}, Std={std:.6f}, Skew={skewness:.3f}, Kurt={kurtosis:.3f}",
        fontsize=FONTSIZE_SUBTITLE,
        y=_SUPTITLE_Y_OFFSET,
    )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "log_returns_distribution.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info("Distribution plot saved: %s", filepath)

    return fig


def run_normality_tests(
    series: pd.Series,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """
    Run normality tests (Jarque-Bera and Shapiro-Wilk) on a time series.

    Jarque-Bera: H0 = Normal distribution (based on skewness and kurtosis).
    Shapiro-Wilk: H0 = Normal distribution (limited to n < 5000).

    Args:
        series: Time series data (e.g., log-returns).
        output_dir: Directory to save results as JSON.

    Returns:
        Dictionary with test results for each normality test.
    """
    import json

    from scipy.stats import jarque_bera, shapiro

    from src.constants import DEFAULT_RANDOM_STATE

    data = series.dropna().values
    n_obs = len(data)

    results: dict[str, object] = {
        "n_observations": n_obs,
    }

    # Jarque-Bera test
    try:
        jb_stat, jb_pvalue = jarque_bera(data)
        jb_conclusion = "Non normale" if jb_pvalue < 0.05 else "Compatible avec normale"  # type: ignore[operator]
        results["Jarque_Bera"] = {
            "statistic": float(jb_stat),  # type: ignore[arg-type]
            "p_value": float(jb_pvalue),  # type: ignore[arg-type]
            "conclusion": jb_conclusion,
            "hypothesis": "H0: Distribution normale (p < 0.05 => rejeter H0 => non normale)",
        }
    except Exception as e:
        logger.warning("Jarque-Bera test failed: %s", e)
        results["Jarque_Bera"] = {"error": str(e)}

    # Shapiro-Wilk test (limited to 5000 observations)
    shapiro_limit = 5000
    try:
        if n_obs <= shapiro_limit:
            sw_stat, sw_pvalue = shapiro(data)
            sw_conclusion = "Non normale" if sw_pvalue < 0.05 else "Compatible avec normale"
            results["Shapiro_Wilk"] = {
                "statistic": float(sw_stat),
                "p_value": float(sw_pvalue),
                "conclusion": sw_conclusion,
                "hypothesis": "H0: Distribution normale (p < 0.05 => rejeter H0 => non normale)",
            }
        else:
            # Subsample for Shapiro-Wilk
            rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
            sample = rng.choice(data.tolist(), size=shapiro_limit, replace=False)
            sw_stat, sw_pvalue = shapiro(sample)
            sw_conclusion = "Non normale" if sw_pvalue < 0.05 else "Compatible avec normale"
            results["Shapiro_Wilk"] = {
                "statistic": float(sw_stat),
                "p_value": float(sw_pvalue),
                "conclusion": sw_conclusion,
                "hypothesis": "H0: Distribution normale (test sur 5000 obs. aleatoires)",
                "note": "Sous-echantillonne a 5000 observations (limite Shapiro-Wilk)",
            }
    except Exception as e:
        logger.warning("Shapiro-Wilk test failed: %s", e)
        results["Shapiro_Wilk"] = {"error": str(e)}

    # Log results
    logger.info("=" * 60)
    logger.info("TESTS DE NORMALITE")
    logger.info("=" * 60)
    for test_name in ["Jarque_Bera", "Shapiro_Wilk"]:
        test_res = results.get(test_name, {})
        if isinstance(test_res, dict):
            if "error" not in test_res and "statistic" in test_res:
                logger.info(
                    "  %s: stat=%.4f, p=%.6f -> %s",
                    test_name,
                    test_res["statistic"],
                    test_res["p_value"],
                    test_res["conclusion"],
                )
            elif "error" in test_res:
                logger.info("  %s: Erreur - %s", test_name, test_res["error"])

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "normality_tests.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Normality tests saved: %s", filepath)

    return results
