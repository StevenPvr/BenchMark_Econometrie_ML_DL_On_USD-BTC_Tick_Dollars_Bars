"""GARCH data visualization and exploration module.

Implements methodology for GARCH data visualization:
- Plot distribution of ARIMA residuals
- Plot ACF of squared ARIMA residuals (for ARCH effect detection)

Note: Functions for returns visualization have been removed as the
returns clustering plot was deleted from the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

from src.constants import GARCH_ACF_LAGS_DEFAULT, GARCH_DATA_VISU_PLOTS_DIR, GARCH_LM_LAGS_DEFAULT
from src.garch.structure_garch.utils import compute_acf, _compute_lm_statistic, _plot_acf_squared
from src.utils import get_logger
from src.visualization import create_figure_canvas, save_canvas

logger = get_logger(__name__)
_FONTSIZE_TITLE = 14


def _clean_residuals(residuals: np.ndarray) -> np.ndarray:
    """Clean residuals array by removing non-finite values.

    Args:
        residuals: Raw residuals array

    Returns:
        Cleaned residuals array

    Raises:
        ValueError: If residuals is None or empty after cleaning
    """
    if residuals is None:
        raise ValueError("residuals cannot be None")

    residuals_clean = np.asarray(residuals, dtype=float)
    residuals_clean = residuals_clean[np.isfinite(residuals_clean)]

    if residuals_clean.size == 0:
        raise ValueError("No finite residuals found after cleaning")

    return residuals_clean


def _create_distribution_figure() -> tuple[Any, Any, Any]:
    """Create figure for residuals distribution visualization.

    Returns:
        Tuple of (figure, canvas, axis)
    """
    fig, canvas, axes = create_figure_canvas((12, 6), n_rows=1, n_cols=1)
    return fig, canvas, axes[0] if isinstance(axes, list) else axes


def _plot_histogram_with_normal(ax: Any, residuals: np.ndarray) -> None:
    """Plot histogram of residuals with normal distribution overlay.

    Args:
        ax: Matplotlib axis
        residuals: Cleaned residuals array
    """
    import scipy.stats as stats

    # Plot histogram
    n, bins, patches = ax.hist(
        residuals, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black"
    )

    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Distribution normale")

    ax.set_xlabel("Résidus ARIMA")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution des résidus ARIMA")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _save_distribution_figure(fig: Any, canvas: Any, out_path: Path, title: str) -> None:
    """Save distribution figure to file.

    Args:
        fig: Matplotlib figure
        canvas: Matplotlib canvas
        out_path: Output file path
        title: Figure title
    """
    fig.suptitle(title, fontsize=_FONTSIZE_TITLE, fontweight="bold")
    save_canvas(canvas, out_path, format="png")
    logger.info(f"Saved residuals distribution plot: {out_path}")


def plot_residuals_distribution(
    residuals: np.ndarray,
    *,
    outdir: Path | str = GARCH_DATA_VISU_PLOTS_DIR,
    filename: str = "arima_residuals_distribution.png",
) -> Path:
    """Plot distribution of ARIMA residuals with normal distribution overlay.

    Creates a histogram of residuals with a fitted normal distribution curve.

    Args:
        residuals: Array of ARIMA residuals
        outdir: Output directory for plot
        filename: Output filename

    Returns:
        Path to saved plot

    Raises:
        ValueError: If residuals array is empty or invalid
    """
    out_path = Path(outdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    residuals_clean = _clean_residuals(residuals)

    if residuals_clean.size == 0:
        raise ValueError("Cannot create plot: no finite residuals after cleaning")

    fig, canvas, ax = _create_distribution_figure()
    _plot_histogram_with_normal(ax, residuals_clean)
    _save_distribution_figure(fig, canvas, out_path, "Distribution des résidus ARIMA")

    return out_path


def compute_squared_acf(residuals: np.ndarray, nlags: int = GARCH_ACF_LAGS_DEFAULT) -> np.ndarray:
    """Compute autocorrelation of squared residuals.

    Inspects autocorrelation structure in squared residuals ε_t^2.
    A significant autocorrelation indicates that a GARCH model is relevant.

    Args:
        residuals: Residual series εt from mean model (ARIMA).
        nlags: Maximum lag.

    Returns:
        ACF(ε^2) for lags 1..nlags.
    """
    e2 = np.asarray(residuals, dtype=float) ** 2
    return compute_acf(e2, nlags=nlags)


def _create_acf_squared_figure() -> tuple[Any, Any, Any]:
    """Create figure for squared residuals ACF visualization.

    Returns:
        Tuple of (figure, canvas, axis)
    """
    fig, canvas, axes = create_figure_canvas((12, 6), n_rows=1, n_cols=1)
    return fig, canvas, axes[0] if isinstance(axes, list) else axes


def _save_acf_squared_figure(fig: Any, canvas: Any, out_path: Path, title: str) -> None:
    """Save ACF squared figure to file.

    Args:
        fig: Matplotlib figure
        canvas: Matplotlib canvas
        out_path: Output file path
        title: Figure title
    """
    fig.suptitle(title, fontsize=_FONTSIZE_TITLE, fontweight="bold")
    save_canvas(canvas, out_path, format="png")
    logger.info(f"Saved squared residuals ACF plot: {out_path}")


def plot_squared_residuals_acf(
    residuals: np.ndarray,
    *,
    nlags: int = GARCH_ACF_LAGS_DEFAULT,
    outdir: Path | str = GARCH_DATA_VISU_PLOTS_DIR,
    filename: str = "arima_squared_residuals_acf.png",
) -> Path:
    """Plot ACF of squared ARIMA residuals for ARCH effect detection.

    Creates a bar plot of autocorrelation function for squared residuals.
    Significant autocorrelations indicate potential ARCH/GARCH effects.

    Args:
        residuals: Array of ARIMA residuals
        nlags: Maximum number of lags to compute ACF
        outdir: Output directory for plot
        filename: Output filename

    Returns:
        Path to saved plot

    Raises:
        ValueError: If residuals array is empty or invalid
    """
    out_path = Path(outdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    residuals_clean = _clean_residuals(residuals)

    if residuals_clean.size == 0:
        raise ValueError("Cannot create plot: no finite residuals after cleaning")

    # Compute ACF of squared residuals
    acf_sq = compute_squared_acf(residuals_clean, nlags=nlags)

    # Compute confidence level (approximate)
    n = len(residuals_clean)
    conf = 1.96 / np.sqrt(n)  # 95% confidence level

    fig, canvas, ax = _create_acf_squared_figure()
    _plot_acf_squared(ax, acf_sq, conf, nlags)
    _save_acf_squared_figure(fig, canvas, out_path, "ACF des résidus ARIMA au carré")

    return out_path


def test_arch_effect(
    residuals: np.ndarray,
    lags: int = GARCH_LM_LAGS_DEFAULT,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Engle's ARCH-LM test for ARCH effect detection.

    Tests for ARCH effect by regressing squared residuals on lagged squared residuals:
    ε_t² ~ const + lags(ε_t²)

    Args:
        residuals: Residual series εt from mean model (ARIMA).
        lags: Number of lags in regression.
        alpha: Significance level.

    Returns:
        Dict with lm_stat, p_value, df, arch_present.
    """
    e2 = np.asarray(residuals, dtype=float) ** 2
    e2 = e2[np.isfinite(e2)]
    n = int(e2.size)

    if n <= lags:
        return {
            "lm_stat": float("nan"),
            "p_value": float("nan"),
            "df": float(lags),
            "arch_present": False,
        }

    lm_stat = _compute_lm_statistic(e2, lags, n)
    p_value = float(chi2.sf(lm_stat, lags))  # survival function (1 - cdf)
    arch_present = bool(p_value < alpha)

    return {
        "lm_stat": lm_stat,
        "p_value": p_value,
        "df": float(lags),
        "arch_present": arch_present,
    }
