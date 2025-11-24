"""Plotting utilities for GARCH evaluation (variance and VaR visuals)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd # type: ignore[import-untyped]

from src.path import (
    GARCH_EVAL_VAR_RESIDUALS_PLOT,
    GARCH_EVAL_VAR_SCATTER_PLOT,
    GARCH_EVAL_VAR_TIMESERIES_PLOT,
)
from src.garch.garch_eval.data_loading import (
    extract_aligned_test_indices,
    load_dataset_for_metrics,
    prepare_residuals_from_dataset,
)
from src.garch.garch_eval.helpers import _setup_plot_style
from src.garch.garch_eval.utils import load_test_resid_sigma2, var_quantile
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)
_FIGSIZE_DEFAULT = (10, 4)
_FIGSIZE_SCATTER = (5, 5)
_FIGSIZE_RESIDUALS = (10, 3)
_PLOT_LIMIT_MULTIPLIER = 1.05


def _validate_plot_data(
    dates: pd.Series | np.ndarray,
    e_test: np.ndarray,
    s2_test: np.ndarray,
) -> bool:
    """Validate plot input data consistency.

    Args:
    ----
        dates: Date array.
        e_test: Test residuals.
        s2_test: Test variance forecasts.

    Returns:
    -------
        True if validation passes, False otherwise.

    """
    if e_test.size == 0 or s2_test.size == 0:
        logger.warning(
            "Cannot plot: empty data arrays (e_test.size=%d, s2_test.size=%d)",
            e_test.size,
            s2_test.size,
        )
        return False

    if e_test.size != s2_test.size:
        logger.warning(
            "Cannot plot: size mismatch (e_test.size=%d, s2_test.size=%d)",
            e_test.size,
            s2_test.size,
        )
        return False

    if len(dates) != e_test.size:
        logger.warning(
            "Cannot plot: dates size mismatch (dates.size=%d, e_test.size=%d)",
            len(dates),
            e_test.size,
        )
        return False

    return True


def plot_variance_timeseries(
    dates: pd.Series | np.ndarray,
    e_test: np.ndarray,
    s2_test: np.ndarray,
    output: Path,
) -> None:
    """Plot time series of realized variance (e^2) vs predicted variance (σ²)."""
    plt, _ = _setup_plot_style()

    # Validate inputs
    if not _validate_plot_data(dates, e_test, s2_test):
        return

    ensure_output_dir(output)
    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)
    ax.plot(dates, e_test**2, label="Réalisé e²", color="#1f77b4", linewidth=1.2)
    ax.plot(dates, s2_test, label="Prévu σ²", color="#ff7f0e", linewidth=1.2)
    ax.set_title("Variance réalisée vs prévision GARCH (test)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Variance")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info("Saved variance time series plot to: %s", output)


def _safe_nanmax(arr: np.ndarray) -> float:
    """Compute safe nanmax handling empty or all-NaN arrays."""
    if arr.size == 0:
        return float("nan")
    m = np.nanmax(arr)
    return float(m) if np.isfinite(m) else float("nan")


def _compute_plot_limits(x: np.ndarray, y: np.ndarray) -> float:
    """Compute plot limits from data."""
    xm = _safe_nanmax(x)
    ym = _safe_nanmax(y)
    base = max([v for v in [xm, ym] if np.isfinite(v)], default=1.0)
    return float(base * _PLOT_LIMIT_MULTIPLIER) if base > 0 else 1.0


def plot_variance_scatter(
    e_test: np.ndarray,
    s2_test: np.ndarray,
    output: Path,
) -> None:
    """Scatter e² vs σ² with y=x reference line."""
    plt, _ = _setup_plot_style()

    # Validate inputs
    if e_test.size == 0 or s2_test.size == 0 or e_test.size != s2_test.size:
        logger.warning(
            "Cannot plot: invalid data (e_test.size=%d, s2_test.size=%d)",
            e_test.size,
            s2_test.size,
        )
        return

    ensure_output_dir(output)
    y = e_test**2
    x = s2_test
    lim = _compute_plot_limits(x, y)
    fig, ax = plt.subplots(figsize=_FIGSIZE_SCATTER)
    ax.scatter(x, y, s=10, alpha=0.6, color="#1f77b4", label="Points (e² vs σ²)")
    ax.plot([0, lim], [0, lim], color="#2ca02c", linestyle="--", label="y = x")
    ax.set_title("Dispersion e² (réalisé) vs σ² (prévu)")
    ax.set_xlabel("σ² (prévu)")
    ax.set_ylabel("e² (réalisé)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info("Saved variance scatter plot to: %s", output)


def plot_variance_residuals(
    dates: pd.Series | np.ndarray,
    e_test: np.ndarray,
    s2_test: np.ndarray,
    output: Path,
) -> None:
    """Plot residuals (e² - σ²) over time."""
    plt, _ = _setup_plot_style()

    # Validate inputs
    if not _validate_plot_data(dates, e_test, s2_test):
        return

    ensure_output_dir(output)
    fig, ax = plt.subplots(figsize=_FIGSIZE_RESIDUALS)
    residuals = (e_test**2) - s2_test
    ax.plot(dates, residuals, color="#d62728", linewidth=1.0, label="e² - σ²")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Résidus de variance (e² - σ²) sur test")
    ax.set_xlabel("Date")
    ax.set_ylabel("Différence")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info("Saved variance residuals plot to: %s", output)


def _compute_var_violations(
    e_test: np.ndarray,
    s2_test: np.ndarray,
    alpha: float,
    dist: str,
    nu: float | None,
    lambda_skew: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute VaR series and violation mask."""
    q = var_quantile(alpha, dist, nu, lambda_skew)
    var_series = q * np.sqrt(s2_test)
    hits = e_test < var_series
    return var_series, hits


def plot_var_violations_enhanced(
    dates: pd.Series | np.ndarray,
    resid: np.ndarray,
    sigma2: np.ndarray,
    alpha: float,
    dist: str,
    nu: float | None,
    output: Path,
    lambda_skew: float | None = None,
) -> None:
    """Plot VaR violations with prominent red markers for violations.

    Args:
        dates: Date array for x-axis.
        resid: Test residuals (realized returns).
        sigma2: Variance forecasts (sigma2_garch).
        alpha: VaR confidence level (e.g., 0.01 for 1%).
        dist: Distribution type ('student' or 'skewt').
        nu: Degrees of freedom for Student-t/Skew-t.
        output: Output path for the plot.
        lambda_skew: Skewness parameter for Skew-t.
    """
    plt, _ = _setup_plot_style()

    if not _validate_plot_data(dates, resid, sigma2):
        return

    ensure_output_dir(output)
    var_series, hits = _compute_var_violations(resid, sigma2, alpha, dist, nu, lambda_skew)

    n_violations = int(np.sum(hits))
    n_total = int(resid.size)
    violation_rate = float(n_violations / n_total) if n_total > 0 else 0.0
    expected_rate = float(alpha)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot returns
    ax.plot(dates, resid, color="#2E86AB", linewidth=0.8, alpha=0.7, label="Returns")

    # Plot VaR threshold
    ax.plot(
        dates, var_series, color="#A23B72", linewidth=1.5, linestyle="-", label=f"VaR {alpha:.0%}"
    )

    # Mark violations with prominent red markers
    if n_violations > 0:
        violation_dates = np.asarray(dates)[hits]
        violation_returns = resid[hits]
        ax.scatter(
            violation_dates,
            violation_returns,
            color="#F24236",
            s=50,
            marker="x",
            linewidth=2,
            zorder=5,
            label=f"Violations ({n_violations})",
        )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)

    ax.set_title(
        f"VaR {alpha:.0%} Violations - Test Split\n"
        f"Observed: {violation_rate:.2%} | Expected: {expected_rate:.2%}",
        fontsize=11,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "Saved enhanced VaR violations plot (α=%.2f) to: %s | %d violations",
        alpha,
        output,
        n_violations,
    )


def _generate_variance_plots(
    dates_test: np.ndarray,
    e_test: np.ndarray,
    s2_test: np.ndarray,
) -> None:
    """Generate core variance plots.

    Args:
    ----
        dates_test: Test dates.
        e_test: Test residuals.
        s2_test: Test variance.

    """
    plot_variance_timeseries(dates_test, e_test, s2_test, GARCH_EVAL_VAR_TIMESERIES_PLOT)
    plot_variance_scatter(e_test, s2_test, GARCH_EVAL_VAR_SCATTER_PLOT)
    plot_variance_residuals(dates_test, e_test, s2_test, GARCH_EVAL_VAR_RESIDUALS_PLOT)


def _load_test_data_for_plots(
    params: dict[str, float],
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare test data for plotting with aligned dates.

    Aligns dates with the filtered residual positions used to build e_test/s2_test
    to avoid empty or misaligned plots when residuals contain invalid values.
    """
    dataset_df = load_dataset_for_metrics()
    df_sorted = dataset_df.sort_values("date").reset_index(drop=True)

    # Compute test series using the leak-free path
    e_test, s2_test = load_test_resid_sigma2(
        params,
        df_sorted,
        model_name=model_name,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )

    # Derive aligned dates for TEST using the same filtering and mapping
    try:
        _df_sorted2, _resid_all, valid_mask, _resid_f = prepare_residuals_from_dataset(df_sorted)
        pos_test = extract_aligned_test_indices(_df_sorted2, valid_mask)
        if pos_test.size == 0:
            dates_test = np.array([], dtype="datetime64[ns]")
        else:
            idx_all = np.arange(_df_sorted2.shape[0])
            idx_valid = idx_all[valid_mask]
            idx_test_valid = idx_valid[pos_test]
            dates_test = _df_sorted2.loc[idx_test_valid, "date"].to_numpy()
    except Exception as ex:
        logger.warning("Failed to align dates for plotting: %s", ex)
        dates_test = np.array([], dtype="datetime64[ns]")

    return df_sorted, e_test, s2_test, dates_test
