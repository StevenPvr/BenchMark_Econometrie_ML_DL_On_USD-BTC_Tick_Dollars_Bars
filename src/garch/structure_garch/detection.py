"""Identification and pre-diagnostics for ARCH/GARCH.

Implements methodology for detecting conditional heteroskedasticity:
1. Extract residuals εt from ARIMA model (mean model)
2. Test for ARCH effect using Lagrange Multiplier test (ARCH-LM)
3. Inspect autocorrelation of squared residuals

A significant autocorrelation structure in squared residuals indicates
that a GARCH model is relevant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

# Constants imported explicitly - no defaults used
from src.garch.structure_garch.utils import (
    compute_arch_lm_test,
    render_with_matplotlib,
    resolve_out_path,
    safe_import_matplotlib,
    verify_or_fallback,
)
from src.utils import ensure_output_dir, get_logger, write_placeholder_file

logger = get_logger(__name__)
_Z_CONF = 1.96


def detect_heteroskedasticity(
    residuals: np.ndarray,
    *,
    lags: int,
    acf_lags: int,
    alpha: float,
) -> dict[str, object]:
    """Detect conditional heteroskedasticity (ARCH/GARCH effect).

    Runs ARCH-LM test (Lagrange Multiplier test) for ARCH effect and
    analyzes ACF of squared residuals.

    Args:
        residuals: Residual series εt from mean model (ARIMA).
        lags: Lags for ARCH-LM test.
        acf_lags: Maximum lag for ACF computation of squared residuals.
        alpha: Significance level for ARCH-LM test.

    Returns:
        Dict summarizing test statistics and boolean flags.
    """
    from src.garch.structure_garch.utils import compute_squared_acf

    lm = compute_arch_lm_test(residuals, lags=lags)

    arch_present = bool(np.isfinite(lm["p_value"]) and lm["p_value"] < alpha)

    # Compute ACF of squared residuals
    finite_mask = np.isfinite(residuals)
    finite_residuals = residuals[finite_mask]

    if len(finite_residuals) == 0:
        acf_squared: np.ndarray = np.array([])
        acf_significance_level = 0.0
        acf_significant = False
    else:
        acf_squared = compute_squared_acf(finite_residuals, nlags=acf_lags)
        # 95% confidence level (1.96 standard deviations)
        acf_significance_level = 1.96 * np.sqrt(1.0 / len(finite_residuals))
        # Check if any ACF values exceed significance threshold
        acf_significant = bool(np.any(np.abs(acf_squared) > acf_significance_level))

    return {
        "arch_lm": lm,
        "arch_effect_present": arch_present,
        "acf_squared": acf_squared.tolist(),
        "acf_significance_level": acf_significance_level,
        "acf_significant": acf_significant,
    }


def _resolve_and_ensure_output_path(out_path: Path | None) -> Path:
    """Resolve output path and ensure parent directory exists.

    Args:
        out_path: Optional output path.

    Returns:
        Resolved output path.
    """
    resolved_path = resolve_out_path(out_path)
    ensure_output_dir(resolved_path)
    return resolved_path


def _save_diagnostics_plot(
    x: np.ndarray,
    acf_sq: np.ndarray,
    conf: float,
    acf_lags: int,
    out_path: Path,
) -> None:
    """Save diagnostics plot using matplotlib or placeholder.

    Args:
        x: Finite residual series.
        acf_sq: ACF of squared residuals.
        conf: Confidence level for significance bands.
        acf_lags: Maximum lag for ACF plot.
        out_path: Output file path.
    """
    Figure, FigureCanvas, have_matplotlib = safe_import_matplotlib()

    if have_matplotlib:
        render_with_matplotlib(
            Figure=cast(Any, Figure),
            FigureCanvas=cast(Any, FigureCanvas),
            x=x,
            acf_sq=acf_sq,
            conf=conf,
            acf_lags=acf_lags,
            out_path=out_path,
        )
    else:
        write_placeholder_file(out_path)

    verify_or_fallback(out_path)


def plot_arch_diagnostics(
    residuals: np.ndarray,
    *,
    acf_lags: int,
    out_path: Path | None = None,
) -> Path:
    """Plot ARCH/GARCH diagnostics for residuals.

    Creates a plot showing residuals and ACF of squared residuals
    to identify conditional heteroskedasticity.

    Args:
        residuals: Residual series εt from mean model (ARIMA).
        acf_lags: Maximum lag for ACF plot.
        out_path: Optional output path. Uses default if None.

    Returns:
        Path to saved plot file.
    """
    resolved_path = _resolve_and_ensure_output_path(out_path)

    # Compute ACF of squared residuals
    from src.garch.structure_garch.utils import compute_squared_acf

    acf_sq = compute_squared_acf(residuals, nlags=acf_lags)

    # Filter finite residuals for plotting
    finite_mask = np.isfinite(residuals)
    x = residuals[finite_mask]

    if len(x) == 0:
        write_placeholder_file(resolved_path)
        return resolved_path

    # Use 95% confidence level (1.96 standard deviations)
    conf = 1.96 * np.sqrt(1.0 / len(x))

    _save_diagnostics_plot(x, acf_sq, conf, acf_lags, resolved_path)
    return resolved_path


__all__ = [
    "detect_heteroskedasticity",
    "plot_arch_diagnostics",
]
