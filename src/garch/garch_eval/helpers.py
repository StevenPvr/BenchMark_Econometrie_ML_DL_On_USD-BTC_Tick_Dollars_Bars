"""Helper utilities for GARCH evaluation.

This module consolidates various utility functions for file operations,
parsing, assembly, and statistical functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import GARCH_FORECASTS_FILE, GARCH_MODEL_FILE
from src.utils import chi2_sf, ensure_output_dir, get_logger

logger = get_logger(__name__)


# ============================================================================
# Statistical utilities
# ============================================================================


def _validate_and_filter_arrays(
    e: np.ndarray | list[float],
    sigma2: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and filter residuals and variance arrays.

    Converts inputs to arrays, checks for finite values and positive variance,
    and returns filtered arrays with a boolean mask.

    Args:
    ----
        e: Residuals array or list.
        sigma2: Variance array or list.

    Returns:
    -------
        Tuple of (filtered_residuals, filtered_variance, mask).

    """
    e_arr = np.asarray(e, dtype=float).ravel()
    s2_arr = np.asarray(sigma2, dtype=float).ravel()
    mask = np.isfinite(e_arr) & np.isfinite(s2_arr) & (s2_arr > 0)
    return e_arr[mask], s2_arr[mask], mask


def _clip_probability(p: float, eps: float) -> float:
    """Clip probability to avoid log(0) or log(1).

    Args:
    ----
        p: Probability value.
        eps: Small epsilon value for clipping.

    Returns:
    -------
        Clipped probability in range [eps, 1-eps].

    """
    return min(max(p, eps), 1.0 - eps)


# ============================================================================
# Parsing utilities
# ============================================================================


def parse_alphas(alphas_str: str) -> list[float]:
    """Parse comma-separated alpha values.

    Args:
    ----
        alphas_str: Comma-separated string of alpha values.

    Returns:
    -------
        List of parsed alpha values.

    """
    return [float(a) for a in str(alphas_str).split(",") if a]


def load_best_model() -> tuple[dict[str, float], str, str, float | None, float | None]:
    """Load best model from trained model file.

    Returns
    -------
        Tuple of (params, name, dist, nu, lambda_skew).

    Raises:
    ------
        FileNotFoundError: If trained model file is missing.

    """
    # Import locally to avoid circular dependency
    import joblib

    if not GARCH_MODEL_FILE.exists():
        msg = f"Trained model file not found: {GARCH_MODEL_FILE}"
        raise FileNotFoundError(msg)

    model_data = joblib.load(GARCH_MODEL_FILE)
    params = model_data["params"]
    dist = model_data["dist"]

    # Build model name from order and distribution
    name = f"egarch_{dist}"

    # Extract distribution parameters if present
    nu = params.get("nu")
    # Support both keys for skew parameter to avoid format mismatches
    lambda_skew = params.get("lambda_skew") or params.get("lambda")

    return params, name, dist, nu, lambda_skew


# ============================================================================
# File operations
# ============================================================================


def save_forecast_results(
    out: pd.DataFrame,
    use_mz_calibration: bool,
    mz_intercept: float,
    mz_slope: float,
) -> None:
    """Save forecast results to CSV file.

    Args:
    ----
        out: Forecast results DataFrame.
        use_mz_calibration: Whether MZ calibration was applied.
        mz_intercept: MZ intercept value.
        mz_slope: MZ slope value.

    """
    out["mz_calibrated"] = use_mz_calibration
    if use_mz_calibration:
        out["mz_intercept"] = mz_intercept
        out["mz_slope"] = mz_slope

    garch_forecasts_file = GARCH_FORECASTS_FILE
    ensure_output_dir(garch_forecasts_file)
    out.to_csv(garch_forecasts_file, index=False)
    logger.info("Saved GARCH forecasts to: %s", garch_forecasts_file)


def to_numpy(series_like: list[float] | np.ndarray) -> np.ndarray:
    """Convert series-like object to numpy array.

    Args:
    ----
        series_like: Input series or array.

    Returns:
    -------
        Numpy array.

    """
    return np.asarray(list(series_like), dtype=float)


# ============================================================================
# Assembly utilities
# ============================================================================


def _build_forecast_row(
    h: int,
    s2: float,
    level: float,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> dict[str, float | str]:
    """Build a single forecast row with PI and VaR.

    Args:
    ----
        h: Horizon step.
        s2: Variance forecast.
        level: Prediction interval level.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Dictionary with forecast row data.

    """
    # Import locally to avoid circular dependency
    from src.garch.garch_eval.eval import prediction_interval, value_at_risk

    lo, hi = prediction_interval(0.0, s2, level=level, dist=dist, nu=nu, lambda_skew=lambda_skew)
    var_l = value_at_risk(
        1.0 - level, mean=0.0, variance=s2, dist=dist, nu=nu, lambda_skew=lambda_skew
    )
    return {
        "h": int(h),
        "sigma2_forecast": float(s2),
        "sigma_forecast": float(np.sqrt(s2)),
        "pi_level": float(level),
        "pi_lower": float(lo),
        "pi_upper": float(hi),
        "var_left_alpha": float(1.0 - level),
        "VaR": float(var_l),
        "dist": dist,
        "nu": float(nu) if nu is not None else np.nan,
        "lambda": float(lambda_skew) if lambda_skew is not None else np.nan,
    }


def assemble_forecast_results(
    s2_h: np.ndarray,
    s2_1: float,
    level: float,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> pd.DataFrame:
    """Assemble forecast results into DataFrame with PI and VaR.

    Args:
    ----
        s2_h: Multi-step variance forecasts.
        s2_1: One-step variance forecast.
        level: Prediction interval level.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        DataFrame with forecast results.

    """
    rows = [
        _build_forecast_row(h, s2, level, dist, nu, lambda_skew)
        for h, s2 in enumerate(s2_h, start=1)
    ]
    out = pd.DataFrame(rows)
    # Sanity check: include one-step as first row consistency
    if out.shape[0] >= 1:
        out.loc[out.index[0], "sigma2_one_step_check"] = s2_1
    return out


# ============================================================================
# Plotting utilities
# ============================================================================


def _setup_plot_style() -> tuple:
    """Setup matplotlib and seaborn for plotting.

    Returns
    -------
        Tuple of (plt, sns) modules.

    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    return plt, sns


# Update __all__
__all__ = [
    # Statistical
    "chi2_sf",
    "_validate_and_filter_arrays",
    "_clip_probability",
    # Parsing
    "parse_alphas",
    "load_best_model",
    # File operations
    "save_forecast_results",
    "to_numpy",
    # Assembly
    "assemble_forecast_results",
    # Plotting
    "_setup_plot_style",
]
