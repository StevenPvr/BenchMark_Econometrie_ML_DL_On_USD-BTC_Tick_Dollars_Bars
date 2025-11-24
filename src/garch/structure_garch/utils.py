"""Utility functions for GARCH structure detection.

Contains helper functions for:
- Loading GARCH datasets
- Preparing residuals
- Computing ACF (Autocorrelation Function)
- Computing ARCH-LM test statistics
- Plotting utilities
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f, t  # type: ignore[import]

from src.constants import GARCH_ACF_LAGS_DEFAULT, GARCH_DATASET_FILE, GARCH_LM_LAGS_DEFAULT
from src.path import GARCH_STRUCTURE_PLOT
from src.utils import (
    chi2_sf,
    get_logger,
    load_dataframe,
    validate_file_exists,
    validate_required_columns,
)

logger = get_logger(__name__)

_STRUCTURE_FIGURE_SIZE: tuple[int, int] = (10, 6)
_STRUCTURE_PLOT_COLOR_RESIDUALS = "#1f77b4"
_STRUCTURE_PLOT_COLOR_ACF = "#ff7f0e"
_STRUCTURE_PLOT_COLOR_ZERO_LINE = "black"
_STRUCTURE_PLOT_COLOR_CONFIDENCE = "red"
_STRUCTURE_PLOT_BAR_WIDTH = 0.8
_STRUCTURE_PLOT_LINEWIDTH = 0.8
_STRUCTURE_PLOT_FONTSIZE_TITLE = 12
_Z_CONF = 1.96


# ============================================================================
# Data Loading Utilities
# ============================================================================


def load_garch_dataset(path: str | None = None) -> pd.DataFrame:
    """Load dataset for GARCH training/diagnostics.

    Delegates to src.utils functions for consistency.

    Args:
        path: Optional CSV path. Defaults to `GARCH_DATASET_FILE` from constants.

    Returns:
        DataFrame with required columns.

    Raises:
        FileNotFoundError: If dataset is missing.
        ValueError: If required columns are absent.
    """
    csv_path = GARCH_DATASET_FILE if path is None else Path(path)
    validate_file_exists(csv_path, "GARCH dataset")

    df = load_dataframe(csv_path, date_columns=["date"], validate_not_empty=False)
    required = {"date", "split", "weighted_log_return"}
    try:
        validate_required_columns(df, required, "GARCH dataset")
    except KeyError as err:
        # Standardize error type for tests expecting ValueError
        raise ValueError("GARCH dataset missing required columns") from err
    return df


def _find_residual_column(data: pd.DataFrame) -> str:
    """Find the name of the residual column in the dataframe.

    Args:
        data: Input dataframe.

    Returns:
        Column name ('arima_resid' or 'sarima_resid').

    Raises:
        ValueError: If neither 'arima_resid' nor 'sarima_resid' column is present.

    """
    if "arima_resid" in data.columns:
        return "arima_resid"
    if "sarima_resid" in data.columns:
        return "sarima_resid"
    msg = "Required column 'arima_resid' or 'sarima_resid' not found in dataset."
    raise ValueError(msg) from None


def _extract_residuals_from_column(data: pd.DataFrame, col_name: str) -> np.ndarray:
    """Extract residuals from a specific column.

    Args:
        data: Input dataframe.
        col_name: Name of the residual column.

    Returns:
        Residual array.

    Raises:
        ValueError: If column contains no valid values.

    """
    series = pd.to_numeric(data[col_name], errors="coerce")
    resid = np.asarray(series, dtype=float)
    if not bool(pd.Series(series).notna().any()):
        msg = f"Column '{col_name}' contains no valid residual values."
        raise ValueError(msg) from None
    return resid


def _compute_residuals_from_fitted(data: pd.DataFrame) -> np.ndarray | None:
    """Compute residuals from fitted values for train data.

    Args:
        data: Input dataframe with fitted values.

    Returns:
        Computed residual array if successful, None otherwise.
    """
    if "arima_fitted_in_sample" not in data.columns:
        return None
    if "weighted_log_return" not in data.columns:
        return None

    returns_series = pd.to_numeric(data["weighted_log_return"], errors="coerce")
    fitted_series = pd.to_numeric(data["arima_fitted_in_sample"], errors="coerce")
    returns_arr = np.asarray(returns_series, dtype=float)
    fitted_arr = np.asarray(fitted_series, dtype=float)
    computed_residuals = returns_arr - fitted_arr
    if bool(np.any(np.isfinite(computed_residuals))):
        logger.info("Computed train residuals from weighted_log_return - arima_fitted_in_sample")
        return computed_residuals
    return None


def _filter_test_data(df: pd.DataFrame, use_test_only: bool) -> pd.DataFrame:
    """Filter dataframe to test split if requested.

    Args:
        df: Input dataframe.
        use_test_only: Whether to filter to test split.

    Returns:
        Filtered dataframe.
    """
    data: pd.DataFrame = df.copy()
    if use_test_only and "split" in data.columns:
        data = data[data["split"] == "test"].copy()  # type: ignore[assignment]
    return data


def _try_extract_from_columns(data: pd.DataFrame) -> np.ndarray | None:
    """Try to extract residuals from direct columns.

    Args:
        data: Input dataframe.

    Returns:
        Residuals array if successful, None otherwise.
    """
    try:
        col_name = _find_residual_column(data)
        return _extract_residuals_from_column(data, col_name)
    except ValueError:
        return None


def _try_compute_from_fitted(data: pd.DataFrame) -> np.ndarray | None:
    """Try to compute residuals from fitted values.

    Args:
        data: Input dataframe with fitted values.

    Returns:
        Computed residuals if successful, None otherwise.
    """
    computed_residuals = _compute_residuals_from_fitted(data)
    if computed_residuals is not None:
        logger.info("Using computed residuals from weighted_log_return - arima_fitted_in_sample")
    return computed_residuals


def prepare_residuals(df: pd.DataFrame, use_test_only: bool = True) -> np.ndarray:
    """Extract residuals εt from ARIMA model (mean model).

    Tries multiple methods to extract residuals:
    1. Direct columns: arima_resid or sarima_resid
    2. Fallback: compute from weighted_log_return - arima_fitted_in_sample (train only)

    Args:
        df: Input dataframe with ARIMA residuals or raw data.
        use_test_only: Restrict to test split if True.

    Returns:
        1D residual array εt.

    Raises:
        ValueError: When no residual column is present and cannot compute from fitted values.
    """
    data = _filter_test_data(df, use_test_only)

    # Try to find residual column first
    residuals = _try_extract_from_columns(data)
    if residuals is not None:
        return residuals

    # Fallback: try to compute from fitted values (only works for train data)
    if use_test_only:
        # For test data, we need the residual column directly
        msg = (
            "Required residual column 'arima_resid' or 'sarima_resid' not found in dataset. "
            "Cannot compute test residuals from fitted values."
        )
        raise ValueError(msg) from None

    residuals = _try_compute_from_fitted(data)
    if residuals is not None:
        return residuals

    msg = (
        "Cannot extract residuals: neither 'arima_resid'/'sarima_resid' columns exist, "
        "nor can compute from 'weighted_log_return' - 'arima_fitted_in_sample'."
    )
    raise ValueError(msg) from None


# ============================================================================
# Autocorrelation Function Utilities
# ============================================================================


def compute_acf(series: np.ndarray, nlags: int = GARCH_ACF_LAGS_DEFAULT) -> np.ndarray:
    """Compute sample ACF for a 1D series (lags 1..nlags).

    Wrapper around autocorr() from garch_diagnostic.autocorrelation
    to maintain backward compatibility. Returns only lags 1..nlags.

    Args:
        series: 1D numeric array.
        nlags: Maximum lag.

    Returns:
        ACF values for lags 1..nlags.
    """
    from src.garch.garch_diagnostic.statistics import autocorr

    r = autocorr(series, nlags)
    return r[1 : nlags + 1]  # Exclude lag 0


def compute_squared_acf(residuals: np.ndarray, nlags: int = GARCH_ACF_LAGS_DEFAULT) -> np.ndarray:
    """Compute autocorrelation of squared residuals.

    Args:
        residuals: Residual series εt from mean model (ARIMA).
        nlags: Maximum lag.

    Returns:
        ACF(ε^2) for lags 1..nlags.
    """
    e2 = np.asarray(residuals, dtype=float) ** 2
    return compute_acf(e2, nlags=nlags)


def prepare_plot_series(
    residuals: np.ndarray,
    *,
    acf_lags: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Prepare data for plotting ARCH diagnostics.

    Args:
        residuals: Residual series εt from mean model (ARIMA).
        acf_lags: Maximum lag for ACF computation.

    Returns:
        Tuple of (finite_residuals, acf_squared, confidence_level).
    """
    # Filter finite residuals
    finite_mask = np.isfinite(residuals)
    x = residuals[finite_mask]

    if len(x) == 0:
        x = np.array([])
        acf_sq = np.array([])
        conf = 0.0
    else:
        # Compute ACF of squared residuals
        acf_sq = compute_squared_acf(x, nlags=acf_lags)
        # Use 95% confidence level (1.96 standard deviations)
        conf = 1.96 * np.sqrt(1.0 / len(x))

    return x, acf_sq, conf


# ============================================================================
# ARCH-LM Test Utilities
# ============================================================================


def _compute_lm_statistic(e2: np.ndarray, lags: int, n: int) -> float:
    """Compute LM statistic from squared residuals regression.

    Args:
        e2: Squared residuals array.
        lags: Number of lags in regression.
        n: Sample size.

    Returns:
        LM statistic value, or NaN if computation fails.
    """
    Y = e2[lags:]
    X = np.ones((n - lags, lags + 1), dtype=float)
    for j in range(1, lags + 1):
        X[:, j] = e2[lags - j : n - j]

    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        Y_hat = X @ beta
        ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
        ss_res = float(np.sum((Y - Y_hat) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return (n - lags) * max(0.0, r2)
    except Exception as exc:
        logger.warning("Failed to compute ARCH-LM statistic: %s", exc)
        return float("nan")


def compute_arch_lm_test(
    residuals: np.ndarray, lags: int = GARCH_LM_LAGS_DEFAULT
) -> dict[str, float]:
    """Engle's ARCH-LM test (Lagrange Multiplier test) for ARCH effect.

    Tests for ARCH effect by regressing squared residuals on lagged squared residuals:
    ε_t^2 ~ const + lags(ε_t^2)

    Uses OLS R^2 to compute the LM statistic.

    Args:
        residuals: Residual series εt from mean model (ARIMA).
        lags: Number of lags in regression.

    Returns:
        Dict with lm_stat, p_value, df.
    """
    e2 = np.asarray(residuals, dtype=float) ** 2
    e2 = e2[~np.isnan(e2)]
    n = int(e2.size)
    if n <= lags:
        return {"lm_stat": float("nan"), "p_value": float("nan"), "df": float(lags)}

    lm = _compute_lm_statistic(e2, lags, n)
    p_val = chi2_sf(lm, lags)
    return {"lm_stat": float(lm), "p_value": float(p_val), "df": float(lags)}


# ============================================================================
# Engle-Ng Sign Bias Tests (for GARCH asymmetry)
# ============================================================================


def _compute_sign_indicators(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute sign indicator variables for Engle-Ng tests.

    Args:
        residuals: Raw residuals εₜ (not standardized).

    Returns:
        Tuple of (S_neg, S_pos) where:
        - S_neg[t] = 1 if ε[t] < 0, else 0
        - S_pos[t] = 1 if ε[t] >= 0, else 0
    """
    eps = np.asarray(residuals, dtype=float)
    s_neg = (eps < 0).astype(float)
    s_pos = (eps >= 0).astype(float)
    return s_neg, s_pos


def _preprocess_residuals_for_regression(std_residuals: np.ndarray) -> tuple[np.ndarray, bool]:
    """Preprocess standardized residuals for regression analysis.

    Args:
        std_residuals: Standardized residuals array.

    Returns:
        Tuple of (processed_residuals, has_enough_data).
        If has_enough_data is False, processed_residuals should not be used.
    """
    z = np.asarray(std_residuals, dtype=float)
    z = z[np.isfinite(z)]
    has_enough_data = len(z) >= 3
    return z, has_enough_data


def _create_nan_regression_result(n: int) -> dict[str, float]:
    """Create regression result dictionary with NaN values.

    Args:
        n: Sample size.

    Returns:
        Dict with coef, t_stat, p_value, n.
    """
    return {
        "coef": float("nan"),
        "t_stat": float("nan"),
        "p_value": float("nan"),
        "n": n,
    }


def _run_ols_regression(y: np.ndarray, X: np.ndarray) -> dict[str, float]:
    """Run OLS regression and return t-statistic and p-value for slope.

    Args:
        y: Dependent variable (n,).
        X: Independent variable (n,) - will be augmented with intercept.

    Returns:
        Dict with keys: coef, t_stat, p_value, n.
    """
    n = len(y)
    if n < 3:
        return {
            "coef": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "n": n,
        }

    # Add intercept column
    X_with_const = np.column_stack([np.ones(n), X])

    try:
        # OLS estimation: β = (X'X)^(-1) X'y
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        y_hat = X_with_const @ beta
        residuals = y - y_hat

        # Compute standard errors
        dof = n - 2  # n observations - 2 parameters (intercept + slope)
        rss = np.sum(residuals**2)
        mse = rss / dof if dof > 0 else float("nan")

        # Variance-covariance matrix: σ² (X'X)^(-1)
        XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        var_beta = mse * XtX_inv

        # Standard error of slope (second coefficient)
        se_slope = np.sqrt(var_beta[1, 1])

        # t-statistic for H0: slope = 0
        t_stat = beta[1] / se_slope if se_slope > 0 else float("nan")

        # Two-tailed p-value
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), dof))

        return {
            "coef": float(beta[1]),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "n": n,
        }
    except Exception as exc:
        logger.warning("OLS regression failed: %s", exc)
        return {
            "coef": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "n": n,
        }


def engle_ng_sign_bias_test(
    std_residuals: np.ndarray,
) -> dict[str, float]:
    """Engle-Ng (1993) sign bias test for asymmetric volatility effects.

    Tests whether positive and negative shocks have different impacts on volatility.

    Regression: z²ₜ = α + β·S⁻ₜ₋₁ + εₜ
    where:
    - zₜ = standardized residuals (εₜ/σₜ)
    - S⁻ₜ₋₁ = 1 if εₜ₋₁ < 0, else 0

    H0: β = 0 (no sign bias)
    H1: β ≠ 0 (sign bias exists)

    Args:
        std_residuals: Standardized residuals zₜ from GARCH model.

    Returns:
        Dict with coef, t_stat, p_value, n.
    """
    z, has_enough_data = _preprocess_residuals_for_regression(std_residuals)
    if not has_enough_data:
        return _create_nan_regression_result(len(z))

    # Dependent variable: z²ₜ
    z_squared = z[1:] ** 2

    # Independent variable: S⁻ₜ₋₁
    s_neg, _ = _compute_sign_indicators(z[:-1])

    return _run_ols_regression(z_squared, s_neg)


def engle_ng_negative_size_bias_test(
    std_residuals: np.ndarray,
) -> dict[str, float]:
    """Engle-Ng (1993) negative size bias test.

    Tests whether the magnitude of negative shocks affects volatility differently.

    Regression: z²ₜ = α + β·(S⁻ₜ₋₁ · εₜ₋₁) + εₜ
    where:
    - S⁻ₜ₋₁ = 1 if εₜ₋₁ < 0, else 0

    H0: β = 0 (no negative size bias)
    H1: β ≠ 0 (negative size bias exists)

    Args:
        std_residuals: Standardized residuals zₜ from GARCH model.

    Returns:
        Dict with coef, t_stat, p_value, n.
    """
    z, has_enough_data = _preprocess_residuals_for_regression(std_residuals)
    if not has_enough_data:
        return _create_nan_regression_result(len(z))

    # Dependent variable: z²ₜ
    z_squared = z[1:] ** 2

    # Independent variable: S⁻ₜ₋₁ · εₜ₋₁
    s_neg, _ = _compute_sign_indicators(z[:-1])
    neg_size_bias_var = s_neg * z[:-1]

    return _run_ols_regression(z_squared, neg_size_bias_var)


def engle_ng_positive_size_bias_test(
    std_residuals: np.ndarray,
) -> dict[str, float]:
    """Engle-Ng (1993) positive size bias test.

    Tests whether the magnitude of positive shocks affects volatility differently.

    Regression: z²ₜ = α + β·(S⁺ₜ₋₁ · εₜ₋₁) + εₜ
    where:
    - S⁺ₜ₋₁ = 1 if εₜ₋₁ ≥ 0, else 0

    H0: β = 0 (no positive size bias)
    H1: β ≠ 0 (positive size bias exists)

    Args:
        std_residuals: Standardized residuals zₜ from GARCH model.

    Returns:
        Dict with coef, t_stat, p_value, n.
    """
    z, has_enough_data = _preprocess_residuals_for_regression(std_residuals)
    if not has_enough_data:
        return _create_nan_regression_result(len(z))

    # Dependent variable: z²ₜ
    z_squared = z[1:] ** 2

    # Independent variable: S⁺ₜ₋₁ · εₜ₋₁
    _, s_pos = _compute_sign_indicators(z[:-1])
    pos_size_bias_var = s_pos * z[:-1]

    return _run_ols_regression(z_squared, pos_size_bias_var)


def engle_ng_joint_test(
    std_residuals: np.ndarray,
) -> dict[str, float]:
    """Engle-Ng (1993) joint test for all three bias effects.

    Tests all three effects simultaneously:
    z²ₜ = α + β₁·S⁻ₜ₋₁ + β₂·(S⁻ₜ₋₁·εₜ₋₁) + β₃·(S⁺ₜ₋₁·εₜ₋₁) + εₜ

    H0: β₁ = β₂ = β₃ = 0 (no asymmetry)
    H1: At least one βᵢ ≠ 0

    Uses F-test for joint significance.

    Args:
        std_residuals: Standardized residuals zₜ from GARCH model.

    Returns:
        Dict with f_stat, p_value, df_num, df_denom, n.
    """
    z = np.asarray(std_residuals, dtype=float)
    z = z[np.isfinite(z)]
    n = len(z)
    if n < 5:  # Need at least 5 obs for meaningful F-test
        return {
            "f_stat": float("nan"),
            "p_value": float("nan"),
            "df_num": 3.0,
            "df_denom": float("nan"),
            "n": n,
        }

    # Dependent variable: z²ₜ
    z_squared = z[1:] ** 2

    # Independent variables
    s_neg, s_pos = _compute_sign_indicators(z[:-1])
    X1 = s_neg  # Sign bias
    X2 = s_neg * z[:-1]  # Negative size bias
    X3 = s_pos * z[:-1]  # Positive size bias

    # Stack all regressors with intercept
    X_full = np.column_stack([np.ones(n - 1), X1, X2, X3])

    try:
        # Full model (with asymmetry terms)
        beta_full = np.linalg.lstsq(X_full, z_squared, rcond=None)[0]
        y_hat_full = X_full @ beta_full
        rss_full = np.sum((z_squared - y_hat_full) ** 2)

        # Restricted model (intercept only)
        X_restricted = np.ones((n - 1, 1))
        beta_restricted = np.linalg.lstsq(X_restricted, z_squared, rcond=None)[0]
        y_hat_restricted = X_restricted @ beta_restricted
        rss_restricted = np.sum((z_squared - y_hat_restricted) ** 2)

        # F-statistic
        df_num = 3  # Number of restrictions (3 coefficients)
        df_denom = n - 1 - 4  # n - k - 1, where k=4 (intercept + 3 regressors)

        if df_denom <= 0 or rss_full <= 0:
            return {
                "f_stat": float("nan"),
                "p_value": float("nan"),
                "df_num": float(df_num),
                "df_denom": float(df_denom),
                "n": n,
            }

        f_stat = ((rss_restricted - rss_full) / df_num) / (rss_full / df_denom)
        p_value = 1 - f.cdf(f_stat, df_num, df_denom)

        return {
            "f_stat": float(f_stat),
            "p_value": float(p_value),
            "df_num": float(df_num),
            "df_denom": float(df_denom),
            "n": n,
        }
    except Exception as exc:
        logger.warning("Joint F-test failed: %s", exc)
        return {
            "f_stat": float("nan"),
            "p_value": float("nan"),
            "df_num": 3.0,
            "df_denom": float("nan"),
            "n": n,
        }


def run_all_engle_ng_tests(
    std_residuals: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Run all four Engle-Ng (1993) tests for asymmetric volatility.

    Convenience function that runs:
    1. Sign bias test
    2. Negative size bias test
    3. Positive size bias test
    4. Joint test

    Args:
        std_residuals: Standardized residuals zₜ from GARCH model.

    Returns:
        Dict with keys: sign_bias, negative_size_bias, positive_size_bias, joint.
        Each value is a dict with test statistics and p-values.
    """
    return {
        "sign_bias": engle_ng_sign_bias_test(std_residuals),
        "negative_size_bias": engle_ng_negative_size_bias_test(std_residuals),
        "positive_size_bias": engle_ng_positive_size_bias_test(std_residuals),
        "joint": engle_ng_joint_test(std_residuals),
    }


# ============================================================================
# Plotting Utilities
# ============================================================================


def safe_import_matplotlib() -> tuple[Any | None, Any | None, bool]:
    """Import Matplotlib Agg primitives defensively.

    Returns:
        (Figure, FigureCanvas, available_flag)
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # type: ignore
        from matplotlib.figure import Figure  # type: ignore

        return Figure, FigureCanvas, True
    except Exception:
        return None, None, False


def resolve_out_path(out_path: Path | None) -> Path:
    """Resolve output path, defaulting to project constant.

    Args:
        out_path: Optional output path.

    Returns:
        Resolved output path.
    """
    if out_path is not None:
        return out_path
    return GARCH_STRUCTURE_PLOT


# ensure_output_dir moved to src.utils


def _create_figure_and_axes(*, Figure: Any, FigureCanvas: Any) -> tuple[Any, Any, Any, Any]:
    """Create figure and axes for diagnostics plot.

    Args:
        Figure: Matplotlib Figure class.
        FigureCanvas: Matplotlib FigureCanvas class.

    Returns:
        Tuple of (figure, canvas, ax1, ax2).
    """
    fig = Figure(figsize=_STRUCTURE_FIGURE_SIZE, constrained_layout=True)
    canvas = FigureCanvas(fig)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    return fig, canvas, ax1, ax2


def _plot_residuals_series(ax: Any, residuals: np.ndarray) -> None:
    """Plot residuals time series on given axis.

    Args:
        ax: Matplotlib axis object.
        residuals: Finite residual series.
    """
    ax.plot(
        residuals,
        color=_STRUCTURE_PLOT_COLOR_RESIDUALS,
        linewidth=_STRUCTURE_PLOT_LINEWIDTH,
    )
    ax.set_title("Residuals (test)")
    ax.set_xlabel("t")
    ax.set_ylabel("ε_t")


def _plot_acf_squared(ax: Any, acf_sq: np.ndarray, conf: float, acf_lags: int) -> None:
    """Plot ACF of squared residuals with confidence bands.

    Args:
        ax: Matplotlib axis object.
        acf_sq: ACF values for squared residuals.
        conf: Confidence level for significance bands.
        acf_lags: Maximum lag for ACF plot.
    """
    lags = np.arange(1, acf_lags + 1)
    ax.bar(
        lags,
        acf_sq,
        color=_STRUCTURE_PLOT_COLOR_ACF,
        width=_STRUCTURE_PLOT_BAR_WIDTH,
    )
    ax.axhline(
        0.0,
        color=_STRUCTURE_PLOT_COLOR_ZERO_LINE,
        linewidth=_STRUCTURE_PLOT_LINEWIDTH,
    )
    ax.axhline(
        conf,
        color=_STRUCTURE_PLOT_COLOR_CONFIDENCE,
        linestyle="--",
        linewidth=_STRUCTURE_PLOT_LINEWIDTH,
    )
    ax.axhline(
        -conf,
        color=_STRUCTURE_PLOT_COLOR_CONFIDENCE,
        linestyle="--",
        linewidth=_STRUCTURE_PLOT_LINEWIDTH,
    )
    ax.set_title("ACF of squared residuals")
    ax.set_xlabel("lag")
    ax.set_ylabel("acf(e_t^2)")


def render_with_matplotlib(
    *,
    Figure: Any,
    FigureCanvas: Any,
    x: np.ndarray,
    acf_sq: np.ndarray,
    conf: float,
    acf_lags: int,
    out_path: Path,
) -> None:
    """Render diagnostics plot using Matplotlib Agg backend.

    Args:
        Figure: Matplotlib Figure class.
        FigureCanvas: Matplotlib FigureCanvas class.
        x: Finite residual series.
        acf_sq: ACF of squared residuals.
        conf: Confidence level for significance bands.
        acf_lags: Maximum lag for ACF plot.
        out_path: Output file path.
    """
    fig, canvas, ax1, ax2 = _create_figure_and_axes(Figure=Figure, FigureCanvas=FigureCanvas)
    _plot_residuals_series(ax1, x)
    _plot_acf_squared(ax2, acf_sq, conf, acf_lags)
    fig.suptitle("ARCH/GARCH identification structure", fontsize=_STRUCTURE_PLOT_FONTSIZE_TITLE)
    canvas.print_png(str(out_path))


def verify_or_fallback(path: Path) -> None:
    """Ensure output exists and is non-empty; fallback to placeholder if needed.

    Args:
        path: Output file path.

    Raises:
        RuntimeError: If file cannot be created.
    """
    try:
        ok = path.exists() and path.stat().st_size > 0
    except Exception as exc:
        logger.warning("Failed to check output file status: %s", exc)
        ok = False
    if ok:
        return
    logger.warning(
        "Plot save did not create a file; creating a minimal placeholder at %s",
        path,
    )
    try:
        path.write_bytes(b"placeholder")
    except Exception as exc:
        msg = f"Failed to write diagnostics plot to {path}"
        raise RuntimeError(msg) from exc


__all__ = [
    # Data loading utilities
    "load_garch_dataset",
    "prepare_residuals",
    # ACF utilities
    "compute_acf",
    "compute_squared_acf",
    "prepare_plot_series",
    # ARCH-LM test utilities
    "compute_arch_lm_test",
    # Engle-Ng asymmetry tests
    "engle_ng_sign_bias_test",
    "engle_ng_negative_size_bias_test",
    "engle_ng_positive_size_bias_test",
    "engle_ng_joint_test",
    "run_all_engle_ng_tests",
    # Plotting utilities
    "safe_import_matplotlib",
    "resolve_out_path",
    "render_with_matplotlib",
    "verify_or_fallback",
]
