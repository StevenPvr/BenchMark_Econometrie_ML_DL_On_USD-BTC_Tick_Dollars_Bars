"""Metrics and statistical utilities for the project.

Provides functions for computing statistical metrics, residuals, and log returns.
Used across all model evaluation pipelines (ARIMA, GARCH, LightGBM).
"""

from __future__ import annotations

from typing import Iterable, cast

import numpy as np
import pandas as pd

__all__ = [
    "chi2_sf",
    "compute_log_returns",
    "compute_residuals",
]


def chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function P[X >= x].

    Uses SciPy when available; returns NaN if SciPy is unavailable.
    This function is used for computing p-values in statistical tests.

    Args:
        x: Test statistic value.
        df: Degrees of freedom.

    Returns:
        P-value (survival function value), or NaN if SciPy unavailable.
    """
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(x, df))
    except Exception:
        return float("nan")


def compute_log_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    *,
    group_by: str | None = "ticker",
    output_col: str = "log_return",
    remove_first: bool = True,
) -> pd.DataFrame:
    """Compute log returns with automatic grouping and NaN handling.

    Calculates log(price_t / price_{t-1}) for each group. Automatically handles
    grouping, NaN removal, and invalid values (inf, negative prices).

    Args:
        df: DataFrame with price data (must be sorted by date within each group).
        price_col: Column containing prices.
        group_by: Column to group by (e.g., "ticker"). None for no grouping.
        output_col: Name for output log return column.
        remove_first: If True, remove first row per group (NaN returns).

    Returns:
        DataFrame with log_return column added.

    Examples:
        Compute returns per ticker:
        >>> df = compute_log_returns(df, price_col="close", group_by="ticker")

        Compute returns for single series:
        >>> df = compute_log_returns(df, group_by=None)

    Usage in project:
        - Replaces data_preparation/computations.py:25-49
        - Pattern repeated 5+ times across the project
    """
    df = df.copy()

    def _safe_log_return(x: pd.Series) -> pd.Series:
        x = x.astype(float)
        prev = x.shift(1)
        ratio = x / prev
        # Clean problematic ratios before log
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        ratio = ratio.mask(ratio <= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return cast(pd.Series, np.log(ratio))

    if group_by is not None:
        df[output_col] = df.groupby(group_by)[price_col].transform(_safe_log_return)
    else:
        df[output_col] = _safe_log_return(df[price_col])

    if remove_first:
        df = df.dropna(subset=[output_col]).reset_index(drop=True)

    return df


def compute_residuals(
    y_true: np.ndarray | pd.Series | Iterable[float],
    y_pred: np.ndarray | pd.Series | Iterable[float],
) -> np.ndarray:
    """Return residuals y_true - y_pred as numpy array.

    Computes residuals for model evaluation. Works with both numpy arrays and pandas Series.
    Used across all forecasting models (ARIMA, GARCH, LightGBM).

    Args:
        y_true: Actual values (array or Series).
        y_pred: Predicted values (array or Series).

    Returns:
        Residuals as numpy array (y_true - y_pred).

    Examples:
        Basic residual computation:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 1.9, 3.2])
        >>> residuals = compute_residuals(y_true, y_pred)
        >>> residuals
        array([-0.1,  0.1, -0.2])

        Works with Series:
        >>> y_true_s = pd.Series([1.0, 2.0, 3.0])
        >>> y_pred_s = pd.Series([1.1, 1.9, 3.2])
        >>> compute_residuals(y_true_s, y_pred_s)
        array([-0.1,  0.1, -0.2])

    Usage in project:
        - Replaces src/arima/evaluation_arima/evaluation_arima.py:compute_residuals
        - Used in all model evaluation modules for residual analysis
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return yt - yp
