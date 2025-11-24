"""Utility functions for GARCH training.

This module contains helper functions used by the training module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd # type: ignore[import-untyped]

from src.path import GARCH_OPTIMIZATION_RESULTS_FILE
from src.utils import get_logger

logger = get_logger(__name__)


def _empty_diagnostics() -> dict[str, float | int]:
    """Return empty diagnostics dictionary.

    Returns:
        Dictionary with empty diagnostics.
    """
    return {
        "n": 0,
        "mean": float("nan"),
        "var": float("nan"),
        "std": float("nan"),
        "abs_gt_2": 0,
        "abs_gt_3": 0,
    }


def _compute_std_resid_diagnostics(  # noqa: Used in training.py
    z: np.ndarray,
    out: dict[str, float | int] | None = None,
) -> dict[str, float | int]:
    """Compute basic diagnostics on standardized residuals.

    Returns mean, variance, std, and tail counts for |z|>2 and |z|>3.

    Args:
        z: Standardized residuals array.
        out: Optional output dictionary to update.

    Returns:
        Dictionary with diagnostics metrics.
    """
    if out is None:
        out = {}

    if z.size == 0:
        empty = _empty_diagnostics()
        out.update(empty)
        return out

    zf = np.asarray(z, dtype=float)
    zf = zf[np.isfinite(zf)]
    n = int(zf.size)
    if n == 0:
        empty = _empty_diagnostics()
        out.update(empty)
        return out

    mean = float(np.mean(zf))
    var = float(np.var(zf))
    std = float(np.sqrt(var))
    abs_gt_2 = int(np.sum(np.abs(zf) > 2.0))
    abs_gt_3 = int(np.sum(np.abs(zf) > 3.0))

    out.update(
        {
            "n": n,
            "mean": mean,
            "var": var,
            "std": std,
            "abs_gt_2": abs_gt_2,
            "abs_gt_3": abs_gt_3,
        }
    )
    return out


def _extract_direct_residuals(df_train: pd.DataFrame) -> np.ndarray:
    """Extract direct residuals from DataFrame.

    Tries multiple methods:
    1. Direct columns: arima_resid or sarima_resid
    2. Fallback: compute from weighted_log_return - arima_fitted_in_sample

    Args:
        df_train: Training DataFrame.

    Returns:
        Residuals array.

    Raises:
        ValueError: If no residual column is present and cannot compute from fitted values.

    """
    # Try direct residual columns first
    col_name = None
    if "arima_resid" in df_train.columns:
        col_name = "arima_resid"
    elif "sarima_resid" in df_train.columns:
        col_name = "sarima_resid"

    if col_name is not None:
        series_train = pd.to_numeric(df_train[col_name], errors="coerce")
        resid = np.asarray(series_train, dtype=float)
        if np.any(np.isfinite(resid)):
            return resid

    # Fallback: compute from fitted values
    from src.model.econometrie.garch.structure_garch.utils import _compute_residuals_from_fitted # type: ignore[import-untyped]

    computed_residuals = _compute_residuals_from_fitted(df_train)
    if computed_residuals is not None and np.any(np.isfinite(computed_residuals)):
        logger.info("Using computed residuals from weighted_log_return - arima_fitted_in_sample")
        return computed_residuals

    msg = (
        "Cannot extract residuals: neither 'arima_resid'/'sarima_resid' columns exist "
        "with valid values, nor can compute from 'weighted_log_return' - 'arima_fitted_in_sample'."
    )
    raise ValueError(msg)


def _extract_train_residuals_with_mask(
    df_train: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract train residuals and valid mask.

    Args:
        df_train: Training DataFrame sorted by date.

    Returns:
        Tuple of (residuals_array, valid_mask_array).

    Raises:
        ValueError: If 'arima_resid' column is missing or contains no valid values.

    """
    resid_train_full = _extract_direct_residuals(df_train)
    valid_mask_train = np.isfinite(resid_train_full)
    if not np.any(valid_mask_train):
        msg = "Column 'arima_resid' contains no valid training residuals."
        raise ValueError(msg)

    return resid_train_full, valid_mask_train


def _prepare_training_data(  # noqa: Used in training.py
    df_sorted: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare training data and extract residuals.

    Args:
        df_sorted: Full DataFrame sorted by date.

    Returns:
        Tuple of (df_train_sorted, resid_train, valid_mask_train, resid_train_full).

    Raises:
        ValueError: If no training data found or residuals cannot be extracted.
    """
    df_train = cast(pd.DataFrame, df_sorted.loc[df_sorted["split"] == "train"].copy())
    if df_train.empty:
        msg = "No training data found in dataset"
        raise ValueError(msg)

    df_train_sorted = df_train.sort_values("date").reset_index(drop=True)
    resid_train_full, valid_mask_train = _extract_train_residuals_with_mask(df_train_sorted)
    resid_train = resid_train_full[valid_mask_train]

    return df_train_sorted, resid_train, valid_mask_train, resid_train_full


def validate_required_columns(
    df: pd.DataFrame, required_columns: set[str], context: str = "DataFrame"
) -> None:
    """Validate that DataFrame has all required columns.

    Args:
        df: DataFrame to validate.
        required_columns: Set of required column names.
        context: Context description for error message.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = required_columns - set(df.columns)
    if missing:
        msg = f"{context} missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def count_splits(df: pd.DataFrame) -> tuple[int, int]:
    """Count number of train and test observations in DataFrame.

    Args:
        df: DataFrame with 'split' column.

    Returns:
        Tuple of (n_train, n_test) counts.
    """
    if "split" not in df.columns:
        return 0, 0

    n_train = int((df["split"] == "train").sum())
    n_test = int((df["split"] == "test").sum())
    return n_train, n_test


def load_optimized_hyperparameters() -> dict[str, Any]:
    """Load optimized GARCH hyperparameters.

    Returns:
        Dictionary with optimized hyperparameters.

    Raises:
        FileNotFoundError: If optimization results not found.
        ValueError: If results are invalid.
    """
    # Prefer the module-level path to allow direct monkeypatching in tests.
    # Do not re-fetch from constants here, as it would override the test patch.
    opt_results_path = Path(GARCH_OPTIMIZATION_RESULTS_FILE)
    if not opt_results_path.exists():
        # Strict: no implicit fallback. Optimization must be run first.
        msg = f"Optimization results not found: {opt_results_path}. Run garch optimization first."
        raise FileNotFoundError(msg)

    with open(opt_results_path) as f:
        results = json.load(f)

    if "best_params" not in results:
        msg = "No 'best_params' in optimization results"
        raise ValueError(msg)

    best_params = results["best_params"]
    logger.info(
        "Loaded optimized hyperparameters from %s: "
        "o=%d, p=%d, distribution=%s, refit_freq=%d, window_type=%s, window_size=%s",
        opt_results_path,
        best_params.get("o"),
        best_params.get("p"),
        best_params.get("distribution"),
        best_params.get("refit_freq"),
        best_params.get("window_type"),
        best_params.get("window_size"),
    )

    return best_params
