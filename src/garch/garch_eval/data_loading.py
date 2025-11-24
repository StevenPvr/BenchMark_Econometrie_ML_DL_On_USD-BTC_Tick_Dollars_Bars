"""Data loading utilities for GARCH evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.garch.garch_eval.models import choose_best_from_estimation
from src.garch.structure_garch.utils import prepare_residuals
from src.path import GARCH_DATASET_FILE, GARCH_ESTIMATION_FILE, GARCH_VARIANCE_OUTPUTS_FILE
from src.utils import get_logger, load_dataframe, load_json_data, validate_file_exists

logger = get_logger(__name__)


def _map_model_name_to_distribution(model_name: str) -> str:
    """Map model name to distribution type.

    Args:
    ----
        model_name: Model name string.

    Returns:
    -------
        Distribution type ('skewt', 'student').

    """
    if "skewt" in model_name:
        return "skewt"
    if "student" in model_name:
        return "student"
    return "student"


def load_model_params() -> tuple[
    dict[str, float],
    str,
    str,
    float | None,
    float | None,
    float | None,
]:
    """Load and extract model parameters from estimation file.

    Returns
    -------
        Tuple of (params_dict, model_name, dist, nu, gamma, lambda_skew).

    Raises:
    ------
        FileNotFoundError: If estimation file is missing.
        ValueError: If estimation file is invalid.

    """
    validate_file_exists(GARCH_ESTIMATION_FILE, "Estimation file")

    est = load_json_data(GARCH_ESTIMATION_FILE)
    best, name, nu, lambda_skew = choose_best_from_estimation(est)

    dist = _map_model_name_to_distribution(name)

    # Extract parameters
    omega = float(best["omega"])  # type: ignore[index]
    alpha = float(best["alpha"])  # type: ignore[index]
    beta = float(best["beta"])  # type: ignore[index]
    gamma_val = best.get("gamma")
    gamma = float(gamma_val) if gamma_val is not None else None

    params: dict[str, float] = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
    }
    return params, name, dist, nu, gamma, lambda_skew


def load_dataset_for_metrics() -> pd.DataFrame:
    """Load dataset for metrics computation, preferring variance outputs CSV.

    Returns
    -------
        Dataset DataFrame.

    Raises
    ------
        FileNotFoundError: If neither variance outputs file nor dataset file exists.

    """
    if GARCH_VARIANCE_OUTPUTS_FILE.exists():
        dataset_df = load_dataframe(
            GARCH_VARIANCE_OUTPUTS_FILE, date_columns=["date"], validate_not_empty=False
        )
        return dataset_df

    if not GARCH_DATASET_FILE.exists():
        msg = (
            f"Neither variance outputs file ({GARCH_VARIANCE_OUTPUTS_FILE}) "
            f"nor dataset file ({GARCH_DATASET_FILE}) exists."
        )
        raise FileNotFoundError(msg)

    dataset_df = load_dataframe(GARCH_DATASET_FILE, date_columns=["date"], validate_not_empty=False)
    return dataset_df


def load_and_prepare_residuals() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load dataset and prepare filtered residuals for train and all data.

    Returns
    -------
        Tuple of (dataframe, train_residuals, all_residuals).
        train_residuals: Only training residuals (for forecast initialization).
        all_residuals: All residuals (for variance path computation if needed).

    """
    data = load_dataframe(GARCH_DATASET_FILE, date_columns=["date"], validate_not_empty=False)

    # Get train residuals for forecast initialization (no data leakage)
    df_train = pd.DataFrame(data.loc[data["split"] == "train"].copy())
    resid_train = prepare_residuals(df_train, use_test_only=False)
    resid_train = resid_train[np.isfinite(resid_train)]

    # Get all residuals (for potential future use, but not for forecast init)
    resid_all = prepare_residuals(data, use_test_only=False)
    resid_all = resid_all[np.isfinite(resid_all)]

    return data, resid_train, resid_all


def prepare_residuals_from_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare and filter residuals from dataset, preserving index alignment.

    Args:
    ----
        dataset: Input dataset DataFrame.

    Returns:
    -------
        Tuple of (sorted_dataframe, all_residuals, valid_mask, filtered_residuals).
        Returns empty arrays if no valid residuals found.

    """
    df_sorted = dataset.sort_values("date").reset_index(drop=True)

    # Build residual series and preserve index alignment
    # Accept either arima_resid or sarima_resid
    from src.garch.structure_garch.utils import _find_residual_column

    try:
        col_name = _find_residual_column(df_sorted)
    except ValueError:
        msg = (
            "Required ARIMA residuals column ('arima_resid' or 'sarima_resid') "
            "not found in dataset."
        )
        raise ValueError(msg) from None
    series = pd.to_numeric(df_sorted[col_name], errors="coerce")
    resid = np.asarray(series, dtype=float)
    valid_mask = np.isfinite(resid)

    if not np.any(valid_mask):
        msg = (
            f"No valid residuals found in column '{col_name}'. "
            "Dataset must contain valid ARIMA residuals."
        )
        raise ValueError(msg)

    # Filtered contiguous residuals for variance recursion
    resid_f = resid[valid_mask]
    return df_sorted, resid, valid_mask, resid_f


def extract_aligned_test_indices(
    df_sorted: pd.DataFrame,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Extract aligned test indices from sorted dataset.

    Args:
    ----
        df_sorted: Sorted dataset DataFrame.
        valid_mask: Boolean mask for valid residuals.

    Returns:
    -------
        Array of positions in filtered arrays for test data, or empty array if none.

    """
    # Extract aligned test block using masks on the original index
    test_mask = (df_sorted["split"].astype(str) == "test").to_numpy()
    idx_all = np.arange(df_sorted.shape[0])
    idx_valid = idx_all[valid_mask]
    idx_test = idx_all[test_mask]
    idx_test_valid = np.intersect1d(idx_valid, idx_test, assume_unique=False)
    if idx_test_valid.size == 0:
        return np.array([], dtype=int)

    # Map original indices -> positions in filtered arrays
    pos_in_valid = -np.ones(df_sorted.shape[0], dtype=int)
    pos_in_valid[idx_valid] = np.arange(idx_valid.size)
    pos_test = pos_in_valid[idx_test_valid]
    # Keep order of time by sorting positions
    pos_test.sort()
    return pos_test


__all__ = [
    "load_model_params",
    "load_dataset_for_metrics",
    "load_and_prepare_residuals",
    "prepare_residuals_from_dataset",
    "extract_aligned_test_indices",
]
