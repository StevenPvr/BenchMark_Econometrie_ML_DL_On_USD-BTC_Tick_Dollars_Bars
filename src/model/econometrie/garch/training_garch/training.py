"""Refactored GARCH training using new architecture.

This is the simplified version of training.py that uses:
- EGARCHForecaster for forecasts
- VarianceFilter for diagnostics
- RefitManager (via EGARCHForecaster)

All functions are ≤40 lines as per AGENTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib # type: ignore[import-untyped]
import numpy as np # type: ignore[import-untyped]
import pandas as pd # type: ignore[import-untyped]

from src.constants import (
    DEFAULT_RANDOM_STATE,
    GARCH_INITIAL_WINDOW_SIZE_DEFAULT,
    GARCH_MIN_WINDOW_SIZE,
)
from src.path import (
    GARCH_MODEL_FILE,
    GARCH_MODEL_METADATA_FILE,
    GARCH_VARIANCE_OUTPUTS_FILE,
)
from src.garch.garch_params.models import create_egarch_params_from_dict
from src.garch.training_garch.forecaster import EGARCHForecaster
from src.garch.training_garch.orchestration import (
    load_optimized_hyperparameters as load_hyperparameters,
)
from src.garch.training_garch.predictions_io import save_estimation_results
from src.garch.training_garch.utils import _compute_std_resid_diagnostics, _prepare_training_data
from src.garch.training_garch.variance_filter import VarianceFilter
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)


def create_egarch_forecaster(
    hyperparams: dict[str, Any],
    *,
    min_window_size: int | None = None,
    initial_window_size: int | None = None,
) -> EGARCHForecaster:
    """Create EGARCHForecaster from hyperparameters.

    Why: Tests construct the forecaster from a hyperparameter dict alone. We
    explicitly resolve `min_window_size` and `initial_window_size` from
    constants when not provided, avoiding hidden behavior while keeping the
    API simple for common cases. For rolling windows, an explicit
    `window_size` in `hyperparams` remains required to protect against silent
    misconfiguration.

    Args:
        hyperparams: Hyperparameter dictionary with keys: 'o', 'p',
            'distribution', 'window_type', 'refit_freq', and optional
            'window_size' for rolling windows.
        min_window_size: Optional explicit minimum window size to start
            forecasting; defaults to `GARCH_MIN_WINDOW_SIZE`.
        initial_window_size: Optional explicit initial training window size;
            defaults to `GARCH_INITIAL_WINDOW_SIZE_DEFAULT` for expanding
            windows, or to `window_size` when using a rolling window.

    Returns:
        Configured EGARCHForecaster.
    """
    o = int(hyperparams["o"])
    p = int(hyperparams["p"])
    dist = str(hyperparams["distribution"])
    window_type = str(hyperparams["window_type"])
    window_size = hyperparams.get("window_size")
    refit_freq = int(hyperparams["refit_freq"])

    # Resolve window sizes explicitly from constants if not provided by caller.
    resolved_min = GARCH_MIN_WINDOW_SIZE if min_window_size is None else int(min_window_size)
    if window_type == "rolling":
        # For rolling windows, initial size aligns with rolling window_size.
        resolved_initial = (
            int(window_size)
            if window_size is not None
            else (
                int(initial_window_size)
                if initial_window_size is not None
                else GARCH_INITIAL_WINDOW_SIZE_DEFAULT
            )
        )
    else:
        resolved_initial = (
            GARCH_INITIAL_WINDOW_SIZE_DEFAULT
            if initial_window_size is None
            else int(initial_window_size)
        )

    return EGARCHForecaster(
        o=o,
        p=p,
        dist=dist,
        refit_frequency=refit_freq,
        window_type=window_type,
        window_size=window_size,
        initial_window_size=resolved_initial,
        min_window_size=resolved_min,
    )


def compute_diagnostics_with_filter(
    forecaster: EGARCHForecaster,
    resid_train: np.ndarray,
    final_params_dict: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Compute diagnostic variance and standardized residuals.

    Returns:
        Tuple of (sigma2_filtered, z_standardized, diagnostics).
    """
    final_params_obj = create_egarch_params_from_dict(
        final_params_dict,
        o=forecaster.o,
        p=forecaster.p,
        dist=forecaster.dist,
    )

    variance_filter = VarianceFilter(final_params_obj)
    sigma2_filtered = variance_filter.filter_variance(resid_train)
    z_standardized = variance_filter.compute_standardized_residuals(resid_train)

    z_valid = z_standardized[np.isfinite(z_standardized)]
    diagnostics = _compute_std_resid_diagnostics(z_valid)

    return sigma2_filtered, z_standardized, diagnostics


def train_egarch_from_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Train EGARCH using EGARCHForecaster, VarianceFilter, and RefitManager.

    Args:
        df: DataFrame with date, split, and residual columns.

    Returns:
        Dictionary with training summary.
    """
    # Setup environment
    df_sorted, hyperparams = _setup_training_environment(df)
    df_train_sorted, resid_train, valid_mask_train, _ = _prepare_training_data(df_sorted)

    # Resolve window sizes
    resolved_min_window, resolved_initial_window = _resolve_training_windows(resid_train)

    # Perform training
    forecaster, result, final_params_dict = _perform_training(
        df_train_sorted, resid_train, hyperparams, resolved_min_window, resolved_initial_window
    )

    # Compute diagnostics
    sigma2_filtered, z_standardized, diagnostics = compute_diagnostics_with_filter(
        forecaster, resid_train, final_params_dict
    )

    # Save artifacts and create summary
    save_training_artifacts(
        forecaster,
        final_params_dict,
        hyperparams,
        sigma2_filtered,
        z_standardized,
        result,
        diagnostics,
        df_train_sorted,
        df_sorted,
        valid_mask_train,
    )

    return create_training_summary(
        forecaster=forecaster,
        final_params=final_params_dict,
        hyperparams=hyperparams,
        result=result,
        diagnostics=diagnostics,
        n_train=len(resid_train),
    )


def save_model_and_metadata(
    forecaster: EGARCHForecaster,
    final_params: dict[str, Any],
    hyperparams: dict[str, Any],
    result: Any,
    diagnostics: dict[str, float],
    n_train: int,
) -> None:
    """Save model file and metadata (≤40 lines)."""
    # Save model
    model_data = {
        "o": forecaster.o,
        "p": forecaster.p,
        "dist": forecaster.dist,
        "params": final_params,
        "hyperparams": hyperparams,
    }
    model_path = Path(GARCH_MODEL_FILE)
    # Ensure parent directory exists before writing to avoid FileNotFoundError.
    ensure_output_dir(model_path)
    joblib.dump(model_data, model_path)
    logger.info("Saved model to %s", model_path)

    # Save metadata
    metadata = {
        "o": forecaster.o,
        "p": forecaster.p,
        "distribution": forecaster.dist,
        "window_type": forecaster.window_type,
        "window_size": forecaster.window_size,
        "refit_frequency": forecaster.refit_frequency,
        "initial_window_size": forecaster.initial_window_size,
        "n_train": n_train,
        "n_refits": result.n_refits,
        "convergence_rate": float(result.convergence_rate),
        "diagnostics": diagnostics,
        "params": final_params,
    }
    metadata_path = Path(GARCH_MODEL_METADATA_FILE)
    # Ensure parent directory exists before writing metadata.
    ensure_output_dir(metadata_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", metadata_path)


def save_training_artifacts(
    forecaster: EGARCHForecaster,
    final_params: dict[str, Any],
    hyperparams: dict[str, Any],
    sigma2_filtered: np.ndarray,
    z_standardized: np.ndarray,
    result: Any,
    diagnostics: dict[str, float],
    df_train_sorted: pd.DataFrame,
    df_sorted: pd.DataFrame,
    valid_mask_train: np.ndarray,
) -> None:
    """Save training artifacts (≤40 lines by delegating)."""
    save_model_and_metadata(
        forecaster, final_params, hyperparams, result, diagnostics, len(df_train_sorted)
    )
    save_variance_outputs(df_sorted, valid_mask_train, sigma2_filtered, z_standardized)
    save_estimation_artifact(
        forecaster=forecaster,
        final_params=final_params,
        result=result,
        n_observations=int(np.count_nonzero(valid_mask_train)),
    )


def save_variance_outputs(
    df_sorted: pd.DataFrame,
    valid_mask_train: np.ndarray,
    sigma2_filtered: np.ndarray,
    z_standardized: np.ndarray,
) -> None:
    """Save variance outputs with data leakage warning."""
    n_total = len(df_sorted)
    sigma2_full = np.full(n_total, np.nan)
    z_full = np.full(n_total, np.nan)

    train_indices = df_sorted[df_sorted["split"] == "train"].index
    valid_train_indices = train_indices[valid_mask_train]

    sigma2_full[valid_train_indices] = sigma2_filtered
    z_full[valid_train_indices] = z_standardized

    outputs = df_sorted.copy()
    outputs["sigma2_garch"] = sigma2_full
    outputs["sigma_garch"] = np.sqrt(np.maximum(sigma2_full, 0))
    outputs["std_resid_garch"] = z_full

    variance_path = Path(GARCH_VARIANCE_OUTPUTS_FILE)
    # Ensure parent directory exists before writing variance outputs.
    ensure_output_dir(variance_path)
    outputs.to_csv(variance_path, index=False)
    logger.warning(
        "⚠️  SAVED FILTERED VARIANCE to %s - DO NOT USE AS ML FEATURES (data leakage)",
        variance_path,
    )


def save_estimation_artifact(
    *,
    forecaster: EGARCHForecaster,
    final_params: dict[str, Any],
    result: Any,
    n_observations: int,
) -> None:
    """Persist estimation.json for downstream diagnostics."""
    loglik = final_params.get("loglik")
    log_likelihood = float(loglik) if loglik is not None else float("nan")
    fits = {
        forecaster.dist: {
            "params": final_params,
            "converged": bool(final_params.get("converged", True)),
            "log_likelihood": log_likelihood,
            "iterations": int(getattr(result, "n_refits", 0)),
            "convergence_message": "Generated from final EGARCH training parameters",
        }
    }
    save_estimation_results(fits, n_observations=n_observations)


def _setup_training_environment(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Setup training environment and load hyperparameters."""
    np.random.seed(DEFAULT_RANDOM_STATE)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    hyperparams = load_hyperparameters()

    logger.info(
        "Training EGARCH with optimized hyperparameters: "
        "EGARCH(%d,%d), distribution=%s, refit_freq=%d, window_type=%s, window_size=%s",
        hyperparams["o"],
        hyperparams["p"],
        hyperparams["distribution"],
        hyperparams["refit_freq"],
        hyperparams["window_type"],
        hyperparams.get("window_size"),
    )

    return df_sorted, hyperparams


def _resolve_training_windows(
    resid_train: np.ndarray,
) -> tuple[int, int]:
    """Resolve training window sizes based on available data."""
    n_train = int(resid_train.size)
    if n_train <= 0:
        raise ValueError("No valid training residuals available for EGARCH training")

    # Resolve sizes explicitly: start as soon as feasible, but never exceed TRAIN length.
    # min_window_size cannot exceed n_train; initial window aligns with n_train for expanding.
    resolved_min_window = min(GARCH_MIN_WINDOW_SIZE, n_train)
    resolved_initial_window = min(GARCH_INITIAL_WINDOW_SIZE_DEFAULT, n_train)

    logger.info(
        "Resolved EGARCH windows from TRAIN size: n_train=%d, min_window_size=%d, "
        "initial_window_size=%d",
        n_train,
        resolved_min_window,
        resolved_initial_window,
    )

    return resolved_min_window, resolved_initial_window


def _perform_training(
    df_train_sorted: pd.DataFrame,
    resid_train: np.ndarray,
    hyperparams: dict[str, Any],
    resolved_min_window: int,
    resolved_initial_window: int,
) -> tuple[EGARCHForecaster, Any, dict[str, Any]]:
    """Perform the actual EGARCH training."""
    forecaster = create_egarch_forecaster(
        hyperparams,
        min_window_size=resolved_min_window,
        initial_window_size=resolved_initial_window,
    )
    dates_train = pd.DatetimeIndex(df_train_sorted["date"])
    result = forecaster.forecast_expanding(resid_train, dates=dates_train)

    # If no expanding steps occurred (e.g., very small TRAIN), use initial-fit params
    if result.params_history:
        final_params_dict = result.params_history[-1]
    else:
        logger.info("No expanding steps performed; using initial-fit parameters from RefitManager.")
        final_params_dict = forecaster.refit_manager.get_current_params()

    return forecaster, result, final_params_dict


def create_training_summary(
    *,
    forecaster: EGARCHForecaster,
    final_params: dict[str, Any],
    hyperparams: dict[str, Any],
    result: Any,
    diagnostics: dict[str, float],
    n_train: int,
) -> dict[str, Any]:
    """Create training summary dictionary (≤40 lines)."""
    return {
        "dist": forecaster.dist,
        "params": final_params,
        "hyperparams": hyperparams,
        "n_train": n_train,
        "window_type": forecaster.window_type,
        "window_size": forecaster.window_size,
        "refit_freq": forecaster.refit_frequency,
        "n_refits": result.n_refits,
        "convergence_rate": float(result.convergence_rate),
        "std_resid_diagnostics": diagnostics,
        "model_file": str(Path(GARCH_MODEL_FILE)),
        "metadata_file": str(Path(GARCH_MODEL_METADATA_FILE)),
        "outputs_file": str(Path(GARCH_VARIANCE_OUTPUTS_FILE)),
    }
