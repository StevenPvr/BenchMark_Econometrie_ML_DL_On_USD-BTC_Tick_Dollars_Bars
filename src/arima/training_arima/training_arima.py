"""ARIMA model training module."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.arima.models.arima_model import FittedARIMAModel, fit_arima_model
from src.path import (
    ARIMA_TRAINED_MODEL_FILE,
    ARIMA_TRAINED_MODEL_METADATA_FILE,
)
from src.utils import (
    ensure_output_dir,
    get_logger,
    load_json_data,
    save_json_pretty,
    validate_file_exists,
)
from src.constants import (
    ARIMA_DEFAULT_ORDER,
    ARIMA_DEFAULT_REFIT_EVERY,
    ARIMA_DEFAULT_TREND,
    ARIMA_EMPTY_TRAINING_SERIES_MSG,
)

from .utils import validate_arima_parameters

logger = get_logger(__name__)


"""
Note: ARIMA optimization has been deprecated. Training now fits a fixed
ARIMA(0,0,0) model which serves ONLY to generate white noise residuals for GARCH.

IMPORTANT: ARIMA(0,0,0) has NO forecasting capability - this is intentional.
The model simply computes residuals (log_returns - mean) which are then used
by GARCH to model conditional volatility. The refit cadence during evaluation
is ARIMA_DEFAULT_REFIT_EVERY (21 days).

Why ARIMA(0,0,0)?
- Simple baseline: no autoregressive or moving average components
- Direct residuals: residuals ≈ log_returns (after removing mean)
- GARCH focus: all temporal patterns are captured by GARCH, not ARIMA
"""


def train_arima_model(
    train_series: pd.Series,
    order: tuple[int, int, int],
) -> FittedARIMAModel:
    """
    Train an ARIMA model with specified order.

    Args:
        train_series: Training time series data
        order: ARIMA order (p, d, q)

    Returns:
        Fitted ARIMA model (SARIMAXResults - seasonal_order=(0,0,0,0))

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If model training fails
    """
    validate_arima_parameters(train_series, order)

    logger.info(f"Training ARIMA{order} model on {len(train_series)} observations")
    try:
        fitted_model = fit_arima_model(train_series, order=order, verbose=False)
        logger.info(f"Model trained successfully - AIC: {fitted_model.aic:.2f}")
        return fitted_model
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        msg = f"Failed to train ARIMA{order} model: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


def train_best_model(
    train_series: pd.Series,
) -> tuple[FittedARIMAModel, dict[str, Any]]:
    """
    Train the fixed ARIMA model with explicit defaults (no optimization).

    Args:
        train_series: Training time series data (must be training set only)

    Returns:
        Tuple of (fitted_model, model_info). model_info contains:
        - order: tuple[int, int, int]
        - params: dict with keys {p,d,q,trend,refit_every}

    Raises:
        ValueError: If training series is empty
        RuntimeError: If model training fails
    """
    if train_series.empty:
        raise ValueError(ARIMA_EMPTY_TRAINING_SERIES_MSG)

    order = ARIMA_DEFAULT_ORDER
    trend = ARIMA_DEFAULT_TREND
    refit_every = ARIMA_DEFAULT_REFIT_EVERY

    logger.info(f"Training fixed ARIMA{order} model (no optimization)")
    fitted_model = train_arima_model(train_series, order)

    model_info: dict[str, Any] = {
        "order": order,
        "params": {
            "p": int(order[0]),
            "d": int(order[1]),
            "q": int(order[2]),
            "trend": trend,
            "refit_every": int(refit_every),
        },
    }

    return fitted_model, model_info


def _save_model_file(fitted_model: FittedARIMAModel) -> None:
    """
    Save fitted model to disk.

    Args:
        fitted_model: Fitted ARIMA model (SARIMAXResults)

    Raises:
        RuntimeError: If saving fails
    """
    ensure_output_dir(ARIMA_TRAINED_MODEL_FILE)
    joblib.dump(fitted_model, ARIMA_TRAINED_MODEL_FILE)
    logger.info(f"Saved trained model: {ARIMA_TRAINED_MODEL_FILE}")


def _save_model_metadata(model_info: dict[str, Any]) -> None:
    """
    Save model metadata to disk.

    Delegates to src.utils.save_json_pretty() for consistency.

    Args:
        model_info: Dictionary with model information

    Raises:
        RuntimeError: If saving fails
    """
    save_json_pretty(model_info, ARIMA_TRAINED_MODEL_METADATA_FILE)
    logger.info(f"Saved model metadata: {ARIMA_TRAINED_MODEL_METADATA_FILE}")


def save_trained_model(fitted_model: FittedARIMAModel, model_info: dict[str, Any] | None) -> None:
    """
    Save trained ARIMA model to disk.

    Args:
        fitted_model: Fitted ARIMA model (SARIMAXResults)
        model_info: Dictionary with model information. Must not be None.

    Raises:
        ValueError: If fitted_model or model_info is None
        RuntimeError: If saving fails
    """
    if fitted_model is None:
        msg = "fitted_model cannot be None"
        raise ValueError(msg)

    if model_info is None:
        msg = "model_info cannot be None"
        raise ValueError(msg)

    try:
        _save_model_file(fitted_model)
        _save_model_metadata(model_info)
    except Exception as e:
        msg = f"Failed to save trained model: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


def load_trained_model() -> tuple[FittedARIMAModel, dict[str, Any]]:
    """
    Load trained ARIMA model from disk.

    Returns:
        Tuple of (fitted_model, model_info)

    Raises:
        FileNotFoundError: If model file or metadata file doesn't exist
        RuntimeError: If loading fails
    """
    validate_file_exists(ARIMA_TRAINED_MODEL_FILE, "Trained model file")
    validate_file_exists(ARIMA_TRAINED_MODEL_METADATA_FILE, "Model metadata file")

    try:
        fitted_model = joblib.load(ARIMA_TRAINED_MODEL_FILE)
        logger.info(f"Loaded trained model: {ARIMA_TRAINED_MODEL_FILE}")

        model_info = load_json_data(ARIMA_TRAINED_MODEL_METADATA_FILE)
        logger.info(f"Loaded model metadata: {ARIMA_TRAINED_MODEL_METADATA_FILE}")

        return fitted_model, model_info
    except Exception as e:
        msg = f"Failed to load trained model: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
