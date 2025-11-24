"""Model evaluation for ARIMA (fit, diagnostics, walk-forward backtest)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils import get_logger

logger = get_logger(__name__)

# Suppress common statsmodels warnings during optimization
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.statespace.sarimax")
warnings.filterwarnings("ignore", message=".*Non-invertible starting MA parameters.*")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

# Type alias for fitted ARIMA model (SARIMAXResults)
FittedARIMAModel = Any  # statsmodels.tsa.statespace.sarimax.SARIMAXResults


@dataclass(frozen=True)
class ArimaParams:
    """ARIMA model parameters.

    Attributes:
        p: AR order (non-negative integer).
        d: Differencing order (non-negative integer).
        q: MA order (non-negative integer).
        trend: Trend component ("n", "c", "t", or "ct").
        refit_every: Frequency of model refitting during walk-forward backtest (must be >= 1).
    """

    p: int
    d: int
    q: int
    trend: str
    refit_every: int


def _configure_warnings_for_fitting() -> None:
    """Configure warnings to suppress during ARIMA model fitting."""
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    warnings.filterwarnings("ignore", message=".*Non-invertible starting MA parameters.*")
    warnings.filterwarnings(
        "ignore", message=".*Maximum Likelihood optimization failed to converge.*"
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def _create_arima_model(
    y: pd.Series,
    params: ArimaParams,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> SARIMAX:
    """Create ARIMA model with given parameters.

    Args:
        y: Time series data to fit.
        params: ARIMA parameters.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.

    Returns:
        SARIMAX model instance.
    """
    return SARIMAX(
        y,
        order=(params.p, params.d, params.q),
        seasonal_order=(0, 0, 0, 0),  # No seasonal components for ARIMA
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        trend=params.trend,
    )


def _fit_model_with_powell(model: SARIMAX, disp: bool) -> FittedARIMAModel:
    """Fit ARIMA model using Powell optimization method.

    Args:
        model: SARIMAX model to fit.
        disp: Whether to display optimization output.

    Returns:
        Fitted model results.

    Raises:
        RuntimeError: If Powell optimization fails.
    """
    try:
        return model.fit(method="powell", disp=disp, maxiter=200)
    except Exception as e:
        logger.error("Powell optimization method failed: %s", str(e))
        raise RuntimeError(
            "Powell optimization failed for ARIMA model. This ensures reproducibility "
            "but may require different parameter values."
        ) from e


def _check_model_convergence(res: FittedARIMAModel, params: ArimaParams) -> None:
    """Check if fitted model converged and log warning if not.

    Args:
        res: Fitted model results.
        params: ARIMA parameters used for fitting.
    """
    if hasattr(res, "mle_retvals"):
        converged = res.mle_retvals.get("converged", False)  # type: ignore
        if not converged:
            logger.warning(
                "ARIMA model (p=%d,d=%d,q=%d) did not converge. "
                "AIC/BIC values may be unreliable. Consider different parameter values.",
                params.p,
                params.d,
                params.q,
            )


def _fit_arima(
    y: pd.Series,
    params: ArimaParams,
    enforce_stationarity: bool = False,  # Disabled for stationary log-returns
    enforce_invertibility: bool = True,
    disp: bool = False,
) -> FittedARIMAModel:
    """Fit an ARIMA model with given parameters.

    Args:
        y: Time series data to fit.
        params: ARIMA parameters.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.
        disp: Whether to display optimization output.

    Returns:
        Fitted ARIMA model (SARIMAXResults object).
    """

    # Suppress warnings during model fitting
    with warnings.catch_warnings():
        _configure_warnings_for_fitting()

        model = _create_arima_model(y, params, enforce_stationarity, enforce_invertibility)
        res = _fit_model_with_powell(model, disp)

    # Check convergence after fitting
    _check_model_convergence(res, params)

    return res


def _validate_backtest_inputs(y: pd.Series, n_splits: int, test_size: int) -> int:
    """Validate backtest inputs and return training end index.

    Args:
        y: Time series data.
        n_splits: Number of splits for backtesting.
        test_size: Size of each test set.

    Returns:
        Training end index.

    Raises:
        ValueError: If inputs are invalid.
    """
    if len(y) == 0:
        raise ValueError("Time series is empty.")
    if n_splits <= 0:
        raise ValueError("n_splits must be positive.")
    if test_size <= 0:
        raise ValueError("test_size must be positive.")

    train_end = len(y) - n_splits * test_size
    if train_end <= 0:
        raise ValueError("Training segment is empty; reduce n_splits or test_size.")

    return train_end


def _prepare_time_series(y: pd.Series) -> pd.Series:
    """Prepare time series for backtesting.

    Args:
        y: Input time series.

    Returns:
        Prepared time series with RangeIndex.
    """
    y_prepared = y.copy()
    y_prepared.index = pd.RangeIndex(len(y_prepared))
    return y_prepared


def _predict_single_split(
    y: pd.Series,
    params: ArimaParams,
    test_start: int,
    test_end: int,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    refit_every: int,
) -> tuple[list[float], list[float]]:
    """Make predictions for a single backtest split.

    Args:
        y: Prepared time series data.
        params: ARIMA parameters.
        test_start: Start index of test split.
        test_end: End index of test split.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.
        refit_every: Number of observations between refits during walk-forward backtest.

    Returns:
        Tuple of (predictions, actuals) for this split.

    Raises:
        ValueError: If insufficient training data.
    """
    predictions: list[float] = []
    actuals: list[float] = []

    for offset in range(0, test_end - test_start, max(1, refit_every)):
        block_start = test_start + offset
        block_end = min(block_start + refit_every, test_end)

        y_train = y.iloc[:block_start]
        if len(y_train) <= 0:
            raise ValueError("Insufficient training data before first validation block.")

        # Fit model for this block
        fitted = _fit_arima(
            y_train,
            params,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )

        # Make predictions for this block
        predictions_block, actuals_block = _predict_block(y, fitted, block_start, block_end)
        predictions.extend(predictions_block)
        actuals.extend(actuals_block)

    return predictions, actuals


def _predict_block(
    y: pd.Series, fitted: Any, block_start: int, block_end: int
) -> tuple[list[float], list[float]]:
    """Make true walk-forward one-step-ahead predictions efficiently.

    Uses statsmodels' apply() method to compute all one-step-ahead forecasts
    for the block in a single vectorized call. This is much faster than
    calling forecast(1) + append() in a loop while maintaining true
    walk-forward behavior where each forecast uses only prior observations.

    Args:
        y: Time series data.
        fitted: Fitted ARIMA model.
        block_start: Start index of block.
        block_end: End index of block.

    Returns:
        Tuple of (predictions, actuals) for this block.
    """
    predictions: list[float] = []
    actuals: list[float] = []

    # Get the block of new observations to predict
    block_size = min(block_end - block_start, len(y) - block_start)
    if block_size <= 0:
        return predictions, actuals

    # Get new observations for this block (as array, not Series, to avoid index issues)
    new_obs = y.iloc[block_start : block_start + block_size].values

    # Apply the new observations to update filter state and get one-step-ahead forecasts
    # This is much faster than a loop of forecast(1) + append(refit=False)
    applied_results = fitted.apply(new_obs)

    # Extract one-step-ahead forecasts from the filter
    # forecasts is shape (1, n_obs) where each column is the forecast for that observation
    forecasts = applied_results.forecasts[0, :]

    # Store predictions and actuals
    for i in range(block_size):
        predictions.append(float(forecasts[i]))
        actuals.append(float(y.iloc[block_start + i]))

    return predictions, actuals


def _calculate_validation_metrics(
    predictions: list[float], actuals: list[float]
) -> Dict[str, float]:
    """Calculate validation metrics from predictions and actuals.

    Args:
        predictions: List of predicted values.
        actuals: List of actual values.

    Returns:
        Dictionary with validation metrics.
    """
    if len(predictions) == 0:
        return {"rmse": float("inf"), "mae": float("inf"), "mean_error": float("inf")}

    pred_array = np.array(predictions)
    actual_array = np.array(actuals)
    errors = actual_array - pred_array

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    mean_error = float(np.mean(errors))

    return {"rmse": rmse, "mae": mae, "mean_error": mean_error}


def walk_forward_backtest(
    y: pd.Series,
    params: ArimaParams,
    n_splits: int,
    test_size: int,
    enforce_stationarity: bool = False,  # Disabled for stationary log-returns
    enforce_invertibility: bool = True,
    refit_every: int | None = None,
) -> Dict[str, float]:
    """Perform strict time-aware walk-forward backtest with periodic refits.

    This function is used to test model robustness across different time periods
    during optimization. The validation metrics (RMSE, MAE) are stored for
    information but are NOT used in the optimization score (only AIC is used).

    Args:
        y: Full time series data (training data only - no test data leakage).
        params: ARIMA parameters to use (includes refit_every).
        n_splits: Number of time splits for backtesting.
        test_size: Size of each test set.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.
        refit_every: Optional override for refit frequency. Defaults to params.refit_every.

    Returns:
        Dictionary with validation metrics: rmse, mae, mean_error.

    Raises:
        ValueError: If training segment is empty or insufficient data.
    """
    # Validate inputs and prepare data
    train_end = _validate_backtest_inputs(y, n_splits, test_size)
    effective_refit_every = params.refit_every if refit_every is None else int(refit_every)
    if effective_refit_every < 1:
        raise ValueError("refit_every must be >= 1 for walk-forward backtest.")
    y_prepared = _prepare_time_series(y)

    # Collect all predictions and actuals across splits
    all_predictions: list[float] = []
    all_actuals: list[float] = []

    for split in range(n_splits):
        test_start = train_end + split * test_size
        test_end = test_start + test_size

        split_predictions, split_actuals = _predict_single_split(
            y_prepared,
            params,
            test_start,
            test_end,
            enforce_stationarity,
            enforce_invertibility,
            effective_refit_every,
        )
        all_predictions.extend(split_predictions)
        all_actuals.extend(split_actuals)

    # Calculate and return validation metrics
    return _calculate_validation_metrics(all_predictions, all_actuals)
