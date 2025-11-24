"""ARIMA model creation and fitting functions."""

from __future__ import annotations

from typing import Any, cast
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils import get_logger

logger = get_logger(__name__)

# Type alias for fitted ARIMA model (SARIMAXResults)
FittedARIMAModel = Any  # statsmodels.tsa.statespace.sarimax.SARIMAXResults


def _extract_parameters_from_tuple(order: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Extract individual parameters from tuple format.

    Args:
        order: ARIMA order (p, d, q)

    Returns:
        Tuple of (p, d, q)
    """
    p_val, d_val, q_val = order
    return p_val, d_val, q_val


def _determine_arima_parameters(
    order: tuple[int, int, int] | None,
    p: int | None,
    d: int | None,
    q: int | None,
) -> tuple[int, int, int]:
    """
    Determine ARIMA parameters from either tuple or individual format.

    Args:
        order: ARIMA order (p, d, q) - alternative to p, d, q
        p: AR order
        d: Differencing order
        q: MA order

    Returns:
        Tuple of (p, d, q)

    Raises:
        ValueError: If parameters are invalid or missing
    """
    if order is not None:
        return _extract_parameters_from_tuple(order)
    if all(param is not None for param in [p, d, q]):
        # Type checker: we've verified all params are not None
        return cast(int, p), cast(int, d), cast(int, q)
    msg = "Must provide either (order) tuple or all individual parameters (p, d, q)"
    raise ValueError(msg)


def _create_and_fit_model(
    train_series: pd.Series,
    p_val: int,
    d_val: int,
    q_val: int,
    verbose: bool,
) -> FittedARIMAModel:
    """
    Create and fit ARIMA model with given parameters.

    Args:
        train_series: Training time series data
        p_val: AR order
        d_val: Differencing order
        q_val: MA order
        verbose: Deprecated parameter, kept for backward compatibility (not used).

    Returns:
        Fitted ARIMA model

    Raises:
        RuntimeError: If model fitting fails
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            model = SARIMAX(
                train_series,
                order=(p_val, d_val, q_val),
                seasonal_order=(0, 0, 0, 0),  # No seasonal components for ARIMA
            )
            # Use disp=False to suppress L-BFGS-B optimization output
            fitted_model = model.fit(disp=False)
            # Use getattr for type safety (aic may not be recognized by type checker)
            aic_value = getattr(fitted_model, "aic", None)
            if aic_value is not None:
                logger.debug(f"Model fitted successfully - AIC: {aic_value:.2f}")
            return fitted_model
        except Exception as e:
            msg = f"Failed to fit ARIMA({p_val},{d_val},{q_val}) model: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e


def fit_arima_model(
    train_series: pd.Series,
    order: tuple[int, int, int] | None = None,
    p: int | None = None,
    d: int | None = None,
    q: int | None = None,
    verbose: bool = False,
) -> FittedARIMAModel:
    """
    Fit an ARIMA model with given parameters.

    Can accept parameters either as tuple (order) or as individual parameters (p, d, q).

    Args:
        train_series: Training time series data
        order: ARIMA order (p, d, q) - alternative to p, d, q
        p: AR order
        d: Differencing order
        q: MA order
        verbose: Whether to show fitting progress

    Returns:
        Fitted ARIMA model (SARIMAXResults)

    Raises:
        ValueError: If parameters are invalid or missing
        RuntimeError: If model fitting fails
    """
    p_val, d_val, q_val = _determine_arima_parameters(order, p, d, q)

    logger.debug(f"Fitting ARIMA({p_val},{d_val},{q_val}) on {len(train_series)} observations")

    return _create_and_fit_model(train_series, p_val, d_val, q_val, verbose)
