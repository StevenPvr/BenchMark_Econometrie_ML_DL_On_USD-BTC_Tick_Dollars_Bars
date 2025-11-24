"""Variance filter for EGARCH models.

This module provides VarianceFilter class for computing filtered variance
(σ²_t|t) and standardized residuals from EGARCH parameters.

⚠️  WARNING: Filtered variance uses information up to time t, which causes
data leakage if used for forecasting. Use EGARCHForecaster for forecasts.
"""

from __future__ import annotations

import numpy as np

from src.garch.garch_params.core import compute_kappa, compute_variance_path
from src.garch.garch_params.models import EGARCHParams
from src.utils import get_logger

logger = get_logger(__name__)


class VarianceFilter:
    """Filter variance and compute standardized residuals from EGARCH parameters.

    ⚠️  WARNING: This class computes filtered variance (σ²_t|t) which uses
    information up to time t. This causes data leakage if used for forecasting.
    Use EGARCHForecaster for out-of-sample forecasts.

    Attributes:
        params: EGARCH parameters.
        kappa: E[|Z|] constant for distribution.
    """

    def __init__(self, params: EGARCHParams) -> None:
        """Initialize variance filter with EGARCH parameters.

        Args:
            params: EGARCH parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        self.params = params
        self.kappa = compute_kappa(
            dist=params.dist,
            nu=params.nu,
            lambda_skew=params.lambda_skew,
        )
        logger.warning(
            "⚠️  VarianceFilter computes filtered variance (data leakage). "
            "Use EGARCHForecaster for forecasts."
        )

    def filter_variance(self, residuals: np.ndarray) -> np.ndarray:
        """Compute filtered variance (σ²_t|t) from residuals.

        Filtered variance uses information up to time t, which causes
        data leakage if used for forecasting.

        Args:
            residuals: Residual series (ε₁, ε₂, ..., εₜ).

        Returns:
            Filtered variance path [σ²₁, σ²₂, ..., σ²ₜ].
        """
        residuals_arr = np.asarray(residuals, dtype=float).ravel()
        alpha, gamma, beta = self.params.extract_for_variance()

        sigma2 = compute_variance_path(
            residuals=residuals_arr,
            omega=self.params.omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            kappa=self.kappa,
            init=None,
            o=self.params.o,
            p=self.params.p,
        )

        return sigma2

    def compute_standardized_residuals(self, residuals: np.ndarray) -> np.ndarray:
        """Compute standardized residuals z_t = ε_t / σ_t.

        Args:
            residuals: Residual series (ε₁, ε₂, ..., εₜ).

        Returns:
            Standardized residuals [z₁, z₂, ..., zₜ].
        """
        residuals_arr = np.asarray(residuals, dtype=float).ravel()
        sigma2 = self.filter_variance(residuals_arr)
        sigma = np.sqrt(np.maximum(sigma2, 0))

        with np.errstate(divide="ignore", invalid="ignore"):
            z = residuals_arr / sigma
            z[~np.isfinite(z)] = np.nan

        return z
