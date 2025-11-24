"""EGARCH forecaster with leak-free variance forecasts.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.constants import (
    DEFAULT_PLACEHOLDER_DATE,
    GARCH_FIT_MIN_SIZE,
    GARCH_MIN_WINDOW_SIZE,
)
from src.garch.garch_params.core import egarch_variance
from src.garch.garch_params.models import EGARCHParams, create_egarch_params_from_dict
from src.garch.garch_params.refit.refit_manager import RefitManager
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    """Result of variance forecasting.

    Attributes:
        dates: Date index for forecasts.
        forecasts: One-step-ahead variance forecasts σ²_t+1|t.
        volatility: Square root of forecasts (volatility).
        params_history: Parameter values used for each forecast.
        refit_mask: Boolean array indicating where refits occurred.
        n_refits: Total number of refits performed.
        convergence_rate: Proportion of successful refits.
    """

    dates: pd.DatetimeIndex
    forecasts: np.ndarray
    volatility: np.ndarray
    params_history: list[dict[str, float]]
    refit_mask: np.ndarray
    n_refits: int
    convergence_rate: float


class EGARCHForecaster:
    """EGARCH forecaster with periodic refit and anti-leakage guarantees.

    This class generates one-step-ahead variance forecasts suitable for use
    as features in machine learning models.

    Key principles:
    1. Forecast at position t uses only data up to position t-1 (exclusive)
    2. RefitManager handles parameter re-estimation
    3. Explicit anti-leakage assertions throughout
    4. Convergence tracking and reporting
    """

    def _validate_and_normalize_params(
        self,
        o: int,
        p: int,
        dist: str,
        refit_frequency: int,
        window_type: str,
        window_size: int | None,
        initial_window_size: int,
        min_window_size: int,
    ) -> tuple[int, int, str, int, str, int | None, int, int]:
        """Validate and normalize initialization parameters."""
        if o not in (1, 2):
            msg = f"ARCH order o={o} not supported (only 1 or 2)"
            raise ValueError(msg)
        if p not in (1, 2):
            msg = f"GARCH order p={p} not supported (only 1 or 2)"
            raise ValueError(msg)
        if window_type not in ("expanding", "rolling"):
            msg = f"window_type={window_type} not supported (only 'expanding' or 'rolling')"
            raise ValueError(msg)
        if window_type == "rolling" and window_size is None:
            msg = "window_size required for rolling window"
            raise ValueError(msg)
        if min_window_size < GARCH_FIT_MIN_SIZE:
            msg = f"min_window_size={min_window_size} too small " f"(min={GARCH_FIT_MIN_SIZE})"
            raise ValueError(msg)
        if min_window_size > initial_window_size:
            # Align minimum window with initial window for small-test scenarios.
            logger.info(
                "Adjusting min_window_size (%d) to initial_window_size (%d)",
                min_window_size,
                initial_window_size,
            )
            min_window_size = initial_window_size
        if initial_window_size < GARCH_FIT_MIN_SIZE:
            msg = (
                f"initial_window_size={initial_window_size} too small "
                f"(min={GARCH_FIT_MIN_SIZE})"
            )
            raise ValueError(msg)

        return (
            o,
            p,
            dist,
            refit_frequency,
            window_type,
            window_size,
            initial_window_size,
            min_window_size,
        )

    def __init__(
        self,
        *,
        o: int,
        p: int,
        dist: str,
        refit_frequency: int,
        window_type: str,
        window_size: int | None = None,
        initial_window_size: int,
        min_window_size: int = GARCH_MIN_WINDOW_SIZE,
        use_fixed_params: bool = False,
        anchor_at_min_window: bool = False,
    ) -> None:
        """Initialize EGARCH forecaster.

        Args:
            o: ARCH order (1 or 2).
            p: GARCH order (1 or 2).
            dist: Innovation distribution ('student', 'skewt').
            refit_frequency: Refit every N observations.
            window_type: Window type ('expanding' or 'rolling').
            window_size: Rolling window size (required if window_type='rolling').
            initial_window_size: Initial training window size (target size for rolling).
            min_window_size: Minimum window size to start forecasting.
            use_fixed_params: Use fixed pre-trained parameters (no refitting).
            anchor_at_min_window: If True, anchor initial fit and refit schedule
                at ``min_window_size`` instead of ``initial_window_size`` or
                ``window_size``. This is intended for evaluation/feature
                generation to start forecasts earlier (e.g., at 50 obs), then
                naturally transition to the configured windowing as history
                grows. Defaults to False to preserve training behavior.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Validate and normalize parameters
        (
            o,
            p,
            dist,
            refit_frequency,
            window_type,
            window_size,
            initial_window_size,
            min_window_size,
        ) = self._validate_and_normalize_params(
            o,
            p,
            dist,
            refit_frequency,
            window_type,
            window_size,
            initial_window_size,
            min_window_size,
        )

        self.o = o
        self.p = p
        self.dist = dist
        self.refit_frequency = refit_frequency
        self.window_type = window_type
        self.window_size = window_size
        self.initial_window_size = initial_window_size
        self.min_window_size = min_window_size
        self.anchor_at_min_window = anchor_at_min_window
        self.seed_with_current_params = False

        # Create refit manager
        self.refit_manager = RefitManager(
            frequency=refit_frequency,
            window_type=window_type,
            window_size=window_size,
            o=o,
            p=p,
            dist=dist,
            use_fixed_params=use_fixed_params,
        )

    def _validate_expanding_residuals(self, residuals: np.ndarray) -> int:
        """Validate residuals for expanding window forecasting.

        Args:
            residuals: Residual array to validate.

        Returns:
            Number of residuals.

        Raises:
            ValueError: If residuals are insufficient.
        """
        n = residuals.size
        if n < self.min_window_size:
            msg = f"Insufficient residuals: {n} < {self.min_window_size} " f"(min_window_size)"
            raise ValueError(msg)
        return n

    def _initialize_expanding_arrays(
        self, n: int
    ) -> tuple[np.ndarray, list[dict[str, float]], np.ndarray]:
        """Initialize arrays for expanding window forecasting.

        Args:
            n: Number of observations.

        Returns:
            Tuple of (forecasts, params_history, refit_mask).
        """
        forecasts = np.full(n, np.nan, dtype=float)
        params_history: list[dict[str, float]] = []
        refit_mask = np.zeros(n, dtype=bool)
        return forecasts, params_history, refit_mask

    def _perform_initial_model_fit(self, residuals: np.ndarray) -> EGARCHParams:
        """Perform initial EGARCH model fit.

        Args:
            residuals: Full residual series.

        Returns:
            Initial EGARCH parameters.
        """
        # Check for fixed pre-trained parameters
        fixed_params = self._get_fixed_params_if_available()
        if fixed_params is not None:
            return fixed_params

        # Check for hybrid-mode seeding
        hybrid_params = self._get_hybrid_params_if_available()
        if hybrid_params is not None:
            return hybrid_params

        # Perform standard MLE fit
        return self._perform_standard_initial_fit(residuals)

    def _get_fixed_params_if_available(self) -> EGARCHParams | None:
        """Get fixed pre-trained parameters if available.

        Returns:
            Fixed parameters or None if not available.
        """
        if self.refit_manager.use_fixed_params and self.refit_manager.current_params is not None:
            logger.info("Using pre-trained parameters for initial fit")
            return create_egarch_params_from_dict(
                self.refit_manager.current_params, o=self.o, p=self.p, dist=self.dist
            )
        return None

    def _get_hybrid_params_if_available(self) -> EGARCHParams | None:
        """Get hybrid-mode seeding parameters if available.

        Returns:
            Hybrid parameters or None if not available.
        """
        if (
            not self.refit_manager.use_fixed_params
            and self.seed_with_current_params
            and self.refit_manager.current_params is not None
        ):
            logger.info("Seeding initial state with pre-trained parameters (hybrid mode)")
            return create_egarch_params_from_dict(
                self.refit_manager.current_params, o=self.o, p=self.p, dist=self.dist
            )
        return None

    def _perform_standard_initial_fit(self, residuals: np.ndarray) -> EGARCHParams:
        """Perform standard MLE initial fit.

        Args:
            residuals: Full residual series.

        Returns:
            Fitted EGARCH parameters.
        """
        initial_pos = self._calculate_initial_fit_position()
        logger.info("Performing initial fit at position %d", initial_pos)

        params, convergence = self.refit_manager.perform_refit(residuals, initial_pos)

        if not convergence.converged:
            msg = (
                "Initial EGARCH MLE did not converge at position "
                f"{initial_pos}: {convergence.message or 'unknown reason'}"
            )
            raise RuntimeError(msg)

        return create_egarch_params_from_dict(params, o=self.o, p=self.p, dist=self.dist)

    def _calculate_initial_fit_position(self) -> int:
        """Calculate the initial position for model fitting.

        Returns:
            Initial fit position.
        """
        # In evaluation mode (anchor_at_min_window=True), start earlier at min_window_size
        # to avoid discarding many early rows downstream.
        if self.anchor_at_min_window:
            return max(self.min_window_size, GARCH_FIT_MIN_SIZE)

        # Training-default behavior: anchor at initial_window_size
        # When rolling, ensure we are at least at window_size.
        if self.window_type == "rolling" and self.window_size is not None:
            return max(self.window_size, self.initial_window_size)

        return max(self.min_window_size, self.initial_window_size)

    def _run_expanding_forecast_loop(
        self,
        residuals: np.ndarray,
        n: int,
        current_params: EGARCHParams,
        forecasts: np.ndarray,
        params_history: list[dict[str, float]],
        refit_mask: np.ndarray,
        start_pos: int,
    ) -> EGARCHParams:
        """Run main expanding window forecast loop.

        Args:
            residuals: Full residual series.
            n: Number of observations.
            current_params: Initial EGARCH parameters.
            forecasts: Array to store forecasts (modified in place).
            params_history: List to store parameter history (modified in place).
            refit_mask: Array to store refit indicators (modified in place).

        Returns:
            Final EGARCH parameters.
        """
        for t in range(start_pos, n):
            # ANTI-LEAKAGE: Refit uses ONLY residuals[:t] (exclusive of t)
            # This ensures that refitting at position t does not use information from t
            if self.refit_manager.should_refit(t, start_pos):
                logger.debug("Refit at position %d", t)
                # ANTI-LEAKAGE ASSERTION: Refit window excludes position t
                params, convergence = self.refit_manager.perform_refit(residuals, t)

                if convergence.converged:
                    current_params = create_egarch_params_from_dict(
                        params, o=self.o, p=self.p, dist=self.dist
                    )
                else:
                    logger.warning(
                        "Refit at position %d did not converge: %s",
                        t,
                        convergence.message or "unknown reason",
                    )

                refit_mask[t] = True

            # ANTI-LEAKAGE: Forecast σ²_{t+1}|t using residuals up to (not including) t
            # This ensures that forecast at position t does not use information from t
            assert t < n, f"Position {t} exceeds residuals length {n}"
            # ANTI-LEAKAGE ASSERTION: We use residuals[:t], not residuals[:t+1]
            assert len(residuals[:t]) < len(
                residuals
            ), f"Forecast at {t} must use fewer residuals than total length {len(residuals)}"

            forecast_t = self._compute_one_step_forecast(
                residuals[:t],  # ANTI-LEAKAGE: Use residuals UP TO (not including) t
                current_params,
            )

            # CRITICAL TEMPORAL INDEXING CONVENTION:
            # forecasts[t] stores σ²_{t+1|t} - the forecast FOR position t+1 made AT position t
            # This means: forecasts[t] predicts variance at t+1 using only data up to t-1
            # When aligning with LightGBM features:
            #   - At row with date D_t, use forecasts[t-1] which contains σ²_{t|t-1}
            #   - This ensures the feature predicts variance AT D_t using data BEFORE D_t
            forecasts[t] = forecast_t
            params_history.append(current_params.to_dict())

        return current_params

    def _create_expanding_result(
        self,
        n: int,
        forecasts: np.ndarray,
        params_history: list[dict[str, float]],
        refit_mask: np.ndarray,
        dates: pd.DatetimeIndex | None,
    ) -> ForecastResult:
        """Create ForecastResult from expanding window forecasting.

        Args:
            n: Number of observations.
            forecasts: Forecast array.
            params_history: Parameter history.
            refit_mask: Refit indicator array.
            dates: Optional date index.

        Returns:
            ForecastResult object.
        """
        convergence_rate = self.refit_manager.get_convergence_rate()

        logger.info(
            "Expanding forecasts complete: generated %d forecasts, %d refits, "
            "convergence rate=%.1f%%",
            n - self.min_window_size,
            self.refit_manager.get_summary()["n_refits"],
            convergence_rate * 100,
        )

        # Create date index if not provided
        if dates is None:
            dates = pd.date_range(start=DEFAULT_PLACEHOLDER_DATE, periods=n, freq="D")

        return ForecastResult(
            dates=dates,
            forecasts=forecasts,
            volatility=np.sqrt(np.maximum(forecasts, 0.0)),
            params_history=params_history,
            refit_mask=refit_mask,
            n_refits=self.refit_manager.get_summary()["n_refits"],
            convergence_rate=convergence_rate,
        )

    def forecast_expanding(
        self,
        residuals: np.ndarray,
        dates: pd.DatetimeIndex | None = None,
    ) -> ForecastResult:
        """Generate one-step-ahead forecasts on expanding window.

        This method generates forecasts for positions [min_window_size, len(residuals)].
        Forecast at position t uses residuals[:t] (excluding position t).

        ANTI-LEAKAGE GUARANTEES:
        - Forecast at position t uses ONLY residuals[:t] (exclusive of t)
        - Refit at position t uses ONLY residuals[:t] (exclusive of t)
        - No future information leaks into forecasts
        - Suitable for use as features in machine learning models

        Args:
            residuals: Full residual series (e.g., ARIMA residuals on TRAIN).
            dates: Optional date index. If None, integer index used.

        Returns:
            ForecastResult with one-step-ahead variance forecasts.

        Raises:
            ValueError: If residuals are insufficient or invalid.

        Example:
            >>> forecaster = EGARCHForecaster(refit_frequency=20, window_type='expanding',
            ...                               min_window_size=250, initial_window_size=1000)
            >>> residuals_train = load_arima_residuals()  # e.g., 2000 obs
            >>> result = forecaster.forecast_expanding(residuals_train)
            >>> # result.forecasts[i] = σ²_{i+1}|i using residuals[:i] for i >= 250
            >>> # Safe to use as features for predicting at position i
        """
        # Set total data length for anti-leakage checks
        self._total_data_length = len(residuals)

        # ANTI-LEAKAGE: Validate residuals before processing
        n = self._validate_expanding_residuals(residuals)

        # ANTI-LEAKAGE ASSERTION: Ensure we have enough residuals for min window
        assert (
            n >= self.min_window_size
        ), f"Insufficient residuals: {n} < {self.min_window_size} (min_window_size)"

        logger.info(
            "Starting forecasts with %s window: n=%d, o=%d, p=%d, dist=%s, "
            "refit_freq=%d, min_window=%d, initial_window=%d, window_size=%s",
            self.window_type,
            n,
            self.o,
            self.p,
            self.dist,
            self.refit_frequency,
            self.min_window_size,
            self.initial_window_size,
            self.window_size,
        )

        # Initialize arrays
        forecasts, params_history, refit_mask = self._initialize_expanding_arrays(n)

        # Compute start position for the forecasting loop.
        # - Evaluation: start at min_window_size (early forecasts).
        # - Training default: anchor at initial_window_size; for rolling, also
        #   require window_size to be reached.
        if self.anchor_at_min_window:
            start_pos = max(self.min_window_size, GARCH_FIT_MIN_SIZE)
        else:
            if self.window_type == "rolling" and self.window_size is not None:
                start_pos = max(self.window_size, self.initial_window_size)
            else:
                start_pos = max(self.min_window_size, self.initial_window_size)

        # ANTI-LEAKAGE: Perform initial fit using ONLY residuals[:start_pos]
        current_params = self._perform_initial_model_fit(residuals)

        # Sanity check: start position must be within residuals length
        assert start_pos <= n, f"Start position {start_pos} exceeds residuals length {n}"

        # Run forecast loop from start_pos
        # ANTI-LEAKAGE: Each forecast at position t uses ONLY residuals[:t]
        self._run_expanding_forecast_loop(
            residuals, n, current_params, forecasts, params_history, refit_mask, start_pos
        )

        # Create and return result
        return self._create_expanding_result(n, forecasts, params_history, refit_mask, dates)

    def _run_test_forecast_loop(
        self,
        residuals_full: np.ndarray,
        n_train: int,
        n_total: int,
        current_params: EGARCHParams,
        forecasts_test: np.ndarray,
        params_history_test: list[dict[str, float]],
        refit_mask_test: np.ndarray,
        start_pos: int,
    ) -> None:
        """Run forecast loop on TEST data.

        Args:
            residuals_full: Concatenated train+test residuals.
            n_train: Number of training observations.
            n_total: Total observations (train+test).
            current_params: Current EGARCH parameters.
            forecasts_test: Array to store test forecasts (modified in place).
            params_history_test: List to store parameter history (modified in place).
            refit_mask_test: Array to store refit indicators (modified in place).
        """
        for i, t in enumerate(range(n_train, n_total)):
            test_pos = i  # Position in test array

            if self.refit_manager.should_refit(t, start_pos):
                logger.debug("Refit at position %d (test pos %d)", t, test_pos)
                params, convergence = self.refit_manager.perform_refit(residuals_full, t)

                if convergence.converged:
                    current_params = create_egarch_params_from_dict(
                        params, o=self.o, p=self.p, dist=self.dist
                    )
                else:
                    logger.warning(
                        "Refit at position %d (test pos %d) did not converge: %s",
                        t,
                        test_pos,
                        convergence.message or "unknown reason",
                    )

                refit_mask_test[test_pos] = True

            forecast_t = self._compute_one_step_forecast(
                residuals_full[:t],
                current_params,
            )

            forecasts_test[test_pos] = forecast_t
            params_history_test.append(current_params.to_dict())

    def forecast_continuing(
        self,
        residuals_train: np.ndarray,
        residuals_test: np.ndarray,
        dates_test: pd.DatetimeIndex,
    ) -> ForecastResult:
        """Continue forecasts on TEST data using current forecaster state.

        This method continues forecasting from the current state without regenerating
        forecasts on TRAIN. It uses the current parameters and refit manager state.

        Args:
            residuals_train: Training residuals (used for refit windows).
            residuals_test: Test residuals to forecast.
            dates_test: Test dates.

        Returns:
            ForecastResult with test forecasts only.

        Raises:
            RuntimeError: If refit manager has no current parameters.
        """
        n_train = len(residuals_train)
        n_test = len(residuals_test)
        n_total = n_train + n_test

        residuals_full = np.concatenate([residuals_train, residuals_test])

        # Set total data length for anti-leakage checks
        self._total_data_length = len(residuals_full)

        current_params_dict = self.refit_manager.get_current_params()
        current_params = create_egarch_params_from_dict(
            current_params_dict, o=self.o, p=self.p, dist=self.dist
        )

        forecasts_test = np.full(n_test, np.nan, dtype=float)
        params_history_test: list[dict[str, float]] = []
        refit_mask_test = np.zeros(n_test, dtype=bool)

        # Use the same scheduling anchor strategy as expanding forecast.
        if self.anchor_at_min_window:
            start_pos = max(self.min_window_size, GARCH_FIT_MIN_SIZE)
        else:
            start_pos = max(self.min_window_size, self.initial_window_size)

        self._run_test_forecast_loop(
            residuals_full,
            n_train,
            n_total,
            current_params,
            forecasts_test,
            params_history_test,
            refit_mask_test,
            start_pos,
        )

        n_refits_test = int(np.sum(refit_mask_test))
        convergence_rate = self.refit_manager.get_convergence_rate()

        logger.info(
            "Test forecasts complete: generated %d forecasts, %d refits, "
            "convergence rate=%.1f%%",
            n_test,
            n_refits_test,
            convergence_rate * 100,
        )

        return ForecastResult(
            dates=dates_test,
            forecasts=forecasts_test,
            volatility=np.sqrt(np.maximum(forecasts_test, 0.0)),
            params_history=params_history_test,
            refit_mask=refit_mask_test,
            n_refits=n_refits_test,
            convergence_rate=convergence_rate,
        )

    def _prepare_residuals_for_variance_computation(
        self, residuals_history: np.ndarray
    ) -> np.ndarray:
        """Prepare residuals for variance computation based on window type.

        Args:
            residuals_history: Historical residuals.

        Returns:
            Residuals to use for variance computation.
        """
        if self.window_type == "rolling" and self.window_size is not None:
            # Safety check for data leakage
            if hasattr(self, "_total_data_length"):
                assert len(residuals_history) < self._total_data_length, (
                    f"LEAKAGE DETECTED: residuals_history length ({len(residuals_history)}) "
                    f"must be < total data length ({self._total_data_length}) to ensure "
                    "no future information is included in rolling window"
                )

            effective_window_size = min(self.window_size, len(residuals_history))
            assert effective_window_size <= len(residuals_history), (
                f"LOGIC ERROR: effective_window_size ({effective_window_size}) "
                f"cannot exceed residuals_history length ({len(residuals_history)})"
            )

            return residuals_history[-effective_window_size:]
        else:
            # For expanding windows, use all history
            return residuals_history

    def _compute_variance_forecast(
        self, residuals_for_variance: np.ndarray, params: EGARCHParams
    ) -> float:
        """Compute variance forecast from prepared residuals.

        Args:
            residuals_for_variance: Residuals for variance computation.
            params: EGARCH parameters.

        Returns:
            One-step-ahead variance forecast.
        """
        alpha, gamma, beta = params.extract_for_variance()

        sigma2_path = egarch_variance(
            residuals_for_variance,
            omega=params.omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            dist=params.dist,
            nu=params.nu,
            lambda_skew=params.lambda_skew,
            init=None,  # Let it auto-initialize
            o=params.o,
            p=params.p,
        )

        if sigma2_path.size == 0:
            msg = "Variance path computation returned empty array"
            raise ValueError(msg)

        return float(sigma2_path[-1])

    def _validate_residuals_input(self, residuals_for_variance: np.ndarray) -> None:
        """Validate input residuals for NaN values.

        Args:
            residuals_for_variance: Residuals to validate.

        Raises:
            ValueError: If NaN values are detected.
        """
        if not np.all(np.isfinite(residuals_for_variance)):
            nan_count = np.sum(~np.isfinite(residuals_for_variance))
            logger.error("NaN values detected in input residuals:")
            logger.error(f"  - Total residuals: {len(residuals_for_variance)}")
            logger.error(f"  - NaN count: {nan_count}")
            logger.error(f"  - First 10 residuals: {residuals_for_variance[:10]}")
            logger.error("  - This usually indicates missing ARIMA residuals in training data")
            logger.error(
                "  - Run ARIMA evaluation first: python -m src.arima.evaluation_arima.main"
            )
            msg = f"NaN residuals detected in EGARCH input: {nan_count} NaN values"
            raise ValueError(msg)

    def _validate_variance_forecast_output(
        self,
        sigma2_forecast: float,
        residuals_history: np.ndarray,
        residuals_for_variance: np.ndarray,
        sigma2_path: np.ndarray,
        params: EGARCHParams,
    ) -> None:
        """Validate variance forecast output.

        Args:
            sigma2_forecast: Computed variance forecast.
            residuals_history: Original residuals history.
            residuals_for_variance: Residuals used for computation.
            sigma2_path: Full variance path.
            params: EGARCH parameters.

        Raises:
            ValueError: If forecast is invalid.
        """
        if not np.isfinite(sigma2_forecast) or sigma2_forecast <= 0:
            logger.error("Invalid variance forecast detected:")
            logger.error(f"  - Forecast value: {sigma2_forecast}")
            logger.error(f"  - Original residuals length: {len(residuals_history)}")
            logger.error(
                f"  - Effective residuals length for variance: {len(residuals_for_variance)}"
            )
            logger.error(f"  - Window type: {self.window_type}, Window size: {self.window_size}")

            alpha, gamma, beta = params.extract_for_variance()
            logger.error(
                f"  - Params: omega={params.omega}, alpha={alpha}, gamma={gamma}, beta={beta}"
            )

            last_residuals = (
                residuals_for_variance[-5:]
                if len(residuals_for_variance) >= 5
                else residuals_for_variance
            )
            logger.error(f"  - Last 5 residuals used: {last_residuals}")

            last_variances = sigma2_path[-5:] if len(sigma2_path) >= 5 else sigma2_path
            logger.error(f"  - Last 5 variances: {last_variances}")

            msg = f"Invalid variance forecast: {sigma2_forecast}"
            raise ValueError(msg)

    def _compute_one_step_forecast(
        self,
        residuals_history: np.ndarray,
        params: EGARCHParams,
    ) -> float:
        """Compute one-step-ahead variance forecast.

        Args:
            residuals_history: Historical residuals up to (not including) forecast point.
            params: EGARCH parameters.

        Returns:
            One-step-ahead variance forecast σ²_{t+1}|t.

        Note:
            For EGARCH, the one-step forecast is σ²_{t+1}|t = exp(E_t[log(σ²_{t+1})]).
            We compute the variance path through t, then the forecast is the last variance value.
        """
        if residuals_history.size == 0:
            msg = "Cannot compute one-step forecast without residuals history"
            raise ValueError(msg)

        # Prepare residuals for variance computation
        residuals_for_variance = self._prepare_residuals_for_variance_computation(residuals_history)

        # Compute variance forecast
        sigma2_forecast = self._compute_variance_forecast(residuals_for_variance, params)

        # Validate inputs and outputs
        self._validate_residuals_input(residuals_for_variance)

        # Note: We need to recompute sigma2_path for validation - this is a trade-off
        # for cleaner separation of concerns
        alpha, gamma, beta = params.extract_for_variance()
        sigma2_path = egarch_variance(
            residuals_for_variance,
            omega=params.omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            dist=params.dist,
            nu=params.nu,
            lambda_skew=params.lambda_skew,
            init=None,
            o=params.o,
            p=params.p,
        )

        self._validate_variance_forecast_output(
            sigma2_forecast, residuals_history, residuals_for_variance, sigma2_path, params
        )

        return sigma2_forecast

    def clear_history(self) -> None:
        """Clear refit history."""
        self.refit_manager.clear_history()

    def get_summary(self) -> dict[str, Any]:
        """Get forecaster summary statistics.

        Returns:
            Dictionary with forecasting statistics.
        """
        return {
            "o": self.o,
            "p": self.p,
            "dist": self.dist,
            "refit_frequency": self.refit_frequency,
            "window_type": self.window_type,
            "window_size": self.window_size,
            "initial_window_size": self.initial_window_size,
            "min_window_size": self.min_window_size,
            **self.refit_manager.get_summary(),
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"EGARCHForecaster(EGARCH({self.o},{self.p}), dist={self.dist}, "
            f"refit_freq={self.refit_frequency}, window={self.window_type})"
        )
