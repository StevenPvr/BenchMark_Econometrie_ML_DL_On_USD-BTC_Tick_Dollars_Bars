"""Walk-forward cross-validation for EGARCH models.

This module implements temporal cross-validation with:
- Burn-in period to avoid early instability
- Rolling/expanding windows
- QLIKE out-of-sample evaluation
- One-step ahead variance forecasting
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GARCH_LOG_VAR_EXPLOSION_THRESHOLD,
    GARCH_MIN_INIT_VAR,
    GARCH_OPTIMIZATION_BURN_IN_RATIO,
    GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE,
    GARCH_QLIKE_MAX_ACCEPTABLE,
)
from src.garch.garch_eval.metrics import qlike_loss
from src.garch.garch_params.core.distributions import compute_kappa
from src.garch.garch_params.core.variance import clip_and_exp_log_variance, compute_variance_path
from src.garch.garch_params.estimation import estimate_egarch_mle
from src.garch.garch_params.models import create_egarch_params_from_dict
from src.utils import get_logger

logger = get_logger(__name__)


def _compute_forecast_step(
    omega: float,
    alpha1: float,
    alpha2: float,
    gamma1: float,
    gamma2: float,
    beta1: float,
    beta2: float,
    kappa: float,
    s2_state: np.ndarray,
    e_state: np.ndarray,
    o: int,
    p: int,
) -> float:
    """Compute one-step variance forecast using EGARCH recursion.

    Args:
        omega: Omega parameter.
        alpha1: Alpha1 parameter.
        alpha2: Alpha2 parameter.
        gamma1: Gamma1 parameter.
        gamma2: Gamma2 parameter.
        beta1: Beta1 parameter.
        beta2: Beta2 parameter.
        kappa: Kappa constant.
        s2_state: Variance state array.
        e_state: Residual state array.
        o: ARCH order.
        p: GARCH order.

    Returns:
        Next variance forecast value.

    Raises:
        RuntimeError: If forecast is invalid or variance explosion detected.
    """
    ln_next = omega

    if p >= 1:
        s2_prev1 = max(float(s2_state[-1]), GARCH_MIN_INIT_VAR)
        ln_next += beta1 * np.log(s2_prev1)
    if p >= 2:
        s2_prev2 = max(float(s2_state[-2]), GARCH_MIN_INIT_VAR)
        ln_next += beta2 * np.log(s2_prev2)

    if o >= 1:
        s2_prev1 = max(float(s2_state[-1]), GARCH_MIN_INIT_VAR)
        z_prev1 = float(e_state[-1] / np.sqrt(s2_prev1))
        ln_next += alpha1 * (abs(z_prev1) - kappa) + gamma1 * z_prev1
    if o >= 2:
        s2_prev2 = max(float(s2_state[-2]), GARCH_MIN_INIT_VAR)
        z_prev2 = float(e_state[-2] / np.sqrt(s2_prev2))
        ln_next += alpha2 * (abs(z_prev2) - kappa) + gamma2 * z_prev2

    # Check for variance explosion BEFORE clipping
    # This indicates non-stationarity in the EGARCH recursion
    if abs(ln_next) > GARCH_LOG_VAR_EXPLOSION_THRESHOLD:
        msg = (
            f"Variance explosion detected: log-variance={ln_next:.2f} exceeds "
            f"threshold={GARCH_LOG_VAR_EXPLOSION_THRESHOLD}. "
            "This indicates non-stationary EGARCH parameters."
        )
        raise RuntimeError(msg)

    s2_next = clip_and_exp_log_variance(ln_next)
    if not np.isfinite(s2_next) or s2_next <= 0:
        msg = "Invalid variance forecast computed"
        raise RuntimeError(msg)

    return s2_next


def _extract_individual_params(
    alpha: float | tuple[float, float],
    gamma: float | tuple[float, float],
    beta: float | tuple[float, float] | tuple[float, float, float],
    o: int,
    p: int,
) -> tuple[float, float, float, float, float, float]:
    """Extract individual parameter values for forecast computation.

    Args:
        alpha: Alpha parameter(s).
        gamma: Gamma parameter(s).
        beta: Beta parameter(s).
        o: ARCH order.
        p: GARCH order.

    Returns:
        Tuple of (alpha1, alpha2, gamma1, gamma2, beta1, beta2).
        Missing parameters are set to 0.0.
    """
    # Extract alpha parameters
    if o == 1:
        alpha1 = float(alpha) if not isinstance(alpha, tuple) else alpha[0]
        alpha2 = 0.0
    else:  # o == 2
        if isinstance(alpha, tuple):
            alpha1, alpha2 = float(alpha[0]), float(alpha[1])
        else:
            alpha1, alpha2 = float(alpha), 0.0

    # Extract gamma parameters
    if o == 1:
        gamma1 = float(gamma) if not isinstance(gamma, tuple) else gamma[0]
        gamma2 = 0.0
    else:  # o == 2
        if isinstance(gamma, tuple):
            gamma1, gamma2 = float(gamma[0]), float(gamma[1])
        else:
            gamma1, gamma2 = float(gamma), 0.0

    # Extract beta parameters
    if p == 1:
        beta1 = float(beta) if not isinstance(beta, tuple) else beta[0]
        beta2 = 0.0
    else:  # p == 2
        if isinstance(beta, tuple):
            beta1, beta2 = float(beta[0]), float(beta[1])
        else:
            beta1, beta2 = float(beta), 0.0

    return alpha1, alpha2, gamma1, gamma2, beta1, beta2


def _initialize_forecast_state(
    resid_train: np.ndarray,
    s2_train: np.ndarray,
    o: int,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize forecast state from training data.

    Args:
        resid_train: Training residuals.
        s2_train: Training variance path.
        o: ARCH order.
        p: GARCH order.

    Returns:
        Tuple of (s2_state, e_state) containing last max(o,p) values.
    """
    max_lag = max(o, p)
    s2_state = s2_train[-max_lag:].copy()
    e_state = resid_train[-max_lag:].copy()
    return s2_state, e_state


def _run_forecast_loop(
    resid_val: np.ndarray,
    omega: float,
    alpha1: float,
    alpha2: float,
    gamma1: float,
    gamma2: float,
    beta1: float,
    beta2: float,
    kappa: float,
    s2_state: np.ndarray,
    e_state: np.ndarray,
    o: int,
    p: int,
) -> np.ndarray:
    """Run forecast loop for validation set.

    Args:
        resid_val: Validation residuals.
        omega: Omega parameter.
        alpha1: First alpha parameter.
        alpha2: Second alpha parameter (0.0 if o=1).
        gamma1: First gamma parameter.
        gamma2: Second gamma parameter (0.0 if o=1).
        beta1: First beta parameter.
        beta2: Second beta parameter (0.0 if p=1).
        kappa: Distribution kappa value.
        s2_state: Initial variance state.
        e_state: Initial residual state.
        o: ARCH order.
        p: GARCH order.

    Returns:
        Variance forecasts for validation set.
    """
    s2_forecast = np.empty(resid_val.size, dtype=float)
    for i in range(resid_val.size):
        s2_next = _compute_forecast_step(
            omega,
            alpha1,
            alpha2,
            gamma1,
            gamma2,
            beta1,
            beta2,
            kappa,
            s2_state,
            e_state,
            o,
            p,
        )
        s2_forecast[i] = s2_next

        # Update state arrays
        s2_state = np.append(s2_state[1:], s2_next)
        e_state = np.append(e_state[1:], float(resid_val[i]))

    return s2_forecast


def _compute_variance_forecast(
    resid_train: np.ndarray,
    resid_val: np.ndarray,
    params: dict[str, float],
    dist: str,
    o: int,
    p: int,
) -> np.ndarray:
    """Compute variance forecasts for validation set.

    Args:
        resid_train: Training residuals for fitting.
        resid_val: Validation residuals (for alignment only).
        params: Estimated EGARCH parameters.
        dist: Distribution name.
        o: ARCH order.
        p: GARCH order.

    Returns:
        Variance forecasts for validation set.

    Raises:
        RuntimeError: If variance computation fails.
    """
    # Use the provided distribution (don't infer from parameters)
    # The dist parameter is already known from the optimization process
    egarch_params = create_egarch_params_from_dict(params, o=o, p=p, dist=dist)
    omega = egarch_params.omega
    alpha, gamma, beta = egarch_params.extract_for_variance()
    nu = egarch_params.nu
    lambda_skew = egarch_params.lambda_skew

    # Fixed initial variance using first 30 points only to avoid future info leakage
    # Using all resid_train would contaminate early variance estimates with future data
    init_window_size = min(30, len(resid_train))
    init_var = float(np.var(resid_train[:init_window_size], ddof=0))

    # Compute kappa and variance path
    kappa = compute_kappa(dist, nu, lambda_skew)
    s2_train = compute_variance_path(
        resid_train,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        kappa=kappa,
        init=init_var,
        o=o,
        p=p,
    )

    # Validate variance path
    if not np.all(np.isfinite(s2_train)) or np.any(s2_train <= 0):
        msg = "Invalid variance path computed on training data"
        raise RuntimeError(msg)

    # Initialize forecast state
    s2_state, e_state = _initialize_forecast_state(resid_train, s2_train, o, p)

    # Extract individual parameters for forecast loop
    alpha1, alpha2, gamma1, gamma2, beta1, beta2 = _extract_individual_params(
        alpha, gamma, beta, o, p
    )

    # Run forecast loop
    return _run_forecast_loop(
        resid_val,
        omega,
        alpha1,
        alpha2,
        gamma1,
        gamma2,
        beta1,
        beta2,
        kappa,
        s2_state,
        e_state,
        o,
        p,
    )


def _process_cv_fold(
    train_local: np.ndarray,
    val_local: np.ndarray,
    val_start: int,
    dist: str,
    o: int,
    p: int,
) -> float | None:
    """Process a single CV fold and return QLIKE score.

    Args:
        train_local: Training window residuals.
        val_local: Validation window residuals.
        val_start: Starting index of validation window.
        dist: Distribution name.
        o: ARCH order.
        p: GARCH order.

    Returns:
        QLIKE score if successful, None otherwise.

    Note:
        Returns None if QLIKE exceeds GARCH_QLIKE_MAX_ACCEPTABLE, indicating
        severe model misspecification. This prevents extreme values from
        contaminating optimization results.
    """
    try:
        params, convergence = estimate_egarch_mle(train_local, dist=dist, o=o, p=p)

        if not convergence.converged:
            logger.warning(
                "EGARCH fit failed to converge at fold starting at index %d "
                "(train_size=%d, val_size=%d, dist=%s, o=%d, p=%d)",
                val_start,
                train_local.size,
                val_local.size,
                dist,
                o,
                p,
            )
            return None

        s2_forecast = _compute_variance_forecast(train_local, val_local, params, dist, o, p)
        qlike_val = qlike_loss(val_local, s2_forecast)

        if not np.isfinite(qlike_val):
            logger.warning(
                "Invalid QLIKE at fold starting at index %d "
                "(qlike=%.6f, train_size=%d, val_size=%d)",
                val_start,
                qlike_val,
                train_local.size,
                val_local.size,
            )
            return None

        # Check for extreme QLIKE values indicating model misspecification
        # This is a scientifically justified bound: QLIKE > 100 is unrealistic
        # for any well-specified GARCH model on financial returns
        if qlike_val > GARCH_QLIKE_MAX_ACCEPTABLE:
            logger.debug(
                "Extreme QLIKE at fold %d: %.2f > %.1f (dist=%s, o=%d, p=%d)",
                val_start,
                qlike_val,
                GARCH_QLIKE_MAX_ACCEPTABLE,
                dist,
                o,
                p,
            )
            return None

        return float(qlike_val)

    except Exception as ex:
        logger.debug(
            "CV fold %d failed: %s (dist=%s, o=%d, p=%d)",
            val_start,
            ex,
            dist,
            o,
            p,
        )
        return None


def _compute_train_window_bounds(
    val_start: int, window_type: str, window_size: int | None
) -> tuple[int, int]:
    """Compute training window bounds for CV fold.

    Args:
        val_start: Starting index of validation window.
        window_type: Window type ('expanding' or 'rolling').
        window_size: Rolling window size (required if window_type='rolling').

    Returns:
        Tuple of (train_start, train_end).

    Raises:
        ValueError: If window_size is required but not provided.
    """
    if window_type == "expanding":
        train_start = 0
    else:  # rolling
        if window_size is None:
            msg = "window_size required for rolling window"
            raise ValueError(msg)
        train_start = max(0, val_start - window_size)

    train_end = val_start
    return train_start, train_end


def _validate_cv_parameters(
    resid_train: np.ndarray, window_type: str, window_size: int | None
) -> None:
    """Validate cross-validation parameters.

    Args:
        resid_train: Training residuals.
        window_type: Window type.
        window_size: Rolling window size.

    Raises:
        ValueError: If parameters are invalid.
    """
    if resid_train.size < GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE:
        msg = (
            f"Insufficient training data: {resid_train.size} < "
            f"{GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE}"
        )
        raise ValueError(msg)

    if window_type == "rolling" and window_size is None:
        msg = "window_size required for rolling window"
        raise ValueError(msg)

    if window_type not in ("expanding", "rolling"):
        msg = f"Invalid window_type: {window_type}"
        raise ValueError(msg)


def _compute_burn_in_size(resid_train: np.ndarray) -> int:
    """Compute burn-in size for cross-validation.

    Args:
        resid_train: Training residuals.

    Returns:
        Burn-in size.
    """
    burn_in_size = int(resid_train.size * GARCH_OPTIMIZATION_BURN_IN_RATIO)
    if burn_in_size < GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE:
        burn_in_size = GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE
    return burn_in_size


def _process_cv_folds(
    resid_train: np.ndarray,
    start_idx: int,
    end_idx: int,
    refit_freq: int,
    window_type: str,
    window_size: int | None,
    dist: str,
    o: int,
    p: int,
) -> list[float]:
    """Process all CV folds and collect QLIKE scores.

    Args:
        resid_train: Training residuals.
        start_idx: Starting index for validation.
        end_idx: Ending index for validation.
        refit_freq: Refit frequency.
        window_type: Window type.
        window_size: Rolling window size.
        dist: Distribution name.
        o: ARCH order.
        p: GARCH order.

    Returns:
        List of QLIKE scores.
    """
    qlike_scores: list[float] = []

    # Handle refit_freq=0 (no refit, use initial fit only)
    if refit_freq == 0:
        # Single validation fold using initial fit
        val_start = start_idx
        val_end = end_idx
        train_start, train_end = _compute_train_window_bounds(val_start, window_type, window_size)
        train_local = resid_train[train_start:train_end]
        if train_local.size < GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE:
            return [float("inf")]
        val_local = resid_train[val_start:val_end]
        qlike_val = _process_cv_fold(train_local, val_local, val_start, dist, o, p)
        return [qlike_val] if qlike_val is not None and np.isfinite(qlike_val) else [float("inf")]

    for val_start in range(start_idx, end_idx, refit_freq):
        val_end = min(val_start + refit_freq, end_idx)
        if val_end - val_start < GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE:
            continue

        train_start, train_end = _compute_train_window_bounds(val_start, window_type, window_size)
        train_local = resid_train[train_start:train_end]

        if train_local.size < GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE:
            continue

        val_local = resid_train[val_start:val_end]
        qlike_val = _process_cv_fold(train_local, val_local, val_start, dist, o, p)

        if qlike_val is not None:
            qlike_scores.append(qlike_val)

    return qlike_scores


def _validate_refit_freq(refit_freq: int) -> None:
    """Validate refit frequency parameter.

    Args:
        refit_freq: Refit frequency.

    Raises:
        ValueError: If refit_freq is negative.
    """
    if refit_freq < 0:
        raise ValueError(f"refit_freq must be >= 0, got {refit_freq}")


def _validate_data_sufficiency(validation_range: int, n_splits: int) -> None:
    """Validate sufficient data for CV splits.

    Args:
        validation_range: Size of validation range.
        n_splits: Number of splits.

    Raises:
        ValueError: If insufficient data.
    """
    if validation_range < n_splits:
        msg = (
            f"Insufficient data for {n_splits} splits: validation range "
            f"({validation_range}) < n_splits ({n_splits})"
        )
        raise ValueError(msg)


def walk_forward_cv(
    resid_train: np.ndarray,
    *,
    o: int,
    p: int,
    dist: str,
    n_splits: int,
    window_type: str,
    window_size: int | None = None,
    refit_freq: int,
) -> float:
    """Perform walk-forward cross-validation on TRAIN data.

    Implements temporal CV with burn-in period and QLIKE evaluation.

    Args:
        resid_train: Training residuals (TRAIN split only).
        o: ARCH order (1 or 2).
        p: GARCH order (1 or 2).
        dist: Distribution name ('student', 'skewt').
        n_splits: Number of walk-forward CV splits.
        window_type: Window type ('expanding' or 'rolling').
        window_size: Rolling window size (required if window_type='rolling').
        refit_freq: Refit frequency. Mandatory; no default.

    Returns:
        Mean QLIKE loss across all validation folds.

    Raises:
        ValueError: If parameters are invalid.
        RuntimeError: If CV fails or no valid folds.
    """
    _validate_cv_parameters(resid_train, window_type, window_size)
    _validate_refit_freq(refit_freq)

    # Validate rolling window size against minimum validation size
    if window_type == "rolling" and window_size is not None:
        if window_size < GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE:
            msg = (
                f"Rolling window size ({window_size}) must be >= "
                f"GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE "
                f"({GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE}). "
                "This ensures sufficient training data for EGARCH estimation."
            )
            raise ValueError(msg)

    burn_in_size = _compute_burn_in_size(resid_train)
    start_idx = burn_in_size
    end_idx = resid_train.size

    _validate_data_sufficiency(end_idx - start_idx, n_splits)

    qlike_scores = _process_cv_folds(
        resid_train, start_idx, end_idx, int(refit_freq), window_type, window_size, dist, o, p
    )

    if len(qlike_scores) == 0:
        msg = "No valid CV folds completed"
        raise RuntimeError(msg)

    mean_qlike = float(np.mean(qlike_scores))
    logger.info(
        "Walk-forward CV completed: %d folds, mean QLIKE=%.6f", len(qlike_scores), mean_qlike
    )

    return mean_qlike
