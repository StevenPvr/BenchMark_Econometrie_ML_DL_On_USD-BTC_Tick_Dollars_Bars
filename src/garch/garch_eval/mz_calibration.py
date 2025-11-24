"""Mincer-Zarnowitz calibration utilities for GARCH evaluation."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from src.constants import GARCH_EVAL_DEFAULT_SLOPE
from src.garch.garch_eval.data_loading import prepare_residuals_from_dataset
from src.utils import get_logger

logger = get_logger(__name__)


def compute_mz_pvalues(
    beta: np.ndarray,
    x_mat: np.ndarray,
    ss_res: float,
) -> dict[str, float]:
    """Compute p-values for Mincer-Zarnowitz regression coefficients.

    Args:
    ----
        beta: Regression coefficients.
        x_mat: Design matrix.
        ss_res: Sum of squared residuals.

    Returns:
    -------
        Dictionary with t-statistics and p-values.

    """
    try:
        from scipy.stats import t as student_t  # type: ignore

        n, k = x_mat.shape
        s2_err = ss_res / max(1, n - k)
        xtx_inv = np.linalg.inv(x_mat.T @ x_mat)
        se = np.sqrt(np.diag(xtx_inv) * s2_err)
        t_intercept = float(beta[0] / se[0]) if se[0] > 0 else float("nan")
        t_slope = float(beta[1] / se[1]) if se[1] > 0 else float("nan")
        dof = max(1, n - k)
        p_intercept = float(2.0 * (1.0 - student_t.cdf(abs(t_intercept), df=dof)))
        p_slope = float(2.0 * (1.0 - student_t.cdf(abs(t_slope), df=dof)))
        return {
            "t_intercept": t_intercept,
            "t_slope": t_slope,
            "p_intercept": p_intercept,
            "p_slope": p_slope,
        }
    except Exception as ex:
        logger.debug("SciPy unavailable for MZ p-values; continuing without: %s", ex)
        return {}


def filter_test_data(
    e_test: np.ndarray,
    s2_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter test data to keep only finite and positive variance values.

    Args:
    ----
        e_test: Test residuals.
        s2_test: Test variance.

    Returns:
    -------
        Tuple of (filtered_residuals, filtered_variance).

    """
    m = np.isfinite(e_test) & np.isfinite(s2_test) & (s2_test > 0)
    return e_test[m], s2_test[m]


def _extract_split_masks(df_sorted: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract train and test masks from dataset."""
    if "split" not in df_sorted.columns:
        logger.warning("Dataset missing 'split' column. Cannot determine train/test split.")
        return None

    train_mask = (df_sorted["split"].astype(str) == "train").to_numpy()
    test_mask = (df_sorted["split"].astype(str) == "test").to_numpy()
    return train_mask, test_mask


def _map_to_filtered_positions(
    df_sorted: pd.DataFrame,
    valid_mask: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map original indices to positions in filtered residual array."""
    idx_all = np.arange(df_sorted.shape[0])
    idx_valid = idx_all[valid_mask]
    idx_train = idx_all[train_mask & valid_mask]
    idx_test = idx_all[test_mask & valid_mask]

    logger.debug("Valid split counts: train=%d, test=%d", idx_train.size, idx_test.size)

    # Map to positions in filtered array
    pos_in_valid = -np.ones(df_sorted.shape[0], dtype=int)
    pos_in_valid[idx_valid] = np.arange(idx_valid.size)
    pos_train = pos_in_valid[idx_train]
    pos_test = pos_in_valid[idx_test]
    pos_train = pos_train[pos_train >= 0]
    pos_test = pos_test[pos_test >= 0]
    pos_train.sort()
    pos_test.sort()

    return pos_train, pos_test


def _extract_train_test_positions(
    df_sorted: pd.DataFrame,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract train/test positions from filtered residuals.

    Args:
    ----
        df_sorted: Sorted DataFrame with split column.
        valid_mask: Boolean mask of valid residuals.

    Returns:
    -------
        Tuple of (pos_train, pos_test) or None if extraction fails.

    """
    masks = _extract_split_masks(df_sorted)
    if masks is None:
        return None
    train_mask, test_mask = masks

    pos_train, pos_test = _map_to_filtered_positions(df_sorted, valid_mask, train_mask, test_mask)

    if pos_test.size == 0:
        logger.warning("No test data found after filtering")
        return None

    if pos_train.size == 0:
        logger.warning("No train data found for variance computation")
        return None

    return pos_train, pos_test


def _compute_variance_up_to_test(
    resid_up_to_test: np.ndarray,
    model_name: str,
    params: dict[str, float],
    dist: str,
    nu: float | None,
    lambda_skew: float | None,
) -> np.ndarray | None:
    """Compute variance path up to test period."""
    try:
        from src.garch.garch_eval.variance_path import compute_variance_path_for_test

        s2_path = compute_variance_path_for_test(
            resid_up_to_test, model_name, params, dist, nu, lambda_skew
        )
    except ValueError as e:
        logger.warning("Failed to compute variance path up to test period: %s", e)
        return None

    if s2_path.size != resid_up_to_test.size:
        logger.warning("Variance path size mismatch")
        return None

    return s2_path


# Keep the original API but we'll only use it for validation side-effects
def _initialize_test_variance_state(
    resid_f: np.ndarray,
    pos_train: np.ndarray,
    pos_test: np.ndarray,
    model_name: str,
    params: dict[str, float],
    dist: str,
    nu: float | None,
    lambda_skew: float | None,
) -> tuple[float, float] | None:
    """Initialize variance state for test period forecasts.

    Args:
    ----
        resid_f: Filtered residuals.
        pos_train: Train positions.
        pos_test: Test positions.
        model_name: Model name.
        params: Model parameters.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
    -------
        Tuple of (e_current, s2_current) or None if initialization fails.

    """
    pos_test_start = int(pos_test[0])
    resid_up_to_test = resid_f[:pos_test_start]

    if resid_up_to_test.size == 0:
        logger.warning("No residuals available up to test period")
        return None

    s2_path = _compute_variance_up_to_test(
        resid_up_to_test, model_name, params, dist, nu, lambda_skew
    )
    if s2_path is None:
        return None

    return float(resid_up_to_test[-1]), float(s2_path[-1])


def _compute_walkforward_test_forecasts(
    resid_history_init: np.ndarray,
    resid_test: np.ndarray,
    params: dict[str, float],
    *,
    dist: str,
    nu: float | None,
    lambda_skew: float | None,
) -> np.ndarray:
    """Compute walk-forward one-step forecasts for EGARCH(o,p) using full recursion.

    For each step t in TEST, compute σ²_{t|t-1} by running egarch_variance on
    the available residual history up to t-1, then append the realized residual
    to the history (leak-free ex-ante forecasts).
    """
    o, p = _infer_egarch_orders(params)
    alpha_param, gamma_param, beta_param = _assemble_egarch_parameters(params, o, p)

    history = np.asarray(resid_history_init, dtype=float).ravel()
    s2_forecasts = np.empty(len(resid_test), dtype=float)

    for i in range(len(resid_test)):
        forecast = _compute_single_step_forecast(
            history, params, alpha_param, gamma_param, beta_param, dist, nu, lambda_skew, o, p
        )
        s2_forecasts[i] = forecast
        # Update history with realized residual
        history = np.append(history, float(resid_test[i]))

    return s2_forecasts


def _infer_egarch_orders(params: dict[str, float]) -> tuple[int, int]:
    """Infer EGARCH model orders (o, p) from parameter dictionary.

    Args:
        params: Parameter dictionary.

    Returns:
        Tuple of (o, p) orders.
    """
    # Infer asymmetry order (o)
    has_a2 = "alpha2" in params or "gamma2" in params
    o = 2 if has_a2 else 1

    # Infer persistence order (p)
    if "beta3" in params:
        p = 3
    elif "beta2" in params:
        p = 2
    elif "beta1" in params or ("beta" in params and np.isfinite(params.get("beta", np.nan))):
        p = 1
    else:
        raise ValueError("Missing GARCH beta parameters for EGARCH(o,p) walk-forward forecasts")

    return o, p


def _assemble_egarch_parameters(
    params: dict[str, float],
    o: int,
    p: int,
) -> tuple[
    Union[float, tuple[float, float]],
    Union[float, tuple[float, float]],
    Union[float, tuple[float, float], tuple[float, float, float]],
]:
    """Assemble EGARCH parameters according to model orders.

    Args:
        params: Parameter dictionary.
        o: Asymmetry order.
        p: Persistence order.

    Returns:
        Tuple of (alpha_param, gamma_param, beta_param).
    """
    # Assemble alpha and gamma parameters
    if o == 1:
        alpha_param: Union[float, tuple[float, float]] = float(params["alpha"])
        gamma_param: Union[float, tuple[float, float]] = float(params.get("gamma", 0.0) or 0.0)
    else:
        alpha_param = (float(params["alpha1"]), float(params["alpha2"]))
        gamma_param = (float(params["gamma1"]), float(params["gamma2"]))

    # Assemble beta parameters
    if p == 1:
        beta_value = params.get("beta1") or params.get("beta")
        if beta_value is None:
            raise ValueError("Missing beta parameter for EGARCH(1,1) model")
        beta_param: Union[float, tuple[float, float], tuple[float, float, float]] = float(
            beta_value
        )
    elif p == 2:
        beta_param = (float(params["beta1"]), float(params["beta2"]))
    else:
        beta_param = (float(params["beta1"]), float(params["beta2"]), float(params["beta3"]))

    return alpha_param, gamma_param, beta_param


def _compute_single_step_forecast(
    history: np.ndarray,
    params: dict[str, float],
    alpha_param: Union[float, tuple[float, float]],
    gamma_param: Union[float, tuple[float, float]],
    beta_param: Union[float, tuple[float, float], tuple[float, float, float]],
    dist: str,
    nu: float | None,
    lambda_skew: float | None,
    o: int,
    p: int,
) -> float:
    """Compute single one-step variance forecast.

    Args:
        history: Residual history up to current point.
        params: Full parameter dictionary.
        alpha_param: Assembled alpha parameters.
        gamma_param: Assembled gamma parameters.
        beta_param: Assembled beta parameters.
        dist: Distribution name.
        nu: Degrees of freedom parameter.
        lambda_skew: Skewness parameter.
        o: Asymmetry order.
        p: Persistence order.

    Returns:
        One-step variance forecast.

    Raises:
        ValueError: If forecast is invalid.
    """
    from src.garch.garch_params.core import egarch_variance

    sigma2_path = egarch_variance(
        history,
        omega=float(params["omega"]),
        alpha=alpha_param,
        gamma=gamma_param,
        beta=beta_param,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
        init=None,
        o=o,
        p=p,
    )

    if sigma2_path.size == 0 or not np.isfinite(sigma2_path[-1]) or sigma2_path[-1] <= 0:
        raise ValueError("Invalid one-step variance forecast in walk-forward computation")

    return float(sigma2_path[-1])


def _prepare_test_data(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Prepare and extract train/test data from dataset."""
    df_sorted, _resid, valid_mask, resid_f = prepare_residuals_from_dataset(dataset)
    if resid_f.size == 0:
        return None

    result = _extract_train_test_positions(df_sorted, valid_mask)
    if result is None:
        return None
    pos_train, pos_test = result

    return df_sorted, resid_f, valid_mask, pos_train, pos_test


def load_test_resid_sigma2(
    params: dict[str, float],
    dataset: pd.DataFrame,
    *,
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned residuals and sigma² forecasts on test split.

    DATA LEAKAGE PREVENTION:
    - Computes variance path only on train data (in-sample)
    - For test period, computes one-step-ahead forecasts using walk-forward
    - Each forecast uses only data up to that point (ex-ante)
    - Does NOT use test residuals to compute test variance a posteriori

    Args:
    ----
        params: Model parameters dictionary.
        dataset: Input dataset DataFrame.
        model_name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Tuple of (test_residuals, test_variance_forecasts).

    """
    empty = (np.array([], dtype=float), np.array([], dtype=float))

    # Prepare residuals and extract train/test positions
    test_data = _prepare_test_data(dataset)
    if test_data is None:
        return empty
    _df_sorted, resid_f, _valid_mask, pos_train, pos_test = test_data

    # Build initial residual history up to start of test
    pos_test_start = int(pos_test[0])
    resid_history_init = resid_f[:pos_test_start]
    if resid_history_init.size == 0:
        logger.warning("No residual history available before test period")
        return empty

    # Validate ability to compute variance up to test (finiteness/positivity checks)
    _state = _initialize_test_variance_state(
        resid_f, pos_train, pos_test, model_name, params, dist, nu, lambda_skew
    )
    if _state is None:
        return empty

    e_test = resid_f[pos_test]
    s2_forecasts = _compute_walkforward_test_forecasts(
        resid_history_init,
        e_test,
        params,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )

    return e_test.astype(float), s2_forecasts.astype(float)


def _compute_mz_calibration_params(
    params: dict[str, float],
    data: pd.DataFrame,
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> tuple[float, float]:
    """Compute MZ calibration parameters from test data.

    Args:
    ----
        params: Model parameters.
        data: Dataset DataFrame.
        model_name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Tuple of (mz_intercept, mz_slope).

    """
    # Import locally to avoid circular dependency
    from src.garch.garch_eval.metrics import mz_calibration_params

    e_test, s2_test = load_test_resid_sigma2(
        params,
        data,
        model_name=model_name,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )
    if e_test.size == 0 or s2_test.size == 0:
        return 0.0, GARCH_EVAL_DEFAULT_SLOPE

    mz_params = mz_calibration_params(e_test, s2_test)
    mz_intercept = mz_params.get("intercept", 0.0)
    mz_slope = mz_params.get("slope", GARCH_EVAL_DEFAULT_SLOPE)
    return mz_intercept, mz_slope


def _apply_calibration_to_forecasts(
    s2_1: float,
    s2_h: np.ndarray,
    mz_intercept: float,
    mz_slope: float,
) -> tuple[float, np.ndarray]:
    """Apply MZ calibration to variance forecasts.

    Args:
    ----
        s2_1: One-step variance forecast.
        s2_h: Multi-step variance forecasts.
        mz_intercept: MZ intercept.
        mz_slope: MZ slope.

    Returns:
    -------
        Tuple of (calibrated_s2_1, calibrated_s2_h).

    """
    # Import locally to avoid circular dependency
    from src.garch.garch_eval.metrics import apply_mz_calibration

    s2_1_calibrated = float(
        apply_mz_calibration(np.array([s2_1]), mz_intercept, mz_slope, use_intercept=True)[0],
    )
    s2_h_calibrated = apply_mz_calibration(s2_h, mz_intercept, mz_slope, use_intercept=True)
    return s2_1_calibrated, s2_h_calibrated


def apply_mz_calibration_to_forecasts(
    s2_1: float,
    s2_h: np.ndarray,
    params: dict[str, float],
    data: pd.DataFrame,
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> tuple[float, np.ndarray, float, float]:
    """Apply MZ calibration to variance forecasts.

    WARNING: This function computes MZ calibration parameters using TEST data
    (via load_test_resid_sigma2). Using calibrated forecasts for evaluation
    introduces data leakage and invalidates out-of-sample performance metrics.
    This function should ONLY be used for diagnostic purposes, not for production
    scoring or model comparison.

    Args:
    ----
        s2_1: One-step variance forecast.
        s2_h: Multi-step variance forecasts.
        params: Model parameters.
        data: Dataset DataFrame.
        model_name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Tuple of (calibrated_s2_1, calibrated_s2_h, mz_intercept, mz_slope).

    """
    mz_intercept = 0.0
    mz_slope = GARCH_EVAL_DEFAULT_SLOPE

    try:
        mz_intercept, mz_slope = _compute_mz_calibration_params(
            params, data, model_name, dist, nu, lambda_skew
        )
        logger.info(
            "Computed MZ calibration: intercept=%.6f, slope=%.3f",
            mz_intercept,
            mz_slope,
        )
        s2_1, s2_h = _apply_calibration_to_forecasts(s2_1, s2_h, mz_intercept, mz_slope)
        logger.info("Applied MZ calibration (multiplicative + additive with floor) to forecasts")
    except Exception as ex:
        logger.warning(
            "Failed to compute/apply MZ calibration: %s. Using uncalibrated forecasts.",
            ex,
        )

    return s2_1, s2_h, mz_intercept, mz_slope


__all__ = [
    "compute_mz_pvalues",
    "filter_test_data",
    "load_test_resid_sigma2",
    "apply_mz_calibration_to_forecasts",
]
