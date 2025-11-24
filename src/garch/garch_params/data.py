"""Data loading and preparation utilities for GARCH estimation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import cast

import numpy as np
import pandas as pd

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_ESTIMATION_MIN_OBSERVATIONS,
    GARCH_ESTIMATION_PARALLEL_WORKERS,
)
from src.garch.structure_garch.utils import load_garch_dataset, prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)


def _validate_residuals(resid_train: np.ndarray) -> None:
    """Validate training residuals meet minimum requirements.

    Args:
        resid_train: Training residuals array.

    Raises:
        ValueError: If insufficient observations.
    """
    if resid_train.size < GARCH_ESTIMATION_MIN_OBSERVATIONS:
        msg = (
            f"Insufficient training residuals: {resid_train.size} < "
            f"{GARCH_ESTIMATION_MIN_OBSERVATIONS}"
        )
        logger.error(msg)
        raise ValueError(msg)


def load_and_prepare_data() -> tuple[np.ndarray, np.ndarray]:
    """Load dataset and prepare training/test residuals.

    Ensures no look-ahead bias: train uses only train split,
    test uses only test split.

    Returns:
        Tuple of (training_residuals, test_residuals).

    Raises:
        ValueError: If data loading or preparation fails.
    """
    try:
        df = load_garch_dataset(str(GARCH_DATASET_FILE))
    except Exception as ex:
        logger.error("Failed to load GARCH dataset: %s", ex)
        raise

    df_train = cast(pd.DataFrame, df.loc[df["split"] == "train"].copy())
    if df_train.empty:
        msg = "No training data found in dataset"
        logger.error(msg)
        raise ValueError(msg)

    try:
        # Training: use only train split (no test data)
        resid_train = prepare_residuals(df_train, use_test_only=False)
        # Test: use only test split (no train data)
        resid_test = prepare_residuals(df, use_test_only=True)
    except Exception as ex:
        logger.error("Failed to prepare residuals: %s", ex)
        raise

    resid_train = resid_train[np.isfinite(resid_train)]
    resid_test = resid_test[np.isfinite(resid_test)]

    _validate_residuals(resid_train)

    return resid_train, resid_test


def estimate_single_model(resid_train: np.ndarray, dist: str) -> tuple[str, dict[str, float]]:
    """Estimate a single EGARCH model (helper for parallel execution).

    Args:
        resid_train: Training residuals from ARIMA model.
        dist: Distribution name ('student', 'skewt').

    Returns:
        Tuple of (distribution_name, parameter_dict).

    Raises:
        RuntimeError: If EGARCH MLE estimation fails.
    """
    from src.garch.garch_params.estimation import estimate_egarch_mle

    try:
        logger.info("Optimizing EGARCH(1,1) with %s innovations...", dist.capitalize())
        params, convergence = estimate_egarch_mle(resid_train, dist=dist)
        return dist, params
    except Exception as ex:
        msg = f"EGARCH MLE failed for dist={dist}"
        logger.error(msg)
        raise RuntimeError(msg) from ex


def estimate_egarch_models(
    resid_train: np.ndarray,
) -> tuple[
    dict[str, float],
    dict[str, float],
]:
    """Estimate EGARCH models for student and skewt distributions.

    Optimizes both models in parallel using conditional MLE.
    Both estimations must succeed.

    Args:
        resid_train: Training residuals from ARIMA model.

    Returns:
        Tuple of (egarch_student, egarch_skewt) parameter dicts.

    Raises:
        RuntimeError: If any EGARCH model estimation fails.
    """
    distributions = ["student", "skewt"]
    results: dict[str, dict[str, float]] = {}

    # Optimize both models in parallel
    logger.info(
        "Starting parallel optimization of %d EGARCH models...",
        GARCH_ESTIMATION_PARALLEL_WORKERS,
    )
    # Prefer process-based parallelism, but fall back to threads if unsupported
    executor_ctx: ProcessPoolExecutor | ThreadPoolExecutor
    try:
        executor_ctx = ProcessPoolExecutor(max_workers=GARCH_ESTIMATION_PARALLEL_WORKERS)
    except Exception as ex:  # pragma: no cover - environment-specific
        logger.warning("ProcessPool unavailable (%s); falling back to ThreadPoolExecutor", ex)
        executor_ctx = ThreadPoolExecutor(max_workers=GARCH_ESTIMATION_PARALLEL_WORKERS)

    with executor_ctx as executor:
        future_to_dist = {
            executor.submit(estimate_single_model, resid_train, dist): dist
            for dist in distributions
        }

        for future in as_completed(future_to_dist):
            try:
                dist, result = future.result()
                results[dist] = result
            except Exception as ex:
                msg = f"Failed to estimate EGARCH model for {future_to_dist[future]}"
                logger.error(msg)
                raise RuntimeError(msg) from ex

    # Ensure both distributions succeeded
    if len(results) != 2:
        missing = set(distributions) - set(results.keys())
        msg = f"Missing EGARCH estimations for distributions: {missing}"
        logger.error(msg)
        raise RuntimeError(msg)

    return results["student"], results["skewt"]
