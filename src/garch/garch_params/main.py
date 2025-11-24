"""CLI for EGARCH parameter estimation and optimization.

Stages:
- ``estimation``: batch EGARCH(1,1) MLE for each innovation distribution.
- ``optimization``: Optuna-based hyperparameter search (no warm-start dependency).
"""

from __future__ import annotations
import math
from pathlib import Path
import sys
from typing import Any

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_ESTIMATION_FILE,
    GARCH_OPTIMIZATION_DISTRIBUTIONS,
    GARCH_OPTIMIZATION_REFIT_FREQ_OPTIONS,
    GARCH_OPTIMIZATION_RESULTS_FILE,
    GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES,
    GARCH_OPTIMIZATION_WINDOW_TYPES,
)
from src.garch.garch_params.data import load_and_prepare_data
from src.garch.garch_params.estimation import estimate_egarch_mle
from src.garch.garch_params.optimization import (
    optimize_egarch_hyperparameters,
    save_optimization_results,
)
from src.garch.garch_params.core import initialize_variance
from src.garch.training_garch.predictions_io import save_estimation_results
from src.utils import get_logger

logger = get_logger(__name__)


# No CLI stage selection: running this module executes full pipeline sequentially.


def _count_model_parameters(o: int, p: int, dist: str) -> int:
    """Count EGARCH(o,p) parameters including distributional terms."""

    n_params = 1 + 2 * o + p
    if dist == "student":
        n_params += 1
    elif dist == "skewt":
        n_params += 2
    return n_params


def _build_fit_entry(
    dist: str,
    params: dict[str, float],
    convergence: Any,
    n_obs: int,
    o: int,
    p: int,
) -> dict[str, Any]:
    """Build structured estimation payload for persistence."""

    n_params = _count_model_parameters(o, p, dist)
    loglik = float(convergence.final_loglik)
    aic = -2.0 * loglik + 2.0 * n_params
    bic = -2.0 * loglik + math.log(n_obs) * n_params

    entry: dict[str, Any] = {
        "params": params,
        "converged": bool(convergence.converged),
        "log_likelihood": loglik,
        "aic": aic,
        "bic": bic,
    }
    if convergence.n_iterations is not None:
        entry["iterations"] = int(convergence.n_iterations)
    if convergence.message:
        entry["convergence_message"] = convergence.message
    return entry


def run_estimation_stage() -> None:
    """Run EGARCH(1,1) MLE for each innovation distribution and persist results."""

    logger.info("=" * 60)
    logger.info("STAGE 1: EGARCH BATCH ESTIMATION (MLE)")
    logger.info("Source dataset: %s", GARCH_DATASET_FILE)
    resid_train, resid_test = load_and_prepare_data()
    logger.info("Training observations: %d", resid_train.size)
    logger.info("Test observations (unused): %d", resid_test.size)

    # Log explicit variance initialization used by MLE
    var_init = initialize_variance(resid_train, None)
    logger.info("Initial variance for MLE init: %.6f", var_init)

    fits: dict[str, dict[str, Any]] = {}
    n_obs = int(resid_train.size)
    for dist in GARCH_OPTIMIZATION_DISTRIBUTIONS:
        logger.info("Estimating EGARCH(1,1) under %s innovations", dist)
        params, convergence = estimate_egarch_mle(resid_train, dist=dist, o=1, p=1)
        fits[dist] = _build_fit_entry(dist, params, convergence, n_obs, 1, 1)

    # Detailed AIC/BIC/loglik per distribution
    logger.info("AIC/BIC summary (Stage 1 - Estimation):")
    for dist, entry in fits.items():
        logger.info(
            "  %s → AIC=%.2f | BIC=%.2f | loglik=%.2f | converged=%s",
            dist,
            float(entry.get("aic", float("nan"))),
            float(entry.get("bic", float("nan"))),
            float(entry.get("log_likelihood", float("nan"))),
            bool(entry.get("converged", False)),
        )

    # Log winner by AIC before saving
    try:
        aic_items = [(d, float(fits[d]["aic"])) for d in fits if "aic" in fits[d]]
        if aic_items:
            best_dist, best_aic = min(aic_items, key=lambda x: x[1])
            logger.info("Best distribution by AIC: %s (AIC=%.2f)", best_dist, best_aic)
    except Exception as ex:
        logger.warning("Could not compute AIC winner: %s", ex)

    save_estimation_results(fits, n_observations=n_obs, output_path=GARCH_ESTIMATION_FILE)
    logger.info("Saved EGARCH estimation diagnostics to %s", GARCH_ESTIMATION_FILE)


def run_optimization_stage() -> None:
    """Optimize EGARCH hyperparameters with Optuna (no estimation warm-start)."""

    logger.info("=" * 60)
    logger.info("STAGE 2: EGARCH HYPERPARAMETER OPTIMIZATION")
    order_options = (1, 2)
    arch_options = (1, 2)
    base_combos = (
        len(order_options)
        * len(arch_options)
        * len(GARCH_OPTIMIZATION_DISTRIBUTIONS)
        * len(GARCH_OPTIMIZATION_REFIT_FREQ_OPTIONS)
    )
    window_breakdown: dict[str, int] = {}
    total_combos = 0
    for window_type in GARCH_OPTIMIZATION_WINDOW_TYPES:
        multiplier = 1
        if window_type == "rolling":
            multiplier = len(GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES)
        window_breakdown[window_type] = base_combos * multiplier
        total_combos += window_breakdown[window_type]
    logger.info(
        "Hyperparameter grid: o∈%s, p∈%s, distributions=%s, refit_freq=%s, window_types=%s",
        order_options,
        arch_options,
        GARCH_OPTIMIZATION_DISTRIBUTIONS,
        GARCH_OPTIMIZATION_REFIT_FREQ_OPTIONS,
        GARCH_OPTIMIZATION_WINDOW_TYPES,
    )
    logger.info(
        "Rolling window sizes (applies when window_type='rolling'): %s",
        GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES,
    )
    logger.info(
        "Grid size: %d combinations (%s)",
        total_combos,
        ", ".join(f"{name}={count}" for name, count in window_breakdown.items()),
    )
    resid_train, resid_test = load_and_prepare_data()
    logger.info("Training data: %d observations", resid_train.size)
    logger.info("Test data: %d observations (not used in optimization)", resid_test.size)
    # If no estimation baselines, continue silently without warm trials.

    n_trials = None
    results = optimize_egarch_hyperparameters(resid_train, n_trials=n_trials)

    results["source"] = str(GARCH_DATASET_FILE)
    results["methodology"] = "Optuna optimization with walk-forward CV on TRAIN"
    results["n_obs_train"] = int(resid_train.size)
    results["n_obs_test"] = int(resid_test.size)
    # No estimation baselines attached

    save_optimization_results(results, GARCH_OPTIMIZATION_RESULTS_FILE)
    logger.info("Optimization completed; results saved to: %s", GARCH_OPTIMIZATION_RESULTS_FILE)


def main() -> None:
    """Run full GARCH params pipeline: estimation then optimization."""
    run_estimation_stage()
    run_optimization_stage()


if __name__ == "__main__":
    main()
