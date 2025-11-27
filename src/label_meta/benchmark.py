"""Benchmarking module for primary models with pre-trained meta-model.

This module handles benchmarking different primary models using a
pre-trained meta-model for arbitrage.

The workflow:
1. Meta-model is trained ONCE (via train_meta_model_once)
2. For each primary model in benchmark:
   - Optimize primary model params (optional)
   - Train primary model
   - Apply pre-trained meta-model for arbitrage
   - Compute metrics
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE
from src.label_meta.config import BenchmarkConfig, BenchmarkResult
from src.label_meta.meta_labeling import get_meta_labels
from src.optimisation.hyperparams import get_hyperparam_space
from src.optimisation.optuna_optimizer import OptunaOptimizer, OptimizationConfig
from src.optimisation.walk_forward_cv import WalkForwardConfig

logger = get_logger(__name__)

__all__ = [
    "run_benchmark_for_primary",
    "compute_metrics",
]


def _get_model_class(model_name: str) -> type:
    """Get model class by name."""
    model_name = model_name.lower()

    if model_name == "lightgbm":
        from src.model.machine_learning.lightgbm.lightgbm_model import LightGBMModel
        return LightGBMModel

    elif model_name == "xgboost":
        from src.model.machine_learning.xgboost.xgboost_model import XGBoostModel
        return XGBoostModel

    elif model_name == "catboost":
        from src.model.machine_learning.catboost.catboost_model import CatBoostModel
        return CatBoostModel

    elif model_name in ("rf", "random_forest", "randomforest"):
        from src.model.machine_learning.rf.random_forest_model import RandomForestModel
        return RandomForestModel

    elif model_name == "ridge":
        from src.model.econometrie.ridge.ridge import RidgeModel
        return RidgeModel

    elif model_name == "lasso":
        from src.model.econometrie.lasso.lasso import LassoModel
        return LassoModel

    elif model_name in ("logistic", "logistic_regression"):
        from src.model.econometrie.logistic.logistic import LogisticModel
        return LogisticModel

    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary of metrics.
    """
    # Filter valid (non-NaN) predictions
    valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_true_valid = y_true[valid_mask].astype(int)
    y_pred_valid = y_pred[valid_mask].astype(int)

    if len(y_true_valid) == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "mcc": 0.0,
            "n_samples": 0,
        }

    return {
        "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
        "f1_macro": float(f1_score(
            y_true_valid,
            y_pred_valid,
            average="macro",
            zero_division="warn",
        )),
        "mcc": float(matthews_corrcoef(y_true_valid, y_pred_valid)),
        "n_samples": len(y_true_valid),
    }


def run_benchmark_for_primary(
    X_train: np.ndarray | pd.DataFrame,
    y_train: pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
    events_train: pd.DataFrame,
    events_test: pd.DataFrame,
    meta_model: Any,
    config: BenchmarkConfig,
    meta_params: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run benchmark for a single primary model.

    This function optimizes (optionally) and trains a primary model,
    then applies the pre-trained meta-model for arbitrage.

    Args:
        X_train: Training features.
        y_train: Training labels (-1, 0, +1).
        X_test: Test features.
        y_test: Test labels (-1, 0, +1).
        events_train: Training events with 'ret' column.
        events_test: Test events with 'ret' column.
        meta_model: Pre-trained meta-model.
        config: Benchmark configuration.
        meta_params: Meta-model parameters (for result tracking).

    Returns:
        BenchmarkResult with all predictions and metrics.
    """
    if config.verbose:
        logger.info("=" * 60)
        logger.info("BENCHMARK: %s", config.primary_model_name.upper())
        logger.info("=" * 60)

    # Get model class
    model_class = _get_model_class(config.primary_model_name)

    # Filter valid labels
    valid_train_mask = ~y_train.isna()
    valid_train_mask_arr = np.asarray(valid_train_mask)
    X_train_valid = np.asarray(X_train)[valid_train_mask_arr]
    y_train_valid = np.asarray(y_train[valid_train_mask]).astype(int)

    valid_test_mask = ~y_test.isna()
    valid_test_mask_arr = np.asarray(valid_test_mask)
    X_test_valid = np.asarray(X_test)[valid_test_mask_arr]
    y_test_valid = np.asarray(y_test[valid_test_mask]).astype(int)

    # Optimize primary model params (optional)
    if config.skip_primary_optim:
        primary_params: dict[str, Any] = dict(config.primary_params or {})
        if config.verbose:
            if primary_params:
                logger.info("Using provided primary params (skip optimization): %s", primary_params)
            else:
                logger.info("Skipping primary model optimization (no params provided)")
    else:
        if config.verbose:
            logger.info("Optimizing %s params (%d trials)...",
                       config.primary_model_name, config.n_trials)

        # If provided (e.g., from label_primaire), reuse them instead of re-optimizing
        if config.primary_params:
            primary_params = dict(config.primary_params)
            if config.verbose:
                logger.info("Using provided primary params: %s", primary_params)
        else:
            primary_params = _optimize_primary_model(
                model_class=model_class,
                model_name=config.primary_model_name,
                X=X_train_valid,
                y=y_train_valid,
                n_trials=config.n_trials,
                n_splits=config.n_splits,
                purge_gap=config.purge_gap,
                random_state=config.random_state,
            )

        if config.verbose:
            logger.info("Best params: %s", primary_params)

    # Train primary model
    if config.verbose:
        logger.info("Training %s on %d samples...",
                   config.primary_model_name, len(y_train_valid))

    cleaned_params = dict(primary_params)
    cleaned_params.pop("random_state", None)  # avoid duplicate

    primary_model = model_class(random_state=config.random_state, **cleaned_params)
    primary_model.fit(X_train_valid, y_train_valid)

    # Get primary predictions
    train_predictions = primary_model.predict(X_train_valid)
    test_predictions = primary_model.predict(X_test_valid)

    # Compute train metrics
    train_metrics = compute_metrics(y_train_valid, train_predictions)

    if config.verbose:
        logger.info("Train metrics: %s", train_metrics)

    # Compute test metrics (primary only)
    test_metrics = compute_metrics(y_test_valid, test_predictions)

    if config.verbose:
        logger.info("Test metrics (primary): %s", test_metrics)

    # Apply meta-model for arbitrage
    if config.verbose:
        logger.info("Applying meta-model arbitrage...")

    # Create meta-features for test set
    X_meta_test = np.column_stack([X_test_valid, test_predictions])

    # Get meta-model predictions (0 = don't trade, 1 = trade)
    meta_predictions = meta_model.predict(X_meta_test)

    # Apply arbitrage: final = primary * meta
    final_predictions = test_predictions * meta_predictions

    # Compute final metrics (after arbitrage)
    final_metrics = compute_metrics(y_test_valid, final_predictions)

    if config.verbose:
        logger.info("Test metrics (after meta): %s", final_metrics)
        logger.info("Trades filtered: %d / %d (%.1f%%)",
                   (meta_predictions == 0).sum(),
                   len(meta_predictions),
                   100 * (meta_predictions == 0).sum() / len(meta_predictions))

    # Build result
    result = BenchmarkResult(
        primary_model_name=config.primary_model_name,
        primary_params=primary_params,
        meta_params=meta_params or {},
        primary_model=primary_model,
        meta_model=meta_model,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        meta_predictions=meta_predictions,
        final_predictions=final_predictions,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        final_metrics=final_metrics,
    )

    if config.verbose:
        logger.info("=" * 60)
        _print_benchmark_summary(result)
        logger.info("=" * 60)

    return result


def _optimize_primary_model(
    model_class: type,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    n_splits: int,
    purge_gap: int,
    random_state: int,
) -> dict[str, Any]:
    """Optimize primary model hyperparameters.

    Args:
        model_class: Model class to optimize.
        model_name: Name of the model for hyperparameter space lookup.
        X: Training features.
        y: Training labels.
        n_trials: Number of Optuna trials.
        n_splits: Number of CV splits.
        purge_gap: Gap between train/test in CV.
        random_state: Random seed.

    Returns:
        Best hyperparameters.
    """
    cv_config = WalkForwardConfig(
        n_splits=n_splits,
        purge_gap=purge_gap,
    )

    hyperparam_space = get_hyperparam_space(model_name)

    optimization_config = OptimizationConfig(
        n_trials=n_trials,
        random_state=random_state,
        direction="maximize",
    )

    optimizer = OptunaOptimizer(
        model_class=model_class,
        hyperparam_space=hyperparam_space,
        cv_config=cv_config,
        optimization_config=optimization_config,
    )

    result = optimizer.optimize(X=X, y=y, metric="mcc", verbose=False)
    return result.best_params


def _print_benchmark_summary(result: BenchmarkResult) -> None:
    """Print benchmark summary."""
    logger.info("BENCHMARK SUMMARY: %s", result.primary_model_name.upper())
    logger.info("-" * 40)
    logger.info("Primary params: %s", result.primary_params)
    logger.info("")
    logger.info("TRAIN:")
    logger.info("  Accuracy: %.4f", result.train_metrics.get("accuracy", 0))
    logger.info("  F1 (macro): %.4f", result.train_metrics.get("f1_macro", 0))
    logger.info("  MCC: %.4f", result.train_metrics.get("mcc", 0))
    logger.info("")
    logger.info("TEST (Primary only):")
    logger.info("  Accuracy: %.4f", result.test_metrics.get("accuracy", 0))
    logger.info("  F1 (macro): %.4f", result.test_metrics.get("f1_macro", 0))
    logger.info("  MCC: %.4f", result.test_metrics.get("mcc", 0))
    logger.info("")
    logger.info("TEST (After meta-arbitrage):")
    logger.info("  Accuracy: %.4f", result.final_metrics.get("accuracy", 0))
    logger.info("  F1 (macro): %.4f", result.final_metrics.get("f1_macro", 0))
    logger.info("  MCC: %.4f", result.final_metrics.get("mcc", 0))
    logger.info("")
    logger.info("Improvement from meta-model:")
    primary_mcc = result.test_metrics.get("mcc", 0)
    final_mcc = result.final_metrics.get("mcc", 0)
    improvement = final_mcc - primary_mcc
    logger.info("  MCC: %+.4f (%s)", improvement,
               "improved" if improvement > 0 else "degraded" if improvement < 0 else "unchanged")
