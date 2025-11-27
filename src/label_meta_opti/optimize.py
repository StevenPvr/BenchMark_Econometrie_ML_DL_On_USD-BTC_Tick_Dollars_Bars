"""Meta-model hyperparameter optimization.

This module optimizes LightGBM hyperparameters for the meta-model,
which is a binary classifier predicting whether a primary model's
trade signal will be correct (1) or wrong (0).

The optimization is done separately from the primary model to allow
different hyperparameters that are better suited for binary classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE
from src.optimisation.walk_forward_cv import (
    create_cv_config,
    walk_forward_cv,
)
from src.utils import save_json_pretty

logger = get_logger(__name__)

__all__ = [
    "LightGBMMetaHyperparams",
    "MetaOptimizationConfig",
    "MetaOptimizationResult",
    "MetaOptimizer",
    "optimize_meta_model",
    "save_meta_optimization_results",
]


@dataclass
class LightGBMMetaHyperparams:
    """Hyperparameter space for LightGBM meta-model (binary classification).

    Note: This is separate from the primary model's hyperparameter space
    to allow independent optimization for the binary classification task.
    """

    learning_rate_range: tuple[float, float] = (0.01, 0.3)
    num_leaves_range: tuple[int, int] = (10, 100)
    max_depth_range: tuple[int, int] = (3, 8)
    min_data_in_leaf_range: tuple[int, int] = (10, 100)
    feature_fraction_range: tuple[float, float] = (0.5, 1.0)
    bagging_fraction_range: tuple[float, float] = (0.5, 1.0)
    reg_alpha_range: tuple[float, float] = (0.0, 1.0)
    reg_lambda_range: tuple[float, float] = (0.0, 1.0)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest LightGBM hyperparameters for a trial."""
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", self.learning_rate_range[0], self.learning_rate_range[1], log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", self.num_leaves_range[0], self.num_leaves_range[1]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", self.max_depth_range[0], self.max_depth_range[1]
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", self.min_data_in_leaf_range[0], self.min_data_in_leaf_range[1]
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", self.feature_fraction_range[0], self.feature_fraction_range[1]
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", self.bagging_fraction_range[0], self.bagging_fraction_range[1]
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", self.reg_alpha_range[0], self.reg_alpha_range[1]
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", self.reg_lambda_range[0], self.reg_lambda_range[1]
            ),
            "bagging_freq": 1,
            "verbosity": -1,
            "random_state": DEFAULT_RANDOM_STATE,
        }

    def get_param_keys(self) -> set[str]:
        """Return the set of parameter keys for this model."""
        return {
            "learning_rate", "num_leaves", "max_depth",
            "min_data_in_leaf", "feature_fraction", "bagging_fraction",
            "reg_alpha", "reg_lambda"
        }


@dataclass
class MetaOptimizationConfig:
    """Configuration for meta-model optimization.

    Attributes:
        n_trials: Number of Optuna trials.
        n_splits: Number of CV splits.
        purge_gap: Gap between train/test in CV.
        min_train_size: Minimum training size.
        direction: Optimization direction.
        sampler: Optuna sampler type.
        random_state: Random seed.
        metric: Metric to optimize (mcc, f1, accuracy, roc_auc).
    """

    n_trials: int = 100
    n_splits: int = 5
    purge_gap: int = 5
    min_train_size: int = 200
    direction: str = "maximize"
    sampler: str = "tpe"
    random_state: int = DEFAULT_RANDOM_STATE
    metric: str = "mcc"


@dataclass
class MetaOptimizationResult:
    """Result from meta-model optimization.

    Attributes:
        best_params: Best hyperparameters for the meta-model.
        best_score: Best score achieved.
        best_trial_number: Trial number of best result.
        n_trials: Total trials.
        n_completed: Completed trials.
        metric: Metric that was optimized.
        study: Optuna study object.
    """

    best_params: dict[str, Any]
    best_score: float
    best_trial_number: int
    n_trials: int
    n_completed: int
    metric: str
    study: optuna.Study


def compute_class_weights(y: np.ndarray | pd.Series) -> dict[Any, float]:
    """Balanced class weights: n_samples / (n_classes * count_c)."""
    y_arr = np.asarray(y).ravel()
    classes, counts = np.unique(y_arr, return_counts=True)
    n_samples = len(y_arr)

    if len(classes) == 0:
        return {}

    weights = n_samples / (len(classes) * counts.astype(float))
    return {cls: float(w) for cls, w in zip(classes, weights)}


class MetaOptimizer:
    """Optimizer for meta-model hyperparameters.

    This optimizer finds optimal LightGBM hyperparameters for the
    binary classification task of predicting trade correctness.
    """

    def __init__(
        self,
        hyperparams: LightGBMMetaHyperparams | None = None,
        config: MetaOptimizationConfig | None = None,
    ) -> None:
        """Initialize the meta optimizer.

        Args:
            hyperparams: Hyperparameter space.
            config: Optimization configuration.
        """
        self.hyperparams = hyperparams or LightGBMMetaHyperparams()
        self.config = config or MetaOptimizationConfig()

        self._study: optuna.Study | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        if self._X is None or self._y is None:
            raise RuntimeError("Data not set.")

        from src.model.machine_learning.lightgbm.lightgbm_model import LightGBMModel

        # Suggest hyperparameters
        params = self.hyperparams.suggest(trial)

        # Add class weights for imbalanced data
        class_weight = compute_class_weights(self._y)
        params["class_weight"] = class_weight

        # Create CV config
        cv_config = create_cv_config(
            n_splits=self.config.n_splits,
            purge_gap=self.config.purge_gap,
            min_train_size=self.config.min_train_size,
        )

        try:
            model = LightGBMModel(**params)
            cv_result = walk_forward_cv(
                model=model,
                X=self._X,
                y=self._y,
                config=cv_config,
                metric=self.config.metric,
                verbose=False,
            )

            score = cv_result.mean_score

            if not np.isfinite(score):
                raise optuna.TrialPruned()

            return score

        except Exception as e:
            logger.debug("Trial %d: CV failed: %s", trial.number, e)
            raise optuna.TrialPruned() from e

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        show_progress_bar: bool = True,
        verbose: bool = True,
    ) -> MetaOptimizationResult:
        """Run meta-model optimization.

        Args:
            X: Feature matrix (meta-features including primary signal).
            y: Binary meta-labels (0=wrong, 1=correct).
            show_progress_bar: Show progress bar.
            verbose: Log progress.

        Returns:
            MetaOptimizationResult with best params.
        """
        self._X = np.asarray(X)
        self._y = np.asarray(y).astype(int)

        np.random.seed(self.config.random_state)

        if self.config.sampler == "tpe":
            sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(seed=self.config.random_state)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.config.random_state)

        self._study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            study_name="meta_model_optimization",
        )

        if verbose:
            logger.info(
                "Starting meta-model optimization: %d trials, metric=%s",
                self.config.n_trials,
                self.config.metric,
            )
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            show_progress_bar=show_progress_bar,
        )

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        completed = len([
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])

        if completed == 0:
            raise RuntimeError("No trials completed")

        best_trial = self._study.best_trial

        if best_trial.value is None:
            raise RuntimeError("Best trial has no value")

        best_score: float = best_trial.value

        # Extract best params (only model params, not class_weight)
        best_params = {
            k: v for k, v in best_trial.params.items()
            if k in self.hyperparams.get_param_keys()
        }
        best_params["random_state"] = self.config.random_state

        if verbose:
            logger.info("Optimization complete: best %s=%.6f", self.config.metric, best_score)
            logger.info("Best meta-model params: %s", best_params)

        return MetaOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_trial_number=best_trial.number,
            n_trials=len(self._study.trials),
            n_completed=completed,
            metric=self.config.metric,
            study=self._study,
        )


def optimize_meta_model(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 100,
    n_splits: int = 5,
    purge_gap: int = 5,
    metric: str = "mcc",
    verbose: bool = True,
) -> MetaOptimizationResult:
    """Convenience function for meta-model optimization.

    Args:
        X: Feature matrix (meta-features).
        y: Binary meta-labels.
        n_trials: Number of trials.
        n_splits: CV splits.
        purge_gap: Gap in CV.
        metric: Metric to optimize.
        verbose: Log progress.

    Returns:
        MetaOptimizationResult with best params.
    """
    optimizer = MetaOptimizer(
        config=MetaOptimizationConfig(
            n_trials=n_trials,
            n_splits=n_splits,
            purge_gap=purge_gap,
            metric=metric,
        ),
    )

    return optimizer.optimize(
        X=X,
        y=y,
        verbose=verbose,
    )


def save_meta_optimization_results(
    result: MetaOptimizationResult,
    output_path: Path | str,
) -> None:
    """Save meta optimization results to JSON.

    Args:
        result: Optimization result.
        output_path: Output file path.
    """
    output = {
        "best_params": result.best_params,
        "best_score": result.best_score,
        "best_trial_number": result.best_trial_number,
        "n_trials": result.n_trials,
        "n_completed": result.n_completed,
        "metric": result.metric,
    }
    save_json_pretty(output, output_path)
    logger.info("Saved meta optimization results to %s", output_path)
