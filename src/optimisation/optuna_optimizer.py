"""Generic Optuna-based hyperparameter optimization.

This module provides a unified interface for hyperparameter optimization
using Optuna with walk-forward cross-validation for any model type.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Type

import numpy as np
import optuna
import pandas as pd  # type: ignore[import-untyped]

from src.constants import DEFAULT_RANDOM_STATE
from src.optimisation.hyperparams import HyperparamSpace, get_hyperparam_space
from src.optimisation.walk_forward_cv import (
    WalkForwardConfig,
    create_cv_config,
    walk_forward_cv,
)
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_N_TRIALS: int = 100
DEFAULT_TIMEOUT: int | None = None  # No timeout by default
DEFAULT_N_JOBS: int = 1


@dataclass
class OptimizationConfig:
    """Configuration for Optuna optimization.

    Attributes:
        n_trials: Number of optimization trials.
        timeout: Timeout in seconds (None = no timeout).
        n_jobs: Number of parallel jobs.
        sampler: Optuna sampler ("tpe", "random", "cmaes").
        pruner: Whether to use median pruning.
        direction: "minimize" or "maximize".
        study_name: Optional name for the study.
        random_state: Random seed for reproducibility.
    """

    n_trials: int = DEFAULT_N_TRIALS
    timeout: int | None = DEFAULT_TIMEOUT
    n_jobs: int = DEFAULT_N_JOBS
    sampler: str = "tpe"
    pruner: bool = False
    direction: str = "minimize"
    study_name: str | None = None
    random_state: int = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.sampler not in ("tpe", "random", "cmaes"):
            raise ValueError(f"sampler must be 'tpe', 'random', or 'cmaes', got {self.sampler}")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(f"direction must be 'minimize' or 'maximize', got {self.direction}")


# ============================================================================
# OPTIMIZATION RESULTS
# ============================================================================


@dataclass
class OptimizationResult:
    """Result from hyperparameter optimization.

    Attributes:
        best_params: Best hyperparameters found.
        best_score: Best objective value achieved.
        best_trial_number: Trial number of the best result.
        n_trials: Total number of trials run.
        n_completed: Number of completed trials.
        n_pruned: Number of pruned trials.
        cv_config: Cross-validation configuration used.
        optimization_config: Optimization configuration used.
        study: The Optuna study object.
    """

    best_params: dict[str, Any]
    best_score: float
    best_trial_number: int
    n_trials: int
    n_completed: int
    n_pruned: int
    cv_config: WalkForwardConfig
    optimization_config: OptimizationConfig
    study: optuna.Study


# ============================================================================
# MODEL FACTORY
# ============================================================================


def create_model_factory(
    model_class: Type[Any],
    fixed_params: dict[str, Any] | None = None,
) -> Callable[[dict[str, Any]], Any]:
    """Create a model factory function.

    Args:
        model_class: Model class to instantiate.
        fixed_params: Parameters that won't be optimized.

    Returns:
        Function that creates model instances with given params.
    """
    fixed_params = fixed_params or {}

    def factory(params: dict[str, Any]) -> Any:
        all_params = {**fixed_params, **params}
        return model_class(**all_params)

    return factory


# ============================================================================
# OPTUNA OPTIMIZER
# ============================================================================


def _create_sampler(config: OptimizationConfig) -> optuna.samplers.BaseSampler:
    """Create Optuna sampler based on configuration."""
    if config.sampler == "tpe":
        return optuna.samplers.TPESampler(seed=config.random_state)
    elif config.sampler == "random":
        return optuna.samplers.RandomSampler(seed=config.random_state)
    elif config.sampler == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=config.random_state)
    else:
        raise ValueError(f"Unknown sampler: {config.sampler}")


def _create_pruner(config: OptimizationConfig) -> optuna.pruners.BasePruner | None:
    """Create Optuna pruner based on configuration."""
    if config.pruner:
        return optuna.pruners.MedianPruner()
    return optuna.pruners.NopPruner()


def _create_study(config: OptimizationConfig) -> optuna.Study:
    """Create Optuna study with configured sampler and pruner."""
    return optuna.create_study(
        direction=config.direction,
        sampler=_create_sampler(config),
        pruner=_create_pruner(config),
        study_name=config.study_name,
    )


class OptunaOptimizer:
    """Generic hyperparameter optimizer using Optuna.

    This optimizer works with any model that implements the FittableModel protocol
    (fit, predict, set_params, get_params methods).

    Example:
        >>> from src.model.machine_learning.xgboost_model import XGBoostModel
        >>> from src.optimisation.hyperparams import XGBoostHyperparams
        >>>
        >>> optimizer = OptunaOptimizer(
        ...     model_class=XGBoostModel,
        ...     hyperparam_space=XGBoostHyperparams(),
        ...     cv_config=create_cv_config(n_splits=5, purge_gap=5),
        ...     optimization_config=OptimizationConfig(n_trials=50),
        ... )
        >>> result = optimizer.optimize(X_train, y_train, metric="mse")
    """

    def __init__(
        self,
        model_class: Type[Any],
        hyperparam_space: HyperparamSpace,
        cv_config: WalkForwardConfig,
        optimization_config: OptimizationConfig | None = None,
        fixed_model_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            model_class: Class of the model to optimize.
            hyperparam_space: Hyperparameter search space.
            cv_config: Walk-forward CV configuration.
            optimization_config: Optuna configuration.
            fixed_model_params: Model params that won't be optimized.
        """
        self.model_class = model_class
        self.hyperparam_space = hyperparam_space
        self.cv_config = cv_config
        self.optimization_config = optimization_config or OptimizationConfig()
        self.fixed_model_params = fixed_model_params or {}

        self._study: optuna.Study | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._metric: str = "mse"
        self._fit_kwargs: dict[str, Any] = {}

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial object.

        Returns:
            Cross-validation score.
        """
        if self._X is None or self._y is None:
            raise RuntimeError("Data not set. Call optimize() first.")

        # Suggest hyperparameters
        params = self.hyperparam_space.suggest(trial)

        # Create model with suggested params
        all_params = {**self.fixed_model_params, **params}
        model = self.model_class(**all_params)

        # Run walk-forward CV
        try:
            cv_result = walk_forward_cv(
                model=model,
                X=self._X,
                y=self._y,
                config=self.cv_config,
                metric=self._metric,
                verbose=False,
                fit_kwargs=self._fit_kwargs,
            )

            if not np.isfinite(cv_result.mean_score):
                raise optuna.TrialPruned()

            return cv_result.mean_score

        except Exception as e:
            logger.debug("Trial %d failed: %s", trial.number, e)
            raise optuna.TrialPruned() from e

    def optimize(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        metric: str = "mse",
        fit_kwargs: dict[str, Any] | None = None,
        show_progress_bar: bool = True,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run hyperparameter optimization.

        Args:
            X: Training features.
            y: Training target.
            metric: Metric to optimize ("mse", "rmse", "mae", "qlike").
            fit_kwargs: Additional kwargs for model.fit().
            show_progress_bar: Show Optuna progress bar.
            verbose: Log optimization progress.

        Returns:
            OptimizationResult with best params and study info.
        """
        # Store data for objective function
        self._X = np.asarray(X)
        self._y = np.asarray(y).ravel()
        self._metric = metric
        self._fit_kwargs = fit_kwargs or {}

        # Set random seed
        np.random.seed(self.optimization_config.random_state)

        # Create study
        self._study = _create_study(self.optimization_config)

        if verbose:
            logger.info(
                "Starting optimization: %d trials, metric=%s, %d CV folds, purge_gap=%d",
                self.optimization_config.n_trials,
                metric,
                self.cv_config.n_splits,
                self.cv_config.purge_gap,
            )

        # Suppress Optuna logs if not verbose
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        self._study.optimize(
            self._objective,
            n_trials=self.optimization_config.n_trials,
            timeout=self.optimization_config.timeout,
            n_jobs=self.optimization_config.n_jobs,
            show_progress_bar=show_progress_bar,
        )

        # Restore logging
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        # Count trial states
        completed = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED])

        if completed == 0:
            raise RuntimeError("No trials completed successfully")

        best_trial = self._study.best_trial

        if best_trial.value is None:
            raise RuntimeError("Best trial has no value")

        best_score: float = best_trial.value

        if verbose:
            logger.info(
                "Optimization complete: best %s=%.6f (trial %d)",
                metric,
                best_score,
                best_trial.number,
            )
            logger.info("Best parameters: %s", best_trial.params)

        return OptimizationResult(
            best_params=best_trial.params,
            best_score=best_score,
            best_trial_number=best_trial.number,
            n_trials=len(self._study.trials),
            n_completed=completed,
            n_pruned=pruned,
            cv_config=self.cv_config,
            optimization_config=self.optimization_config,
            study=self._study,
        )

    @property
    def study(self) -> optuna.Study | None:
        """Get the Optuna study."""
        return self._study


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def optimize_model(
    model_class: Type[Any],
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    model_name: str | None = None,
    hyperparam_space: HyperparamSpace | None = None,
    n_splits: int = 5,
    purge_gap: int = 1,
    min_train_size: int = 100,
    n_trials: int = 100,
    metric: str = "mse",
    fixed_params: dict[str, Any] | None = None,
    verbose: bool = True,
) -> OptimizationResult:
    """Convenience function to optimize a model.

    Args:
        model_class: Model class to optimize.
        X: Training features.
        y: Training target.
        model_name: Name to lookup default hyperparams (if space not given).
        hyperparam_space: Custom hyperparameter space.
        n_splits: Number of CV splits.
        purge_gap: Samples to purge between train/test.
        min_train_size: Minimum training size.
        n_trials: Number of Optuna trials.
        metric: Metric to optimize.
        fixed_params: Model params that won't be optimized.
        verbose: Log progress.

    Returns:
        OptimizationResult with best parameters.

    Example:
        >>> from src.model.machine_learning.xgboost_model import XGBoostModel
        >>> result = optimize_model(
        ...     XGBoostModel,
        ...     X_train, y_train,
        ...     model_name="xgboost",
        ...     n_splits=5,
        ...     purge_gap=5,
        ...     n_trials=50,
        ... )
        >>> print(result.best_params)
    """
    # Get hyperparameter space
    if hyperparam_space is None:
        if model_name is None:
            # Try to infer from class name
            model_name = model_class.__name__.lower().replace("model", "")
        hyperparam_space = get_hyperparam_space(model_name)

    # Create configs
    cv_config = create_cv_config(
        n_splits=n_splits,
        purge_gap=purge_gap,
        min_train_size=min_train_size,
    )
    opt_config = OptimizationConfig(n_trials=n_trials)

    # Run optimization
    optimizer = OptunaOptimizer(
        model_class=model_class,
        hyperparam_space=hyperparam_space,
        cv_config=cv_config,
        optimization_config=opt_config,
        fixed_model_params=fixed_params,
    )

    return optimizer.optimize(X, y, metric=metric, verbose=verbose)


def save_optimization_results(
    result: OptimizationResult,
    output_file: str | Path,
) -> None:
    """Save optimization results to JSON file.

    Args:
        result: Optimization result to save.
        output_file: Output file path.
    """
    output_path = Path(output_file)

    data = {
        "best_params": result.best_params,
        "best_score": result.best_score,
        "best_trial_number": result.best_trial_number,
        "n_trials": result.n_trials,
        "n_completed": result.n_completed,
        "n_pruned": result.n_pruned,
        "cv_config": {
            "n_splits": result.cv_config.n_splits,
            "purge_gap": result.cv_config.purge_gap,
            "min_train_size": result.cv_config.min_train_size,
            "window_type": result.cv_config.window_type,
            "rolling_window_size": result.cv_config.rolling_window_size,
        },
        "optimization_config": {
            "n_trials": result.optimization_config.n_trials,
            "sampler": result.optimization_config.sampler,
            "direction": result.optimization_config.direction,
        },
    }

    save_json_pretty(data, output_path)
    logger.info("Saved optimization results to %s", output_path)


def load_and_apply_best_params(
    model_class: Type[Any],
    result: OptimizationResult,
    fixed_params: dict[str, Any] | None = None,
) -> Any:
    """Create a model with the best parameters from optimization.

    Args:
        model_class: Model class to instantiate.
        result: Optimization result with best params.
        fixed_params: Additional fixed parameters.

    Returns:
        Model instance with best parameters.
    """
    all_params = {**(fixed_params or {}), **result.best_params}
    return model_class(**all_params)
