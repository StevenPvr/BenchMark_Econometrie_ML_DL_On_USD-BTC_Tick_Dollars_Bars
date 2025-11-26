"""Hyperparameter optimization module with walk-forward cross-validation.

This module provides a unified interface for hyperparameter optimization
compatible with all model types in the project:
- Econometric models (Ridge, Lasso)
- Machine Learning models (XGBoost, LightGBM, CatBoost, RandomForest)
- Deep Learning models (LSTM)

Key Features:
- Walk-forward cross-validation with purge gap to prevent data leakage
- Optuna-based hyperparameter search
- Pre-defined search spaces for all model types
- Support for expanding and rolling windows

Example Usage:
    >>> from src.model.machine_learning.xgboost_model import XGBoostModel
    >>> from src.optimisation import optimize_model, create_cv_config
    >>>
    >>> # Quick optimization
    >>> result = optimize_model(
    ...     XGBoostModel,
    ...     X_train, y_train,
    ...     model_name="xgboost",
    ...     n_splits=5,
    ...     purge_gap=5,
    ...     n_trials=50,
    ... )
    >>> print(f"Best MSE: {result.best_score}")
    >>> print(f"Best params: {result.best_params}")
    >>>
    >>> # Use best params to create optimized model
    >>> best_model = XGBoostModel(**result.best_params)
    >>> best_model.fit(X_train, y_train)

For LSTM models:
    >>> from src.model.deep_learning.lstm_model import LSTMModel
    >>> result = optimize_model(
    ...     LSTMModel,
    ...     X_train, y_train,
    ...     model_name="lstm",
    ...     n_splits=3,
    ...     purge_gap=10,  # Larger purge for sequence models
    ...     n_trials=30,
    ... )

For custom search spaces:
    >>> from src.optimisation import (
    ...     OptunaOptimizer,
    ...     XGBoostHyperparams,
    ...     create_cv_config,
    ...     OptimizationConfig,
    ... )
    >>>
    >>> # Custom hyperparameter ranges
    >>> custom_space = XGBoostHyperparams(
    ...     n_estimators_range=(100, 200),
    ...     max_depth_range=(3, 6),
    ... )
    >>>
    >>> optimizer = OptunaOptimizer(
    ...     model_class=XGBoostModel,
    ...     hyperparam_space=custom_space,
    ...     cv_config=create_cv_config(n_splits=5, purge_gap=5),
    ...     optimization_config=OptimizationConfig(n_trials=100),
    ... )
    >>> result = optimizer.optimize(X_train, y_train)
"""

from __future__ import annotations

# Walk-forward cross-validation
from src.optimisation.walk_forward_cv import (
    CVFold,
    CVResult,
    FittableModel,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardSplitter,
    create_cv_config,
    mae,
    mse,
    qlike,
    rmse,
    walk_forward_cv,
)

# Hyperparameter spaces
from src.optimisation.hyperparams import (
    CatBoostHyperparams,
    HyperparamSpace,
    LassoHyperparams,
    LightGBMHyperparams,
    LSTMHyperparams,
    RandomForestHyperparams,
    RidgeHyperparams,
    XGBoostHyperparams,
    get_hyperparam_space,
    list_available_models,
)

# Optuna optimizer
from src.optimisation.optuna_optimizer import (
    OptimizationConfig,
    OptimizationResult,
    OptunaOptimizer,
    load_and_apply_best_params,
    optimize_model,
    save_optimization_results,
)

__all__ = [
    # Walk-forward CV
    "CVFold",
    "CVResult",
    "FittableModel",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardSplitter",
    "create_cv_config",
    "walk_forward_cv",
    # Metrics
    "mae",
    "mse",
    "qlike",
    "rmse",
    # Hyperparameter spaces
    "CatBoostHyperparams",
    "HyperparamSpace",
    "LassoHyperparams",
    "LightGBMHyperparams",
    "LSTMHyperparams",
    "RandomForestHyperparams",
    "RidgeHyperparams",
    "XGBoostHyperparams",
    "get_hyperparam_space",
    "list_available_models",
    # Optuna optimizer
    "OptimizationConfig",
    "OptimizationResult",
    "OptunaOptimizer",
    "load_and_apply_best_params",
    "optimize_model",
    "save_optimization_results",
]
