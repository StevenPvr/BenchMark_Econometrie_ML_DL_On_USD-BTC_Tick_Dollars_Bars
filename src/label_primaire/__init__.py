"""Label Primaire - Triple-Barrier Labeling Module.

This module handles:
    1. Triple-barrier labeling (De Prado methodology)
    2. Joint optimization of labeling + primary model parameters
    3. Label generation on full dataset

Supported primary models:
    - lightgbm (default)
    - xgboost
    - catboost
    - rf (Random Forest)
    - ridge
    - lasso
    - logistic

Usage:
    python -m src.label_primaire.main

The optimized model parameters are saved and reused by the
label_meta module for the meta-model.
"""

from src.label_primaire.config import (
    LabelGenerationConfig,
    LabelingOptimizationConfig,
    LabelingOptimizationResult,
)
from src.label_primaire.optimize import (
    AVAILABLE_MODELS,
    CatBoostHyperparams,
    JointOptimizationConfig,
    JointOptimizationResult,
    JointOptimizer,
    LabelingHyperparams,
    LassoHyperparams,
    LightGBMHyperparams,
    LogisticHyperparams,
    ModelHyperparams,
    RandomForestHyperparams,
    RidgeHyperparams,
    XGBoostHyperparams,
    compute_class_weights,
    compute_sharpe_ratio,
    get_model_class,
    get_model_hyperparams,
    normalize_volatility_scale,
    optimize_joint_params,
    save_optimization_results,
)
from src.label_primaire.triple_barrier import (
    apply_triple_barrier_labels,
    get_triple_barrier_events,
    get_vertical_barriers,
)

__all__ = [
    # Triple-barrier
    "get_triple_barrier_events",
    "get_vertical_barriers",
    "apply_triple_barrier_labels",
    # Configuration
    "LabelingHyperparams",
    "LabelingOptimizationConfig",
    "LabelingOptimizationResult",
    "LabelGenerationConfig",
    # Model hyperparams
    "ModelHyperparams",
    "LightGBMHyperparams",
    "XGBoostHyperparams",
    "CatBoostHyperparams",
    "RandomForestHyperparams",
    "RidgeHyperparams",
    "LassoHyperparams",
    "LogisticHyperparams",
    # Joint optimization
    "JointOptimizationConfig",
    "JointOptimizationResult",
    "JointOptimizer",
    "optimize_joint_params",
    "save_optimization_results",
    # Utilities
    "compute_class_weights",
    "compute_sharpe_ratio",
    "normalize_volatility_scale",
    "get_model_class",
    "get_model_hyperparams",
    "AVAILABLE_MODELS",
]
