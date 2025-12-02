"""
Labeling module for De Prado methodology.

This module provides implementations of labeling methods from
"Advances in Financial Machine Learning" by Marcos Lopez de Prado.

Modules:
--------
- label_primaire: Primary model labeling pipeline (direction prediction)
- label_meta: Meta-labeling for bet sizing (Chapter 3.6)
- triple_barriere: Numba-optimized triple-barrier labeling and relabeling
"""

from src.labelling.label_primaire import (
    # Core De Prado functions
    get_daily_volatility,
    get_events_primary_fast,
    # Optimization
    optimize_model,
    OptimizationConfig,
    OptimizationResult,
    WalkForwardCV,
    # Training
    train_model,
    load_optimized_params,
    TrainingConfig,
    TrainingResult,
    # Data utilities
    load_model_class,
    get_dataset_for_model,
    load_dollar_bars,
    # Registry
    MODEL_REGISTRY,
    RISK_REWARD_RATIO,
)

__all__ = [
    # Core De Prado functions (Numba-optimized)
    "get_daily_volatility",
    "get_events_primary_fast",
    # Optimization
    "optimize_model",
    "OptimizationConfig",
    "OptimizationResult",
    "WalkForwardCV",
    # Training
    "train_model",
    "load_optimized_params",
    "TrainingConfig",
    "TrainingResult",
    # Data utilities
    "load_model_class",
    "get_dataset_for_model",
    "load_dollar_bars",
    # Registry
    "MODEL_REGISTRY",
    "RISK_REWARD_RATIO",
]
