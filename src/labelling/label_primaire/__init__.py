"""
Label Primaire - Primary Model Pipeline (De Prado Chapter 3).

This module implements the PRIMARY model that predicts trade DIRECTION.

The primary model learns to predict:
- +1 (Long): Price will increase
- -1 (Short): Price will decrease
- 0 (Neutral): No significant move expected

Pipeline:
1. opti: Optimize primary model hyperparameters + triple barrier params
2. train: Train primary model with optimized parameters
3. eval: Evaluate primary model performance (TODO)

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
"""

from src.labelling.label_primaire.utils import (
    # Registry
    MODEL_REGISTRY,
    TRIPLE_BARRIER_SEARCH_SPACE,
    # Dataclasses
    OptimizationConfig,
    OptimizationResult,
    # Data utilities
    load_model_class,
    get_dataset_for_model,
    load_dollar_bars,
    # Volatility
    get_daily_volatility,
)

from src.labelling.label_primaire.opti import (
    # Core De Prado functions
    apply_pt_sl_on_t1,
    get_events_primary,
    # Optimization
    optimize_model,
    WalkForwardCV,
)

from src.labelling.label_primaire.train import (
    train_model,
    load_optimized_params,
    TrainingConfig,
    TrainingResult,
)

__all__ = [
    # Core De Prado functions
    "get_daily_volatility",
    "apply_pt_sl_on_t1",
    "get_events_primary",
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
    "TRIPLE_BARRIER_SEARCH_SPACE",
]
