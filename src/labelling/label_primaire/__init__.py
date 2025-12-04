"""Label Primaire - Primary Model Pipeline (De Prado Chapter 3).

This module implements the PRIMARY model that predicts trade DIRECTION.

The primary model learns to predict:
- +1 (Long): Price will increase
- -1 (Short): Price will decrease
- 0 (Neutral): No significant move expected

Pipeline:
1. Triple barrier labeling is done SEPARATELY by triple_barriere/relabel_datasets.py
2. opti: Optimize primary model hyperparameters (uses pre-calculated labels)
3. train: Train primary model with optimized parameters
4. eval: Evaluate primary model performance (TODO)

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
"""

from src.labelling.label_primaire.utils import (
    # Registry
    MODEL_REGISTRY,
    RISK_REWARD_RATIO,
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

# Fast Numba-optimized labeling (use this for triple-barrier events)
from src.labelling.triple_barriere import get_events_primary_fast

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
