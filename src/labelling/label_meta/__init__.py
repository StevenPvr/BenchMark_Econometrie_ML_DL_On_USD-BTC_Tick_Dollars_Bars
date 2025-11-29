"""
Label Meta - Meta-Labeling Pipeline (De Prado Chapter 3.6).

This module implements the META model that filters false positives
from the primary model's predictions.

The meta model learns to predict:
- 1: Take the trade (primary model prediction is likely correct)
- 0: Skip the trade (primary model prediction is likely wrong)

Pipeline:
1. opti: Optimize meta model hyperparameters + triple barrier params
2. train: Train meta model with optimized parameters (TODO)
3. eval: Evaluate meta model performance (TODO)

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
"""

from src.labelling.label_meta.utils import (
    # Registry
    MODEL_REGISTRY,
    TRIPLE_BARRIER_SEARCH_SPACE,
    # Dataclasses
    MetaOptimizationConfig,
    MetaOptimizationResult,
    # Data utilities
    load_model_class,
    get_dataset_for_model,
    load_dollar_bars,
    load_primary_model,
    get_available_primary_models,
    # Volatility
    get_daily_volatility,
)

from src.labelling.label_meta.opti import (
    # Core De Prado functions
    get_events_meta,
    get_bins,
    # Optimization
    optimize_meta_model,
    WalkForwardCV,
)

__all__ = [
    # Core De Prado functions
    "get_daily_volatility",
    "get_events_meta",
    "get_bins",
    # Optimization
    "optimize_meta_model",
    "MetaOptimizationConfig",
    "MetaOptimizationResult",
    "WalkForwardCV",
    # Data utilities
    "load_model_class",
    "get_dataset_for_model",
    "load_dollar_bars",
    "load_primary_model",
    "get_available_primary_models",
    # Registry
    "MODEL_REGISTRY",
    "TRIPLE_BARRIER_SEARCH_SPACE",
]
