"""Label Meta - Meta-Labeling Pipeline (De Prado Chapter 3.6).

This module implements the META model that filters false positives
from the primary model's predictions.

The meta model learns to predict:
- 1: Take the trade (primary model prediction is likely correct)
- 0: Skip the trade (primary model prediction is likely wrong)

Pipeline:
1. opti: Optimize meta model hyperparameters
2. train: Train meta model with optimized parameters (TODO)
3. eval: Evaluate meta model performance (TODO)

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
"""

from __future__ import annotations

from src.labelling.label_meta.utils import (
    MODEL_REGISTRY,
    MetaOptimizationConfig,
    MetaOptimizationResult,
    get_dataset_for_model,
    load_model_class,
    load_primary_model,
    get_available_primary_models,
)

from src.labelling.label_meta.opti import (
    optimize_model,
    WalkForwardCV,
)

__all__ = [
    "optimize_model",
    "MetaOptimizationConfig",
    "MetaOptimizationResult",
    "WalkForwardCV",
    "load_model_class",
    "get_dataset_for_model",
    "load_primary_model",
    "get_available_primary_models",
    "MODEL_REGISTRY",
]
