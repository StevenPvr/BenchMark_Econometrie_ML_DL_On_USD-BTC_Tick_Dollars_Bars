"""Label Meta Optimization module.

This module optimizes the hyperparameters for the meta-model (LightGBM)
separately from the primary model. The meta-model is a binary classifier
that predicts whether a primary model's trade signal will be correct.
"""

from src.label_meta_opti.optimize import (
    optimize_meta_model,
    MetaOptimizationResult,
    save_meta_optimization_results,
)

__all__ = [
    "optimize_meta_model",
    "MetaOptimizationResult",
    "save_meta_optimization_results",
]
