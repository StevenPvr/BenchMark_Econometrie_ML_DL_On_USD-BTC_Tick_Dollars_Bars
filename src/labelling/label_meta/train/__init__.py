"""Training package for meta-label models."""

from src.labelling.label_meta.train.logic import (
    MetaEvaluationMetrics,
    MetaTrainingConfig,
    MetaTrainingResult,
    build_meta_features,
    compute_metrics,
    get_available_meta_optimizations,
    get_yes_no_input,
    load_optimized_params,
    select_meta_model,
    select_primary_model,
    train_meta_model,
)
from src.labelling.label_meta.train.main import main

__all__ = [
    "MetaEvaluationMetrics",
    "MetaTrainingConfig",
    "MetaTrainingResult",
    "build_meta_features",
    "compute_metrics",
    "get_available_meta_optimizations",
    "get_yes_no_input",
    "load_optimized_params",
    "main",
    "select_meta_model",
    "select_primary_model",
    "train_meta_model",
]
