"""Training package for primary labeling models with OOF predictions."""

from src.labelling.label_primaire.train.logic import (
    PrimaryEvaluationMetrics,
    TrainingConfig,
    TrainingResult,
    WalkForwardKFold,
    compute_metrics,
    evaluate_model,
    generate_oof_predictions,
    get_available_optimized_models,
    get_yes_no_input,
    load_optimized_params,
    select_model,
    train_model,
)
from src.labelling.label_primaire.train.main import main

__all__ = [
    "PrimaryEvaluationMetrics",
    "TrainingConfig",
    "TrainingResult",
    "WalkForwardKFold",
    "compute_metrics",
    "evaluate_model",
    "generate_oof_predictions",
    "get_available_optimized_models",
    "get_yes_no_input",
    "load_optimized_params",
    "main",
    "select_model",
    "train_model",
]
