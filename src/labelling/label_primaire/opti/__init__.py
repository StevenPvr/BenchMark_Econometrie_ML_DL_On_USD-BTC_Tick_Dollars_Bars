"""Optimization package for primary labeling models."""

from src.labelling.label_primaire.opti.logic import (
    WalkForwardCV,
    create_objective,
    optimize_model,
    run_parallel,
    run_sequential,
    select_models_interactive,
    print_final_summary,
)
from src.labelling.label_primaire.utils import (
    OptimizationConfig,
    OptimizationResult,
)

__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "WalkForwardCV",
    "create_objective",
    "optimize_model",
    "print_final_summary",
    "run_parallel",
    "run_sequential",
    "select_models_interactive",
]
