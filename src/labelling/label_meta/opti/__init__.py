"""Optimization package for meta-labeling models."""

from src.labelling.label_meta.opti.logic import (
    MetaOptimizationConfig,
    MetaOptimizationResult,
    WalkForwardCV,
    get_bins,
    get_events_meta,
    optimize_model,
)
from src.labelling.label_meta.opti.main import (
    main,
)

__all__ = [
    "MetaOptimizationConfig",
    "MetaOptimizationResult",
    "WalkForwardCV",
    "get_bins",
    "get_events_meta",
    "optimize_model",
    "main",
]
