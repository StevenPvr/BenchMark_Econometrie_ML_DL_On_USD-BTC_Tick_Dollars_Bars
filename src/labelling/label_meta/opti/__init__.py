"""Optimization package for meta-labeling models."""

from __future__ import annotations

from src.labelling.label_meta.opti.logic import (
    WalkForwardCV,
    optimize_model,
)
from src.labelling.label_meta.opti.main import main
from src.labelling.label_meta.utils import (
    MetaOptimizationConfig,
    MetaOptimizationResult,
)

__all__ = [
    "MetaOptimizationConfig",
    "MetaOptimizationResult",
    "WalkForwardCV",
    "optimize_model",
    "main",
]
