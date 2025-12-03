"""Evaluation package for meta-label models."""

from src.labelling.label_meta.eval.logic import (
    CombinedEvaluationResult,
)

from src.labelling.label_meta.eval.main import (
    main,
)

__all__ = [
    "CombinedEvaluationResult",
    "main",
]
