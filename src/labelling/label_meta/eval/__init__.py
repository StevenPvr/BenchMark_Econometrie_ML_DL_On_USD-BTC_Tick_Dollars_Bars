"""Evaluation package for meta-label models."""

from __future__ import annotations

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
