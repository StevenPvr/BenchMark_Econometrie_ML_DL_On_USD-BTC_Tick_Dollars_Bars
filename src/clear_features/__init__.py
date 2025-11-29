"""Clear features module - PCA reduction, log transform, and normalization.

Pipeline:
1. Group-based PCA reduction (fit on train, transform all)
2. Log transformation for non-stationary features
3. Normalization (z-score for linear, minmax for LSTM)
"""

from src.clear_features.pca_reducer import (
    GroupPCAResult,
    PCAReductionSummary,
    GroupPCAReducer,
    WeightedPCAReducer,  # Alias for backward compatibility
)
from src.clear_features.log_transformer import LogTransformer, LogTransformResult
from src.clear_features.scaler_applier import ScalerApplier
from src.clear_features.main import run_full_pipeline

__all__ = [
    # PCA reduction
    "GroupPCAResult",
    "PCAReductionSummary",
    "GroupPCAReducer",
    "WeightedPCAReducer",
    # Log transform
    "LogTransformer",
    "LogTransformResult",
    # Scaler
    "ScalerApplier",
    # Main pipeline
    "run_full_pipeline",
]
