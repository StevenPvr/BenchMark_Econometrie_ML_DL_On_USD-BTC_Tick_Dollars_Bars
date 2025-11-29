"""Clear features module - Correlation analysis, PCA reduction, log transform, and normalization.

Pipeline:
1. Non-linear correlation analysis (Spearman) to identify correlated feature groups
2. PCA reduction to merge correlated features
3. Log transformation for non-stationary features
4. Normalization (z-score for linear, minmax for LSTM)
"""

from src.clear_features.nonlinear_correlation import (
    NonLinearCorrelationAnalyzer,
    CorrelationCluster,
    CorrelationAnalysisResult,
)
from src.clear_features.pca_reducer import (
    ClusterPCAResult,
    PCAReductionSummary,
    WeightedPCAReducer,
)
from src.clear_features.log_transformer import LogTransformer, LogTransformResult
from src.clear_features.scaler_applier import ScalerApplier
from src.clear_features.main import run_full_pipeline

__all__ = [
    # Correlation analysis
    "NonLinearCorrelationAnalyzer",
    "CorrelationCluster",
    "CorrelationAnalysisResult",
    # PCA reduction
    "ClusterPCAResult",
    "PCAReductionSummary",
    "WeightedPCAReducer",
    # Log transform
    "LogTransformer",
    "LogTransformResult",
    # Scaler
    "ScalerApplier",
    # Main pipeline
    "run_full_pipeline",
]
