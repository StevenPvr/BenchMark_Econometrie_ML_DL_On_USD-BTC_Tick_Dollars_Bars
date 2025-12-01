"""Feature Analysis Module.

This package provides comprehensive feature analysis for the volatility
forecasting project. It includes:

- Correlation Analysis: Spearman, Distance Correlation, Mutual Information
- Stationarity Analysis: ADF and KPSS tests
- Multicollinearity Analysis: VIF, Condition Number
- Target Analysis: Feature-target relationships
- Clustering: Hierarchical, t-SNE, UMAP
- Temporal Analysis: ACF/PACF, Rolling correlations

Usage:
    # Run all analyses
    python -m src.analyse_features.main

    # Or import specific functions
    from src.analyse_features.correlation import run_correlation_analysis
    from src.analyse_features.stationarity import run_stationarity_analysis
"""

# Import analysis functions (not from main.py to avoid circular import warning)
from src.analyse_features.correlation import run_correlation_analysis
from src.analyse_features.stationarity import run_stationarity_analysis
from src.analyse_features.multicollinearity import run_multicollinearity_analysis
from src.analyse_features.target_analysis import run_target_analysis
from src.analyse_features.clustering import run_clustering_analysis
from src.analyse_features.temporal import run_temporal_analysis

__all__ = [
    "run_correlation_analysis",
    "run_stationarity_analysis",
    "run_multicollinearity_analysis",
    "run_target_analysis",
    "run_clustering_analysis",
    "run_temporal_analysis",
]
