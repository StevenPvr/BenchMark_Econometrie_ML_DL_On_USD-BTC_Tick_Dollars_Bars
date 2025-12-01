"""Configuration for feature analysis module.

This module centralizes all parameters for the feature analysis pipeline:
- Parallelization settings (n_jobs, chunk_size)
- Analysis thresholds
- Output paths
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from src.path import (
    DATA_DIR,
    DATASET_FEATURES_CLEAR_PARQUET,
    DATASET_FEATURES_LINEAR_CLEAR_PARQUET,
    DATASET_FEATURES_LSTM_CLEAR_PARQUET,
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
)

# ============================================================================
# PARALLELIZATION SETTINGS
# ============================================================================

# Number of parallel jobs (-1 = all cores)
N_JOBS: int = -1 if os.cpu_count() else 1

# Actual number of cores for explicit usage
N_CORES: int = os.cpu_count() or 1

# Chunk size for batch processing (features per batch)
CHUNK_SIZE: int = 50

# Backend for joblib parallelization
JOBLIB_BACKEND: str = "loky"  # 'loky' (default), 'multiprocessing', 'threading'

# Verbosity level for joblib (0=silent, 10=verbose)
JOBLIB_VERBOSITY: int = 0

# ============================================================================
# OUTPUT PATHS
# ============================================================================

# Base output directory for analysis results
ANALYSE_FEATURES_DIR: Path = DATA_DIR / "analyse_features"

# JSON output files (human-readable results)
RESULTS_JSON_DIR: Path = ANALYSE_FEATURES_DIR / "json"
CORRELATION_RESULTS_JSON: Path = RESULTS_JSON_DIR / "correlation_results.json"
STATIONARITY_RESULTS_JSON: Path = RESULTS_JSON_DIR / "stationarity_results.json"
MULTICOLLINEARITY_RESULTS_JSON: Path = RESULTS_JSON_DIR / "multicollinearity_results.json"
TARGET_RESULTS_JSON: Path = RESULTS_JSON_DIR / "target_results.json"
CLUSTERING_RESULTS_JSON: Path = RESULTS_JSON_DIR / "clustering_results.json"
TEMPORAL_RESULTS_JSON: Path = RESULTS_JSON_DIR / "temporal_results.json"
SUMMARY_JSON: Path = RESULTS_JSON_DIR / "analysis_summary.json"

# Input datasets (from clear_features/ with '_clear' suffix)
INPUT_DATASETS = {
    "tree_based": DATASET_FEATURES_CLEAR_PARQUET,
    "linear": DATASET_FEATURES_LINEAR_CLEAR_PARQUET,
    "lstm": DATASET_FEATURES_LSTM_CLEAR_PARQUET,
}

# Output datasets (with '_final' suffix - ready for labelling)
OUTPUT_DATASETS = {
    "tree_based": DATASET_FEATURES_FINAL_PARQUET,
    "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
}

# Plot directories
PLOTS_HTML_DIR: Path = ANALYSE_FEATURES_DIR / "plots" / "html"
PLOTS_PNG_DIR: Path = ANALYSE_FEATURES_DIR / "plots" / "png"

# Cache directory for intermediate results
CACHE_DIR: Path = ANALYSE_FEATURES_DIR / "cache"

# ============================================================================
# DATA SAMPLING SETTINGS
# ============================================================================

# Fraction of dataset to use for analysis (0.2 = 20%)
DATASET_SAMPLE_FRACTION: float = 0.2

# ============================================================================
# CORRELATION ANALYSIS SETTINGS
# ============================================================================

# Correlation thresholds (used for summary and filtering)
# - SIGNIFICANT: minimum correlation to consider relevant
# - HIGH: threshold for "highly correlated" pairs
# - VERY_HIGH: threshold for potential redundancy (consider removing)
CORRELATION_SIGNIFICANT_THRESHOLD: float = 0.5
CORRELATION_HIGH_THRESHOLD: float = 0.7
SPEARMAN_HIGH_CORR_THRESHOLD: float = 0.7  # Legacy alias, use CORRELATION_HIGH_THRESHOLD
# Threshold for dropping one feature from highly correlated groups
CORRELATION_DROP_THRESHOLD: float = 0.7

# Mutual information settings
MI_N_NEIGHBORS: int = 5  # k-NN for MI estimation
MI_RANDOM_STATE: int = 42

# ============================================================================
# STATIONARITY ANALYSIS SETTINGS
# ============================================================================

# Significance level for stationarity tests
STATIONARITY_ALPHA: float = 0.05

# ADF regression type: 'c' (constant), 'ct' (constant + trend), 'n' (no constant)
ADF_REGRESSION: Literal['c', 'ct', 'n'] = "c"

# Maximum lags for ADF (None = auto)
ADF_MAX_LAGS: int | None = None

# KPSS regression type: 'c' (constant), 'ct' (constant + trend)
KPSS_REGRESSION: Literal['c', 'ct'] = "c"

# ============================================================================
# CLUSTERING SETTINGS
# ============================================================================

# Hierarchical clustering linkage method
CLUSTER_LINKAGE: str = "ward"  # 'ward', 'complete', 'average', 'single'

# Distance metric for clustering (1 - |corr|)
CLUSTER_METRIC: str = "correlation"

# UMAP parameters
UMAP_N_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.1
UMAP_N_COMPONENTS: int = 2
UMAP_METRIC: str = "correlation"

# t-SNE parameters
TSNE_PERPLEXITY: int = 30
TSNE_N_ITER: int = 1000
TSNE_RANDOM_STATE: int = 42

# ============================================================================
# TEMPORAL ANALYSIS SETTINGS
# ============================================================================

# ACF/PACF lags
ACF_LAGS: int = 40

# Rolling correlation window sizes
ROLLING_WINDOWS: list[int] = [50, 100, 250, 500]

# Temporal split for heatmap (number of periods)
N_TEMPORAL_PERIODS: int = 10

# ============================================================================
# TARGET ANALYSIS SETTINGS
# ============================================================================

# Target column name
TARGET_COLUMN: str = "log_return"

# Top N features to display in plots
TOP_N_FEATURES: int = 30

# Scatter plot sample size
SCATTER_SAMPLE_SIZE: int = 10000

# ============================================================================
# MULTICOLLINEARITY SETTINGS
# ============================================================================

# VIF threshold for high multicollinearity
VIF_HIGH_THRESHOLD: float = 10.0
VIF_CRITICAL_THRESHOLD: float = 100.0

# Condition number thresholds
CONDITION_NUMBER_MODERATE: float = 30.0
CONDITION_NUMBER_HIGH: float = 100.0

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Figure sizes
FIGSIZE_HEATMAP: tuple[int, int] = (16, 14)
FIGSIZE_BAR: tuple[int, int] = (12, 8)
FIGSIZE_SCATTER: tuple[int, int] = (10, 8)
FIGSIZE_DENDROGRAM: tuple[int, int] = (16, 10)
FIGSIZE_ACF: tuple[int, int] = (14, 6)

# DPI for PNG exports
PNG_DPI: int = 150

# Color schemes
COLORSCALE_CORRELATION: str = "RdBu_r"  # Red-Blue diverging
COLORSCALE_SEQUENTIAL: str = "Viridis"


def ensure_directories() -> None:
    """Create all output directories if they don't exist."""
    dirs_to_create = [
        ANALYSE_FEATURES_DIR,
        RESULTS_JSON_DIR,
        PLOTS_HTML_DIR,
        PLOTS_PNG_DIR,
        CACHE_DIR,
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
