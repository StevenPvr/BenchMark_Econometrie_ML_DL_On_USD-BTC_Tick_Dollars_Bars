"""Configuration for clear_features module - PCA reduction, log transform, and scaling."""

from __future__ import annotations

from pathlib import Path

from src.path import (
    DATA_DIR,
    FEATURES_DIR,
    DATASET_FEATURES_PARQUET,
    DATASET_FEATURES_LINEAR_PARQUET,
    DATASET_FEATURES_LSTM_PARQUET,
    SCALERS_DIR,
    ZSCORE_SCALER_FILE,
    MINMAX_SCALER_FILE,
)

# Input/Output directories
ANALYSE_FEATURES_DIR = DATA_DIR / "analyse_features"
CLUSTERING_RESULTS_FILE = ANALYSE_FEATURES_DIR / "json" / "clustering_results.json"
CORRELATION_RESULTS_FILE = ANALYSE_FEATURES_DIR / "json" / "correlation_results.json"
STATIONARITY_RESULTS_FILE = ANALYSE_FEATURES_DIR / "json" / "stationarity_results.json"

# Output directory for PCA artifacts
CLEAR_FEATURES_DIR = DATA_DIR / "clear_features"
PCA_ARTIFACTS_DIR = CLEAR_FEATURES_DIR / "pca_artifacts"
LOG_TRANSFORM_ARTIFACTS_DIR = CLEAR_FEATURES_DIR / "log_transform_artifacts"

# Datasets (input and output - same files, overwrite)
DATASETS = {
    "tree_based": DATASET_FEATURES_PARQUET,
    "linear": DATASET_FEATURES_LINEAR_PARQUET,
    "lstm": DATASET_FEATURES_LSTM_PARQUET,
}

# Reference dataset (for fitting PCA - use tree_based as it's unscaled)
REFERENCE_DATASET = "tree_based"

# Metadata columns to exclude from transformations
META_COLUMNS = ["bar_id", "datetime_close", "split"]

# Target column (exclude from transformations)
TARGET_COLUMN = "log_return"

# Non-linear correlation configuration
CORRELATION_CONFIG = {
    # Method: 'spearman' (rank-based, recommended) or 'kendall'
    "method": "spearman",
    # Correlation threshold for clustering (finance-appropriate: 0.7-0.9)
    # 0.8 is conservative and recommended for financial data
    "threshold": 0.8,
    # Minimum features in a cluster to apply PCA
    "min_cluster_size": 2,
    # Sample size for correlation computation (None = use all data)
    "sample_size": 50000,
}

# PCA configuration
PCA_CONFIG = {
    # Variance explained threshold to determine n_components
    "variance_explained_threshold": 0.90,
    # Alternative: maximum number of components (None = use variance threshold)
    "max_components": None,
}

# Feature categories file (for group-based PCA)
FEATURE_CATEGORIES_FILE = FEATURES_DIR / "feature_categories.json"

# Weighting configuration (using mutual information from correlation_results.json)
WEIGHTING_CONFIG = {
    # Whether to use MI weighting before PCA
    "use_mi_weighting": True,
    # Default MI weight for features not in MI results
    "default_mi_weight": 0.01,
}

# Log transform configuration
LOG_TRANSFORM_CONFIG = {
    # Stationarity conclusions that trigger log transform
    "non_stationary_conclusions": ["trend_stationary", "non_stationary"],
    # Minimum value threshold for log transform (to avoid log(0))
    "min_value_threshold": 1e-10,
    # Whether to use log1p (log(1+x)) for values near zero
    "use_log1p": True,
}

# Scaler files (fitted in features/main.py)
SCALER_CONFIG = {
    "zscore_scaler_path": ZSCORE_SCALER_FILE,
    "minmax_scaler_path": MINMAX_SCALER_FILE,
}
