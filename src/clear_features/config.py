"""Configuration for clear_features module - PCA reduction, log transform, and scaling.

This module imports constants from constants.py and defines paths from path.py.
"""

from __future__ import annotations

from src.constants import (
    CLEAR_FEATURES_META_COLUMNS,
    CLEAR_FEATURES_TARGET_COLUMN,
    CLEAR_FEATURES_CORRELATION_METHOD,
    CLEAR_FEATURES_CORRELATION_THRESHOLD,
    CLEAR_FEATURES_CORRELATION_MIN_CLUSTER_SIZE,
    CLEAR_FEATURES_CORRELATION_SAMPLE_SIZE,
    CLEAR_FEATURES_PCA_VARIANCE_THRESHOLD,
    CLEAR_FEATURES_LOG_NON_STATIONARY_CONCLUSIONS,
    CLEAR_FEATURES_LOG_MIN_VALUE_THRESHOLD,
    CLEAR_FEATURES_LOG_USE_LOG1P,
    CLEAR_FEATURES_SAVE_BATCH_SIZE,
)
from src.path import (
    DATA_DIR,
    FEATURES_DIR,
    DATASET_FEATURES_PARQUET,
    DATASET_FEATURES_LINEAR_PARQUET,
    DATASET_FEATURES_LSTM_PARQUET,
    DATASET_FEATURES_CLEAR_PARQUET,
    DATASET_FEATURES_LINEAR_CLEAR_PARQUET,
    DATASET_FEATURES_LSTM_CLEAR_PARQUET,
    SCALERS_DIR,
    ZSCORE_SCALER_FILE,
    MINMAX_SCALER_FILE,
)

# Re-export constants with module-level aliases for backward compatibility
META_COLUMNS: list[str] = list(CLEAR_FEATURES_META_COLUMNS)
TARGET_COLUMN: str = CLEAR_FEATURES_TARGET_COLUMN

# Input/Output directories
ANALYSE_FEATURES_DIR = DATA_DIR / "analyse_features"
CLUSTERING_RESULTS_FILE = ANALYSE_FEATURES_DIR / "json" / "clustering_results.json"
CORRELATION_RESULTS_FILE = ANALYSE_FEATURES_DIR / "json" / "correlation_results.json"
STATIONARITY_RESULTS_FILE = ANALYSE_FEATURES_DIR / "json" / "stationarity_results.json"

# Output directory for PCA artifacts
CLEAR_FEATURES_DIR = DATA_DIR / "clear_features"
PCA_ARTIFACTS_DIR = CLEAR_FEATURES_DIR / "pca_artifacts"
LOG_TRANSFORM_ARTIFACTS_DIR = CLEAR_FEATURES_DIR / "log_transform_artifacts"

# Input datasets (from features/)
INPUT_DATASETS = {
    "tree_based": DATASET_FEATURES_PARQUET,
    "linear": DATASET_FEATURES_LINEAR_PARQUET,
    "lstm": DATASET_FEATURES_LSTM_PARQUET,
}

# Output datasets (with '_clear' suffix)
OUTPUT_DATASETS = {
    "tree_based": DATASET_FEATURES_CLEAR_PARQUET,
    "linear": DATASET_FEATURES_LINEAR_CLEAR_PARQUET,
    "lstm": DATASET_FEATURES_LSTM_CLEAR_PARQUET,
}

# Legacy alias for backwards compatibility
DATASETS = INPUT_DATASETS

# Reference dataset (for fitting PCA - use tree_based as it's unscaled)
REFERENCE_DATASET = "tree_based"

# Non-linear correlation configuration (uses constants)
CORRELATION_CONFIG: dict[str, str | float | int | None] = {
    "method": CLEAR_FEATURES_CORRELATION_METHOD,
    "threshold": CLEAR_FEATURES_CORRELATION_THRESHOLD,
    "min_cluster_size": CLEAR_FEATURES_CORRELATION_MIN_CLUSTER_SIZE,
    "sample_size": CLEAR_FEATURES_CORRELATION_SAMPLE_SIZE,
}

# PCA configuration (uses constants)
PCA_CONFIG: dict[str, float | int | None] = {
    "variance_explained_threshold": CLEAR_FEATURES_PCA_VARIANCE_THRESHOLD,
    "max_components": None,
}

# Feature categories file (for group-based PCA)
FEATURE_CATEGORIES_FILE = FEATURES_DIR / "feature_categories.json"

# Log transform configuration (uses constants)
LOG_TRANSFORM_CONFIG: dict[str, list[str] | float | bool] = {
    "non_stationary_conclusions": list(CLEAR_FEATURES_LOG_NON_STATIONARY_CONCLUSIONS),
    "min_value_threshold": CLEAR_FEATURES_LOG_MIN_VALUE_THRESHOLD,
    "use_log1p": CLEAR_FEATURES_LOG_USE_LOG1P,
}

# Scaler files (fitted in features/main.py)
SCALER_CONFIG = {
    "zscore_scaler_path": ZSCORE_SCALER_FILE,
    "minmax_scaler_path": MINMAX_SCALER_FILE,
}

# Batch size for parquet saving
SAVE_BATCH_SIZE = CLEAR_FEATURES_SAVE_BATCH_SIZE
