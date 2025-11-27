"""File and directory paths for the BTC/USD Tick Data Forecasting project."""

from __future__ import annotations

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# BASE DIRECTORIES
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data"
# Keep results/plots under the existing outputs/ tree used in this repo
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

# Legacy artifacts directory (backward compatibility)
ARTIFACTS_DIR = RESULTS_DIR

# ============================================================================
# DATA PIPELINE - File paths
# ============================================================================

# BTC/USD tick data pipeline directories
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PREPARED_DATA_DIR = DATA_DIR / "prepared"

# Raw BTC/USD tick data files
DATASET_RAW_CSV = RAW_DATA_DIR / "dataset_raw.csv"
DATASET_RAW_PARQUET = RAW_DATA_DIR / "dataset_raw.parquet"

# Cleaned BTC/USD tick data files
DATASET_CLEAN_CSV = CLEANED_DATA_DIR / "dataset_clean.csv"
DATASET_CLEAN_PARQUET = CLEANED_DATA_DIR / "dataset_clean.parquet"

# Prepared log-returns (map weighted aliases to existing files)
WEIGHTED_LOG_RETURNS_FILE = PREPARED_DATA_DIR / "log_returns.csv"
WEIGHTED_LOG_RETURNS_SPLIT_FILE = PREPARED_DATA_DIR / "log_returns_split.parquet"
LOG_RETURNS_PARQUET = PREPARED_DATA_DIR / "log_returns.parquet"
LOG_RETURNS_CSV = PREPARED_DATA_DIR / "log_returns.csv"
DOLLAR_BARS_PARQUET = PREPARED_DATA_DIR / "dollar_bars.parquet"
DOLLAR_BARS_CSV = PREPARED_DATA_DIR / "dollar_bars.csv"
DOLLAR_IMBALANCE_BARS_PARQUET = PREPARED_DATA_DIR / "dollar_imbalance_bars.parquet"
DOLLAR_IMBALANCE_BARS_CSV = PREPARED_DATA_DIR / "dollar_imbalance_bars.csv"

# Features dataset files
FEATURES_DIR = DATA_DIR / "features"
SCALERS_DIR = FEATURES_DIR / "scalers"
LOG_RETURNS_SPLIT_PARQUET = PREPARED_DATA_DIR / "log_returns_split.parquet"
DATASET_FEATURES_PARQUET = FEATURES_DIR / "dataset_features.parquet"
DATASET_FEATURES_CSV = FEATURES_DIR / "dataset_features.csv"
DATASET_FEATURES_TRAIN_PARQUET = FEATURES_DIR / "dataset_features_train.parquet"
DATASET_FEATURES_TEST_PARQUET = FEATURES_DIR / "dataset_features_test.parquet"
DATASET_FEATURES_TRAIN_CSV = FEATURES_DIR / "dataset_features_train.csv"
DATASET_FEATURES_TEST_CSV = FEATURES_DIR / "dataset_features_test.csv"
DATASET_FEATURES_LINEAR_PARQUET = FEATURES_DIR / "dataset_features_linear.parquet"
DATASET_FEATURES_LINEAR_CSV = FEATURES_DIR / "dataset_features_linear.csv"
DATASET_FEATURES_LINEAR_TRAIN_PARQUET = FEATURES_DIR / "dataset_features_linear_train.parquet"
DATASET_FEATURES_LINEAR_TEST_PARQUET = FEATURES_DIR / "dataset_features_linear_test.parquet"
DATASET_FEATURES_LINEAR_TRAIN_CSV = FEATURES_DIR / "dataset_features_linear_train.csv"
DATASET_FEATURES_LINEAR_TEST_CSV = FEATURES_DIR / "dataset_features_linear_test.csv"
DATASET_FEATURES_LSTM_PARQUET = FEATURES_DIR / "dataset_features_lstm.parquet"
DATASET_FEATURES_LSTM_CSV = FEATURES_DIR / "dataset_features_lstm.csv"
DATASET_FEATURES_LSTM_TRAIN_PARQUET = FEATURES_DIR / "dataset_features_lstm_train.parquet"
DATASET_FEATURES_LSTM_TEST_PARQUET = FEATURES_DIR / "dataset_features_lstm_test.parquet"
DATASET_FEATURES_LSTM_TRAIN_CSV = FEATURES_DIR / "dataset_features_lstm_train.csv"
DATASET_FEATURES_LSTM_TEST_CSV = FEATURES_DIR / "dataset_features_lstm_test.csv"
# Scalers (fit on train only)
ZSCORE_SCALER_FILE = SCALERS_DIR / "zscore_scaler.joblib"
MINMAX_SCALER_FILE = SCALERS_DIR / "minmax_scaler.joblib"

# Prepared BTC/USD data files
DATA_TICKERS_FULL_FILE = PREPARED_DATA_DIR / "data_btc_full.csv"
DATA_TICKERS_FULL_INDICATORS_FILE = PREPARED_DATA_DIR / "data_btc_full_indicators.csv"
DATA_TICKERS_FULL_INSIGHTS_FILE = PREPARED_DATA_DIR / "data_btc_full_insights.parquet"
DATA_TICKERS_FULL_INSIGHTS_INDICATORS_FILE = PREPARED_DATA_DIR / "data_btc_full_insights_indicators.csv"

# Dataset files (general)
DATASET_FILE = PREPARED_DATA_DIR / "dataset.csv"
DATASET_FILTERED_FILE = PREPARED_DATA_DIR / "dataset_filtered.csv"
DATASET_FILTERED_PARQUET_FILE = PREPARED_DATA_DIR / "dataset_filtered.parquet"

# ============================================================================
# RESULTS DIRECTORIES - Organized by pipeline step/model
# ============================================================================

# Data pipeline results
DATA_RESULTS_DIR = RESULTS_DIR / "data"
STATIONARITY_REPORT_FILE = DATA_RESULTS_DIR / "stationarity_report.json"
FETCH_REPORT_FILE = DATA_RESULTS_DIR / "fetch_report.json"

# GARCH dataset (log returns with train/test split)
# GARCH now works directly on log returns (no ARIMA preprocessing)
GARCH_DATASET_FILE = PREPARED_DATA_DIR / "dataset_garch.csv"

# GARCH results
GARCH_RESULTS_DIR = RESULTS_DIR / "garch"
GARCH_STRUCTURE_DIR = GARCH_RESULTS_DIR / "structure"
GARCH_ESTIMATION_DIR = GARCH_RESULTS_DIR / "estimation"
GARCH_TRAINING_DIR = GARCH_RESULTS_DIR / "training"
GARCH_DIAGNOSTIC_DIR = GARCH_RESULTS_DIR / "diagnostic"
GARCH_EVALUATION_DIR = GARCH_RESULTS_DIR / "evaluation"
GARCH_ROLLING_DIR = GARCH_RESULTS_DIR / "rolling"

# GARCH structure detection files
GARCH_DIAGNOSTICS_FILE = GARCH_STRUCTURE_DIR / "diagnostics.json"
GARCH_NUMERICAL_TESTS_FILE = GARCH_STRUCTURE_DIR / "numerical_tests.json"

# GARCH estimation files
GARCH_ESTIMATION_FILE = GARCH_ESTIMATION_DIR / "estimation.json"

# GARCH optimization files
GARCH_OPTIMIZATION_DIR = GARCH_RESULTS_DIR / "optimization"
GARCH_OPTIMIZATION_RESULTS_FILE = GARCH_OPTIMIZATION_DIR / "hyperparameters.json"

# GARCH training files
GARCH_MODEL_FILE = GARCH_TRAINING_DIR / "model.joblib"
GARCH_MODEL_METADATA_FILE = GARCH_TRAINING_DIR / "model_metadata.json"
GARCH_RESIDUALS_OUTPUTS_FILE = (
    GARCH_TRAINING_DIR / "residuals_outputs.json"
)  # Residuals + variance forecast (safe)
# Legacy: kept for backward compatibility (deprecated, use GARCH_RESIDUALS_OUTPUTS_FILE instead)
GARCH_VARIANCE_OUTPUTS_FILE = GARCH_TRAINING_DIR / "variance_outputs.csv"

# GARCH diagnostic files
GARCH_LJUNGBOX_FILE = GARCH_DIAGNOSTIC_DIR / "ljungbox.json"
GARCH_DISTRIBUTION_DIAGNOSTICS_FILE = GARCH_DIAGNOSTIC_DIR / "distribution_diagnostics.json"

# GARCH evaluation files
GARCH_FORECASTS_FILE = GARCH_EVALUATION_DIR / "garch_forecasts.parquet"
GARCH_EVAL_METRICS_FILE = GARCH_EVALUATION_DIR / "metrics.json"
GARCH_EVAL_VAR_SUMMARY_FILE = GARCH_EVALUATION_DIR / "var_summary.json"
GARCH_EVAL_TEST_METRICS_FILE = GARCH_EVALUATION_DIR / "test_metrics.json"

# GARCH rolling files
GARCH_ROLLING_FORECASTS_FILE = GARCH_ROLLING_DIR / "forecasts.parquet"
GARCH_ROLLING_EVAL_FILE = GARCH_ROLLING_DIR / "metrics.json"
GARCH_ROLLING_VARIANCE_FILE = GARCH_ROLLING_DIR / "variance.parquet"
GARCH_ML_DATASET_FILE = GARCH_ROLLING_DIR / "ml_dataset.parquet"


# ============================================================================
# PLOTS DIRECTORIES - Organized by pipeline step/model
# ============================================================================

# Data pipeline plots
DATA_PLOTS_DIR = PLOTS_DIR / "data"

# GARCH plots
GARCH_PLOTS_DIR = PLOTS_DIR / "garch"
GARCH_DATA_VISU_PLOTS_DIR = GARCH_PLOTS_DIR / "data_visualization"
GARCH_STRUCTURE_PLOTS_DIR = GARCH_PLOTS_DIR / "structure"
GARCH_DIAGNOSTICS_PLOTS_DIR = GARCH_PLOTS_DIR / "diagnostics"
GARCH_EVALUATION_PLOTS_DIR = GARCH_PLOTS_DIR / "evaluation"

# GARCH structure plots
GARCH_STRUCTURE_PLOT = GARCH_STRUCTURE_PLOTS_DIR / "structure.png"

# GARCH diagnostic plots
GARCH_STD_SQUARED_ACF_PACF_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "std_squared_acf_pacf.png"
GARCH_STD_ACF_PACF_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "std_acf_pacf.png"
GARCH_STD_QQ_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "std_residuals_qq.png"
GARCH_STD_HISTOGRAM_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "std_residuals_histogram.png"

# GARCH data visualization plots
GARCH_ACF_SQUARED_PLOT = GARCH_DATA_VISU_PLOTS_DIR / "acf_squared.png"
GARCH_SQUARED_RESIDUALS_ACF_LB_PLOT = (
    GARCH_DATA_VISU_PLOTS_DIR / "squared_residuals_acf_ljungbox.png"
)

# GARCH evaluation plots
GARCH_EVAL_VAR_TIMESERIES_PLOT = GARCH_EVALUATION_PLOTS_DIR / "var_timeseries.png"
GARCH_EVAL_VAR_SCATTER_PLOT = GARCH_EVALUATION_PLOTS_DIR / "var_scatter.png"
GARCH_EVAL_VAR_RESIDUALS_PLOT = GARCH_EVALUATION_PLOTS_DIR / "var_residuals.png"
GARCH_EVAL_VAR_COMBINED_PLOT = GARCH_EVALUATION_PLOTS_DIR / "var_overlay.png"
# Template path for VaR violations plots; will be formatted with alpha
GARCH_EVAL_VAR_VIOLATIONS_TEMPLATE = str(
    GARCH_EVALUATION_PLOTS_DIR / "var_violations_alpha_{alpha}.png"
)

# ============================================================================
# MACHINE LEARNING MODEL DIRECTORIES
# ============================================================================

ML_RESULTS_DIR = RESULTS_DIR / "ml"

# LightGBM
LIGHTGBM_DIR = ML_RESULTS_DIR / "lightgbm"
LIGHTGBM_ARTIFACTS_DIR = LIGHTGBM_DIR / "artifacts"
LIGHTGBM_MODEL_FILE = LIGHTGBM_ARTIFACTS_DIR / "lightgbm_model.joblib"
LIGHTGBM_BEST_PARAMS_FILE = LIGHTGBM_ARTIFACTS_DIR / "best_params.json"
LIGHTGBM_TRAINING_RESULTS_FILE = LIGHTGBM_ARTIFACTS_DIR / "training_results.json"
LIGHTGBM_TEST_EVAL_FILE = LIGHTGBM_ARTIFACTS_DIR / "test_evaluation.json"

# XGBoost
XGBOOST_DIR = ML_RESULTS_DIR / "xgboost"

# CatBoost
CATBOOST_DIR = ML_RESULTS_DIR / "catboost"

# Random Forest
RANDOM_FOREST_DIR = ML_RESULTS_DIR / "random_forest"
RANDOM_FOREST_ARTIFACTS_DIR = RANDOM_FOREST_DIR / "artifacts"
RANDOM_FOREST_MODEL_FILE = RANDOM_FOREST_ARTIFACTS_DIR / "random_forest_model.joblib"
RANDOM_FOREST_BEST_PARAMS_FILE = RANDOM_FOREST_ARTIFACTS_DIR / "best_params.json"
RANDOM_FOREST_TRAINING_RESULTS_FILE = RANDOM_FOREST_ARTIFACTS_DIR / "training_results.json"
RANDOM_FOREST_TEST_EVAL_FILE = RANDOM_FOREST_ARTIFACTS_DIR / "test_evaluation.json"

# ============================================================================
# ECONOMETRIC MODEL DIRECTORIES
# ============================================================================

ECONOMETRIC_DIR = RESULTS_DIR / "econometric"

# Ridge
RIDGE_DIR = ECONOMETRIC_DIR / "ridge"
RIDGE_ARTIFACTS_DIR = RIDGE_DIR / "artifacts"
RIDGE_MODEL_FILE = RIDGE_ARTIFACTS_DIR / "ridge_model.joblib"
RIDGE_BEST_PARAMS_FILE = RIDGE_ARTIFACTS_DIR / "best_params.json"
RIDGE_TRAINING_RESULTS_FILE = RIDGE_ARTIFACTS_DIR / "training_results.json"
RIDGE_TEST_EVAL_FILE = RIDGE_ARTIFACTS_DIR / "test_evaluation.json"

# Lasso
LASSO_DIR = ECONOMETRIC_DIR / "lasso"
LASSO_ARTIFACTS_DIR = LASSO_DIR / "artifacts"
LASSO_MODEL_FILE = LASSO_ARTIFACTS_DIR / "lasso_model.joblib"
LASSO_BEST_PARAMS_FILE = LASSO_ARTIFACTS_DIR / "best_params.json"
LASSO_TRAINING_RESULTS_FILE = LASSO_ARTIFACTS_DIR / "training_results.json"
LASSO_TEST_EVAL_FILE = LASSO_ARTIFACTS_DIR / "test_evaluation.json"

# OLS (Ordinary Least Squares)
OLS_DIR = ECONOMETRIC_DIR / "ols"
OLS_ARTIFACTS_DIR = OLS_DIR / "artifacts"
OLS_MODEL_FILE = OLS_ARTIFACTS_DIR / "ols_model.joblib"
OLS_BEST_PARAMS_FILE = OLS_ARTIFACTS_DIR / "best_params.json"
OLS_TRAINING_RESULTS_FILE = OLS_ARTIFACTS_DIR / "training_results.json"
OLS_TEST_EVAL_FILE = OLS_ARTIFACTS_DIR / "test_evaluation.json"

# ============================================================================
# ECONOMETRIC CLASSIFIER DIRECTORIES
# ============================================================================

# Ridge Classifier
RIDGE_CLASSIFIER_DIR = ECONOMETRIC_DIR / "ridge_classifier"
RIDGE_CLASSIFIER_ARTIFACTS_DIR = RIDGE_CLASSIFIER_DIR / "artifacts"
RIDGE_CLASSIFIER_MODEL_FILE = RIDGE_CLASSIFIER_ARTIFACTS_DIR / "ridge_classifier_model.joblib"
RIDGE_CLASSIFIER_BEST_PARAMS_FILE = RIDGE_CLASSIFIER_ARTIFACTS_DIR / "best_params.json"
RIDGE_CLASSIFIER_TRAINING_RESULTS_FILE = RIDGE_CLASSIFIER_ARTIFACTS_DIR / "training_results.json"
RIDGE_CLASSIFIER_TEST_EVAL_FILE = RIDGE_CLASSIFIER_ARTIFACTS_DIR / "test_evaluation.json"

# Lasso Classifier (L1 Logistic)
LASSO_CLASSIFIER_DIR = ECONOMETRIC_DIR / "lasso_classifier"
LASSO_CLASSIFIER_ARTIFACTS_DIR = LASSO_CLASSIFIER_DIR / "artifacts"
LASSO_CLASSIFIER_MODEL_FILE = LASSO_CLASSIFIER_ARTIFACTS_DIR / "lasso_classifier_model.joblib"
LASSO_CLASSIFIER_BEST_PARAMS_FILE = LASSO_CLASSIFIER_ARTIFACTS_DIR / "best_params.json"
LASSO_CLASSIFIER_TRAINING_RESULTS_FILE = LASSO_CLASSIFIER_ARTIFACTS_DIR / "training_results.json"
LASSO_CLASSIFIER_TEST_EVAL_FILE = LASSO_CLASSIFIER_ARTIFACTS_DIR / "test_evaluation.json"

# Logistic Regression (no penalty)
LOGISTIC_DIR = ECONOMETRIC_DIR / "logistic"
LOGISTIC_ARTIFACTS_DIR = LOGISTIC_DIR / "artifacts"
LOGISTIC_MODEL_FILE = LOGISTIC_ARTIFACTS_DIR / "logistic_model.joblib"
LOGISTIC_BEST_PARAMS_FILE = LOGISTIC_ARTIFACTS_DIR / "best_params.json"
LOGISTIC_TRAINING_RESULTS_FILE = LOGISTIC_ARTIFACTS_DIR / "training_results.json"
LOGISTIC_TEST_EVAL_FILE = LOGISTIC_ARTIFACTS_DIR / "test_evaluation.json"

# ============================================================================
# DEEP LEARNING MODEL DIRECTORIES
# ============================================================================

DL_RESULTS_DIR = RESULTS_DIR / "deep_learning"

# LSTM
LSTM_DIR = DL_RESULTS_DIR / "lstm"

# ============================================================================
# LABEL PRIMAIRE - Triple-Barrier Labeling
# ============================================================================

LABEL_PRIMAIRE_DIR = RESULTS_DIR / "label_primaire"
LABEL_PRIMAIRE_OPTIMIZATION_FILE = LABEL_PRIMAIRE_DIR / "joint_optimization.json"
LABEL_PRIMAIRE_LABELING_PARAMS_FILE = LABEL_PRIMAIRE_DIR / "labeling_params.json"
LABEL_PRIMAIRE_EVENTS_TRAIN_FILE = LABEL_PRIMAIRE_DIR / "events_train.parquet"
LABEL_PRIMAIRE_EVENTS_TEST_FILE = LABEL_PRIMAIRE_DIR / "events_test.parquet"
LABEL_PRIMAIRE_EVALUATION_FILE = LABEL_PRIMAIRE_DIR / "primary_evaluation.json"
LABEL_PRIMAIRE_MODELS_DIR = LABEL_PRIMAIRE_DIR / "models"

# ============================================================================
# LABEL META - Meta-Model & Benchmarking
# ============================================================================

LABEL_META_DIR = RESULTS_DIR / "label_meta"
LABEL_META_MODEL_FILE = LABEL_META_DIR / "meta_model.joblib"
LABEL_META_BENCHMARKS_DIR = LABEL_META_DIR / "benchmarks"
LABEL_META_MODELS_DIR = LABEL_META_DIR / "models"
LABEL_META_PARAMS_DIR = LABEL_META_DIR / "params"

# Refactored Label Meta
LABEL_META_TRAIN_DIR = RESULTS_DIR / "label_meta_train"
LABEL_META_TRAIN_MODELS_DIR = LABEL_META_TRAIN_DIR / "models"

LABEL_META_EVAL_DIR = RESULTS_DIR / "label_meta_eval"
LABEL_META_EVAL_RESULTS_DIR = LABEL_META_EVAL_DIR / "results"

# ============================================================================
# LABELING (LEGACY - DEPRECATED, use LABEL_PRIMAIRE_DIR and LABEL_META_DIR)
# ============================================================================

LABELING_DIR = RESULTS_DIR / "labeling"
LABELING_ARTIFACTS_DIR = LABELING_DIR / "artifacts"
LABELING_OPTIMIZATION_FILE = LABELING_DIR / "labeling_optimization.json"
LABELING_PARAMS_FILE = LABELING_DIR / "optimal_labeling_params.json"

# Two-stage model artifacts
META_LABELING_DIR = LABELING_DIR / "meta_labeling"
PRIMARY_MODEL_FILE = META_LABELING_DIR / "primary_model.joblib"
META_MODEL_FILE = META_LABELING_DIR / "meta_model.joblib"
META_LABELING_RESULTS_FILE = META_LABELING_DIR / "results.json"

# Events files (for meta-labeling)
EVENTS_TRAIN_PARQUET = LABELING_DIR / "events_train.parquet"
EVENTS_TEST_PARQUET = LABELING_DIR / "events_test.parquet"
JOINT_OPTIMIZATION_FILE = LABELING_DIR / "joint_optimization.json"

# ============================================================================
# LABELED DATASETS
# ============================================================================

# Raw features with labels (for tree-based ML: XGBoost, LightGBM, CatBoost, RF)
DATASET_FEATURES_LABEL_PARQUET = FEATURES_DIR / "dataset_features_label.parquet"
DATASET_FEATURES_LABEL_CSV = FEATURES_DIR / "dataset_features_label.csv"

# Z-scored features with labels (for linear models: Ridge, Lasso, OLS)
DATASET_FEATURES_LINEAR_LABEL_PARQUET = FEATURES_DIR / "dataset_features_linear_label.parquet"
DATASET_FEATURES_LINEAR_LABEL_CSV = FEATURES_DIR / "dataset_features_linear_label.csv"

# Min-max scaled features with labels (for deep learning: LSTM)
DATASET_FEATURES_LSTM_LABEL_PARQUET = FEATURES_DIR / "dataset_features_lstm_label.parquet"
DATASET_FEATURES_LSTM_LABEL_CSV = FEATURES_DIR / "dataset_features_lstm_label.csv"
