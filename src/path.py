"""File and directory paths for the S&P 500 Forecasting project."""

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

# ============================================================================
# DATA PIPELINE - File paths
# ============================================================================

# Repo uses raw/cleaned/prepared subfolders; keep legacy aliases for S&P modules
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PREPARED_DATA_DIR = DATA_DIR / "prepared"

# Raw data files
SP500_TICKERS_FILE = DATA_DIR / "sp500_tickers.csv"
DATASET_RAW_CSV = RAW_DATA_DIR / "dataset_raw.csv"
DATASET_RAW_PARQUET = RAW_DATA_DIR / "dataset_raw.parquet"
DATASET_FILE = DATASET_RAW_CSV  # alias expected by S&P modules
DATASET_FILTERED_FILE = CLEANED_DATA_DIR / "dataset_clean.csv"
DATASET_FILTERED_PARQUET_FILE = CLEANED_DATA_DIR / "dataset_clean.parquet"

DATA_TICKERS_FULL_FILE = PREPARED_DATA_DIR / "data_tickers_full.parquet"
DATA_TICKERS_FULL_INSIGHTS_FILE = PREPARED_DATA_DIR / "data_tickers_full_insights.parquet"
DATA_TICKERS_FULL_INDICATORS_FILE = PREPARED_DATA_DIR / "data_tickers_full_indicators.parquet"
DATA_TICKERS_FULL_INSIGHTS_INDICATORS_FILE = (
    PREPARED_DATA_DIR / "data_tickers_full_insights_indicators.parquet"
)

# Prepared log-returns (map weighted aliases to existing files)
WEIGHTED_LOG_RETURNS_FILE = PREPARED_DATA_DIR / "log_returns.csv"
WEIGHTED_LOG_RETURNS_SPLIT_FILE = PREPARED_DATA_DIR / "log_returns_split.csv"
DOLLAR_IMBALANCE_BARS_PARQUET = PREPARED_DATA_DIR / "dollar_imbalance_bars.parquet"
DOLLAR_IMBALANCE_BARS_CSV = PREPARED_DATA_DIR / "dollar_imbalance_bars.csv"

# LightGBM datasets
LIGHTGBM_DATASET_COMPLETE_FILE = DATA_DIR / "lightgbm_dataset_complete.csv"
LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE = DATA_DIR / "lightgbm_dataset_without_insights.csv"
LIGHTGBM_DATASET_WITHOUT_SIGMA2_FILE = DATA_DIR / "lightgbm_dataset_without_sigma2.csv"
LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE = DATA_DIR / "lightgbm_dataset_sigma_plus_base.csv"
LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE = DATA_DIR / "lightgbm_dataset_log_volatility_only.csv"
LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE = DATA_DIR / "lightgbm_dataset_technical_indicators.csv"
LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE = DATA_DIR / "lightgbm_dataset_insights_only.csv"

# Additional technical-variant datasets (used for dedicated model variants)
# - technical_only: technical indicators without ARIMA/GARCH insights and without target lags
# - technical_plus_insights: technical indicators with ARIMA/GARCH insights and without target lags
LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE = (
    DATA_DIR / "lightgbm_dataset_technical_only_no_target_lags.csv"
)
LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE = (
    DATA_DIR / "lightgbm_dataset_technical_plus_insights_no_target_lags.csv"
)

# ============================================================================
# RESULTS DIRECTORIES - Organized by pipeline step/model
# ============================================================================

# Data pipeline results
DATA_RESULTS_DIR = RESULTS_DIR / "data"
STATIONARITY_REPORT_FILE = DATA_RESULTS_DIR / "stationarity_report.json"
FETCH_REPORT_FILE = DATA_RESULTS_DIR / "fetch_report.json"

# ARIMA results
ARIMA_RESULTS_DIR = RESULTS_DIR / "arima"
ARIMA_OPTIMIZATION_DIR = ARIMA_RESULTS_DIR / "optimization"
ARIMA_TRAINING_DIR = ARIMA_RESULTS_DIR / "training"
ARIMA_EVALUATION_DIR = ARIMA_RESULTS_DIR / "evaluation"
ARIMA_OUTPUTS_DIR = ARIMA_RESULTS_DIR / "outputs"
ARIMA_STATS_DIR = ARIMA_RESULTS_DIR / "stats"

# ARIMA optimization files
ARIMA_BEST_MODELS_FILE = ARIMA_OPTIMIZATION_DIR / "arima_best_models.json"
ARIMA_OPTIMIZATION_RESULTS_FILE = ARIMA_OPTIMIZATION_DIR / "optimization_results.csv"

# ARIMA training files
ARIMA_TRAINED_MODEL_FILE = ARIMA_TRAINING_DIR / "model.pkl"
ARIMA_TRAINED_MODEL_METADATA_FILE = ARIMA_TRAINING_DIR / "metadata.json"

# ARIMA evaluation files
ROLLING_PREDICTIONS_ARIMA_FILE = ARIMA_EVALUATION_DIR / "rolling_predictions.csv"
ROLLING_VALIDATION_METRICS_ARIMA_FILE = ARIMA_EVALUATION_DIR / "rolling_metrics.json"
LJUNGBOX_RESIDUALS_ARIMA_FILE = ARIMA_EVALUATION_DIR / "ljungbox_residuals.json"
PREDICTIONS_VS_ACTUAL_ARIMA_PLOT = ARIMA_EVALUATION_DIR / "predictions_vs_actual.png"

# ARIMA outputs (for next pipeline step)
GARCH_DATASET_FILE = ARIMA_OUTPUTS_DIR / "dataset_garch.csv"

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

# LightGBM results
LIGHTGBM_RESULTS_DIR = RESULTS_DIR / "lightgbm"
LIGHTGBM_OPTIMIZATION_DIR = LIGHTGBM_RESULTS_DIR / "optimization"
LIGHTGBM_TRAINING_DIR = LIGHTGBM_RESULTS_DIR / "training"
LIGHTGBM_MODELS_DIR = LIGHTGBM_RESULTS_DIR / "models"
LIGHTGBM_EVALUATION_DIR = LIGHTGBM_RESULTS_DIR / "evaluation"
LIGHTGBM_ABLATION_DIR = LIGHTGBM_RESULTS_DIR / "ablation"

# LightGBM optimization files
LIGHTGBM_OPTIMIZATION_RESULTS_FILE = LIGHTGBM_OPTIMIZATION_DIR / "results.json"

# LightGBM training files
LIGHTGBM_TRAINING_RESULTS_FILE = LIGHTGBM_TRAINING_DIR / "results.json"

# LightGBM evaluation files
LIGHTGBM_EVAL_RESULTS_FILE = LIGHTGBM_EVALUATION_DIR / "results.json"
LIGHTGBM_PERMUTATION_RESULTS_FILE = LIGHTGBM_EVALUATION_DIR / "permutation_importance.json"
LIGHTGBM_LEAKAGE_TEST_RESULTS_FILE = LIGHTGBM_EVALUATION_DIR / "leakage_test_results.json"

# LightGBM ablation files
LIGHTGBM_ABLATION_SIGMA2_RESULTS_FILE = LIGHTGBM_ABLATION_DIR / "sigma2_results.json"


# ============================================================================
# PLOTS DIRECTORIES - Organized by pipeline step/model
# ============================================================================

# Data pipeline plots
DATA_PLOTS_DIR = PLOTS_DIR / "data"

# ARIMA plots
ARIMA_PLOTS_DIR = PLOTS_DIR / "arima"
ARIMA_DATA_VISU_PLOTS_DIR = ARIMA_PLOTS_DIR / "data_visualization"
ARIMA_EVALUATION_PLOTS_DIR = ARIMA_PLOTS_DIR / "evaluation"
ARIMA_SEASONALITY_PLOTS_DIR = ARIMA_PLOTS_DIR / "saisonnalite"

# ARIMA data visualization plots
ARIMA_DATA_VISU_PLOTS_DIR = ARIMA_DATA_VISU_PLOTS_DIR
ARIMA_EVALUATION_PLOTS_DIR = ARIMA_EVALUATION_PLOTS_DIR
ARIMA_RESIDUALS_LJUNGBOX_PLOT = ARIMA_EVALUATION_PLOTS_DIR / "ljungbox_residuals.png"

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

# LightGBM plots
LIGHTGBM_PLOTS_DIR = PLOTS_DIR / "lightgbm"
LIGHTGBM_CORRELATION_PLOTS_DIR = LIGHTGBM_PLOTS_DIR / "correlation"
LIGHTGBM_SHAP_PLOTS_DIR = LIGHTGBM_PLOTS_DIR / "shap"
LIGHTGBM_PERMUTATION_PLOTS_DIR = LIGHTGBM_PLOTS_DIR / "permutation"


# ARIMA artifacts directory (legacy, use ARIMA_OPTIMIZATION_DIR instead)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARIMA_ARTIFACTS_DIR = ARTIFACTS_DIR / "arima"
