"""Constants for the BTC/USD dollar-bar forecasting project."""

from __future__ import annotations

from datetime import datetime

# Column name constants
CLOSE_COLUMN: str = "close"
LOG_RETURN_COLUMN: str = "log_return"

# Re-export available paths from path.py for backward compatibility
from src.path import (  # noqa: F401
    # Basic directories
    DATA_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    ARTIFACTS_DIR,

    # Data directories
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    PREPARED_DATA_DIR,

    # Raw data files
    DATASET_RAW_CSV,
    DATASET_RAW_PARQUET,

    # Cleaned data files
    DATASET_CLEAN_CSV,
    DATASET_CLEAN_PARQUET,

    # Prepared data files
    DATA_TICKERS_FULL_FILE,
    DATA_TICKERS_FULL_INDICATORS_FILE,
    DATA_TICKERS_FULL_INSIGHTS_FILE,
    DATA_TICKERS_FULL_INSIGHTS_INDICATORS_FILE,
    DATASET_FILE,
    DATASET_FILTERED_FILE,
    DATASET_FILTERED_PARQUET_FILE,
    WEIGHTED_LOG_RETURNS_FILE,
    LOG_RETURNS_PARQUET,
    LOG_RETURNS_CSV,
    DOLLAR_BARS_PARQUET,
    DOLLAR_BARS_CSV,

    # Results directories
    DATA_RESULTS_DIR,
    STATIONARITY_REPORT_FILE,
    FETCH_REPORT_FILE,

    # GARCH dataset file
    GARCH_DATASET_FILE,

    # GARCH paths
    GARCH_RESULTS_DIR,
    GARCH_STRUCTURE_DIR,
    GARCH_ESTIMATION_DIR,
    GARCH_TRAINING_DIR,
    GARCH_DIAGNOSTIC_DIR,
    GARCH_EVALUATION_DIR,
    GARCH_ROLLING_DIR,
    GARCH_OPTIMIZATION_DIR,
    GARCH_PLOTS_DIR,
    GARCH_DATA_VISU_PLOTS_DIR,
    GARCH_STRUCTURE_PLOTS_DIR,
    GARCH_DIAGNOSTICS_PLOTS_DIR,
    GARCH_EVALUATION_PLOTS_DIR,

    # GARCH files
    GARCH_DIAGNOSTICS_FILE,
    GARCH_NUMERICAL_TESTS_FILE,
    GARCH_ESTIMATION_FILE,
    GARCH_OPTIMIZATION_RESULTS_FILE,
    GARCH_MODEL_FILE,
    GARCH_MODEL_METADATA_FILE,
    GARCH_RESIDUALS_OUTPUTS_FILE,
    GARCH_VARIANCE_OUTPUTS_FILE,
    GARCH_LJUNGBOX_FILE,
    GARCH_DISTRIBUTION_DIAGNOSTICS_FILE,
    GARCH_FORECASTS_FILE,
    GARCH_EVAL_METRICS_FILE,
    GARCH_EVAL_VAR_SUMMARY_FILE,
    GARCH_EVAL_TEST_METRICS_FILE,
    GARCH_ROLLING_FORECASTS_FILE,
    GARCH_ROLLING_EVAL_FILE,
    GARCH_ROLLING_VARIANCE_FILE,
    GARCH_ML_DATASET_FILE,

    # GARCH plots
    GARCH_STRUCTURE_PLOT,
    GARCH_STD_SQUARED_ACF_PACF_PLOT,
    GARCH_STD_ACF_PACF_PLOT,
    GARCH_STD_QQ_PLOT,
    GARCH_STD_HISTOGRAM_PLOT,
    GARCH_ACF_SQUARED_PLOT,
    GARCH_SQUARED_RESIDUALS_ACF_LB_PLOT,
    GARCH_EVAL_VAR_TIMESERIES_PLOT,
    GARCH_EVAL_VAR_SCATTER_PLOT,
    GARCH_EVAL_VAR_RESIDUALS_PLOT,
    GARCH_EVAL_VAR_COMBINED_PLOT,
    GARCH_EVAL_VAR_VIOLATIONS_TEMPLATE,

    # Project root
    PROJECT_ROOT,
)

# ============================================================================
# DATA VALIDATION
# ============================================================================

# Required columns for BTC/USD tick data validation
REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "price",
    "amount",
)

# ============================================================================
# MODEL PARAMETERS & DEFAULTS
# ============================================================================

# General defaults
DEFAULT_RANDOM_STATE: int = 42  # Seed for reproducibility
DEFAULT_PLACEHOLDER_DATE: str = "2024-01-01"  # Placeholder date for synthetic date ranges

# Dataset split labels
TRAIN_SPLIT_LABEL: str = "train"
TEST_SPLIT_LABEL: str = "test"

# Leakage detection thresholds
LEAKAGE_R2_THRESHOLD: float = 0.1

# LightGBM train/test split ratio
LIGHTGBM_TRAIN_TEST_SPLIT_RATIO: float = 0.8

# LightGBM realized volatility window (for rolling std calculation)
LIGHTGBM_REALIZED_VOL_WINDOW: int = 21  # ~1 trading month

# Evaluation plot constants
EVAL_FIGURE_SIZE: tuple[int, int] = (10, 4)
EVAL_DPI: int = 150
TEXT_POSITION_X: float = 0.02
TEXT_POSITION_Y: float = 0.95

# GARCH defaults and constraints
GARCH_MIN_INIT_VAR: float = 1e-10

GARCH_STUDENT_NU_MIN: float = 2.5
GARCH_STUDENT_NU_MAX: float = 200.0
GARCH_STUDENT_NU_INIT: float = 8.0
GARCH_SKEWT_LAMBDA_MIN: float = -0.99
GARCH_SKEWT_LAMBDA_MAX: float = 0.99
GARCH_SKEWT_LAMBDA_INIT: float = -0.1
# GARCH parameter estimation defaults
GARCH_ESTIMATION_MIN_OBSERVATIONS: int = 10
GARCH_ESTIMATION_PARALLEL_WORKERS: int = 3
GARCH_ESTIMATION_PENALTY_VALUE: float = 1e50
GARCH_ESTIMATION_BETA_MIN: float = -0.999
GARCH_ESTIMATION_BETA_MAX: float = 0.999
GARCH_ESTIMATION_OMEGA_BOUND_MIN: float = -50.0
GARCH_ESTIMATION_OMEGA_BOUND_MAX: float = 50.0
GARCH_ESTIMATION_ALPHA_BOUND_MIN: float = -5.0
GARCH_ESTIMATION_ALPHA_BOUND_MAX: float = 5.0
GARCH_ESTIMATION_GAMMA_BOUND_MIN: float = -5.0
GARCH_ESTIMATION_GAMMA_BOUND_MAX: float = 5.0
GARCH_ESTIMATION_INIT_BETA: float = 0.95
GARCH_ESTIMATION_INIT_BETA2: float = 0.01  # Small initial value for beta2 in EGARCH(o,2)
GARCH_ESTIMATION_INIT_ALPHA: float = 0.1
GARCH_ESTIMATION_INIT_GAMMA: float = 0.0
GARCH_ESTIMATION_NU_MIN_THRESHOLD: float = 2.0
GARCH_ESTIMATION_KAPPA_ADJUSTMENT_COEFF: float = 0.1
GARCH_ESTIMATION_KAPPA_EPSILON: float = 1e-12
# GARCH optimization convergence settings
GARCH_ESTIMATION_MAXITER: int = 1000  # Maximum iterations for SLSQP
GARCH_ESTIMATION_FTOL: float = 1e-7  # Function tolerance for convergence
GARCH_ESTIMATION_EPS: float = 1e-8  # Step size for numerical derivatives
# GARCH Skew-t distribution constants
GARCH_SKEWT_COEFF_A_MULTIPLIER: float = 4.0
GARCH_SKEWT_COEFF_B_SQ_TERM1: float = 1.0
GARCH_SKEWT_COEFF_B_SQ_TERM2: float = 3.0

# GARCH diagnostics parameters (must be specified explicitly)
GARCH_LJUNG_BOX_SPECIFIC_LAGS: list[int] = [10, 20]  # Specific lags for comprehensive diagnostics
GARCH_STD_EPSILON: float = 1e-12  # Small epsilon for numerical stability in standardization
GARCH_LM_LAGS_DEFAULT: int = 5  # Default lags for ARCH-LM test
GARCH_ACF_LAGS_DEFAULT: int = 20  # Default lags for ACF plots in GARCH diagnostics
GARCH_LJUNG_BOX_LAGS_DEFAULT: int = 10  # Default lags for Ljung-Box test

# GARCH numerical tests - minimum observation requirements
GARCH_NUMERICAL_DEFAULT_ALPHA: float = 0.05
GARCH_NUMERICAL_MIN_OBS_BREUSCH_PAGAN: int = 3
GARCH_NUMERICAL_MIN_OBS_WHITE: int = 5

# GARCH numerical tests - polynomial orders for heteroskedasticity tests
GARCH_NUMERICAL_BREUSCH_PAGAN_POLYNOMIAL_ORDER: int = 2
GARCH_NUMERICAL_WHITE_POLYNOMIAL_ORDER: int = 4

# GARCH numerical tests - normalization epsilon for numerical stability
GARCH_NUMERICAL_WHITE_NORMALIZATION_EPSILON: float = 1e-10

GARCH_NUMERICAL_TEST_NAME_LJUNG_BOX_RESIDUALS: str = "Ljung-Box Test sur les résidus"
GARCH_NUMERICAL_TEST_NAME_LJUNG_BOX_SQUARED: str = "Ljung-Box Test sur les résidus au carré"
GARCH_NUMERICAL_TEST_NAME_ENGLE_ARCH_LM: str = "Engle ARCH LM Test"
GARCH_NUMERICAL_TEST_NAME_MCLEOD_LI: str = "McLeod-Li Test"

GARCH_MIN_WINDOW_SIZE: int = 250  # Minimum window size to start forecasting (≈1 year of trading)
GARCH_ESSENTIAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "sigma2_egarch_raw",
    "sigma_garch",
    "sigma2_garch",
)
# For data_tickers_full_insights, only log_sigma_garch is added
GARCH_INSIGHTS_COLUMN: str = "log_sigma_garch"
GARCH_INITIAL_WINDOW_SIZE_DEFAULT: int = 500  # Default initial window size for GARCH training
GARCH_LOG_VAR_MAX: float = 700.0  # Maximum log-variance for numerical stability
GARCH_LOG_VAR_MIN: float = -700.0  # Minimum log-variance for numerical stability
GARCH_LOG_VAR_EXPLOSION_THRESHOLD: float = 300.0
GARCH_QLIKE_MAX_ACCEPTABLE: float = 100.0
GARCH_FIT_MIN_SIZE: int = 10


# GARCH calibration defaults
GARCH_CALIBRATION_EPS: float = 1e-12


# GARCH hyperparameter optimization defaults
GARCH_OPTIMIZATION_BURN_IN_RATIO: float = (
    0.10  # 10% of TRAIN for burn-in (reduced from 17% for faster optimization)
)
GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE: int = 20  # Minimum validation window size
GARCH_OPTIMIZATION_N_TRIALS: int = 30  # Reduced from 100 for faster optimization
GARCH_OPTIMIZATION_N_SPLITS: int = 3  # Reduced from 5 for faster optimization
GARCH_OPTIMIZATION_DATA_FRACTION: float = 0.5  # Use 50% of TRAIN data for optimization
GARCH_OPTIMIZATION_DISTRIBUTIONS: tuple[str, ...] = (
    "student",
    "skewt",
)

# EGARCH order options: (o, p) pairs to test
# (1,1) = standard EGARCH, (2,1) = 2 ARCH lags for better shock capture
GARCH_OPTIMIZATION_ORDER_OPTIONS: tuple[tuple[int, int], ...] = ((1, 1), (2, 1))

GARCH_OPTIMIZATION_REFIT_FREQ_OPTIONS: tuple[int, ...] = (5, 21, 63)  # weekly, monthly, quarterly
GARCH_OPTIMIZATION_WINDOW_TYPES: tuple[str, ...] = ("expanding",)  # Fixed to expanding only
GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES: tuple[int, ...] = (1000,)  # Simplified (not used with expanding)

# GARCH hyperparameter optimization - 3-phase validation split
GARCH_VALIDATION_TRAIN_RATIO: float = 0.7  # 70% for training
GARCH_VALIDATION_VAL_RATIO: float = 0.2  # 20% for validation (hyperparameter tuning)
GARCH_VALIDATION_TEST_RATIO: float = 0.1  # 10% for final test (evaluation only)

GARCH_EVAL_EPSILON: float = 1e-12
GARCH_EVAL_MIN_ALPHA: float = 1e-6
GARCH_EVAL_AIC_MULTIPLIER: float = 2.0
GARCH_EVAL_HALF: float = 0.5
GARCH_EVAL_MIN_OBS: int = 2  # Minimum observations for Mincer-Zarnowitz regression
GARCH_EVAL_DEFAULT_ALPHAS: tuple[float, ...] = (0.01, 0.05)
GARCH_EVAL_DEFAULT_LEVEL: float = 0.95  # Default confidence level for VaR evaluation
GARCH_EVAL_DEFAULT_SLOPE: float = 1.0  # Neutral slope for MZ calibration fallback
GARCH_EVAL_FORCED_MIN_START_SIZE: int = 200
GARCH_EVAL_FORECAST_MODE_NO_REFIT: str = "no_refit"
GARCH_EVAL_FORECAST_MODE_HYBRID: str = "hybrid"
GARCH_EVAL_FORECAST_MODE_CHOICES: tuple[str, ...] = (
    GARCH_EVAL_FORECAST_MODE_NO_REFIT,
    GARCH_EVAL_FORECAST_MODE_HYBRID,
)
GARCH_EVAL_FORECAST_MODE_DEFAULT: str = GARCH_EVAL_FORECAST_MODE_NO_REFIT


GARCH_MODEL_NAMES: tuple[str, ...] = ("egarch_student", "egarch_skewt")
GARCH_MODEL_PARAMS_COUNT: dict[str, int] = {
    "egarch_student": 5,
    "egarch_skewt": 6,
}
# ============================================================================
# DATA PIPELINE CONSTANTS
# ============================================================================

# Crypto tick download defaults (used by ccxt pipeline + tests)
EXCHANGE_ID: str = "binance"  # Try Binance for potentially more data
SYMBOL: str = "BTC/USDT"  # Binance uses USDT instead of USD
START_DATE: str = "2024-02-01"  # 3-month period for more data
END_DATE: str = "2024-02-05"

# Data conversion constants
MAX_ERROR_DATES_DISPLAY: int = 5

# Data preparation constants
TRAIN_RATIO_DEFAULT: float = 0.8
TIMESERIES_SPLIT_N_SPLITS: int = 2
STATIONARITY_RESAMPLE_FREQ: str = "W"

# Time series analysis defaults
ACF_PACF_DEFAULT_LAGS: int = 30
ACF_PACF_MIN_LAGS: int = 1
STATIONARITY_ROLLING_WINDOW_DEFAULT: int = 252
STATIONARITY_DEFAULT_ALPHA: float = 0.05
ZIVOT_ANDREWS_TRIM: float = 0.15

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================

# Plot styling defaults
PLOT_ALPHA_DEFAULT: float = 0.8
PLOT_ALPHA_LIGHT: float = 0.3
PLOT_ALPHA_MEDIUM: float = 0.7
PLOT_ALPHA_FILL: float = 0.2

# Color constants for plots
COLOR_NORMAL_FIT: str = "#A23B72"
COLOR_RESIDUAL: str = "#2E86AB"
COLOR_SERIES_ORIGINAL: str = "blue"
COLOR_TEST: str = "#A23B72"
COLOR_TRAIN: str = "#2E86AB"
COLOR_ACTUAL: str = "#2E86AB"
COLOR_PREDICTION: str = "#F18F01"
COLOR_SPLIT_LINE: str = "red"

# Figure size constants
FIGURE_SIZE_ACF_PACF: tuple[int, int] = (14, 4)
FIGURE_SIZE_DEFAULT: tuple[int, int] = (10, 4)
FIGURE_SIZE_STATIONARITY: tuple[int, int] = (16, 12)
FIGURE_SIZE_WEIGHTED_SERIES: tuple[int, int] = (18, 6)

# Font size constants
FONTSIZE_AXIS: int = 10
FONTSIZE_LABEL: int = 12
FONTSIZE_SUBTITLE: int = 12
FONTSIZE_TITLE: int = 14
FONTSIZE_TEXT: int = 9

# GARCH diagnostic plotting constants
GARCH_Z_CONF: float = 1.96
GARCH_DIAGNOSTIC_FIGURE_SIZE: tuple[int, int] = (10, 6)
GARCH_QQ_FIGURE_SIZE: tuple[int, int] = (6, 6)
GARCH_PLOT_ALPHA: float = 0.8
GARCH_BAR_WIDTH: float = 0.8
GARCH_COLOR_ACF: str = "#1f77b4"
GARCH_COLOR_PACF: str = "#ff7f0e"
GARCH_COLOR_CONFIDENCE: str = "red"
GARCH_COLOR_ZERO_LINE: str = "black"
GARCH_COLOR_GRAY: str = "gray"
GARCH_COLOR_TRAIN: str = "#1f77b4"
GARCH_COLOR_TEST: str = "#ff7f0e"
GARCH_COLOR_TRAIN_STD: str = "#2ca02c"
GARCH_COLOR_TEST_STD: str = "#d62728"
GARCH_LINEWIDTH: float = 1.0
GARCH_LINESTYLE_DASHED: str = "--"
GARCH_LINESTYLE_DOTTED: str = ":"
GARCH_LEGEND_LOC: str = "upper right"
GARCH_SCATTER_SIZE: int = 8
GARCH_QQ_PROB_OFFSET: float = 0.5
GARCH_STD_ERROR_DENOMINATOR: float = 1.0

# Line width constants
LINEWIDTH_BOLD: float = 1.5
LINEWIDTH_DEFAULT: float = 0.6
LINEWIDTH_THIN: float = 0.8

# Distribution and data constants
DISTRIBUTION_HISTOGRAM_BINS: int = 100
RESIDUALS_HISTOGRAM_BINS: int = 50
STATISTICS_PRECISION: int = 6

# Text box styling
TEXTBOX_STYLE_DEFAULT: dict[str, str | float] = {
    "boxstyle": "round",
    "facecolor": "wheat",
    "alpha": 0.8,
}
TEXTBOX_STYLE_INFO: dict = {
    "boxstyle": "round",
    "facecolor": "lightblue",
    "alpha": 0.8,
}

# Statistics and data constants
MAX_POINTS_SUBSAMPLE: int = 500

# CSV output defaults
CSV_ENCODING_DEFAULT: str = "utf-8"
CSV_SEPARATOR_DEFAULT: str = ","
CSV_LINETERMINATOR_DEFAULT: str = "\n"
CSV_QUOTING_DEFAULT: int = 0  # csv.QUOTE_MINIMAL

# Date format defaults
DATE_FORMAT_DEFAULT: str = "%Y-%m-%d"

# Ticker ID generation
TICKER_CRC32_MASK: int = 0x7FFFFFFF  # Mask to positive int32

# Rolling forecast progress reporting
ROLLING_FORECAST_PROGRESS_INTERVAL: int = 100

# Stationarity plot text box position
STATIONARITY_TEXT_BOX_X: float = 0.02
STATIONARITY_TEXT_BOX_Y: float = 0.95

# Seasonality plot constants
SEASONALITY_SEPARATOR_LENGTH: int = 50
SEASONAL_RESAMPLE_FREQ_WEEKLY: str = "W"

# Year range for date validation
YEAR_MIN: int = 1900
YEAR_MAX: int = 2100

# LightGBM dataset file
LIGHTGBM_DATASET_COMPLETE_FILE: str = "lightgbm_dataset_complete.parquet"

# ============================================================================
# LABELING CONSTANTS
# ============================================================================

# Default triple barrier labeling parameters
DEFAULT_PT_MULT: float = 1.0  # Profit-taking multiplier
DEFAULT_SL_MULT: float = 1.0  # Stop-loss multiplier
DEFAULT_MAX_HOLDING: int = 20  # Maximum holding period in bars
DEFAULT_MIN_RETURN: float = 0.0  # Minimum return threshold (percentage)

# Log return scale factor (dollar bars use log_return × 100)
LOG_RETURN_SCALE: float = 100.0

# Volatility estimation
DEFAULT_VOL_WINDOW: int = 21  # Window for daily volatility estimation

# ============================================================================
# OPTIMIZATION CONSTANTS (Label Primaire)
# ============================================================================

# Optuna optimization defaults
OPTI_N_TRIALS_DEFAULT: int = 50  # Number of Optuna trials
OPTI_N_FOLDS_DEFAULT: int = 5  # Number of walk-forward folds
OPTI_TIMEOUT_DEFAULT: int = 3600  # Timeout in seconds (1 hour)
OPTI_METRIC_DEFAULT: str = "mcc_weighted"  # Default optimization metric

# Walk-forward CV parameters
OPTI_MIN_TRAIN_SIZE: int = 500  # Minimum training samples per fold
OPTI_EMBARGO_PCTG: float = 0.01  # Embargo percentage to prevent leakage

# Triple barrier search space bounds
TRIPLE_BARRIER_PT_MULT_MIN: float = 0.5
TRIPLE_BARRIER_PT_MULT_MAX: float = 3.0
TRIPLE_BARRIER_SL_MULT_MIN: float = 0.5
TRIPLE_BARRIER_SL_MULT_MAX: float = 3.0
TRIPLE_BARRIER_MIN_RETURN_MIN: float = 0.1  # 0.1% minimum return
TRIPLE_BARRIER_MIN_RETURN_MAX: float = 2.0  # 2.0% maximum return
TRIPLE_BARRIER_MAX_HOLDING_MIN: int = 5
TRIPLE_BARRIER_MAX_HOLDING_MAX: int = 50
