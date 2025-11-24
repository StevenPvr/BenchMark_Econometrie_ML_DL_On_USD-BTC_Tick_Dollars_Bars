"""Constants for the S&P 500 Forecasting project."""

from __future__ import annotations

from datetime import datetime

# Re-export all paths from path.py for backward compatibility
from src.path import (  # noqa: F401
    ARIMA_ARTIFACTS_DIR,
    ARIMA_BEST_MODELS_FILE,
    ARIMA_DATA_VISU_PLOTS_DIR,
    ARIMA_EVALUATION_DIR,
    ARIMA_EVALUATION_PLOTS_DIR,
    ARIMA_OPTIMIZATION_DIR,
    ARIMA_OPTIMIZATION_RESULTS_FILE,
    ARIMA_OUTPUTS_DIR,
    ARIMA_PLOTS_DIR,
    ARIMA_RESIDUALS_LJUNGBOX_PLOT,
    ARIMA_RESULTS_DIR,
    ARIMA_SEASONALITY_PLOTS_DIR,
    ARIMA_STATS_DIR,
    ARIMA_TRAINED_MODEL_FILE,
    ARIMA_TRAINED_MODEL_METADATA_FILE,
    ARIMA_TRAINING_DIR,
    ARTIFACTS_DIR,
    DATA_DIR,
    DATA_PLOTS_DIR,
    DATA_RESULTS_DIR,
    DATA_TICKERS_FULL_FILE,
    DATA_TICKERS_FULL_INDICATORS_FILE,
    DATA_TICKERS_FULL_INSIGHTS_FILE,
    DATA_TICKERS_FULL_INSIGHTS_INDICATORS_FILE,
    DATASET_FILE,
    DATASET_FILTERED_FILE,
    DATASET_FILTERED_PARQUET_FILE,
    FETCH_REPORT_FILE,
    GARCH_ACF_SQUARED_PLOT,
    GARCH_DATA_VISU_PLOTS_DIR,
    GARCH_DATASET_FILE,
    GARCH_DIAGNOSTIC_DIR,
    GARCH_DIAGNOSTICS_FILE,
    GARCH_DIAGNOSTICS_PLOTS_DIR,
    GARCH_DISTRIBUTION_DIAGNOSTICS_FILE,
    GARCH_ESTIMATION_DIR,
    GARCH_ESTIMATION_FILE,
    GARCH_EVAL_METRICS_FILE,
    GARCH_EVAL_VAR_COMBINED_PLOT,
    GARCH_EVAL_VAR_RESIDUALS_PLOT,
    GARCH_EVAL_VAR_SCATTER_PLOT,
    GARCH_EVAL_VAR_TIMESERIES_PLOT,
    GARCH_EVAL_VAR_SUMMARY_FILE,
    GARCH_EVAL_VAR_VIOLATIONS_TEMPLATE,
    GARCH_EVAL_TEST_METRICS_FILE,
    GARCH_EVALUATION_DIR,
    GARCH_EVALUATION_PLOTS_DIR,
    GARCH_FORECASTS_FILE,
    GARCH_LJUNGBOX_FILE,
    GARCH_ML_DATASET_FILE,
    GARCH_MODEL_FILE,
    GARCH_MODEL_METADATA_FILE,
    GARCH_NUMERICAL_TESTS_FILE,
    GARCH_OPTIMIZATION_DIR,
    GARCH_OPTIMIZATION_RESULTS_FILE,
    GARCH_PLOTS_DIR,
    GARCH_RESIDUALS_OUTPUTS_FILE,
    GARCH_RESULTS_DIR,
    GARCH_ROLLING_DIR,
    GARCH_ROLLING_EVAL_FILE,
    GARCH_ROLLING_FORECASTS_FILE,
    GARCH_ROLLING_VARIANCE_FILE,
    GARCH_SQUARED_RESIDUALS_ACF_LB_PLOT,
    GARCH_STD_ACF_PACF_PLOT,
    GARCH_STD_HISTOGRAM_PLOT,
    GARCH_STD_QQ_PLOT,
    GARCH_STD_SQUARED_ACF_PACF_PLOT,
    GARCH_STRUCTURE_DIR,
    GARCH_STRUCTURE_PLOT,
    GARCH_STRUCTURE_PLOTS_DIR,
    GARCH_TRAINING_DIR,
    GARCH_VARIANCE_OUTPUTS_FILE,
    LIGHTGBM_ABLATION_DIR,
    LIGHTGBM_ABLATION_SIGMA2_RESULTS_FILE,
    LIGHTGBM_CORRELATION_PLOTS_DIR,
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE,
    LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
    LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE,
    LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
    LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
    LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
    LIGHTGBM_DATASET_WITHOUT_SIGMA2_FILE,
    LIGHTGBM_EVAL_RESULTS_FILE,
    LIGHTGBM_EVALUATION_DIR,
    LIGHTGBM_LEAKAGE_TEST_RESULTS_FILE,
    LIGHTGBM_MODELS_DIR,
    LIGHTGBM_OPTIMIZATION_DIR,
    LIGHTGBM_OPTIMIZATION_RESULTS_FILE,
    LIGHTGBM_PERMUTATION_PLOTS_DIR,
    LIGHTGBM_PERMUTATION_RESULTS_FILE,
    LIGHTGBM_PLOTS_DIR,
    LIGHTGBM_RESULTS_DIR,
    LIGHTGBM_SHAP_PLOTS_DIR,
    LIGHTGBM_TRAINING_DIR,
    LIGHTGBM_TRAINING_RESULTS_FILE,
    LJUNGBOX_RESIDUALS_ARIMA_FILE,
    PLOTS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    ROLLING_PREDICTIONS_ARIMA_FILE,
    ROLLING_VALIDATION_METRICS_ARIMA_FILE,
    SP500_TICKERS_FILE,
    STATIONARITY_REPORT_FILE,
    WEIGHTED_LOG_RETURNS_FILE,
    WEIGHTED_LOG_RETURNS_SPLIT_FILE,
)

# ============================================================================
# MODEL PARAMETERS & DEFAULTS
# ============================================================================

# ARIMA defaults (explicitly fixed, no optimization)
# NOTE: ARIMA(0,0,0) = white noise model with NO forecasting capability
# This is intentional - ARIMA serves ONLY to generate residuals for GARCH
# The residuals from ARIMA(0,0,0) are essentially the log returns themselves
# GARCH then models the conditional volatility of these residuals
ARIMA_DEFAULT_ORDER: tuple[int, int, int] = (0, 0, 0)
ARIMA_DEFAULT_REFIT_EVERY: int = 21  # Refit every ~1 trading month (21 business days)
ARIMA_DEFAULT_TREND: str = "c"  # Constant trend (intercept only)

# ARIMA error messages
ARIMA_EMPTY_TRAINING_SERIES_MSG: str = "Training series cannot be empty"

# Dataset split labels
TRAIN_SPLIT_LABEL: str = "train"
TEST_SPLIT_LABEL: str = "test"

# LightGBM model names
LIGHTGBM_COMPLETE_MODEL_NAME: str = "lightgbm_complete"
LIGHTGBM_DATASET_COMPLETE_MODEL_NAME: str = "lightgbm_dataset_complete"

# Leakage detection thresholds
LEAKAGE_R2_THRESHOLD: float = 0.1

# ARIMA parameter ranges (legacy for optimization compatibility; not used now)


# LightGBM train/test split ratio
LIGHTGBM_TRAIN_TEST_SPLIT_RATIO: float = 0.8

ARIMA_MIN_SERIES_LENGTH_DIFFERENCED: int = 10  # Minimum differenced series length for validation
ARIMA_LJUNG_BOX_LAGS_DEFAULT: int = (
    20  # Default Ljung-Box lags used during ARIMA optimisation diagnostics
)

# ARIMA seasonality periods for decomposition
ARIMA_SEASONALITY_DAILY_PERIOD: int = 7  # Daily seasonality (7 days per week)
ARIMA_SEASONALITY_MONTHLY_PERIOD: int = 12  # Monthly seasonality (12 months per year)

# ARIMA minimum training size for residuals analysis
ARIMA_MIN_TRAIN_SIZE_FOR_RESIDUALS: int = (
    30  # Minimum training size for reliable residuals analysis
)

# ARIMA evaluation constants
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
    0.17  # 4% of TRAIN for burn-in (≈406 days for 2400 train days)
)
GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE: int = 20  # Minimum validation window size
GARCH_OPTIMIZATION_N_TRIALS: int = 100
GARCH_OPTIMIZATION_N_SPLITS: int = 5
GARCH_OPTIMIZATION_DISTRIBUTIONS: tuple[str, ...] = (
    "student",
    "skewt",
)

GARCH_OPTIMIZATION_REFIT_FREQ_OPTIONS: tuple[int, ...] = (1, 5, 15, 21, 63)
GARCH_OPTIMIZATION_WINDOW_TYPES: tuple[str, ...] = ("expanding", "rolling")
GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES: tuple[int, ...] = (500, 1000, 1500, 2000)

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

# LightGBM defaults
# Lags restricted to 1–3 to avoid lookahead bias and over-engineering
LIGHTGBM_LAG_WINDOWS: tuple[int, ...] = (1, 2, 3)

# Base features shared across all datasets now include log_volatility (with lags)
LIGHTGBM_BASE_FEATURE_COLUMNS: tuple[str, ...] = ("log_volatility",)

LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS: tuple[str, ...] = ("log_sigma_garch",)
LIGHTGBM_OPTIMIZATION_N_SPLITS: int = 5
LIGHTGBM_OPTIMIZATION_N_TRIALS: int = 100
LIGHTGBM_OPTIMIZATION_SAMPLE_FRACTION: float = 0.8  # Use 80% of train data for optimization
LIGHTGBM_OPTIMIZATION_MAX_WORKERS: int = 4  # Optimize 4 models in parallel


# Custom ML feature windows
LIGHTGBM_VOL_MA_SHORT_WINDOW: int = 5
LIGHTGBM_VOL_MA_LONG_WINDOW: int = 20
LIGHTGBM_TURNOVER_MA_WINDOW: int = 5
LIGHTGBM_REALIZED_VOL_WINDOW: int = 5

# New technical feature set used for lagging (calendar features are excluded from lags)
# Note: realized_vol_5d is not included as we use log_volatility (the target) for lags
LIGHTGBM_TECHNICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "log_return",
    "abs_ret",
    "ret_sq",
    "log_volatility",
    "log_volume",
    "log_volume_rel_ma_5",
    "log_volume_zscore_20",
    "log_turnover",
    "turnover_rel_ma_5",
    "obv",
    "atr",
)

# Calendar features (non-lagged) to include in technical indicator datasets
# Kept separate to avoid creating lags for these discrete/time-indexed signals
LIGHTGBM_CALENDAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "day_of_week",
    "month",
    "is_month_end",
    "day_in_month_norm",
)

# ============================================================================
# DATA PIPELINE CONSTANTS
# ============================================================================

# Data fetching constants
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DATA_FETCH_END_DATE = datetime(2024, 12, 31)
DATA_FETCH_START_DATE = datetime(2013, 1, 1)
WIKIPEDIA_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) " "AppleWebKit/537.36 (KHTML, like Gecko)"
)
WIKIPEDIA_REQUEST_TIMEOUT = 15
REQUIRED_TICKER_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
DATA_FETCH_LOG_PROGRESS_INTERVAL = 50

# Data cleaning constants
TOP_N_TICKERS_REPORT: int = 10

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

# Required column sets for data validation
REQUIRED_OHLCV_COLUMNS: list[str] = ["date", "tickers", "close", "volume"]
REQUIRED_COLS_TICKER_DATA: set[str] = {"date", "ticker", "high", "low", "close", "volume"}
REQUIRED_COLS_WEIGHTED_RETURNS: set[str] = {"date", "weighted_log_return"}
REQUIRED_COLS_SPLIT_DATA: set[str] = {"date", "weighted_log_return", "split"}
REQUIRED_COLS_LOG_RETURN: set[str] = {"ticker", "date", "log_return"}
REQUIRED_COLS_CLOSE_PRICE: set[str] = {"ticker", "date", "close"}

# Column name constants for data fetching and processing
WIKIPEDIA_SYMBOL_COLUMN: str = "Symbol"
WIKIPEDIA_TICKER_COLUMN: str = "ticker"
YF_DATE_COLUMN: str = "Date"
YF_OPEN_COLUMN: str = "Open"
YF_HIGH_COLUMN: str = "High"
YF_LOW_COLUMN: str = "Low"
YF_CLOSE_COLUMN: str = "Close"
YF_ADJ_CLOSE_COLUMN: str = "Adj Close"
YF_VOLUME_COLUMN: str = "Volume"

# Normalized column names (snake_case)
NORMALIZED_DATE_COLUMN: str = "date"
NORMALIZED_HIGH_COLUMN: str = "high"
NORMALIZED_LOW_COLUMN: str = "low"
NORMALIZED_CLOSE_COLUMN: str = "close"
NORMALIZED_ADJ_CLOSE_COLUMN: str = "adj_close"
NORMALIZED_VOLUME_COLUMN: str = "volume"
NORMALIZED_TICKERS_COLUMN: str = "tickers"

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

# Seasonal decomposition constants
SEASONAL_RESAMPLE_FREQ_WEEKLY: str = "W"
SEASONALITY_SEPARATOR_LENGTH: int = 60
STATIONARITY_TEXT_BOX_X: float = 0.02
STATIONARITY_TEXT_BOX_Y: float = 0.98

# Data loading constants
YEAR_MIN: int = 1900
YEAR_MAX: int = 2100

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

# ============================================================================
# GENERAL CONSTANTS
# ============================================================================

DEFAULT_RANDOM_STATE = 42
DATE_FORMAT_DEFAULT: str = "%Y-%m-%d"

# Default placeholder date for fabricated datetime indices
DEFAULT_PLACEHOLDER_DATE: str = "2000-01-01"

# CSV export constants
CSV_ENCODING_DEFAULT: str = "utf-8"
CSV_SEPARATOR_DEFAULT: str = ","
CSV_QUOTING_DEFAULT: int = 0  # csv.QUOTE_MINIMAL
CSV_LINETERMINATOR_DEFAULT: str = "\n"

# Ticker encoding constants
TICKER_CRC32_MASK: int = 0x7FFFFFFF

# Rolling liquidity weights default window
LIQUIDITY_WEIGHTS_WINDOW_DEFAULT: int = 20

# Rolling forecast progress reporting interval
ROLLING_FORECAST_PROGRESS_INTERVAL: int = 10

# Training display separator length
TRAINING_SEPARATOR_LENGTH: int = 70

# LightGBM default verbosity (silent)
LIGHTGBM_DEFAULT_VERBOSITY: int = -1

# Permutation importance defaults
PERMUTATION_DEFAULT_BLOCK_SIZE: int = 20
PERMUTATION_DEFAULT_N_REPEATS: int = 200
PERMUTATION_DEFAULT_SAMPLE_FRACTION: float = 0.3
PERMUTATION_DEFAULT_TOP_K_FEATURES: int = 20
