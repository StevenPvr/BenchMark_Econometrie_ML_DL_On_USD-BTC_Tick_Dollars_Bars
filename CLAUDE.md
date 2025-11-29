# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Volatility forecasting benchmark comparing econometric models (Ridge, Lasso, OLS), ML models (XGBoost, LightGBM, CatBoost, Random Forest), and a lightweight LSTM on BTC-USD tick data using dollar bars.

## Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/data_preparation/test_preparation.py

# Run a single test class/method
python -m pytest tests/data_preparation/test_preparation.py::TestDollarBars::test_compute_dollar_bars_basic
```

## Architecture

### Data Pipeline Flow

```
data_fetching → data_cleaning → data_preparation → arima → garch → model (ML/DL)
```

1. **data_fetching**: Downloads BTC/USD trades via ccxt and persists raw ticks
2. **data_cleaning**: Validates, filters, and cleans raw ticker data
3. **data_preparation**: Computes log returns, builds dollar bars (De Prado methodology), train/test splits
4. **arima**: ARIMA(0,0,0) extracts demeaned residuals for GARCH input (no forecasting capability by design)
5. **garch**: EGARCH with Student-t/Skew-t innovations for volatility forecasting (σ²)
6. **model**: Benchmark models consuming GARCH features

### Key Directories

- `src/constants.py`: All model parameters, thresholds, and defaults
- `src/path.py`: All file paths (import paths from here, not hardcode)
- `src/utils/`: Shared utilities (datetime, financial calculations, metrics, transforms, validation, logging)
- `src/garch/garch_params/`: EGARCH MLE estimation + Optuna hyperparameter optimization
- `src/garch/garch_eval/`: Evaluation metrics, VaR backtests, Mincer-Zarnowitz calibration

### Model Hierarchy

All models inherit from `src/model/base.py::BaseModel`:

- `fit(X, y)` / `predict(X)` / `save(path)` / `load(path)`

Model implementations:

- `src/model/econometrie/`: Ridge, Lasso, GARCH
- `src/model/machine_learning/`: XGBoost, LightGBM, CatBoost, Random Forest
- `src/model/deep_learning/`: LSTM

### Data Files

Data paths defined in `src/path.py`. Key files:

- `data/prepared/log_returns.parquet`: Processed log returns
- `data/prepared/dollar_imbalance_bars.parquet`: Dollar bars (De Prado)
- `data/prepared/data_tickers_full_insights.parquet`: Features with GARCH volatility forecasts

### Anti-Leakage Safeguards

The codebase is designed to prevent data leakage:

- GARCH optimization uses only TRAIN data (60% train, 30% val, 10% internal test)
- Final evaluation uses TEST holdout (20% of full data)
- `--forecast-mode no_refit` (default): frozen parameters, zero leakage
- Walk-forward validation: never uses future data
- Feature lags restricted to 1-3 to avoid lookahead bias

## Running Individual Pipeline Steps

```bash
# ARIMA pipeline
python src/arima/data_visualisation/main.py
python src/arima/stationnarity_check/main.py
python src/arima/training_arima/main.py
python src/arima/evaluation_arima/main.py

# GARCH pipeline
python src/garch/garch_data_visualisation/main.py
python src/garch/garch_numerical_test/main.py
python src/garch/structure_garch/main.py
python src/garch/garch_params/main.py
python src/garch/training_garch/main.py
python src/garch/garch_diagnostic/main.py
python src/garch/garch_eval/main.py --forecast-mode no_refit
```

## Key Libraries

- `arch`: GARCH model estimation
- `statsmodels`: ARIMA, statistical tests
- `optuna`: Hyperparameter optimization
- `lightgbm`, `xgboost`, `catboost`: Gradient boosting
- `torch`: LSTM implementation
- `pandas`, `numpy`: Data manipulation
