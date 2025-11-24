"""Evaluation module for ARIMA models."""

from __future__ import annotations

from src.utils.metrics import compute_residuals

from .evaluation_arima import (
    backtest_full_series,
    calculate_metrics,
    evaluate_model,
    ljung_box_on_residuals,
    plot_residuals_acf_with_ljungbox,
    rolling_forecast,
    save_evaluation_results,
    save_ljung_box_results,
)
from .model_performance import (
    plot_fitted_vs_actual,
    plot_forecast_errors,
    plot_predictions_vs_actual,
)
from .save_data_for_garch import save_garch_dataset
from .utils import detect_value_column

__all__ = [
    "backtest_full_series",
    "calculate_metrics",
    "compute_residuals",
    "evaluate_model",
    "ljung_box_on_residuals",
    "plot_fitted_vs_actual",
    "plot_forecast_errors",
    "plot_predictions_vs_actual",
    "plot_residuals_acf_with_ljungbox",
    "rolling_forecast",
    "save_evaluation_results",
    "save_ljung_box_results",
    "save_garch_dataset",
    "detect_value_column",
]
