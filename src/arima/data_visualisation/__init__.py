"""Data visualization module for ARIMA analysis.

This module provides comprehensive visualizations for ARIMA modeling:
- Pre-modeling: Time series, stationarity, seasonality, ACF/PACF
- Residual diagnostics: Distribution, Q-Q plots, ACF, comprehensive dashboard
- Model performance: Fitted vs actual, predictions, forecast errors
"""

from __future__ import annotations

# Import from main module (which imports from all submodules)
from .main import (
    generate_all_plots,
    generate_performance_plots,
    generate_pre_modeling_plots,
    generate_residual_diagnostics,
    plot_acf_pacf,
    plot_comprehensive_residuals,
    plot_log_returns_distribution,
    plot_residuals_acf,
    plot_residuals_histogram,
    plot_residuals_qq,
    plot_residuals_timeseries,
    plot_stationarity,
    plot_weighted_series,
)

__all__ = [
    # Pre-modeling visualizations
    "plot_weighted_series",
    "plot_log_returns_distribution",
    "plot_acf_pacf",
    "plot_stationarity",
    # Returns visualizations
    # Residual diagnostics
    "plot_residuals_timeseries",
    "plot_residuals_histogram",
    "plot_residuals_qq",
    "plot_residuals_acf",
    "plot_comprehensive_residuals",
    # Orchestration functions
    "generate_pre_modeling_plots",
    "generate_residual_diagnostics",
    "generate_performance_plots",
    "generate_all_plots",
]
