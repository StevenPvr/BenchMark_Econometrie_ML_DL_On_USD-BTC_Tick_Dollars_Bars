"""Data visualisation utilities for dollar bars log-returns analysis."""

from __future__ import annotations

from .analysis import run_full_analysis
from .autocorrelation import (
    compute_autocorrelation,
    compute_autocorrelation_squared,
    run_ljung_box_test,
)
from .distribution import plot_log_returns_distribution, run_normality_tests
from .io import compute_log_returns, load_dollar_bars
from .stationarity import plot_stationarity, run_stationarity_tests
from .time_series import plot_log_returns_time_series
from .trend import (
    compute_trend_statistics,
    mann_kendall_test,
    plot_trend_analysis,
    plot_trend_extraction,
)

__all__ = [
    # I/O
    "load_dollar_bars",
    "compute_log_returns",
    # Distribution
    "plot_log_returns_distribution",
    "run_normality_tests",
    # Stationarity
    "run_stationarity_tests",
    "plot_stationarity",
    # Time series
    "plot_log_returns_time_series",
    # Autocorrelation
    "compute_autocorrelation",
    "compute_autocorrelation_squared",
    "run_ljung_box_test",
    # Trend
    "mann_kendall_test",
    "compute_trend_statistics",
    "plot_trend_extraction",
    "plot_trend_analysis",
    # Full analysis
    "run_full_analysis",
]
