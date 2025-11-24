"""GARCH data visualization module.

Provides functions for visualizing ARIMA residuals distribution and ACF.

Note: Functions for returns visualization have been removed as the
returns clustering plot was deleted from the project.
"""

from __future__ import annotations

from src.garch.garch_data_visualisation.plots import (
    compute_squared_acf,
    plot_residuals_distribution,
    plot_squared_residuals_acf,
    test_arch_effect,
)
from src.garch.garch_data_visualisation.utils import (
    extract_dates_from_dataframe,
    prepare_x_axis,
)

__all__ = [
    "plot_residuals_distribution",
    "plot_squared_residuals_acf",
    "compute_squared_acf",
    "test_arch_effect",
    "extract_dates_from_dataframe",
    "prepare_x_axis",
]
