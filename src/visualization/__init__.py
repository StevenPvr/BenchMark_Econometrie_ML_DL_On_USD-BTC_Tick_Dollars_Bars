"""Visualization utilities package.

Provides common plotting utilities for ARIMA and GARCH visualizations
to maintain DRY principles across the codebase.
"""

from __future__ import annotations

from .plotting_utils import (
    add_confidence_bands,
    add_grid,
    add_legend,
    add_metrics_textbox,
    add_statistics_textbox,
    add_zero_line,
    clean_array,
    create_figure_canvas,
    create_standard_figure,
    ensure_output_directory,
    format_date_axis,
    format_seasonal_axis_daily,
    get_color_palette,
    get_test_result_style,
    load_json_if_exists,
    plot_histogram_with_normal_overlay,
    plot_qq_normal,
    plot_series_with_train_test_split,
    prepare_temporal_axis,
    save_canvas,
    save_figure,
    save_plot_wrapper,
    setup_plot_style,
    subsample_for_plotting,
    validate_plot_arrays,
)

__all__ = [
    # Figure creation
    "create_figure_canvas",
    "create_standard_figure",
    # Saving
    "save_figure",
    "save_canvas",
    "save_plot_wrapper",
    # Date formatting
    "format_date_axis",
    "format_seasonal_axis_daily",
    # Data preparation
    "prepare_temporal_axis",
    "validate_plot_arrays",
    "clean_array",
    "subsample_for_plotting",
    # Plot elements
    "add_zero_line",
    "add_confidence_bands",
    "add_grid",
    "add_legend",
    "add_statistics_textbox",
    "add_metrics_textbox",
    # Generic plotting functions
    "plot_histogram_with_normal_overlay",
    "plot_qq_normal",
    "plot_series_with_train_test_split",
    # Styling
    "setup_plot_style",
    "get_color_palette",
    "get_test_result_style",
    # Utilities
    "ensure_output_directory",
    "load_json_if_exists",
]
