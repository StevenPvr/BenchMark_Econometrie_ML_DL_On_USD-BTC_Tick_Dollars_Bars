"""Utility functions for GARCH evaluation.

This module provides backward compatibility imports from refactored modules.
All functions have been moved to specialized modules for better organization.
"""

from __future__ import annotations

from src.garch.garch_eval.data_loading import (
    extract_aligned_test_indices,
    load_and_prepare_residuals,
    load_dataset_for_metrics,
    load_model_params,
    prepare_residuals_from_dataset,
)

# Import all utilities from specialized modules for backward compatibility
from src.garch.garch_eval.helpers import (
    assemble_forecast_results,
    chi2_sf,
    load_best_model,
    parse_alphas,
    save_forecast_results,
    to_numpy,
)
from src.garch.garch_eval.metrics import (
    add_comparison_metrics,
    apply_mz_calibration_if_requested,
    build_var_series,
    compute_all_metrics,
    compute_variance_metrics,
    quantile,
    var_quantile,
)
from src.garch.garch_eval.models import (
    aic,
    choose_best_from_estimation,
    collect_converged_candidates,
)
from src.garch.garch_eval.mz_calibration import (
    apply_mz_calibration_to_forecasts,
    compute_mz_pvalues,
    filter_test_data,
    load_test_resid_sigma2,
)
from src.garch.garch_eval.variance_path import (
    compute_egarch_forecasts,
    compute_initial_forecasts,
    compute_variance_path,
    compute_variance_path_for_test,
)
from src.utils import ensure_output_dir

__all__ = [
    # Quantiles
    "quantile",
    "var_quantile",
    "build_var_series",
    # Model selection
    "aic",
    "collect_converged_candidates",
    "choose_best_from_estimation",
    # Data loading
    "load_model_params",
    "load_dataset_for_metrics",
    "load_and_prepare_residuals",
    "prepare_residuals_from_dataset",
    "extract_aligned_test_indices",
    # Variance path
    "compute_variance_path",
    "compute_variance_path_for_test",
    "compute_egarch_forecasts",
    "compute_initial_forecasts",
    # MZ calibration
    "compute_mz_pvalues",
    "filter_test_data",
    "load_test_resid_sigma2",
    "apply_mz_calibration_to_forecasts",
    # Statistical
    "chi2_sf",
    # File operations
    "save_forecast_results",
    "ensure_output_dir",
    "to_numpy",
    # Parsing
    "parse_alphas",
    "load_best_model",
    # Assembly
    "assemble_forecast_results",
    # Metrics computation
    "compute_variance_metrics",
    "apply_mz_calibration_if_requested",
    "add_comparison_metrics",
    "compute_all_metrics",
]
