"""GARCH diagnostic module.

Post-estimation diagnostics for GARCH models including:
- ACF/PACF plots for standardized residuals
- Ljung-Box tests
- Distribution adequacy tests
- ARCH-LM test
- Engle-Ng asymmetry tests
- Nyblom stability test
"""

from __future__ import annotations

# Import order: alphabetical by module name
from src.garch.garch_diagnostic.advanced_diagnostics import (
    compute_arch_lm_on_standardized,
    compute_comprehensive_diagnostics,
    compute_ljung_box_at_specific_lags,
    compute_standardized_residual_moments,
    nyblom_stability_test,
    save_comprehensive_diagnostics,
)
from src.garch.garch_diagnostic.diagnostics import (
    compute_distribution_diagnostics,
    compute_ljung_box_on_std,
    compute_ljung_box_on_std_squared,
    save_acf_pacf_std_plots,
    save_acf_pacf_std_squared_plots,
    save_histogram_std_residuals,
    save_qq_plot_std_residuals,
    save_residual_plots,
)
from src.garch.garch_diagnostic.io import (
    choose_best_params,
    extract_nu_from_params,
    extract_params_dict,
    load_and_prepare_residuals,
    load_data_and_params,
    load_estimation_file,
    save_diagnostics_json,
    validate_dict_field,
)
from src.garch.garch_diagnostic.standardization import standardize_residuals

__all__ = [
    # advanced_diagnostics
    "compute_arch_lm_on_standardized",
    "compute_comprehensive_diagnostics",
    "compute_ljung_box_at_specific_lags",
    "compute_standardized_residual_moments",
    "nyblom_stability_test",
    "save_comprehensive_diagnostics",
    # data_loading
    "choose_best_params",
    "extract_nu_from_params",
    "load_and_prepare_residuals",
    "load_data_and_params",
    "load_estimation_file",
    # diagnostics
    "compute_distribution_diagnostics",
    "compute_ljung_box_on_std",
    "compute_ljung_box_on_std_squared",
    "save_acf_pacf_std_plots",
    "save_acf_pacf_std_squared_plots",
    "save_histogram_std_residuals",
    "save_qq_plot_std_residuals",
    "save_residual_plots",
    # io
    "save_diagnostics_json",
    "validate_dict_field",
    "extract_params_dict",
    # standardization
    "standardize_residuals",
]
