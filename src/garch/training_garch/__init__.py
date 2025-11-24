"""GARCH training module.

This module provides the final training pipeline for GARCH models.
Note: Batch estimation has been moved to garch_params module as it is
a preparation step that must run before hyperparameter optimization.
"""

from __future__ import annotations

from src.garch.training_garch.predictions_io import (
    load_estimation_results,
    load_garch_forecasts,
    save_estimation_results,
    save_garch_forecasts,
    save_ml_dataset,
)
from src.garch.training_garch.training import train_egarch_from_dataset
from src.garch.training_garch.variance_filter import VarianceFilter

__all__ = [
    # Training
    "train_egarch_from_dataset",
    # I/O
    "save_garch_forecasts",
    "save_ml_dataset",
    "save_estimation_results",
    "load_garch_forecasts",
    "load_estimation_results",
    # Diagnostics
    "VarianceFilter",
]
