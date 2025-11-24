"""GARCH structure detection module.

Provides utilities for ARCH/GARCH identification:
- Loading GARCH datasets
- Running Engle's ARCH-LM test
- Detecting heteroskedasticity

"""

from __future__ import annotations

from src.garch.structure_garch.detection import detect_heteroskedasticity
from src.garch.structure_garch.utils import (
    chi2_sf,
    compute_acf,
    compute_arch_lm_test,
    load_garch_dataset,
    prepare_residuals,
)

__all__ = [
    # Main detection functions
    "detect_heteroskedasticity",
    # Utility functions
    "compute_acf",
    "compute_arch_lm_test",
    "chi2_sf",
    "load_garch_dataset",
    "prepare_residuals",
]
