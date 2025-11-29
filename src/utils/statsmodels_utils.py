"""Statsmodels utility functions."""

from __future__ import annotations

import warnings


def suppress_statsmodels_warnings() -> None:
    """Suppress common statsmodels warnings."""
    warnings.filterwarnings("ignore", module="statsmodels")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")
    warnings.filterwarnings("ignore", message="ConvergenceWarning")
