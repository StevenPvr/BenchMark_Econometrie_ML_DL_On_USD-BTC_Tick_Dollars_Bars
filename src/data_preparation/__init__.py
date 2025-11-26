"""Data preparation module for financial data pipelines.

This module provides:
- Dollar Bars generation (De Prado methodology)
- Log returns computation
- Data validation utilities
"""

from __future__ import annotations

from src.data_preparation.preparation import (
    compute_dollar_bars,
    generate_dollar_bars,
    prepare_dollar_bars,
    run_dollar_bars_pipeline,
)

__all__ = [
    # Dollar Bars (De Prado methodology)
    "compute_dollar_bars",
    "generate_dollar_bars",
    "prepare_dollar_bars",
    "run_dollar_bars_pipeline",
]
