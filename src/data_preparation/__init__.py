"""Data preparation module for dollar bars generation.

This module implements Dollar Bars following De Prado's methodology
from "Advances in Financial Machine Learning" (Chapter 2).
"""

from __future__ import annotations

from src.data_preparation.preparation import (
    add_log_returns_to_bars_file,
    compute_dollar_bars,
    create_empty_bars_df,
    generate_dollar_bars,
    prepare_dollar_bars,
    run_dollar_bars_pipeline,
    run_dollar_bars_pipeline_batch,
)

__all__ = [
    "add_log_returns_to_bars_file",
    "compute_dollar_bars",
    "create_empty_bars_df",
    "generate_dollar_bars",
    "prepare_dollar_bars",
    "run_dollar_bars_pipeline",
    "run_dollar_bars_pipeline_batch",
]
