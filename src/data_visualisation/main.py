#!/usr/bin/env python3
"""Main entry point for dollar bars visualisation."""

from __future__ import annotations

import os
import sys

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.path import DOLLAR_IMBALANCE_BARS_PARQUET, PLOTS_OUTPUT_DIR
from src.data_visualisation.visualisation import run_full_analysis


def main() -> None:
    """Execute the full dollar bars analysis with plots."""
    run_full_analysis(
        parquet_path=DOLLAR_IMBALANCE_BARS_PARQUET,
        output_dir=PLOTS_OUTPUT_DIR,
        show_plots=True,
    )


if __name__ == "__main__":
    main()
