"""Data cleaning module for BTC/USD tick data.

Provides robust outlier detection methods suitable for financial markets
and dollar bar construction (De Prado methodology).

All methods are CAUSAL (use only past data) to prevent temporal data leakage.

Submodules
----------
outliers : Robust outlier detection (MAD, Z-score, volume, dollar value)
parquet_io : Streaming parquet I/O operations
cleaning : Main cleaning pipeline orchestration
"""

from __future__ import annotations

from src.data_cleaning.cleaning import clean_ticks_data
from src.data_cleaning.outliers import OutlierReport

__all__ = [
    "OutlierReport",
    "clean_ticks_data",
]
