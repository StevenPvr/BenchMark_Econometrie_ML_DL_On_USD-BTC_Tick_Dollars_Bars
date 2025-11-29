"""Data cleaning module for BTC/USD tick data.

Provides robust outlier detection methods suitable for financial markets
and dollar bar construction (De Prado methodology).

Methods:
- MAD (Median Absolute Deviation) - robust to fat-tailed distributions
- Rolling Z-score - adapts to local volatility regimes
- Flash crash/spike detection - identifies transient price anomalies
- Volume anomaly detection - filters dust trades and manipulation
- Dollar value filtering - combined price*volume anomalies
"""

from __future__ import annotations

from src.data_cleaning.cleaning import (
    OutlierReport,
    clean_ticks_data,
)

__all__ = [
    "clean_ticks_data",
    "OutlierReport",
]
