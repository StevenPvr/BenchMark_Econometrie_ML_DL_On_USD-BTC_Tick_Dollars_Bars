"""I/O utilities for data visualisation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import CLOSE_COLUMN
from src.path import DOLLAR_BARS_PARQUET
from src.utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "load_dollar_bars",
    "compute_log_returns",
]


def load_dollar_bars(parquet_path: Path = DOLLAR_BARS_PARQUET) -> pd.DataFrame:
    """
    Load dollar bars from a parquet file.

    Args:
        parquet_path: Path to the parquet file containing dollar bars.

    Returns:
        DataFrame containing the dollar bars data.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dollar bars file not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def compute_log_returns(df: pd.DataFrame, price_col: str = CLOSE_COLUMN) -> pd.Series:
    """
    Compute log-returns from closing prices.

    Calculates: log_return_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Args:
        df: DataFrame containing the dollar bars.
        price_col: Name of the price column. Defaults to "close".

    Returns:
        Series of log-returns (first value is NaN).
    """
    prices = df[price_col]
    log_returns: pd.Series = np.log(prices / prices.shift(1))
    return log_returns
