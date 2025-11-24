"""Evaluation helpers for ARIMA (keeps external `src.*` imports intact)."""

from __future__ import annotations


import pandas as pd

from src.utils import get_logger  # type: ignore

logger = get_logger(__name__)


def detect_value_column(df: pd.DataFrame) -> str:
    """Detect the value column name from DataFrame.

    Args:
        df: DataFrame with data columns.

    Returns:
        Column name for target values.

    Raises:
        ValueError: If no suitable column found.
    """
    candidates = [
        c for c in df.columns if c.lower() in {"y", "target", "log_return", "weighted_log_return"}
    ]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if "return" in c.lower() and c not in {"date", "split"}:
            return c
    msg = (
        "Could not identify the returns column "
        "(expected one of: y, target, log_return, weighted_log_return)."
    )
    raise ValueError(msg)
