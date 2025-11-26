"""Trade classification utilities for order flow analysis.

This module provides methods to classify trades as buy-initiated or sell-initiated:
1. Direct side: Use exchange-provided 'side' field (most accurate)
2. Tick rule: Price up → buy, price down → sell
3. Quote rule: Price > mid → buy, price < mid → sell

Reference:
    Lee, C., & Ready, M. (1991). Inferring Trade Direction from Intraday Data.
    Journal of Finance, 46(2), 733-746.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.config_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "classify_trades_tick_rule",
    "classify_trades_direct",
]


@njit(cache=True)
def _tick_rule_classify(prices: NDArray[np.float64]) -> NDArray[np.int8]:
    """Classify trades using tick rule (numba optimized).

    Tick rule:
    - Price up from previous trade → buy (+1)
    - Price down from previous trade → sell (-1)
    - Price unchanged → use previous classification

    Args:
        prices: Array of trade prices.

    Returns:
        Array of trade classifications: +1 (buy), -1 (sell).
    """
    n = len(prices)
    signs = np.zeros(n, dtype=np.int8)

    if n == 0:
        return signs

    # First trade: assume buy
    signs[0] = 1
    last_sign = 1

    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            signs[i] = 1
            last_sign = 1
        elif diff < 0:
            signs[i] = -1
            last_sign = -1
        else:
            # Price unchanged: use previous classification
            signs[i] = last_sign

    return signs


def classify_trades_tick_rule(df: pd.DataFrame, price_col: str = "price") -> pd.Series:
    """Classify trades as buy/sell using the tick rule.

    Args:
        df: DataFrame with trade data.
        price_col: Name of price column.

    Returns:
        Series with +1 (buy) or -1 (sell) for each trade.
    """
    prices = df[price_col].values.astype(np.float64)
    signs = _tick_rule_classify(prices)
    return pd.Series(signs, index=df.index, name="trade_sign")


def classify_trades_direct(df: pd.DataFrame, side_col: str = "side") -> pd.Series:
    """Classify trades using exchange-provided side field.

    Args:
        df: DataFrame with trade data containing side column.
        side_col: Name of side column (expected values: 'buy', 'sell').

    Returns:
        Series with +1 (buy) or -1 (sell) for each trade.
    """
    mapping: dict[str, int] = {"buy": 1, "sell": -1}
    signs = df[side_col].map(mapping).fillna(0).astype(np.int8)  # type: ignore[arg-type]
    return pd.Series(signs, index=df.index, name="trade_sign")
